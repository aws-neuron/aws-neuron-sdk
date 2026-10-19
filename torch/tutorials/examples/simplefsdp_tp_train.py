import argparse
import copy

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint
from training_common import (
    GLOBAL_BATCH_SIZE,
    HIDDEN_FEATURES,
    INPUT_FEATURES,
    LEARNING_RATE,
    TP_DEGREE,
    add_run_arguments,
    full_cpu,
    initialize,
    make_batch,
    make_model,
    save_results,
    train,
)

COMPILE_BACKEND = "neuron"


# BEGIN SIMPLEFSDP
class SimpleFSDPWeight(torch.nn.Module):
    """Gather a DP-sharded TP weight through differentiable DTensor operations."""

    def __init__(self, mesh, shape, dp_dim, tp_dim):
        super().__init__()
        self.mesh = mesh
        self.dp_mesh = mesh["dp"]
        self.tp_mesh = mesh["tp"]
        self.shape = shape
        self.dp_dim = dp_dim
        self.tp_dim = tp_dim
        tp_shape = list(shape)
        tp_shape[tp_dim] //= self.tp_mesh.size()
        self.tp_shape = tuple(tp_shape)

    def gather(self, weight):
        # Collapse to the DP submesh, gather this TP partition, then restore TP.
        dp_weight = DTensor.from_local(
            weight.to_local(),
            self.dp_mesh,
            (Shard(self.dp_dim),),
            shape=self.tp_shape,
            stride=(self.tp_shape[1], 1),
        )
        tp_local = dp_weight.redistribute(placements=(Replicate(),)).to_local(
            grad_placements=(Partial("sum"),)
        )
        return DTensor.from_local(
            tp_local,
            self.tp_mesh,
            (Shard(self.tp_dim),),
            shape=self.shape,
            stride=(self.shape[1], 1),
        )

    def forward(self, weight):
        # Recompute the gather in backward instead of retaining full weights.
        return checkpoint(self.gather, weight, use_reentrant=False)


class SimpleFSDPTP(torch.nn.Module):
    def __init__(self, reference, mesh, device="neuron"):
        super().__init__()
        self.layers = copy.deepcopy(reference)
        dp_rank, tp_rank = mesh.get_coordinate()
        for index, dp_dim, tp_dim in [(0, 1, 0), (2, 0, 1)]:
            weight = self.layers[index].weight.detach()
            local = weight.chunk(mesh["tp"].size(), dim=tp_dim)[tp_rank]
            local = local.chunk(mesh["dp"].size(), dim=dp_dim)[dp_rank].contiguous()
            self.layers[index].weight = torch.nn.Parameter(
                DTensor.from_local(
                    local.clone().to(device),
                    mesh,
                    (Shard(dp_dim), Shard(tp_dim)),
                    shape=weight.shape,
                    stride=weight.stride(),
                )
            )
            parametrize.register_parametrization(
                self.layers[index],
                "weight",
                SimpleFSDPWeight(mesh, tuple(weight.shape), dp_dim, tp_dim),
                unsafe=True,
            )

    def forward(self, inputs):
        output = self.layers(inputs)
        # Row-parallel output is partial across TP; all-reduce it in this graph.
        return output.redistribute(placements=(Replicate(),)).to_local(
            grad_placements=(Replicate(),)
        )


# END SIMPLEFSDP
def main():
    parser = argparse.ArgumentParser()
    add_run_arguments(parser)
    parser.add_argument("--tp-degree", type=int, default=TP_DEGREE)
    parser.add_argument("--compile-backend", default=COMPILE_BACKEND)
    args = parser.parse_args()
    rank, world_size, device = initialize(args)
    try:
        if args.tp_degree < 1 or world_size % args.tp_degree:
            raise ValueError("The TP degree must divide the world size.")
        dp_degree = world_size // args.tp_degree
        if GLOBAL_BATCH_SIZE % dp_degree:
            raise ValueError("The global batch must divide evenly across DP workers.")
        if HIDDEN_FEATURES % args.tp_degree or INPUT_FEATURES % dp_degree:
            raise ValueError("Weight dimensions must divide evenly across the mesh.")
        mesh = init_device_mesh(
            args.device, (dp_degree, args.tp_degree), mesh_dim_names=("dp", "tp")
        )
        dp_rank, tp_rank = mesh.get_coordinate()
        initial = make_model("simplefsdp", args.seed)
        sharded = SimpleFSDPTP(initial, mesh, device)
        parameters = list(sharded.parameters())
        assert all(isinstance(p, DTensor) for p in parameters)
        assert tuple(parameters[0].to_local().shape) == (
            HIDDEN_FEATURES // args.tp_degree,
            INPUT_FEATURES // dp_degree,
        )
        assert tuple(parameters[1].to_local().shape) == (
            INPUT_FEATURES // dp_degree,
            HIDDEN_FEATURES // args.tp_degree,
        )
        model = torch.compile(sharded, backend=args.compile_backend, fullgraph=True)
        optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE)

        def train_step(step):
            batch, target = make_batch(step)
            local_batch = batch.chunk(dp_degree)[dp_rank].contiguous()
            local_target = target.chunk(dp_degree)[dp_rank].contiguous().to(device)
            inputs = DTensor.from_local(local_batch.to(device), mesh["tp"], (Replicate(),))
            output = model(inputs)
            loss = torch.nn.functional.mse_loss(output, local_target)
            # SimpleFSDP sums gradients across DP; normalize the local mean.
            (loss / dp_degree).backward()
            global_loss = loss.detach().clone()
            dist.all_reduce(global_loss, group=mesh["dp"].get_group())
            global_loss.div_(dp_degree)
            return output, global_loss

        records = train(parameters, optimizer, args.steps, train_step)
        save_results(
            args,
            records,
            model="simplefsdp",
            mode="simplefsdp_tp",
            rank=rank,
            world_size=world_size,
            dp_degree=dp_degree,
            dp_rank=dp_rank,
            tp_degree=args.tp_degree,
            tp_rank=tp_rank,
        )
        dist.barrier()
        print(f"PASS rank={rank} mode=simplefsdp_tp world_size={world_size}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
