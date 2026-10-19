import argparse

import torch
import torch.distributed as dist
from simplefsdp_tp_train import COMPILE_BACKEND, SimpleFSDPTP
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from torch.distributed.tensor import DTensor, Replicate
from training_common import (
    GLOBAL_BATCH_SIZE,
    HIDDEN_FEATURES,
    INPUT_FEATURES,
    LEARNING_RATE,
    RANKS_PER_NODE,
    TP_DEGREE,
    add_run_arguments,
    initialize,
    make_batch,
    make_model,
    save_results,
    train,
)

PP_DEGREE = 2
MICROBATCHES = 4


class PipelineBlock(SimpleFSDPTP):
    """Keep pipeline boundaries as local tensors; use DTensor within each stage."""

    def __init__(self, reference, mesh, device):
        super().__init__(reference, mesh, device)
        self.tp_mesh = mesh["tp"]

    def forward(self, inputs):
        inputs = DTensor.from_local(inputs, self.tp_mesh, (Replicate(),))
        return super().forward(inputs)


def main():
    parser = argparse.ArgumentParser()
    add_run_arguments(parser, expected_world_size=PP_DEGREE * RANKS_PER_NODE)
    parser.add_argument("--tp-degree", type=int, default=TP_DEGREE)
    parser.add_argument("--microbatches", type=int, default=MICROBATCHES)
    parser.add_argument("--compile-backend", default=COMPILE_BACKEND)
    args = parser.parse_args()
    rank, world_size, device = initialize(args)
    try:
        if args.tp_degree < 1 or world_size % (PP_DEGREE * args.tp_degree):
            raise ValueError("The world size must divide into two stages and TP groups.")
        dp_degree = world_size // (PP_DEGREE * args.tp_degree)
        if GLOBAL_BATCH_SIZE % dp_degree:
            raise ValueError("The global batch must divide evenly across DP workers.")
        if HIDDEN_FEATURES % args.tp_degree or INPUT_FEATURES % dp_degree:
            raise ValueError("Weight dimensions must divide evenly across the mesh.")
        local_batch_size = GLOBAL_BATCH_SIZE // dp_degree
        if args.microbatches < 1 or local_batch_size % args.microbatches:
            raise ValueError("The local batch must divide evenly into microbatches.")
        microbatch_size = local_batch_size // args.microbatches
        mesh = init_device_mesh(
            args.device,
            (PP_DEGREE, dp_degree, args.tp_degree),
            mesh_dim_names=("pp", "dp", "tp"),
        )
        pp_rank, dp_rank, tp_rank = mesh.get_coordinate()
        initial = make_model("pp", args.seed)
        sharded = PipelineBlock(initial[pp_rank], mesh["dp", "tp"], device)
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

        # BEGIN PIPELINE
        model = torch.compile(sharded, backend=args.compile_backend, fullgraph=True)
        device = parameters[0].device
        stage = PipelineStage(
            model,
            stage_index=pp_rank,
            num_stages=PP_DEGREE,
            device=device,
            input_args=torch.empty(
                microbatch_size, INPUT_FEATURES, device=device, requires_grad=(pp_rank > 0)
            ),
            output_args=torch.empty(
                microbatch_size, INPUT_FEATURES, device=device, requires_grad=True
            ),
            group=mesh["pp"].get_group(),
        )

        def loss_fn(output, target):
            # SimpleFSDP sums across DP ranks.
            return torch.nn.functional.mse_loss(output, target) / dp_degree

        schedule = Schedule1F1B(
            stage, n_microbatches=args.microbatches, loss_fn=loss_fn, scale_grads=True
        )
        # END PIPELINE
        optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE)

        def train_step(step):
            batch, target = make_batch(step)
            local_batch = batch.chunk(dp_degree)[dp_rank].contiguous().to(device)
            local_target = target.chunk(dp_degree)[dp_rank].contiguous().to(device)
            if pp_rank == 0:
                schedule.step(local_batch)
                return None, None
            losses = []
            output = schedule.step(target=local_target, losses=losses)
            # Losses include /dp_degree. Averaging microbatches then summing
            # across DP restores the global mean; scale_grads averages grads.
            global_loss = torch.stack(losses).mean().detach()
            dist.all_reduce(global_loss, group=mesh["dp"].get_group())
            return output, global_loss

        records = train(parameters, optimizer, args.steps, train_step)
        save_results(
            args,
            records,
            model="pp",
            mode="simplefsdp_tp_pp",
            rank=rank,
            world_size=world_size,
            dp_degree=dp_degree,
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            tp_degree=args.tp_degree,
            tp_rank=tp_rank,
        )
        dist.barrier()
        print(f"PASS rank={rank} mode=simplefsdp_tp_pp world_size={world_size}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
