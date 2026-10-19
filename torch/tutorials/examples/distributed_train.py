"""Run the same distributed training plan on CPU or Neuron."""

import argparse

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.nn.parallel import DistributedDataParallel
from training_common import (
    DEVICE_TYPE,
    GLOBAL_BATCH_SIZE,
    HIDDEN_FEATURES,
    INPUT_FEATURES,
    LEARNING_RATE,
    PROCESS_GROUP_BACKEND,
    add_run_arguments,
    initialize,
    make_batch,
    make_model,
    save_results,
    train,
)


def main(default_device=DEVICE_TYPE, default_backend=PROCESS_GROUP_BACKEND):
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallelism", choices=("ddp", "fsdp", "tp"), required=True)
    add_run_arguments(parser, device=default_device, backend=default_backend)
    args = parser.parse_args()
    rank, world_size, device = initialize(args)
    try:
        model = make_model("eager", args.seed).to(device)
        data_parallel = args.parallelism in ("ddp", "fsdp")
        if data_parallel and GLOBAL_BATCH_SIZE % world_size:
            raise ValueError("The global batch must divide evenly across DP workers.")
        if args.parallelism != "ddp" and (
            HIDDEN_FEATURES % world_size or INPUT_FEATURES % world_size
        ):
            raise ValueError("Linear dimensions must divide evenly across the mesh.")
        if args.parallelism == "ddp":
            model = DistributedDataParallel(model)
        elif args.parallelism == "fsdp":
            mesh = init_device_mesh(args.device, (world_size,), mesh_dim_names=("dp",))
            # Layers 0 and 2 are Linear modules with parameters. Layer 1 is
            # ReLU: it has no parameters to shard, so it needs no FSDP wrapper.
            fully_shard(model[0], mesh=mesh, reshard_after_forward=True)
            fully_shard(model[2], mesh=mesh, reshard_after_forward=True)
            fully_shard(model, mesh=mesh)
        else:
            mesh = init_device_mesh(args.device, (world_size,), mesh_dim_names=("tp",))
            model = parallelize_module(
                model, mesh, {"0": ColwiseParallel(), "2": RowwiseParallel()}
            )
        if args.parallelism != "ddp":
            assert isinstance(model[0].weight, DTensor)
            assert isinstance(model[2].weight, DTensor)
            assert tuple(model[0].weight.to_local().shape) == (
                HIDDEN_FEATURES // world_size,
                INPUT_FEATURES,
            )
            shape = (
                (INPUT_FEATURES // world_size, HIDDEN_FEATURES)
                if args.parallelism == "fsdp"
                else (INPUT_FEATURES, HIDDEN_FEATURES // world_size)
            )
            assert tuple(model[2].weight.to_local().shape) == shape
        parameters = list(model.parameters())
        optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE)

        def train_step(step):
            batch, target = make_batch(step)
            if data_parallel:
                batch = batch.chunk(world_size)[rank].contiguous()
                target = target.chunk(world_size)[rank].contiguous()
            output = model(batch.to(device))
            loss = torch.nn.functional.mse_loss(output, target.to(device))
            loss.backward()
            global_loss = loss.detach().clone()
            if data_parallel:
                dist.all_reduce(global_loss)
                global_loss.div_(world_size)
            return output, global_loss

        records = train(parameters, optimizer, args.steps, train_step)
        save_results(
            args,
            records,
            model="eager",
            mode=args.parallelism,
            rank=rank,
            world_size=world_size,
            dp_degree=world_size if data_parallel else 1,
            dp_rank=rank if data_parallel else 0,
            tp_degree=1 if data_parallel else world_size,
            tp_rank=0 if data_parallel else rank,
        )
        dist.barrier()
        print(f"PASS rank={rank} mode={args.parallelism} world_size={world_size}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
