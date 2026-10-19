"""Model, data, and training logic shared by CPU and Neuron distributed runs."""

from pathlib import Path

import torch
from torch.distributed.tensor import DTensor

# Edit these defaults or use the command-line options in the launch scripts.
DEVICE_TYPE = "neuron"
PROCESS_GROUP_BACKEND = "neuron"
RANKS_PER_NODE = 64
TP_DEGREE = 4
SEED = 0
STEPS = 10
GLOBAL_BATCH_SIZE = 512
INPUT_FEATURES = 128
HIDDEN_FEATURES = 256
LEARNING_RATE = 0.01


def make_model(kind, seed):
    torch.manual_seed(seed)

    def block():
        return torch.nn.Sequential(
            torch.nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES, bias=(kind == "eager")),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_FEATURES, INPUT_FEATURES, bias=(kind == "eager")),
        )

    return torch.nn.Sequential(block(), block()) if kind == "pp" else block()


def make_batch(step):
    batch = torch.arange(GLOBAL_BATCH_SIZE * INPUT_FEATURES, dtype=torch.float32).reshape(
        GLOBAL_BATCH_SIZE, INPUT_FEATURES
    )
    batch = batch.remainder(257) / INPUT_FEATURES - 1 + step / 10
    return batch, batch.cos()


def full_cpu(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.detach().cpu().clone()


def train(parameters, optimizer, steps, train_step):
    """Run one model; train_step performs its forward and backward computation."""
    if steps < 1:
        raise ValueError("Use at least one training step.")
    records = []
    for step in range(steps):
        optimizer.zero_grad()
        before = [full_cpu(p) for p in parameters]
        output, loss = train_step(step)
        if any(p.grad is None for p in parameters):
            raise AssertionError("A model parameter has no gradient.")
        gradients = [full_cpu(p.grad) for p in parameters]
        optimizer.step()
        updated = [full_cpu(p) for p in parameters]
        deltas = [value - initial for value, initial in zip(updated, before, strict=True)]
        if not any(bool(torch.count_nonzero(delta)) for delta in deltas):
            raise AssertionError("The optimizer did not update any parameters.")
        records.append(
            {
                "output": None if output is None else full_cpu(output),
                "loss": None if loss is None else float(loss.detach().cpu()),
                "gradients": gradients,
                "parameters": updated,
                "deltas": deltas,
            }
        )
        print(f"step={step + 1}/{steps} training=completed", flush=True)
    return records


def add_run_arguments(
    parser, expected_world_size=RANKS_PER_NODE, device=DEVICE_TYPE, backend=PROCESS_GROUP_BACKEND
):
    parser.add_argument("--device", choices=("cpu", "neuron"), default=device)
    parser.add_argument("--backend", default=backend)
    parser.add_argument("--expected-world-size", type=int, default=expected_world_size)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output-dir", required=True)


def initialize(args):
    import os
    from datetime import timedelta

    import torch.distributed as dist

    if args.device == "neuron":
        import torch_neuronx
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.expected_world_size:
        raise ValueError(f"Launch exactly {args.expected_world_size} workers.")
    dist.init_process_group(backend=args.backend, timeout=timedelta(minutes=10))
    return int(os.environ["RANK"]), world_size, torch.device(args.device)


def save_results(args, records, **metadata):
    path = Path(args.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    rank = metadata.get("rank", 0)
    torch.save(
        {
            "seed": args.seed,
            "steps": args.steps,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "device": args.device,
            "backend": args.backend,
            **metadata,
            "records": records,
        },
        path / f"rank-{rank}.pt",
    )
