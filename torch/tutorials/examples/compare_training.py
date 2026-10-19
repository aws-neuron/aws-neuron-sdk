"""Compare separately collected CPU and Neuron distributed training results."""

import argparse
from pathlib import Path

import torch


def compare(reference_dir, actual_dir, world_size):
    reference_paths = list(Path(reference_dir).glob("rank-*.pt"))
    if len(reference_paths) != world_size:
        raise AssertionError(
            f"Expected {world_size} CPU result files, found {len(reference_paths)}."
        )
    paths = list(Path(actual_dir).glob("rank-*.pt"))
    if len(paths) != world_size:
        raise AssertionError(f"Expected {world_size} result files, found {len(paths)}.")
    losses = []
    reference_losses = []
    seen = set()
    configuration = None
    for path in sorted(paths):
        actual = torch.load(path, weights_only=True)
        rank = actual["rank"]
        assert rank not in seen and 0 <= rank < world_size
        seen.add(rank)
        assert actual["world_size"] == world_size
        assert path.name == f"rank-{rank}.pt"
        mode = actual["mode"]
        config = tuple(
            actual[key]
            for key in (
                "mode",
                "dp_degree",
                "tp_degree",
                "model",
                "seed",
                "steps",
                "global_batch_size",
            )
        )
        if configuration is None:
            configuration = config
        assert config == configuration, "Inconsistent mesh configuration."
        if mode in ("ddp", "fsdp"):
            assert actual["model"] == "eager"
            assert actual["dp_degree"] == world_size and actual["tp_degree"] == 1
            expected_dp, expected_tp, expected_pp = rank, 0, 0
        elif mode == "tp":
            assert actual["model"] == "eager"
            assert actual["dp_degree"] == 1 and actual["tp_degree"] == world_size
            expected_dp, expected_tp, expected_pp = 0, rank, 0
        elif mode in ("simplefsdp_tp", "simplefsdp_tp_pp"):
            assert actual["model"] == ("pp" if mode.endswith("_pp") else "simplefsdp")
            pp_degree = 2 if actual["model"] == "pp" else 1
            dp_degree, tp_degree = actual["dp_degree"], actual["tp_degree"]
            assert dp_degree > 0 and tp_degree > 0
            assert pp_degree * dp_degree * tp_degree == world_size
            expected_tp = rank % tp_degree
            expected_dp = (rank // tp_degree) % dp_degree
            expected_pp = rank // (dp_degree * tp_degree)
        else:
            raise ValueError(f"Unknown distributed mode: {mode}.")
        assert actual["dp_rank"] == expected_dp
        assert actual["tp_rank"] == expected_tp
        assert actual.get("pp_rank", 0) == expected_pp
        reference_path = Path(reference_dir) / path.name
        if not reference_path.is_file():
            raise AssertionError(f"Missing CPU result file: {reference_path}.")
        reference = torch.load(reference_path, weights_only=True)
        for key in (
            "seed",
            "steps",
            "global_batch_size",
            "model",
            "mode",
            "rank",
            "world_size",
            "dp_degree",
            "dp_rank",
            "tp_degree",
            "tp_rank",
        ):
            assert actual[key] == reference[key], f"Mismatched {key} in {path}."
        assert actual.get("pp_rank", 0) == reference.get("pp_rank", 0)
        assert reference["device"] == "cpu" and reference["backend"] == "gloo"
        assert actual["device"] in ("cpu", "neuron")
        assert actual["backend"] == {"cpu": "gloo", "neuron": "neuron"}[actual["device"]]
        assert len(actual["records"]) == len(reference["records"]) == reference["steps"]
        for step, (record, expected) in enumerate(
            zip(actual["records"], reference["records"], strict=True)
        ):
            if mode == "simplefsdp_tp_pp" and expected_pp == 0:
                assert record["output"] is None and record["loss"] is None
                assert expected["output"] is None and expected["loss"] is None
            else:
                assert record["output"] is not None and record["loss"] is not None
                assert expected["output"] is not None and expected["loss"] is not None
                torch.testing.assert_close(
                    record["output"], expected["output"], rtol=3e-3, atol=3e-4
                )
                torch.testing.assert_close(
                    torch.tensor(record["loss"]),
                    torch.tensor(expected["loss"]),
                    rtol=3e-3,
                    atol=3e-4,
                )
                if not losses:
                    losses = [None] * reference["steps"]
                    reference_losses = [None] * reference["steps"]
                if losses[step] is None:
                    losses[step] = record["loss"]
                    reference_losses[step] = expected["loss"]
            for key in ("gradients", "parameters", "deltas"):
                values = expected[key]
                for value, expected_value in zip(record[key], values, strict=True):
                    torch.testing.assert_close(
                        value,
                        expected_value,
                        rtol=1e-2 if key == "deltas" else 3e-3,
                        atol=2e-6 if key == "deltas" else 3e-4,
                    )
        print(f"PASS rank={rank} steps={reference['steps']} parity=passed", flush=True)
    assert all(loss is not None for loss in losses)
    print("| Step | CPU loss | Distributed loss |")
    print("| --- | ---: | ---: |")
    for step, (reference_loss, loss) in enumerate(zip(reference_losses, losses, strict=True), 1):
        print(f"| {step} | {reference_loss:.9f} | {loss:.9f} |")
    print(f"PASS comparison ranks={world_size} steps={reference['steps']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--actual-dir", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    compare(args.reference_dir, args.actual_dir, args.world_size)


if __name__ == "__main__":
    main()
