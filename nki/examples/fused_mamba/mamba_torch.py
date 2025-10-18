"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

Mamba-v1 PyTorch Reference Implementation.

"""

# NKI_EXAMPLE_24_BEGIN
import torch
import torch_neuronx
import torch_xla.core.xla_model as xm
import os
import argparse

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --model-type=transformer --disable-dge "


def associative_scan(deltaA, deltaB_u):
    """
    Args:
        deltaA: [batch_size, channels, state_size, seq_len]
        deltaB_u: [batch_size, channels, state_size, seq_len]

    Mamba uses an associative scan operator to aggregate information across
    time sequentially (sequence length, e.g. sequence of tokens),
    from the past to the present.
    """
    batch_size, channels, state_size, seq_len = deltaA.shape
    out = torch.empty(batch_size, channels, state_size, seq_len,
                        device=deltaA.device, dtype=deltaA.dtype)
    for i in range(seq_len):
        prev_state = out[..., i - 1] if i > 0 else 0
        out[..., i] = deltaA[..., i] * prev_state + deltaB_u[..., i]
    return out


def mamba_layer(delta, A, B, u, C):
    """
    Args:
        delta: [batch, channels, seq_len]
        u: [batch, channels, seq_len]
        A: [channels, state_size]
        B: [batch, state_size, seq_len]
        C: [batch, state_size, seq_len]
    """
    # expand the tensors so they all have the same dimensions and compute elementwise products (with broadcast)
    # deltaA and deltaB_u have shape [batch_size, channels, state_size, seq_len]
    deltaA = torch.exp(delta[:, :, None, :] * A[None, :, :, None])
    deltaB_u = delta[:, :, None, :] * B[:, None, :, :] * u[:, :, None, :]
    scan_res = associative_scan(deltaA, deltaB_u)
    # y sums over the `state_size` axis and has shape [batch_size, channels, seq_len]
    mamba_out = (C[:, None, :, :] * scan_res).sum(dim=-2)
    return mamba_out


def parse_args():
    parser = argparse.ArgumentParser(
    """Run Mamba PyTorch implementation. Hard-coded small example only since
       PyTorch implementation is very slow for larger configs.
    """)
    parser.add_argument("--mode",
                        choices=["accuracy", "perf"],
                        default="accuracy",
                        help="""Do accuracy test or perf test.
                                Accuracy test compares mamba_v1 kernel against PyTorch implementation.
                                Perf test will generate a NEFF for the PyTorch implementation in local directory
                                for a manual run of neuron-profile.
                             """)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Toy example
    batch = 1
    seq_len = 512
    channels = 256
    state_size = 16

    dtype = torch.float32

    device = xm.xla_device()

    delta = torch.ones(batch, channels, seq_len, dtype=dtype, device=device)
    u = torch.ones(batch, channels, seq_len, dtype=dtype, device=device)

    # For numerical accuracy testing purposes, we choose negative numbers for A on purpose.
    # Otherwise, the associative scan will integrate too fast and overflow, which would
    # mask any real numerical issues in our computation.
    # A negative A will ensure we catch numerical issues when we have them.
    A = -torch.ones(channels, state_size, dtype=dtype, device=device)
    B = torch.ones(batch, state_size, seq_len, dtype=dtype, device=device)

    C = torch.ones(batch, state_size, seq_len, dtype=dtype, device=device)

    xm.mark_step()
    torch_out = mamba_layer(delta, A, B, u, C)
    xm.mark_step()
    print(torch_out)
    # NKI_EXAMPLE_24_END

    if args.mode == "accuracy":
        # Call NKI mamba_v1 kernel to check accuracy
        from mamba_nki_kernels import mamba_v1

        xm.mark_step()
        nki_out = mamba_v1(delta, u, A, B, C)
        xm.mark_step()

        allclose = torch.allclose(torch_out, nki_out, atol=1e-2, rtol=1e-2)

        if allclose:
            print("NKI and Torch match")
        else:
            print("NKI and Torch differ")

        assert allclose
