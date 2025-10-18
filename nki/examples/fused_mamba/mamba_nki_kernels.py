"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

Mamba-v1 NKI kernel implementation.

"""
# NKI_EXAMPLE_25_BEGIN
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np
# NKI_EXAMPLE_25_END
import os
import argparse
import itertools

# NKI_EXAMPLE_25_BEGIN
@nki.jit
def mamba_v1(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)

    _, state_size = A.shape

    # We can relax this using mask paramters in all the NKI API calls
    assert channels % 128 == 0

    # Map channels to the partition dimension
    # Tile channels to comply with NKI tile size constraints
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    # Most outer loop with batch_size, parallel_for
    for i_batch in nl.affine_range(batch_size):
        # partial accumulated scanC result with processed states
        scanC_accum = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

        # Second outer loop with state_size, partial parallel
        for i_state in nl.affine_range(state_size):

            # Inner loop: tiling channels
            for i_channel_tile in nl.affine_range(n_channel_tile):
                channel_start = i_channel_tile * channel_psize

                # Load the relevant tile from delta and A
                delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state])

                # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)

                # Load the relevant tile from u and B
                u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])

                # Step 3: Element-wise multiplication of delta_i, B_i and u_i
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=nl.multiply)
                B_i_bcast = B_i.broadcast_to((channel_psize, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                # Step 4: Associative scan between deltaA and deltaBu
                scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                        op0=np.multiply, op1=np.add)

                # Load the relevant tile from C
                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])

                # Step 5: Element-wise multiplication of scan_res and C_i
                C_i_bcast = C_i.broadcast_to((channel_psize, seq_len))
                scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

                # Step 6: Accumulation of scanC along state_size dimension
                # scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len] = nisa.tensor_tensor(
                #         scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len], scanC, op=nl.add)
                scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len] += scanC

        # Store scanC_accum for a single batch to output
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len])

    return output
# NKI_EXAMPLE_25_END

# NKI_EXAMPLE_26_BEGIN
@nki.jit
def mamba_v2(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)
    _, state_size = A.shape

    assert channels % 128 == 0

    # Map channels to the partition dimension
    # Tile channels to comply with NKI tile size constraints
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    # Most outer loop with batch_size, parallel_for
    for i_batch in nl.affine_range(batch_size):

        # Second outer loop: tiling channels
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            # partial accumulated scanC result with processed states
            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

            # Load delta/u once to be reused across states
            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])

            # Inner loop with state_size, partial parallel
            for i_state in nl.affine_range(state_size):
                # Load the relevant tile from A
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state])

                # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)

                # Load the relevant tile from B
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])

                # Step 3: Element-wise multiplication of delta_i, B_i and u_i
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=nl.multiply)
                B_i_bcast = B_i.broadcast_to((channel_psize, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                # Step 4: Associative scan between deltaA and deltaBu
                scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                        op0=np.multiply, op1=np.add)

                # Load the relevant tile from C
                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])

                # Step 5: Element-wise multiplication of scan_res and C_i
                C_i_bcast = C_i.broadcast_to((channel_psize, seq_len))
                scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

                # Step 6: Accumulation of scanC along state_size dimension
                scanC_accum[0:channel_psize, 0:seq_len] += scanC

            # Store scanC_accum for a single batch to output
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[0:channel_psize, 0:seq_len])

    return output
# NKI_EXAMPLE_26_END


@nki.jit
def mamba_v3(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)
    _, state_size = A.shape

    # Map channels to the partition dimension
    # Tile channels to comply with NKI tile size constraints
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    # Magic number, decided through empiracal profiling data
    seq_len_fsize = 512
    n_seq_len_tile = seq_len // seq_len_fsize

    # Fix this later with mask
    assert channels % channel_psize == 0
    assert seq_len % seq_len_fsize == 0

    # Most outer loop with batch_size, parallel_for
    for i_batch in nl.affine_range(batch_size):

        # Second outer loop: tiling channels
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            # partial accumulated scanC result with processed states
            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

            # Load delta/u once to be reused across states
            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])

            # Inner loop with state_size, partial parallel
            for i_state in nl.affine_range(state_size):
                # Load the relevant tile from A
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state])

                # Last scan result
                scan_init = nl.zeros((channel_psize, 1), dtype=delta_i.dtype)
                # FIXME: sequential_range gives incorrect answer and also much worse perf than static_range
                # for i_seq_len_tile in nl.sequential_range(n_seq_len_tile):
                for i_seq_len_tile in nl.static_range(n_seq_len_tile):
                    seq_len_start = i_seq_len_tile * seq_len_fsize

                    # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
                    deltaA = nisa.activation(op=nl.exp,
                            data=delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            scale=A_i)

                    # Load the relevant tile from B
                    B_i = nl.load(B[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize])

                    # Step 3: Element-wise multiplication of delta_i, B_i and u_i
                    deltaU = nisa.tensor_tensor(delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            u_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            op=nl.multiply)
                    B_i_bcast = B_i.broadcast_to((channel_psize, seq_len_fsize))
                    deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                    # Step 4: Associative scan between deltaA and deltaBu
                    scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=scan_init,
                            op0=np.multiply, op1=np.add)
                    scan_init[...] = scan_res[0:channel_psize, seq_len_fsize-1]

                    # Load the relevant tile from C
                    C_i = nl.load(C[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize])

                    # Step 5: Element-wise multiplication of scan_res and C_i
                    C_i_bcast = C_i.broadcast_to((channel_psize, seq_len_fsize))
                    scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

                    # Step 6: Accumulation of scanC along state_size dimension
                    scanC_accum[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize] += scanC

            # Store scanC_accum for a single batch to output
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[0:channel_psize, 0:seq_len])
    return output


def parse_args():
    parser = argparse.ArgumentParser("Run Mamba NKI kernels.")
    parser.add_argument("--mode",
                        choices=["accuracy", "perf"],
                        default="accuracy",
                        help="""Do accuracy test or perf test.
                                Accuracy test uses mamba_v1 output as golden reference.
                                Accuracy of mamba_v1 is tested by mamba_torch.py against native PyTorch implementation.
                             """)
    parser.add_argument("--version",
            nargs='+',
            default=["v1", "v2", "v3"],
            choices=["v1", "v2", "v3"],
            help="Test versions")

    parser.add_argument("--batch",
            nargs='+',
            default=[1],
            help="Batch size.")
    parser.add_argument("--seq_len",
            nargs='+',
            default=[2048],
            help="Sequence length.")
    parser.add_argument("--channels",
            nargs='+',
            default=[256],
            help="Number of channels.")
    parser.add_argument("--state_size",
            nargs='+',
            default=[16],
            help="State size.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Small test to ensure numerical correctness
    arr_batch = [int(_) for _ in args.batch]
    arr_seq_len = [int(_) for _ in args.seq_len]
    arr_channels = [int(_) for _ in args.channels]
    arr_state_size = [int(_) for _ in args.state_size]

    configs = itertools.product(arr_batch, arr_seq_len, arr_channels, arr_state_size)

    print(f"Running {args.mode} mode.")

    for config in configs:
        batch, seq_len, channels, state_size = config
        print(f">>> batch={batch}, seq_len={seq_len}, channels={channels}, state_size={state_size}")

        # Set up input tensors
        dtype = np.float32
        delta = np.ones((batch, channels, seq_len), dtype=dtype)
        u = np.ones((batch, channels, seq_len), dtype=dtype)
        A = -np.ones((channels, state_size), dtype=dtype)
        B = np.ones((batch, state_size, seq_len), dtype=dtype)
        C = np.ones((batch, state_size, seq_len), dtype=dtype)

        func_dict = {"v1": mamba_v1,
                     "v2": mamba_v2,
                     "v3": mamba_v3,
                    }

        if args.mode == "accuracy":
            # v1: reference kernel
            print(f">>>> Running v1 (reference).")
            nki_out_v1 = mamba_v1(delta, u, A, B, C)

            for version in args.version:
                if version == "v1":
                    # already run, continue
                    continue

                print(f">>>> Running version {version}.")
                func = func_dict[version]
                nki_out_test = func(delta, u, A, B, C)
                print(f">>>> mamba {version} matches?", np.all(nki_out_test == nki_out_v1))
                assert np.all(nki_out_test == nki_out_v1)


        else:
            # perf mode
            for version in args.version:
                print(f">>>> Running version {version}.")
                func = func_dict[version]
                nki.benchmark(func,
                              save_neff_name='file.neff',
                              save_trace_name='profile.ntff')\
                             (delta, u, A, B, C)
                # TODO: rename neff/ntff (bug in nki.benchmark with neff name)
                os.rename("file.neff", f"{version}_b{batch}_sl{seq_len}_c{channels}_ss{state_size}.neff")
                os.rename("profile.ntff", f"{version}_b{batch}_sl{seq_len}_c{channels}_ss{state_size}.ntff")
