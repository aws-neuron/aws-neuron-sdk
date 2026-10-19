"""Collectives example for nki.simulate documentation.

Run with two ranks:

    NEURON_LOGICAL_NC_CONFIG=1 torchrun --nproc-per-node 2 \
        nki_simulate_collectives_example.py
"""

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch.distributed as dist


# NKI_EXAMPLE_SIMULATE_COLLECTIVES_BEGIN
@nki.jit
def sum_and_gather_kernel(x, world_size):
    size = x.shape[0]
    replica_group = nccl.ReplicaGroup.from_partition_op(
        world_size, nccl.rg_partition_op.bitwise_and, 0
    )
    cc_src = nl.ndarray((size,), dtype=x.dtype, buffer=nl.private_hbm)
    cc_sum = nl.ndarray((size,), dtype=x.dtype, buffer=nl.private_hbm)
    cc_gather = nl.ndarray(
        (size * world_size,), dtype=x.dtype, buffer=nl.private_hbm
    )
    nisa.dma_copy(dst=cc_src, src=x)

    # Sum x across all ranks
    nccl.all_reduce(
        srcs=[cc_src], dsts=[cc_sum], replica_group=replica_group, op=nl.add
    )
    # Concatenate every rank's x along dimension 0, in rank order
    nccl.all_gather(
        srcs=[cc_src], dsts=[cc_gather], replica_group=replica_group, collective_dim=0
    )

    out_sum = nl.ndarray((size,), dtype=x.dtype, buffer=nl.private_hbm)
    out_gather = nl.ndarray(
        (size * world_size,), dtype=x.dtype, buffer=nl.private_hbm
    )
    nisa.dma_copy(dst=out_sum, src=cc_sum)
    nisa.dma_copy(dst=out_gather, src=cc_gather)
    return out_sum, out_gather
# NKI_EXAMPLE_SIMULATE_COLLECTIVES_END


# NKI_EXAMPLE_SIMULATE_COLLECTIVES_RUN_BEGIN
# torchrun starts one process per rank; the simulator exchanges data
# between ranks through this process group
dist.init_process_group(backend="gloo")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Each rank contributes a tile filled with its rank ID
x = np.full((128,), rank, dtype=np.float32)

out_sum, out_gather = nki.simulate(sum_and_gather_kernel)(x, world_size)

# Every rank receives the same results
expected_sum = np.full_like(x, sum(range(world_size)))
expected_gather = np.concatenate(
    [np.full_like(x, r) for r in range(world_size)], axis=0
)
np.testing.assert_allclose(out_sum, expected_sum)
np.testing.assert_allclose(out_gather, expected_gather)

dist.destroy_process_group()
# NKI_EXAMPLE_SIMULATE_COLLECTIVES_RUN_END
