from dataclasses import dataclass
from enum import Enum
from typing import *

from nki.language import NKIObject

@dataclass
class ReplicaGroup(NKIObject):
    r"""Defines a group of ranks that participate in a collective operation.

    Sub-groups represented by lists of ranks should not have any overlap."""

    ...

def all_reduce(srcs: List, dsts: List, replica_group: ReplicaGroup, op, priority: Optional[int]=None) -> None:
    r"""Perform an all-reduce on the given replica group and input/output tensors.

    The ``srcs`` and ``dsts`` parameters accept lists of tensors to support coalesced
    collective communication, which allows multiple tensors to be reduced in a single
    collective operation for improved efficiency.

    Tensors can reside on either HBM or SBUF. However, mixing memory spaces is not
    supported: all tensors must be on HBM or all must be on SBUF. Coalesced collective
    communication (multiple tensors) is only supported when tensors are on HBM.

    :param srcs: List of input tensors to reduce
    :param dsts: List of output tensors to store results
    :param replica_group: ReplicaGroup defining rank groups for the collective
    :param op: The reduction operation to perform (``nl.add``, ``nl.minimum``, or ``nl.maximum``)
    :param priority: DMA quality-of-service priority level 0-3 where lower is higher
        priority (NeuronCore-v4+ only)"""
    ...

def all_gather(srcs: List, dsts: List, replica_group: ReplicaGroup, collective_dim: int, priority: Optional[int]=None) -> None:
    r"""Perform an all-gather on the given replica group and input/output tensors.

    The ``srcs`` and ``dsts`` parameters accept lists of tensors to support coalesced
    collective communication, which allows multiple tensors to be gathered in a single
    collective operation for improved efficiency.

    Tensors can reside on either HBM or SBUF. However, mixing memory spaces is not
    supported: all tensors must be on HBM or all must be on SBUF. Coalesced collective
    communication (multiple tensors) is only supported when tensors are on HBM.

    :param srcs: List of input tensors to gather
    :param dsts: List of output tensors to store results
    :param replica_group: ReplicaGroup defining rank groups for the collective
    :param collective_dim: Dimension along which output tensors are concatenated.
        Currently only 0 is supported for HBM tensors. For SBUF tensors, 0 or 1 is
        supported as SBUF collectives currently only operate on 2D tensors with a
        single free dimension.
    :param priority: DMA quality-of-service priority level 0-3 where lower is higher
        priority (NeuronCore-v4+ only)"""
    ...

def reduce_scatter(srcs: List, dsts: List, replica_group: ReplicaGroup, collective_dim: int, op, priority: Optional[int]=None) -> None:
    r"""Perform a reduce-scatter on the given replica group and input/output tensors.

    The ``srcs`` and ``dsts`` parameters accept lists of tensors to support coalesced
    collective communication, which allows multiple tensors to be reduced and scattered
    in a single collective operation for improved efficiency.

    Tensors can reside on either HBM or SBUF. However, mixing memory spaces is not
    supported: all tensors must be on HBM or all must be on SBUF. Coalesced collective
    communication (multiple tensors) is only supported when tensors are on HBM.

    :param srcs: List of input tensors to reduce and scatter
    :param dsts: List of output tensors to store results
    :param replica_group: ReplicaGroup defining rank groups for the collective
    :param collective_dim: Dimension along which input tensors are split.
        Currently only 0 is supported for both HBM and SBUF tensors.
    :param op: The reduction operation to perform (``nl.add``, ``nl.minimum``, or ``nl.maximum``)
    :param priority: DMA quality-of-service priority level 0-3 where lower is higher
        priority (NeuronCore-v4+ only)"""
    ...

def all_to_all(srcs: List, dsts: List, replica_group: ReplicaGroup, collective_dim: int, priority: Optional[int]=None) -> None:
    r"""Perform an all-to-all on the given replica group and input/output tensors.

    The ``srcs`` and ``dsts`` parameters accept lists of tensors to support coalesced
    collective communication, which allows multiple tensors to be redistributed in a
    single collective operation for improved efficiency.

    Tensors must reside on HBM. SBUF is not currently supported for all-to-all.

    :param srcs: List of input tensors to redistribute
    :param dsts: List of output tensors to store results
    :param replica_group: ReplicaGroup defining rank groups for the collective
    :param collective_dim: Dimension along which input tensors are split and output tensors are concatenated.
        Currently only 0 is supported.
    :param priority: DMA quality-of-service priority level 0-3 where lower is higher
        priority (NeuronCore-v4+ only)"""
    ...

def all_to_all_v(srcs: List, dsts: List, replica_group: ReplicaGroup, metadata_tensor, recv_counts_known: bool=False, has_rdispls: bool=False, priority: Optional[int]=None) -> None:
    r"""Executes an all-to-all collective where each rank can send
    a different number of elements, known only at execution time (rather
    than at compile time).

    Unlike ``all_to_all`` which splits/concatenates along a collective
    dimension, ``all_to_all_v`` treats tensors as flat element buffers.
    Per-rank send/recv counts and displacements are supplied via a uint32
    metadata tensor, making per-rank payload sizes dynamic.

    **Current restrictions:**

    On instances with a NeuronSwitch fabric (see `Trn3 architecture
    <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html>`_),
    ``all_to_all_v`` requires LNC=2 and more than one participating
    device. Multiple ranks per device are supported, but for every
    replica-group rank-list, every device participating in that
    rank-list must have all of its ranks (4 under LNC=2) included in
    the same rank-list — each rank-list is a set of sequential ranks
    in the world (e.g. ``[[1, 2, 3, 4], [5, 6, 7, 8]]``). To exclude a
    rank, keep it in the replica group and set its ``send_count`` to 0.

    On other instances, ``all_to_all_v`` currently supports only
    inter-node replica groups: each rank-list contains same-indexed
    ranks from different nodes (a node refers to a different Trn EC2
    instance).

    :param srcs: Input tensor list. Currently supports exactly one tensor.
        Must be HBM-backed.
    :param dsts: Output tensor list. Currently supports exactly one tensor.
        Must be HBM-backed. ``src`` and ``dst`` element counts can be
        different; sizes are validated against the metadata at execution
        time.
    :param replica_group: ReplicaGroup defining which ranks participate.
    :param metadata_tensor: ``uint32`` tensor laid out contiguously in
        memory. Shape depends on backing buffer, where ``rows`` is 3 when
        ``has_rdispls=False`` and 4 when ``has_rdispls=True``:

        - HBM: ``(rows, replica_group_size)``.
        - SBUF: ``(1, rows, replica_group_size)`` — the whole buffer must
          live on a single partition, so a trivial partition dim is
          prepended.

        For each other rank ``r`` in the replica group, the rows are:

        - Row 0 ``send_counts[r]``: number of elements sent to rank ``r``.
          Always an input.
        - Row 1 ``send_displs[r]``: offset in elements within ``src`` where
          the chunk destined for rank ``r`` begins. Always an input.
        - Row 2 ``recv_counts[r]``: number of elements received from rank
          ``r``. Controlled by ``recv_counts_known`` — see that flag.
        - Row 3 ``recv_displs[r]``: offset in elements within ``dst`` where
          the chunk from rank ``r`` is written. Only present when
          ``has_rdispls=True``.

    :param recv_counts_known:
        Controls whether row 2 is populated by the collective during
        execution. Row 2 is never read as input.

        - ``True``: row 2 is left untouched, avoiding a small per-rank
          writeback.
        - ``False`` (default): row 2 is an **output** — per-rank received
          counts are written during execution, and can be read after the
          op to learn received sizes.

    :param has_rdispls:
        - ``True``: row 3 is an **input**; recv_displs must be populated.
          The chunk from sender rank ``r`` is written at
          ``dst[recv_displs[r] : recv_displs[r] + recv_counts[r]]``.
        - ``False``: row 3 may be omitted from ``metadata_tensor`` (pass a
          3-row tensor). Incoming chunks are laid out equally-spaced at
          ``recv_displs[r] = (dst.total_elements / replica_group_size) * r``,
          regardless of the actual recv_count per rank.

    :param priority: DMA QoS priority level 0-3 where lower is higher
        priority (NeuronCore-v4+ only)."""
    ...

def collective_permute(srcs: List, dsts: List, source_target_pairs: List[Tuple[int, int]], priority: Optional[int]=None) -> None:
    r"""Send and receive data between ranks based on explicitly defined source-target pairs.

    Each pair ``(source, target)`` specifies that data from the source rank
    should be sent to the target rank. This gives you full control over the
    communication pattern (e.g., pairwise swaps, arbitrary shuffles).

    Prefer :func:`collective_permute_implicit` when the communication
    follows a ring topology, as the hardware can optimize that pattern.

    Tensors must reside on HBM. SBUF is not currently supported for collective_permute.

    Coalesced collective communication (multiple tensors) is not currently supported;
    each list parameter must contain exactly one tensor.

    :param srcs: List of source tensors to send
    :param dsts: List of destination tensors to receive into
    :param source_target_pairs: List of (source, target) rank ID pairs
    :param priority: DMA quality-of-service priority level 0-3 where lower is higher
        priority (NeuronCore-v4+ only)"""
    ...

def collective_permute_implicit(srcs_by_channel: List[List], dsts_by_channel: List[List], replica_group: ReplicaGroup, channel_ids: List[int]=[0], priority: Optional[int]=None) -> None:
    r"""Send and receive data between ranks in a ring, where sources and destinations are
    implicitly determined by the ring structure during runtime.

    Each rank sends data to its successor and receives from its predecessor in the ring.
    This differs from :func:`collective_permute` where users explicitly specify source-target pairs.

    Since the sources and destinations are implicitly determined, use
    :func:`collective_permute_implicit_current_processing_rank_id` to get the rank ID
    whose data is currently being processed.

    The outer dimension of ``srcs_by_channel`` and ``dsts_by_channel`` corresponds to channels.
    For each channel, the inner list contains exactly one tensor (coalesced collective
    communication is not currently supported).

    **Channels**: Multiple channels enable overlapping communication, allowing concurrent data
    transfers. The number of available channels depends on the replica group and system
    connectivity (see
    `Neuron Collectives <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/about/collectives.html#system-connectivity>`_).
    The maximum number of channels is 4 for replica groups containing all devices inside a node
    and 2 for other supported replica groups.

    :param srcs_by_channel: List of source tensor lists, one per channel. Each inner list must contain exactly one tensor.
    :param dsts_by_channel: List of destination tensor lists, one per channel. Each inner list must contain exactly one tensor.
    :param replica_group: ReplicaGroup defining rank groups for the collective
    :param channel_ids: List of channel IDs to use for communication (default [0] for single channel).
        Currently must be consecutive integers starting from 0.
    :param priority: DMA quality-of-service priority level 0-3 where lower is higher
        priority (NeuronCore-v4+ only)"""
    ...

def collective_permute_implicit_reduce(srcs0_by_channel: List[List], srcs1_by_channel: List[List], dsts_by_channel: List[List], replica_group: ReplicaGroup, op, channel_ids: List[int]=[0], priority: Optional[int]=None) -> None:
    r"""Perform an implicit collective permute with reduction in a ring, where sources and
    destinations are implicitly determined by the ring structure during runtime.

    Combines :func:`collective_permute_implicit` with a reduction operation.
    Each rank reduces its local sources using ``op(srcs0_by_channel[i], srcs1_by_channel[i])``,
    sends the result to its successor, and receives its predecessor's reduced result into
    ``dsts_by_channel[i]``.

    Since the sources and destinations are implicitly determined, use
    :func:`collective_permute_implicit_current_processing_rank_id` to get the rank ID
    whose data is currently being processed.

    The outer dimension of ``srcs0_by_channel``, ``srcs1_by_channel``, and ``dsts_by_channel``
    corresponds to channels. For each channel, the inner list contains exactly one tensor
    (coalesced collective communication is not currently supported).

    **Channels**: Multiple channels enable overlapping communication, allowing concurrent data
    transfers. The number of available channels depends on the replica group and system
    connectivity (see
    `Neuron Collectives <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/about/collectives.html#system-connectivity>`_).
    The maximum number of channels is 4 for replica groups containing all devices inside a node
    and 2 for other supported replica groups.

    :param srcs0_by_channel: List of source tensor lists (left operand of reduction), one per channel. Each inner list must contain exactly one tensor.
    :param srcs1_by_channel: List of source tensor lists (right operand of reduction), one per channel. Each inner list must contain exactly one tensor.
    :param dsts_by_channel: List of destination tensor lists to receive predecessor's reduced result, one per channel. Each inner list must contain exactly one tensor.
    :param replica_group: ReplicaGroup defining rank groups for the collective
    :param op: The reduction operation to perform (``nl.add``, ``nl.minimum``, or ``nl.maximum``)
    :param channel_ids: List of channel IDs to use for communication (default [0] for single channel).
        Currently must be consecutive integers starting from 0.
    :param priority: DMA quality-of-service priority level 0-3 where lower is higher
        priority (NeuronCore-v4+ only)"""
    ...

def collective_permute_implicit_current_processing_rank_id(iteration_id: int, replica_group: ReplicaGroup, channel_id: int=0):
    r"""Returns the rank ID of the data to be processed in the current ring iteration.

    This function is intended to be used in conjunction with
    :func:`collective_permute_implicit` or :func:`collective_permute_implicit_reduce`.
    Since the sources and destinations are implicitly determined in ring algorithms,
    the rank ID of received data can only be determined at runtime.

    At iteration 0, this returns the current rank's own ID (processing local data).
    In subsequent iterations, it returns the rank ID of data received from predecessors,
    progressing around the ring.

    The returned rank ID is a scalar register. To determine the offset of the received
    data chunk within a tensor, use register ALU operations (e.g., multiply the rank ID
    by chunk size), then use dynamic access pattern (``tensor.ap()``) in ISA compute
    operations (e.g., ``nisa.nc_matmul()``).

    **Typical usage pattern**: In each iteration of a ring algorithm, the compute kernel
    uses this function to identify which rank's data is being processed, computes on that
    data while concurrently triggering the next communication step to send already-computed
    chunks to the successor.

    **Channels**: Multiple channels enable overlapping communication, allowing concurrent data
    transfers. The number of available channels depends on the replica group and system
    connectivity (see
    `Neuron Collectives <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/about/collectives.html#system-connectivity>`_).
    The maximum number of channels is 4 for replica groups containing all devices inside a node
    and 2 for other supported replica groups.

    :param iteration_id: Current ring step (typically the loop counter).
    :param replica_group: ReplicaGroup defining the ring topology
    :param channel_id: Channel ID for the communication (0 to num_channels-1)
    :return: Scalar register containing the rank ID of the data to be processed"""
    ...

def rank_id():
    r"""Get the rank ID of the current rank.

    :return: The rank ID of the current rank within the collective group"""
    ...
