"""Auto-generated stub file"""
from enum import Enum
import nki.language as nl
import ml_dtypes

class NKIObject:

    ...

class ReplicaGroup(NKIObject):
    r"""
    Defines a group of ranks that participate in a collective operation.

    Sub-groups represented by lists of ranks should not have any overlap.
    """

    ...

def all_reduce(srcs, dsts, replica_group, op):
    r"""
    Perform an all-reduce on the given replica group and input/output tensors.

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
    """
    ...

def all_gather(srcs, dsts, replica_group, collective_dim):
    r"""
    Perform an all-gather on the given replica group and input/output tensors.

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
    """
    ...

def reduce_scatter(srcs, dsts, replica_group, collective_dim, op):
    r"""
    Perform a reduce-scatter on the given replica group and input/output tensors.

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
        Currently only 0 is supported.
    :param op: The reduction operation to perform (``nl.add``, ``nl.minimum``, or ``nl.maximum``)
    """
    ...

def all_to_all(srcs, dsts, replica_group, collective_dim):
    r"""
    Perform an all-to-all on the given replica group and input/output tensors.

    The ``srcs`` and ``dsts`` parameters accept lists of tensors to support coalesced
    collective communication, which allows multiple tensors to be redistributed in a
    single collective operation for improved efficiency.

    Tensors must reside on HBM. SBUF is not currently supported for all-to-all.

    :param srcs: List of input tensors to redistribute
    :param dsts: List of output tensors to store results
    :param replica_group: ReplicaGroup defining rank groups for the collective
    :param collective_dim: Dimension along which input tensors are split and output tensors are concatenated.
        Currently only 0 is supported.
    """
    ...

def collective_permute(srcs, dsts, source_target_pairs):
    r"""
    Send and receive data between ranks based on explicitly defined source-target pairs.

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
    """
    ...

def collective_permute_implicit(srcs_by_channel, dsts_by_channel, replica_group, channel_ids=[0]):
    r"""
    Send and receive data between ranks in a ring, where sources and destinations are
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
    """
    ...

def collective_permute_implicit_reduce(srcs0_by_channel, srcs1_by_channel, dsts_by_channel, replica_group, op, channel_ids=[0]):
    r"""
    Perform an implicit collective permute with reduction in a ring, where sources and
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
    """
    ...

def rank_id():
    r"""
    Get the rank ID of the current rank.

    :return: The rank ID of the current rank within the collective group
    """
    ...

def collective_permute_implicit_current_processing_rank_id(iteration_id, replica_group, channel_id=0, num_channels=1):
    r"""
    Returns the rank ID of the data to be processed in the current ring iteration.

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
    :param channel_id: Channel ID for the communication (0 to num_channels-1)
    :param num_channels: Total number of channels (use 1 for single-channel)
    :param replica_group: ReplicaGroup defining the ring topology
    :return: Scalar register containing the rank ID of the data to be processed
    """
    ...

