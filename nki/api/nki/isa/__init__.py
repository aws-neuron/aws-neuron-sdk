"""Auto-generated stub file"""
from enum import Enum
import numpy as np
import ml_dtypes

class dge_mode(Enum):
    r"""Neuron Descriptor Generation Engine Mode"""

    unknown = 0
    r"""Unknown DGE mode, i.e., let compiler decide the DGE mode"""

    swdge = 1
    r"""Software DGE"""

    hwdge = 2
    r"""Hardware DGE"""

    none = 3
    r"""Not using DGE"""


class engine(Enum):
    r"""Neuron Device engines"""

    tensor = 1
    r"""Tensor Engine"""

    vector = 5
    r"""Vector Engine"""

    scalar = 2
    r"""Scalar Engine"""

    gpsimd = 3
    r"""GpSIMD Engine"""

    dma = 4
    r"""DMA Engine"""

    sync = 6
    r"""Sync Engine"""

    unknown = 0
    r"""Unknown Engine"""


def get_nc_version():
    r"""Returns the ``nc_version`` of the current target context."""
    ...

class nc_version(Enum):
    r"""NeuronCore version"""

    gen2 = 2
    r"""Trn1/Inf2 target"""

    gen3 = 3
    r"""Trn2 target"""

    gen4 = 4
    r"""Trn3 target"""


class oob_mode(Enum):
    r"""Neuron OOB Access Mode"""

    error = 0

    skip = 1


class reduce_cmd(Enum):
    r"""Engine Register Reduce commands"""

    idle = 0
    r"""Not using the accumulator registers"""

    reset = 1
    r"""Resets the accumulator registers to its initial state"""

    reduce = 2
    r"""Keeps accumulating over the current value of the accumulator registers"""

    reset_reduce = 3
    r"""Resets the accumulator registers then immediately accumulate the results of the current instruction into the accumulators"""

    load_reduce = 4
    r"""Loads a value into the accumulator registers, then accumulate the results of the current instruction into the accumulators"""


def activation(dst, op, data, bias=None, scale=1.0, reduce_op=None, reduce_res=None, reduce_cmd=reduce_cmd.idle, name=None):
    r"""
    Apply an activation function on every element of the input tile using Scalar Engine, with an optional scale/bias operation
    before the activation and an optional reduction operation after the activation in the same instruction.

    The activation function is specified in the ``op`` input field (see :ref:`nki-act-func` for a list of
    supported activation functions and their valid input ranges).

    ``nisa.activation`` can optionally multiply the input ``data`` by a scalar or vector ``scale``
    and then add another vector ``bias`` before the activation function is applied.

    After the activation function
    is applied, Scalar Engine can also reduce along the free dimensions of the activated data per lane, using
    ``reduce_op`` operation. ``reduce_op`` must be ``nl.add``.

    The reduction result is then either stored into or reduced on top of a set of internal engine registers
    called ``reduce_regs`` (one 32-bit register per compute lane, 128 registers in total), controlled by the
    ``reduce_cmd`` field:

    - ``nisa.reduce_cmd.reset``: Reset ``reduce_regs`` to zero only.
    - ``nisa.reduce_cmd.idle``: Do not modify ``reduce_regs``.
    - ``nisa.reduce_cmd.reduce``: Reduce activated data over existing values in ``reduce_regs``.
    - ``nisa.reduce_cmd.reset_reduce``: Reset ``reduce_regs`` to zero and then store the reduction result
      of the activated data.

    ``nisa.activation`` can also emit another instruction to read out ``reduce_regs`` by
    passing an SBUF/PSUM tile in the ``reduce_res`` arguments.
    The ``reduce_regs`` state can persist across multiple ``nisa.activation`` instructions without the need to
    be evicted back to SBUF/PSUM (``reduce_res`` tile).

    The following is the pseudo code for ``nisa.activation``:

    .. math::

        output = op(data * scale + bias)

        if reduce_cmd == nisa.reduce_cmd.reset or reduce_cmd == nisa.reduce_cmd.reset_reduce:
            reduce_regs = 0

        result = reduce\_op(reduce_regs, reduce\_op(output, axis=<FreeAxis>))

        if reduce_cmd == nisa.reduce_cmd.reduce or reduce_cmd == nisa.reduce_cmd.reset_reduce:
            reduce_regs += result

        if reduce_res:
            reduce_res = reduce_regs


    All these optional operations incur no further performance penalty compared to only applying the activation function,
    except reading out ``reduce_regs`` into ``reduce_res`` will have a small overhead due to an extra instruction.

    **Memory types.**

    The input ``data`` tile can be an SBUF or PSUM tile. Similarly, the instruction
    can write the output ``dst`` tile into either SBUF or PSUM.

    **Data types.**

    Both input ``data`` and output ``dst`` tiles can be in any valid NKI data type
    (see :ref:`nki-dtype` for more information).
    The Scalar Engine always performs the math operations in float32 precision.
    Therefore, the engine automatically casts the input ``data`` tile to float32 before
    performing multiply/add/activate specified in the activation instruction.
    The engine is also capable of casting the float32 math results into another
    output data type in ``dst`` at no additional performance cost.
    The ``scale`` parameter must
    have a float32 data type, while the ``bias`` parameter can be float32/float16/bfloat16.

    **Layout.**

    The ``scale`` can either be a compile-time constant scalar or a
    ``[N, 1]`` vector from SBUF/PSUM. ``N`` must be the same as the partition dimension size of ``data``.
    In NeuronCore-v2, the ``bias`` must be a ``[N, 1]`` vector, but starting NeuronCore-v3, ``bias`` can either be
    a compile-time constant scalar or a ``[N, 1]`` vector similar to ``scale``.

    When the ``scale`` (or similarly, ``bias``) is a scalar, the scalar
    is broadcasted to all the elements in the input ``data`` tile to perform the computation.
    When the ``scale`` (or ``bias``) is a vector, the ``scale`` (or ``bias``) value in each partition is broadcast
    along the free dimension of the ``data`` tile.

    **Tile size.**

    The partition dimension size of input ``data`` and output ``dst`` tiles must be the same and must not exceed 128.
    The number of elements per partition of ``data`` and ``dst`` tiles must be the same and must not
    exceed the physical size of each SBUF partition.


    :param dst: the activation output
    :param op: an activation function (see :ref:`nki-act-func` for supported functions)
    :param data: the input tile; layout: (partition axis <= 128, free axis)
    :param scale: a scalar or a vector for multiplication
    :param bias: a scalar (NeuronCore-v3 or newer) or a vector for addition
    :param reduce_op: the reduce operation to perform on the free dimension of the activated data
    :param reduce_res: a tile of shape ``(data.shape[0], 1)`` to hold the final state of ``reduce_regs``.
    :param reduce_cmd: an enum member from ``nisa.reduce_cmd`` to control the state of ``reduce_regs``.

    """
    ...

def activation_reduce(dst, op, data, reduce_op, reduce_res, bias=None, scale=1.0, name=None):
    r"""
    Perform the same computation as ``nisa.activation`` and also a reduction along the free dimension of the
    ``nisa.activation`` result using Scalar Engine. The results for the reduction is stored
    in the reduce_res.

    This API is equivalent to calling ``nisa.activation`` with
    ``reduce_cmd=nisa.reduce_cmd.reset_reduce`` and passing in reduce_res. This API is kept for
    backward compatibility, we recommend using ``nisa.activation`` moving forward.

    Refer to :doc:`nisa.activation <nki.isa.activation>` for semantics of ``op/data/bias/scale``.

    In addition to :doc:`nisa.activation <nki.isa.activation>` computation, this API also performs a reduction
    along the free dimension(s) of the :doc:`nisa.activation <nki.isa.activation>` result, at a small additional
    performance cost. The reduction result is returned in ``reduce_res`` in-place, which must be a
    SBUF/PSUM tile with the same partition axis size as the input tile ``data`` and one element per partition.
    On NeuronCore-v2, the ``reduce_op`` can only be an addition, ``np.add`` or ``nl.add``.

    There are 128 registers on the scalar engine for storing reduction results, corresponding
    to the 128 partitions of the input. These registers are shared between ``activation`` and ``activation_accu`` calls.
    This instruction first resets those
    registers to zero, performs the reduction on the value after activation function is applied,
    stores the results into the registers,
    then reads out the reduction results from the register, eventually store them into ``reduce_res``.

    Note that ``nisa.activation`` can also change the state of the register. It's user's
    responsibility to ensure correct ordering. It's the best practice to not mixing
    the use of ``activation_reduce`` and ``activation``.

    Reduction axis is not configurable in this API. If the input tile has multiple free axis, the API will
    reduce across all of them.

    Mathematically, this API performs the following computation:

    .. math::
          output = f_{act}(data * scale + bias) \\
          reduce\_res = reduce\_op(output, axis=<FreeAxis>)


    :param dst: output tile of the activation instruction; layout: same as input ``data`` tile
    :param op: an activation function (see :ref:`nki-act-func` for supported functions)
    :param data: the input tile; layout: (partition axis <= 128, free axis)
    :param reduce_op: the reduce operation to perform on the free dimension of the activation result
    :param reduce_res: a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                    is the partition axis size of the input ``data`` tile. The result of ``sum(ReductionResult)``
                    is written in-place into the tensor.
    :param bias: a vector with the same partition axis size as ``data``
                 for broadcast add (after broadcast multiply with ``scale``)
    :param scale: a scalar or a vector with the same partition axis size as ``data``
                  for broadcast multiply
    """
    ...

def affine_select(dst, pattern, offset, channel_multiplier, on_true_tile, on_false_value, cmp_op=np.equal, name=None):
    r"""
    Select elements between an input tile ``on_true_tile`` and a scalar value ``on_false_value``
    according to a boolean predicate tile using GpSimd Engine.

    The predicate tile is calculated on-the-fly in the engine by evaluating an affine expression element-by-element.
    The affine expression is defined by a ``pattern``, ``offset``, and ``channel_multiplier``, similar to ``nisa.iota``.
    The ``pattern`` field is a list of lists in the form of
    ``[[step_w, num_w], [step_z, num_z], [step_y, num_y], [step_x, num_x]]``. When fewer than 4D ``pattern``
    is provided, NKI compiler automatically pads remaining dimensions with size of 1.

    Given a 4D pattern (padded if needed), the instruction generates a predicate using the following pseudo code:

    .. code-block:: python

        num_partitions = dst.shape[0]
        [[step_w, num_w], [step_z, num_z], [step_y, num_y], [step_x, num_x]] = pattern

        for channel_id in range(num_partitions):
          for w in range(num_w):
            for z in range(num_z):
              for y in range(num_y):
                for x in range(num_x):
                  affine_value = offset + (channel_id * channel_multiplier) +
                                (w * step_w) + (z * step_z) + (y * step_y) + (x * step_x)

                  predicate = cmp_op(affine_value, 0)  # Compare with 0 using cmp_op

                  if predicate:
                      dst[channel_id, w, z, y, x] = on_true_tile[channel_id, w, z, y, x]
                  else:
                      dst[channel_id, w, z, y, x] = on_false_value

    The above pseudo code assumes ``dst`` has the same size in every dimension ``x/y/z/w`` for simplicity. However,
    the instruction allows any sizes in the free dimension, as long as the number of elements per partition in ``dst``
    matches the product: ``num_w * num_z * num_y * num_x``.

    A common use case for ``affine_select`` is to apply a causal mask on the attention
    scores for transformer decoder models.

    **Memory types.**

    The output ``dst`` tile must be in SBUF. The input ``on_true_tile`` must also be in SBUF.

    **Data types.**

    The input ``on_true_tile`` and output ``dst`` tile can be any valid NKI data type
    (see :ref:`nki-dtype` for more information). If the data type of ``on_true_tile`` differs from
    that of ``dst``, the input elements in ``on_true_tile``, if selected, are first cast to FP32
    before converting to the output data type in ``dst``.
    The ``on_false_value`` must be float32, regardless of the input/output tile data types.


    **Layout.**

    The partition dimension determines the number of active channels for parallel pattern generation and selection.
    The input tile ``on_true_tile``, the calculated boolean predicate tile, and the returned output tile
    must have the same partition dimension size and.


    **Tile size.**

    - The partition dimension size of ``dst`` and ``on_true_tile`` must be the same and must not exceed 128.
    - The number of elements per partition of ``dst`` and ``on_true_tile`` must not
      exceed the physical size of each SBUF partition.
    - The total number of elements in ``pattern`` must match the number of elements
      per partition in the ``dst`` and ``on_true_tile`` tiles.


    :param dst: the output tile in SBUF to store the selected values
    :param pattern: a list of [step, num] to describe up to 4D tensor sizes and strides for affine expression generation
    :param offset: an int32 offset value to be added to every generated affine value
    :param channel_multiplier: an int32 multiplier to be applied to the channel (partition) ID
    :param on_true_tile: an input tile for selection with a ``True`` predicate value
    :param on_false_value: a scalar value for selection with a ``False`` predicate value
    :param cmp_op: comparison operator to use for predicate evaluation (default: np.equal)

    """
    ...

def bn_aggr(dst, data, name=None):
    r"""
    Aggregate one or multiple ``bn_stats`` outputs to generate
    a mean and variance per partition using Vector Engine.

    The input ``data`` tile
    effectively has an array of ``(count, mean, variance*count)`` tuples per partition
    produced by  :doc:`bn_stats <nki.isa.bn_stats>` instructions. Therefore, the number of elements per partition
    of ``data`` must be a modulo of three.

    Note, if you need to aggregate multiple ``bn_stats`` instruction outputs,
    it is recommended to declare a SBUF tensor
    and then make each ``bn_stats`` instruction write its output into the
    SBUF tensor at different offsets.

    Vector Engine performs the statistics aggregation in float32 precision.
    Therefore, the engine automatically casts the input ``data`` tile to float32 before
    performing float32 computation and is capable of casting
    the float32 computation results into another data type specified by the ``dtype`` field,
    at no additional performance cost. If ``dtype`` field is not specified, the instruction
    will cast the float32 results back to the same data type as the input ``data`` tile.



    :param dst: an output tile with two elements per partition: a mean followed by a variance
    :param data: an input tile with results of one or more :doc:`bn_stats <nki.isa.bn_stats>`
    """
    ...

def bn_stats(dst, data, name=None):
    r"""
    Compute mean- and variance-related statistics for each partition of an input tile ``data``
    in parallel using Vector Engine.

    The output tile of the instruction has 6 elements per partition:

    - the ``count`` of the even elements (of the input tile elements from the same partition)
    - the ``mean`` of the even elements
    - ``variance * count`` of the even elements
    - the ``count`` of the odd elements
    - the ``mean`` of the odd elements
    - ``variance * count`` of the odd elements

    To get the final mean and variance of the input tile,
    we need to pass the above ``bn_stats`` instruction output
    into the :doc:`bn_aggr <nki.isa.bn_aggr>`
    instruction, which will output two elements per partition:

    - mean (of the original input tile elements from the same partition)
    - variance

    Due to hardware limitation, the number of elements per partition
    (i.e., free dimension size) of the input ``data`` must not exceed 512 (nl.tile_size.bn_stats_fmax).
    To calculate per-partition mean/variance of a tensor with more than
    512 elements in free dimension, we can invoke ``bn_stats`` instructions
    on each 512-element tile and use a single ``bn_aggr`` instruction to
    aggregate ``bn_stats`` outputs from all the tiles.

    Vector Engine performs the above statistics calculation in float32 precision.
    Therefore, the engine automatically casts the input ``data`` tile to float32 before
    performing float32 computation and is capable of casting
    the float32 computation results into another data type specified by the ``dtype`` field,
    at no additional performance cost. If ``dtype`` field is not specified, the instruction
    will cast the float32 results back to the same data type as the input ``data`` tile.


    :param dst: an output tile with 6-element statistics per partition
    :param data: the input tile (up to 512 elements per partition)



    """
    ...

def core_barrier(data, cores, engine=engine.unknown, name=None):
    r"""
    Synchronize execution across multiple NeuronCores by implementing a barrier mechanism.

    .. note::
      Available only on NeuronCore-v3 or newer.

    This instruction creates a synchronization point where all specified NeuronCores must
    reach before any can proceed. The barrier is implemented using a semaphore-based protocol
    where each NeuronCore writes a semaphore to each other core (remote semaphore update)
    and then waits for the other cores' semaphores before continuing execution (local semaphore wait).

    The use case is when two NeuronCores both need to write to disjoint portions of a
    shared HBM tensor (``data``) and they both need to consume the tensor after both cores
    have finished writing into the tensor. In this case, both cores can perform the write to
    ``data`` in HBM using ``nisa.dma_copy``, and then signal to each other when the write operation is complete
    using ``nisa.core_barrier``.

    This instruction is only allowed in NeuronCore-v3 or newer when
    `LNC (Logical NeuronCore) <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html>`_
    is enabled. Currently only ``cores=(0, 1)`` is supported. This allows synchronization between exactly
    two NeuronCores that share the same HBM stack.

    The ``data`` parameter represents the shared data that all cores need to synchronize on.
    This must be data in shared HBM that multiple cores are accessing.

    The ``engine`` parameter allows specifying which engine inside the NeuronCores should execute the barrier
    instruction (that is, the remote semaphore update and local semaphore wait).

    :param data: the shared data that all cores need to synchronize on; must be data in shared HBM
    :param cores: a tuple of core indices to synchronize; only ``(0, 1)`` is supported when LNC2 is enabled
    :param engine: the engine to execute the barrier instruction on; defaults to automatic selection

    Example:

    .. code-block:: python

        # Synchronize between two cores after each core writes to half of shared tensor
        shared_tensor = nl.ndarray((batch_size, hidden_dim), dtype=nl.float32, buffer=nl.shared_hbm)

        # Each core writes to half of the tensor
        if core_id == 0:
            # Core 0 writes to first half
            core0_data = nl.ndarray((batch_size // 2, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=shared_tensor[:batch_size // 2, :], src=core0_data)
        else:
            # Core 1 writes to second half
            core1_data = nl.ndarray((batch_size // 2, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=shared_tensor[batch_size // 2:, :], src=core1_data)

        core_barrier(data=shared_tensor, cores=(0, 1))

        # Now both cores can safely read the complete tensor

    """
    ...

def dma_compute(dst, srcs, scales, reduce_op, name=None):
    r"""
    Perform math operations using compute logic inside DMA engines with element-wise scaling and reduction.

    This instruction leverages the compute capabilities within DMA engines to perform scaled element-wise operations
    followed by reduction across multiple source tensors. The computation follows the pattern:
    ``dst = reduce_op(srcs[0] * scales[0], srcs[1] * scales[1], ...)``, where each source tensor is first
    multiplied by its corresponding scale factor, then all scaled results are combined using the specified
    reduction operation.
    Currently, only ``nl.add`` is supported for ``reduce_op``, and
    all values in ``scales`` must be ``1.0``.

    The DMA engines perform all computations in float32 precision internally. Input tensors are automatically
    cast from their source data types to float32 before computation, and the final float32 result is cast
    to the output data type in a pipelined fashion.

    **Memory types.**

    Both input ``srcs`` tensors and output ``dst`` tensor can be in HBM or SBUF.
    Both ``srcs`` and ``dst`` tensors must have compile-time known addresses.

    **Data types.**

    All input ``srcs`` tensors and the output ``dst`` tensor can be any supported NKI data types
    (see :ref:`nki-dtype` for more information). The DMA engines automatically cast input data types to float32
    before performing the scaled reduction computation. The float32 computation results are then cast to the
    data type of ``dst`` in a pipelined fashion.

    **Layout.**

    The computation is performed element-wise across all tensors, with the reduction operation applied
    across the scaled source tensors at each element position.

    **Tile size.**

    The element count of each tensor in ``srcs`` and ``dst`` must match exactly.
    The max number of source tensors in ``srcs`` is 16.

    :param dst: the output tensor to store the computed results
    :param srcs: a list of input tensors to be scaled and reduced
    :param scales: a list of scale factors corresponding to each tensor in ``srcs`` (must be [1.0, 1.0, ...])
    :param reduce_op: the reduction operation to apply (currently only ``nl.add`` is supported)

    """
    ...

def dma_copy(dst, src, dst_rmw_op=None, oob_mode=oob_mode.error, dge_mode=dge_mode.unknown, name=None):
    r"""
    Copy data from ``src`` to ``dst`` using DMA engines with optional read-modify-write operations.

    This instruction performs data movement between memory locations (SBUF or HBM) using DMA engines. The basic operation
    copies data from the source tensor to the destination tensor: ``dst = src``. Optionally, a read-modify-write
    operation can be performed where the source data is combined with existing destination data using a specified
    operation: ``dst = dst_rmw_op(dst, src)``.

    Currently, only ``np.add`` is supported for ``dst_rmw_op`` when performing read-modify-write operations.
    When ``dst_rmw_op=None``, the source data directly overwrites the destination data.

    ``nisa.dma_copy`` supports different modes of DMA descritpor generation (DGE):

    - ``nisa.dge_mode.none``: Neuron Runtime generates DMA descriptors and stores them into HBM before NEFF execution.
    - ``nisa.dge_mode.swdge``: Gpsimd Engine generates DMA descriptors as part of the ``nisa.dma_copy`` instruction
      during NEFF execution.
    - ``nisa.dge_mode.hwdge``: Sync Engine or Scalar Engine sequencers invoke DGE hardware block to generate DMA
      descriptors as part of the ``nisa.dma_copy`` instruction during NEFF execution.

    See `Trainium2 arch guide` and `Introduction to DMA with NKI` for more discussion.

    When either ``sw_dge`` or ``hw_dge`` mode is used, the ``src`` and ``dst`` tensors can have a dynamic start address
    which depends on a variable that cannot be resolved at compile time. When ``sw_dge`` is selected, ``nisa.dma_copy``
    can also perform a gather or scatter operation, using a list of **unique** dynamic indices from SBUF.
    In both of these dynamic modes, out-of-bound address checking is turned on automatically during execution.
    By default a runtime error is raised (``oob_mode=oob_mode.error`` as default setting).
    Developers can disable this error and make the nisa.dma_copy instruction skips the DMA transfer for a given dynamic
    address or index when it is out of bound using ``oob_mode=oob_mode.skip``.
    If ``dst_rmw_op`` is specified for these dynamic modes, only ``oob_mode.error`` is allowed.
    See Beta2 NKI kernel migration guide for the latest syntax to handle dynamic addresses or indices.

    ``nisa.dma_copy`` also supports non-unique scatter indices when ``dge_mode=nisa.dge_mode.none``
    and ``dst_rmw_op=nl.add`` are set. An example use case for this is performing embedding table entry updates
    after a training backward pass to calcualte embedding table gradients.

    **Memory types.**

    Both ``src`` and ``dst`` tiles can be in HBM or SBUF. However, if both tiles are in SBUF, consider using
    :doc:`nisa.tensor_copy <nki.isa.tensor_copy>` instead for better performance.

    **Data types.**

    Both ``src`` and ``dst`` tiles can be any supported NKI data types (see :ref:`nki-dtype` for more information).

    The DMA engines automatically handle data type conversion when ``src`` and ``dst`` have different data types.
    The conversion is performed through a two-step process: first casting from ``src.dtype`` to float32, then
    from float32 to ``dst.dtype``.

    If ``dst_rmw_op`` is used, the DMA engines automatically cast input data types to float32
    before performing the read-modify-write computation, and the final float32 result is cast to the output
    data type in a pipelined fashion.


    **Layout.**

    If ``dst_rmw_op`` is used, the computation is done element-wise between ``src`` and `dst`.


    **Tile size.**

    The total number of data elements in ``src`` must match that of ``dst``.


    :param dst: the destination tensor to copy data into
    :param src: the source tensor to copy data from
    :param dst_rmw_op: optional read-modify-write operation (currently only ``np.add`` is supported)
    :param dge_mode: (optional) specify which Descriptor Generation Engine (DGE) mode to use for DMA descriptor generation: ``nki.isa.dge_mode.none`` (turn off DGE) or ``nki.isa.dge_mode.swdge`` (software DGE) or ``nki.isa.dge_mode.hwdge`` (hardware DGE)  or ``nki.isa.dge_mode.unknown`` (by default, let compiler select the best DGE mode). Hardware based DGE is only supported for NeuronCore-v3 or newer. See `Trainium2 arch guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium2_arch.html>`__ for more information.
    :param oob_mode: (optional) Specifies how to handle out-of-bounds (oob) array indices during indirect access operations. Valid modes are:

        - ``oob_mode.error``: (Default) Raises an error when encountering out-of-bounds indices.
        - ``oob_mode.skip``: Silently skips any operations involving out-of-bounds indices.

        For example, when using indirect gather/scatter operations, out-of-bounds indices can occur if the index array contains values that exceed the dimensions of the target array.

    """
    ...

def dma_transpose(dst, src, axes=None, dge_mode=dge_mode.unknown, name=None):
    r"""
    Perform a transpose on input ``src`` using DMA Engine.

    The permutation of transpose follow the rules described below:

    1. For 2-d input tile, the permutation will be [1, 0]
    2. For 3-d input tile, the permutation will be [2, 1, 0]
    3. For 4-d input tile, the permutation will be [3, 1, 2, 0]

    The only valid ``dge_mode`` s are ``unknown`` and ``hwdge``. If ``hwdge``, this instruction will be lowered
    to a Hardware DGE transpose. This has additional restrictions:

    1. ``src.shape[0] == 16``
    2. ``src.shape[-1] % 128 == 0``
    3. ``dtype`` is 2 bytes

    :param src: the source of transpose, must be a tile in HBM or SBUF.
    :param axes: transpose axes where the i-th axis of the transposed tile will correspond to the axes[i] of the source.
                 Supported axes are ``(1, 0)``, ``(2, 1, 0)``, and ``(3, 1, 2, 0)``.
    :param dge_mode: (optional) specify which Descriptor Generation Engine (DGE) mode to use for DMA descriptor generation: ``nki.isa.dge_mode.none`` (turn off DGE) or ``nki.isa.dge_mode.swdge`` (software DGE) or ``nki.isa.dge_mode.hwdge`` (hardware DGE)  or ``nki.isa.dge_mode.unknown`` (by default, let compiler select the best DGE mode). Hardware based DGE is only supported for NeuronCore-v3 or newer. See `Trainium2 arch guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium2_arch.html>`__ for more information.


    """
    ...

def dropout(dst, data, prob, name=None):
    r"""
    Randomly replace some elements of the input tile ``data`` with zeros
    based on input probabilities using Vector Engine.
    The probability of replacing input elements with zeros (i.e., drop probability)
    is specified using the ``prob`` field:
    - If the probability is 1.0, all elements are replaced with zeros.
    - If the probability is 0.0, all elements are kept with their original values.

    The ``prob`` field can be a scalar constant or a tile of shape ``(data.shape[0], 1)``,
    where each partition contains one drop probability value.
    The drop probability value in each partition is applicable to the input
    ``data`` elements from the same partition only.

    Data type of the input ``data`` tile can be any valid NKI data types
    (see :ref:`nki-dtype` for more information).
    However, data type of ``prob`` has restrictions based on the data type of ``data``:

    - If data type of ``data`` is any of the integer types (e.g., int32, int16),
      ``prob`` data type must be float32
    - If data type of data is any of the float types (e.g., float32, bfloat16),
      ``prob`` data can be any valid float type

    The output data type of this instruction is specified by the ``dtype`` field. The output data type
    must match the input data type of ``data`` if input data type is any of the integer types.
    Otherwise, output data type can be any valid NKI data types. If output data type is not specified,
    it is default to be the same as input data type.


    :param dst: an output tile of the dropout result
    :param data: the input tile
    :param prob: a scalar or a tile of shape ``(data.shape[0], 1)`` to indicate the
                 probability of replacing elements with zeros


    """
    ...

def iota(dst, pattern, offset, channel_multiplier=0, name=None):
    r"""
    Generate a constant literal pattern into SBUF using GpSimd Engine.

    The pattern is defined by an int32 ``offset``, a tensor access pattern of up to 4D ``pattern`` and
    an int32 ``channel_multiplier``. The ``pattern`` field is a list of lists in the form of
    ``[[step_w, num_w], [step_z, num_z], [step_y, num_y], [step_x, num_x]]``. When fewer than 4D ``pattern``
    is provided, NKI compiler automatically pads remaining dimensions with size of 1.

    Given a 4D pattern (padded if needed), the instruction generates a stream of values using the following pseudo code:

    .. code-block:: python

        num_partitions = dst.shape[0]
        [[step_w, num_w], [step_z, num_z], [step_y, num_y], [step_x, num_x]] = pattern

        for channel_id in range(num_partitions):
            for w in range(num_w):
                for z in range(num_z):
                    for y in range(num_y):
                        for x in range(num_x):
                            value = offset + (channel_id * channel_multiplier) +
                                    (w * step_w) + (z * step_z) + (y * step_y) + (x * step_x)

                            dst[channel_id, w, z, y, x] = value


    The above pseudo code assumes ``dst`` has the same size in every dimension ``x/y/z/w`` for simplicity. However,
    the instruction allows any sizes in the free dimension, as long as the number of elements per partition in ``dst``
    matches the product: ``num_w * num_z * num_y * num_x``.

    **Memory types.**

    The output ``dst`` tile must be in SBUF.

    **Data types.**

    The generated values are computed in 32-bit integer arithmetic. The GpSimd Engine can cast
    these integer results to any valid NKI data type (see :ref:`nki-dtype` for more information)
    before writing to the output tile. The output data type is determined by the ``dst`` tile's
    data type.

    **Layout.**

    The partition dimension determines the number of active channels for parallel pattern generation.

    **Tile size.**

    The partition dimension size of ``dst`` must not exceed 128. The number of
    elements per partition of ``dst`` must not exceed the physical size of each SBUF partition.
    The total number of elements in ``pattern`` must match the number of elements per partition in the ``dst`` tile.


    :param dst: the output tile in SBUF to store the generated pattern
    :param pattern: a list of [step, num] to describe up to 4D tensor sizes and strides
    :param offset: an int32 offset value to be added to every generated value
    :param channel_multiplier: an int32 multiplier to be applied to the channel (parition) ID


    """
    ...

def local_gather(dst, src_buffer, index, num_elem_per_idx=1, num_valid_indices=None, name=None):
    r"""
    Gather SBUF data in ``src_buffer`` using ``index`` on GpSimd Engine.

    Each of the eight GpSimd cores in GpSimd Engine connects to 16 contiguous SBUF partitions
    (e.g., core[0] connected to partition[0:16]) and performs gather from the connected 16
    SBUF partitions *independently* in parallel. The indices used for gather on each core should also
    come from the same 16 connected SBUF partitions.

    During execution of the instruction, each GpSimd core reads a 16-partition slice from ``index``, flattens
    all indices into a 1D array ``indices_1d`` (along the partition dimension first).
    By default with no ``num_valid_indices`` specified, each GpSimd core
    will treat all indices from its corresponding 16-partition ``index`` slice as valid indices.
    However, when the number of valid indices per core
    is not a multiple of 16, users can explicitly specify the valid index count per core in ``num_valid_indices``.
    Note, ``num_valid_indices`` must not exceed the total element count in each 16-partition ``index`` slice
    (i.e., ``num_valid_indices <= index.size / (index.shape[0] / 16)``).

    Next, each GpSimd core uses the flattened ``indices_1d`` indices as *partition offsets* to gather from
    the connected 16-partition slice of ``src_buffer``. Optionally, this API also allows gathering of multiple
    contiguous elements starting at each index to improve gather throughput, as indicated by ``num_elem_per_idx``.
    Behavior of out-of-bound index access is undefined.

    Even though all eight GpSimd cores can gather with completely different indices, a common use case for
    this API is to make all cores gather with the same set of indices (i.e., partition offsets). In this case,
    users can generate indices into 16 partitions, replicate them eight times to 128 partitions and then feed them into
    ``local_gather``.

    As an example, if ``src_buffer`` is (128, 512) in shape and ``index`` is (128, 4) in shape, where the partition
    dimension size is 128, ``local_gather`` effectively performs the following operation:

    .. nki_example:: ../../test/test_nki_isa_local_gather.py
     :language: python
     :marker:   NUMPY_SEMANTICS

    ``local_gather`` preserves the input data types from ``src_buffer`` in the gather output.
    Therefore, no data type casting is allowed in this API. The indices in ``index`` tile must be uint16 types.

    This API has three tile size constraints [subject to future relaxation]:

    #. The partition axis size of ``src_buffer`` must match that of ``index`` and must
       be a multiple of 16. In other words, ``src_buffer.shape[0] == index.shape[0] and src_buffer.shape[0] % 16 == 0``.
    #. The number of contiguous elements to gather per index per partition ``num_elem_per_idx``
       must be one of the following values: ``[1, 2, 4, 8, 16, 32]``.
    #. The number of indices for gather per core must be less than or equal to 4096.


    :param dst: an output tile of the gathered data
    :param src_buffer: an input tile for gathering.
    :param index: an input tile with indices used for gathering.
    :param num_elem_per_idx: an optional integer value to read multiple contiguous elements per index per partition; default is 1.
    :param num_valid_indices: an optional integer value to specify the number of valid indices per GpSimd core; default is
                              ``index.size / (index.shape[0] / 16)``.


    Click :download:`here <../../test/test_nki_isa_local_gather.py>` to download the
    full NKI code example with equivalent numpy implementation.
    """
    ...

def max8(dst, src, name=None):
    r"""
    Find the 8 largest values in each partition of the source tile.

    This instruction reads the input elements, converts them to fp32 internally, and outputs
    the 8 largest values in descending order for each partition. By default, returns the
    same dtype as the input tensor.

    The source tile can be up to 5-dimensional, while the output tile is always 2-dimensional.
    The number of elements read per partition must be between 8 and 16,384 inclusive.
    The output will always contain exactly 8 elements per partition.
    The source and output must have the same partition dimension size:

    - source: [par_dim, ...]
    - output: [par_dim, 8]


    :param dst: a 2D tile containing the 8 largest values per partition in descending order with shape [par_dim, 8]
    :param src: the source tile to find maximum values from



    """
    ...

def memset(dst, value, engine=engine.unknown, name=None):
    r"""
    Initialize a tile filled with a compile-time constant value using Vector or GpSimd Engine.
    The shape of the tile is specified in the ``shape`` field and the
    initialized value in the ``value`` field.
    The memset instruction supports all valid NKI dtypes
    (see :ref:`nki-dtype`).

    :param dst: a tile with shape `shape` whose elements are initialized to `value`.
    :param value: the constant value to initialize with
    :param engine: specify which engine to use for memset: ``nki.isa.vector_engine`` or ``nki.isa.gpsimd_engine`` ;
                   ``nki.isa.unknown_engine`` by default, lets compiler select the best engine for the given
                   input tile shape



    """
    ...

def nc_find_index8(dst, data, vals, name=None):
    r"""
    Find indices of the 8 given vals in each partition of the data tensor.

    This instruction first loads the 8 values,
    then loads the data tensor and outputs the indices (starting at 0) of the first
    occurrence of each value in the data tensor, for each partition.

    The data tensor can be up to 5-dimensional, while the vals tensor must be up
    to 3-dimensional. The data tensor must have between 8 and 16,384 elements per
    partition. The vals tensor must have exactly 8 elements per partition.
    The output will contain exactly 8 elements per partition and will be uint16 or
    uint32 type. Default output type is uint32.

    Behavior is undefined if vals tensor contains values that are not in
    the data tensor.

    If provided, a mask is applied only to the data tensor.


    :param dst: a 2D tile containing indices (uint16 or uint32) of the 8 values in each partition with shape [par_dim, 8]
    :param data: the data tensor to find indices from
    :param vals: tensor containing the 8 values per partition whose indices will be found



    """
    ...

def nc_match_replace8(dst, data, vals, imm, dst_idx=None, name=None):
    r"""
    Replace first occurrence of each value in ``vals`` with ``imm`` in ``data``
    using the Vector engine and return the replaced tensor. If ``dst_idx``
    tile is provided, the indices of the matched values are written to ``dst_idx``.

    This instruction reads the input ``data``, replaces the first occurrence of each
    of the given values (from ``vals`` tensor) with the specified immediate constant and,
    optionally, output indices of matched values to ``dst_idx``. When performing the operation,
    the free dimensions of both ``data`` and ``vals`` are flattened. However, these dimensions
    are preserved in the replaced output tensor and in ``dst_idx`` respectively. The partition
    dimension defines the parallelization boundary. Match, replace, and index
    generation operations execute independently within each partition.

    The ``data`` tensor can be up to 5-dimensional, while the ``vals`` tensor can be up
    to 3-dimensional. The ``vals`` tensor must have exactly 8 elements per partition.
    The data tensor must have no more than 16,384 elements per partition.
    The replaced output will have the same shape as the input data tensor. ``data`` and ``vals``
    must have the same number of partitions. Both input tensors can come from SBUF
    or PSUM.

    Behavior is undefined if vals tensor contains values that are not in the data
    tensor.

    If provided, a mask is applied to the data tensor.


    **NumPy equivalent:**

    .. code-block:: python

        # Let's assume we work with NumPy, and ``data``, ``vals`` are 2-dimensional arrays
        # (with shape[0] being the partition axis) and imm is a constant float32 value.

        import numpy as np

        # Get original shapes
        data_shape = data.shape
        vals_shape = vals.shape

        # Reshape to 2D while preserving first dimension
        data_2d = data.reshape(data_shape[0], -1)
        vals_2d = vals.reshape(vals_shape[0], -1)

        # Initialize output array for indices
        indices = np.zeros(vals_2d.shape, dtype=np.uint32)

        for i in range(data_2d.shape[0]):
          for j in range(vals_2d.shape[1]):
            val = vals_2d[i, j]
            # Find first occurrence of val in data_2d[i, :]
            matches = np.where(data_2d[i, :] == val)[0]
            if matches.size > 0:
              indices[i, j] = matches[0]  # Take first match
              data_2d[i, matches[0]] = imm

        output = data_2d.reshape(data.shape)
        indices = indices.reshape(vals.shape) # Computed only if ``dst_idx`` is specified

    :param dst: the modified data tensor
    :param data: the data tensor to modify
    :param dst_idx: (optional) the destination tile to write flattened indices of matched values
    :param vals: tensor containing the 8 values per partition to replace
    :param imm: float32 constant to replace matched values with



    """
    ...

def nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, tile_position=(), tile_size=(), psum_accumulate_flag=3, name=None):
    r"""
    Compute ``dst = stationary.T @ moving`` matrix multiplication using Tensor Engine.

    The figure below illustrates how to map a matrix multiplication from a mathematical definition
    to ``nisa.nc_matmul`` on Tensor Engine. For more detailed discussion of Tensor Engine capabilities, see
    `Trainium arch guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium_inferentia2_arch.html>`_.


    .. figure:: ../../img/arch_images/matmul.png
      :align: center
      :width: 100%

      MxKxN Matrix Multiplication Visualization.

    **Performance mode.**

    On NeuronCore-v2, performance mode is not supported.
    On NeuronCore-v3 and NeuronCore-v4, Tensor Engine supports FP8 double performance mode, enabled by setting
    performance mode to ``double_row``.
    See `Trainium2 arch guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium2_arch.html>`_
    for more details.
    ``double_row`` performance mode cannot be combined with Tensor Engine column tiling mode (details below).

    **Tiling mode.**
    NeuronCore Tensor Engine is built upon a systolic array with 128 rows and 128 columns of processing elements (PEs).
    Tensor Engine supports both row and column tiling modes, which allow multiple ``nc_matmul`` instructions with
    a stationary tile size smaller than [128, 128] to run in parallel to improve hardware utilization.
    Row tiling mode slices the 128 PE rows into 2x 64 row
    tiles (NeuronCore-v2 or newer), or 4x 32 row tiles (NeuronCore-v3 or newer). Column tiling mode slices
    the 128 PE columns in the same fashion. The row and column tile sizes can be set independently in the
    ``tile_size`` field as a tuple ``(row_size, column_size)``. The stationary tile size must not exceed the chosen
    ``tile_size``.

    In addition, a given ``nc_matmul`` can also pick the exact row and column tile within the 128x128 systolic
    array, by specifying the starting row and starting column in ``tile_position`` as a
    tuple ``(start_row, start_column)``. The ``start_row`` must be a multiple of ``row_size`` specified in ``tile_size``
    and must not exceed 128. Similarly, the ``start_column`` must be a multiple of ``column_size`` and must not exceed 128.

    For example, setting ``tile_position`` to (64, 0) and ``tile_size`` to (64, 128) means using the bottom half
    of the systolic array.

    Note, ``tile_position`` and ``tile_size`` must both be set to enable tiling mode. If they are not set,
    the default is to use the full systolic array, which is equivalent to ``tile_position=(0, 0)``
    and ``tile_size=(128, 128)``. The values in ``tile_position`` and ``tile_size`` tuples can be
    integers or affine expressions.

    **Transpose mode.**

    Tensor Engine can transpose a tile in SBUF by loading it as a stationary tile and using an identity matrix
    as the moving tile.
    Starting NeuronCore-v3, turning on transpose mode by setting ``is_transpose=True`` enables bit-accurate
    data transpose, which can transpose tensors with NaN/Inf values properly.
    See `Trainium2 arch guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium2_arch.html>`_
    for more details.

    On NeuronCore-v2, Tensor Engine does not support transpose mode natively. However, setting ``is_transpose=True``
    ensures neuron-profile identifies this instruction as a transpose for performance metric accounting purposes.

    **Memory types.**

    The ``nc_matmul`` instruction *must* read inputs from SBUF and
    write outputs to PSUM. Therefore, the ``stationary`` and ``moving`` must be SBUF tiles, and ``dst`` tile
    must be a PSUM tile.

    The ``psum_accumulate_flag`` controls whether the matmul result data should overwrite or accumulate on top of
    the ``dst`` PSUM tile. Multiple ``nisa.nc_matmul`` instructions accumulating into the same PSUM tile
    can form an accumulation group before the PSUM tile content is evicted back to SBUF. The encoding of
    ``psum_accumulate_flag`` is as follows:

    - bit[0] of ``psum_accumulate_flag``: if set, indicates this ``nisa.nc_matmul`` call is the first instruction
      in the accumulation group. The matmul result should overwrite the existing content in the ``dst`` PSUM tile.
    - bit[1] of ``psum_accumulate_flag``: if set, indicates this ``nisa.nc_matmul`` call is the last instruction
      in the accumulation group. The matmul result should accumulate to the existing content in the ``dst`` PSUM tile.
    - bit[2] of ``psum_accumulate_flag``: if set, indicates this ``nisa.nc_matmul`` call is the first instruction
      in the accumulation group. However, the matmul result should accumulate to the existing content in the ``dst`` PSUM tile.

    ``nisa.nc_matmul`` calls that are not the first or last instruction of an accumulation group should not set any bit:
    ``psum_accumulate_flag=0``.

    **Data types.**

    The input ``stationary`` and ``moving`` tiles can be one of these supported data types:
    ``float8_e4m3/float8_e5m2/bfloat16/float16/tfloat32/float32``. The ``stationary`` and ``moving`` tiles
    can have different data types, with one exception: if one of the input tiles is ``tfloat32/float32``,
    the other tile must also be ``tfloat32/float32``.
    On NeuronCore-v3 and NeuronCore-v4, when performance mode is ``double_row``, ``stationary`` and ``moving`` tiles
    must be one of ``float8_e4m3`` or ``float8_e5m2``, but the two input tiles can have different float8 formats.

    The accumulation precision internal to Tensor Engine is float32.
    The ``dst`` tile must be a float32 tile in NeuronCore-v2 and NeuronCore-v3. Starting NeuronCore-v4,
    ``dst`` can either be a float32 or bfloat16 tile.

    **Layout.**

    If performance mode is off, the contraction dimension of the matmul must be along the partition dimension in
    both ``stationary`` and ``moving`` tiles.

    If performance mode is ``double_row``, the contraction dimension of the matmul is split between the partition dimension
    and the first free dimension after the partition dimension in both ``stationary`` and ``moving`` tiles.
    The first free dimension must be 2. For example, to perform a matmul of ``[1, 256]@[256, 3]=[1, 3]``, the stationary
    tile is of shape ``[128, 2, 1]``, while the moving tile is of shape ``[128, 2, 3]``.

    Regardless of performance mode, the free dimension of the ``stationary`` tile matches the partition
    dimension of the output ``dst`` tile in size, while the free dimension of the ``moving`` tile
    matches the free dimension of the ``dst`` tile in size.

    **Tile size.**

    The partition dimension sizes of the ``stationary`` and ``moving`` tiles must be identical. They must not
    exceed 128 when tiling mode is off or ``row_size`` specified in ``tile_size`` when tiling mode is on.
    The free dimension size of ``stationary`` must not exceed 128 when tiling mode is off or ``column_size``
    in ``tile_size`` when tiling mode is on.

    On NeuronCore-v2 and -v3, the free dimension size of ``moving`` tile must not exceed 512, matching the maximum
    number of float32 elements per PSUM bank. Starting NeuronCore-v4, the free dimension size of ``moving`` tile
    can go up to 4096 for float32 ``dst`` or 8192 for bfloat16 ``dst``, matching the size of 8x PSUM banks
    (the entire PSUM).


    Explicit tiling is required when the high-level matmul operation exceeds the tile size limits of ``nc_matmul``.

    :param dst: the matmul output
    :param stationary: the stationary operand
    :param moving: the moving operand
    :param is_stationary_onezero: hints to the compiler whether the ``stationary`` operand is a tile with ones/zeros only;
                           setting this field explicitly could lead to 2x better performance
                           if ``stationary`` tile is in float32; the field has no impact for non-float32 ``stationary``
    :param is_moving_onezero: hints to the compiler whether the ``moving`` operand is a tile with ones/zeros only;
                           setting this field explicitly could lead to 2x better performance
                           if ``moving`` tile is in float32; the field has no impact for non-float32 ``moving``
    :param is_transpose: controls Tensor Engine transpose mode on/off starting NeuronCore-v3
    :param tile_position: a 2D tuple (start_row, start_column) to control starting row in Tensor Engine tiling mode; start_column must be 0
    :param tile_size: a 2D tuple (row_size, column_size) to control row tile size in Tensor Engine tiling mode; column_size must be 128
    :param psum_accumulate_flag: controls PSUM near-memory accumulation in the ``dst`` tile


    """
    ...

def nc_matmul_mx(dst, stationary, moving, stationary_scale, moving_scale, tile_position=None, tile_size=None, psum_accumulate_flag=3, name=None):
    r"""
    Compute matrix multiplication of MXFP8/MXFP4 quantized matrices with integrated dequantization using Tensor Engine.

    .. note::

      Available only on NeuronCore-v4 and newer.

    The NeuronCore-v4 Tensor Engine supports matrix multiplication of MXFP8/MXFP4 quantized matrices as defined in the
    `OCP Microscaling standard <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__.
    This instruction performs matrix multiplication between quantized ``stationary`` and ``moving`` matrices while
    applying dequantization scales during computation. The micro-scaling group size is 32 elements in groupss of
    8 partitions × 4 elements per partition of both ``stationary`` and ``moving`` tensors.
    See `Trainium3 arch guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/about/trainium3_arch.html>`_
    for more detailed discussion.

    **Tiling Mode.**

    NeuronCore Tensor Engine is built upon a systolic array with 128 rows and 128 columns of processing elements (PEs).
    For ``nc_matmul_mx``, Tensor Engine supports only row tiling mode, which allows multiple ``nc_matmul_mx`` instructions with
    a stationary partition dimension size smaller than 128 to run in parallel to improve hardware utilization.
    Row tiling mode slices the 128 PE rows into 2x 64 row tiles or 4x 32 row tiles.

    The row tile size can be set in the ``tile_size`` field as a tuple ``(row_size, column_size)``,
    where ``column_size`` must be 128.
    The stationary tile size must not exceed the chosen ``tile_size``.

    A given ``nc_matmul_mx`` can pick the exact row tile within the 128x128 systolic array by specifying the starting row
    in ``tile_position`` as a tuple ``(start_row, start_column)``, where ``start_column`` must be 0.
    The ``start_row`` must be a multiple of ``row_size`` specified in ``tile_size`` and must not exceed 128.

    For example, setting ``tile_position`` to (64, 0) and ``tile_size`` to (64, 128) means using the bottom half
    of the systolic array.

    Note, ``tile_position`` and ``tile_size`` must both be set to enable tiling mode. If they are not set,
    the default is to use the full systolic array, which is equivalent to ``tile_position=(0, 0)``
    and ``tile_size=(128, 128)``. The values in ``tile_position`` and ``tile_size`` tuples can be
    integers or affine expressions.

    **Memory types.**

    The ``nc_matmul_mx`` instruction must read inputs from SBUF and write outputs to PSUM. Therefore, the
    ``stationary``, ``moving``, ``stationary_scale``, and ``moving_scale`` must be SBUF tiles, and ``dst``
    tile must be a PSUM tile.

    The ``psum_accumulate_flag`` controls whether the matmul result data should overwrite or accumulate on top of
    the ``dst`` PSUM tile. Multiple ``nisa.nc_matmul_mx`` instructions accumulating into the same PSUM tile
    can form an accumulation group before the PSUM tile content is evicted back to SBUF. The

    - bit[0] of ``psum_accumulate_flag``: if set, indicates this ``nisa.nc_matmul_mx`` call is the first instruction
      in the accumulation group. The matmul result should overwrite the existing content in the ``dst`` PSUM tile.
    - bit[1] of ``psum_accumulate_flag``: if set, indicates this ``nisa.nc_matmul_mx`` call is the last instruction
      in the accumulation group. The matmul result should accumulate to the existing content in the ``dst`` PSUM tile.
    - bit[2] of ``psum_accumulate_flag``: if set, indicates this ``nisa.nc_matmul_mx`` call is the first instruction
      in the accumulation group. However, the matmul result should accumulate to the existing content in the ``dst`` PSUM tile.

    ``nisa.nc_matmul_mx`` calls that are not the first or last instruction of an accumulation group should not set any bit:
    ``psum_accumulate_flag=0``.

    **Data types.**

    The input ``stationary`` and ``moving`` tiles must be float8_e5m2_x4, float8_e4m3fn_x4, or float4_e2m1fn_x4
    (4-packed quantized data types). The ``stationary_scale`` and ``moving_scale`` tiles must be uint8.
    The ``dst`` tile can be float32 or bfloat16.

    The 4-packed data types (float8_e5m2_x4/float8_e4m3fn_x4/float4_e2m1fn_x4) pack multiple quantized values
    into single elements. These packed data types are required because 4 microscaling quantized data values
    share 1 scale value and must operate together as a compact group.

    **Layout.**

    The contraction dimension of the matrix multiplication is along the partition dimension of ``stationary``
    and ``moving`` tensors and also the x4 dimension within each packed data type element
    (float8_e5m2_x4, float8_e4m3fn_x4, or float4_e2m1fn_x4).

    The free dimension of the ``stationary`` tile matches the partition
    dimension of the output ``dst`` tile in size, while the free dimension of the ``moving`` tile
    matches the free dimension of the ``dst`` tile in size.

    The scale tensors follow a special layout requirement. See more details in ``nisa.quantize_mx`` API doc.

    *Tile size*

    - The partition dimension size of ``stationary`` and ``moving`` must be identical and be a multiple of 32,
      not exceeding 128.
    - The free dimension size of ``stationary`` must be even and not exceed 128.
    - The free dimension size of ``moving`` must not exceed 512 when ``dst`` is in float32 or 1024 when ``dst`` is in bfloat16.
    - The scale tensors have partition dimensions that depend on whether the data tensors span multiple quadrants.
      See more details in ``nisa.quantize_mx`` API doc.


    :param dst: the matrix multiplication output (PSUM tile)
    :param stationary: the stationary quantized matrix (SBUF tile)
    :param moving: the moving quantized matrix (SBUF tile)
    :param stationary_scale: the dequantization scales for stationary matrix (SBUF tile)
    :param moving_scale: the dequantization scales for moving matrix (SBUF tile)
    :param tile_position: a 2D tuple (start_row, start_column) to control starting row and column in Tensor Engine tiling mode
    :param tile_size: a 2D tuple (row_size, column_size) to control row and column tile sizes in Tensor Engine tiling mode
    :param psum_accumulate_flag: controls PSUM near-memory accumulation in the ``dst`` tile

    """
    ...

def nc_stream_shuffle(dst, src, shuffle_mask, name=None):
    r"""
    Apply cross-partition data movement within a quadrant of 32 partitions from source tile
    ``src`` to destination tile ``dst`` using Vector Engine.

    Both source and destination tiles can be in either SBUF or PSUM, and passed in by reference as arguments.
    In-place shuffle is allowed, i.e., ``dst`` same as ``src``. ``shuffle_mask`` is a 32-element list. Each mask
    element must be in data type int or affine expression. ``shuffle_mask[i]`` indicates which input partition the
    output partition [i] copies from within each 32-partition quadrant. The special value ``shuffle_mask[i]=255``
    means the output tensor in partition [i] will be unmodified. ``nc_stream_shuffle`` can be applied to multiple
    of quadrants. In the case with more than one quadrant, the shuffle is applied to each quadrant independently,
    and the same ``shuffle_mask`` is used for each quadrant. For more information about the cross-partition data movement,
    see :ref:`arch_guide_cross_partition_data_movement`.

    This API has 3 constraints on ``src`` and ``dst``:

    #. ``dst`` must have same data type as ``src``.
    #. ``dst`` must have the same number of elements per partition as ``src``.
    #. The access start partition of ``src`` (``src_start_partition``), does not have to match or be in the same quadrant
       as that of ``dst`` (``dst_start_partition``). However, ``src_start_partition``/``dst_start_partition`` needs to follow
       some special hardware rules with the number of active partitions ``num_active_partitions``.
       ``num_active_partitions = ceil(max(src_num_partitions, dst_num_partitions)/32) * 32``, where ``src_num_partitions`` and
       ``dst_num_partitions`` refer to the number of partitions the ``src`` and ``dst`` tensors access respectively.
       ``src_start_partition``/``dst_start_partition`` is constrained based on the value of ``num_active_partitions``:

      * If ``num_active_partitions`` is 96/128, ``src_start_partition``/``dst_start_partition`` must be 0.

      * If ``num_active_partitions`` is 64, ``src_start_partition``/``dst_start_partition`` must be 0/64.

      * If ``num_active_partitions`` is 32, ``src_start_partition``/``dst_start_partition`` must be 0/32/64/96.


    :param dst: the destination tile
    :param src: the source tile
    :param shuffle_mask: a 32-element list that specifies the shuffle source and destination partition



    """
    ...

def nc_transpose(dst, data, engine=engine.unknown, name=None):
    r"""
    Perform a 2D transpose between the partition axis and the free axis of input ``data`` using Tensor or Vector Engine.

    If the ``data`` tile has more than one free axis, this API implicitly flattens all free axes into one axis
    and then performs a 2D transpose.

    2D transpose on Tensor Engine is implemented by performing a matrix multiplication between ``data`` as the
    stationary tensor and an identity matrix as the moving tensor. This is equivalent to calling ``nisa.nc_matmul``
    directly with ``is_transpose=True``. See :ref:`architecture guide <arch_sec_tensor_engine_alternative_use>`
    for more information. On NeuronCore-v2, Tensor Engine transpose is not bit-accurate if the input ``data``
    contains NaN/Inf.
    You may consider replacing NaN/Inf with regular floats (float_max/float_min/zeros) in the input matrix.
    Starting NeuronCore-v3, all Tensor Engine transpose is bit-accurate.

    **Memory types.**

    Tensor Engine ``nc_transpose`` must read the input tile from SBUF and write the transposed result to PSUM.
    Vector Engine ``nc_transpose`` can read/write from/to either SBUF or PSUM.

    **Data types.**

    The input ``data`` tile can be any valid NKI data type (see :ref:`nki-dtype` for more information).
    The output ``dst`` tile must have the same data type as that of ``data``.

    **Layout.**
    The partition dimension of ``data`` tile becomes the free dimension of the ``dst`` tile.
    Similarly, the free dimension of the ``data`` tile becomes the partition dimension of the ``dst`` tile.

    **Tile size.**
    Tensor Engine ``nc_transpose`` can handle an input tile of shape [128, 128] or smaller, while Vector
    Engine can handle shape [32, 32] or smaller.
    If no ``engine`` is specified, Neuron Compiler will automatically select an engine
    based on the input shape.


    :param dst: the transpose output
    :param data: the input tile to be transposed
    :param engine: specify which engine to use for transpose: ``nki.isa.tensor_engine`` or ``nki.isa.vector_engine``;
                   by default, the best engine will be selected for the given input tile shape

    """
    ...

def quantize_mx(dst, src, dst_scale, name=None):
    r"""
    Quantize FP16/BF16 data to MXFP8 tensors (both data and scales) using Vector Engine.

    .. note::

      Available only on NeuronCore-v4 and newer.

    The resulting MXFP8 tensors, ``dst`` and ``dst_scale`` are as defined in the
    `OCP Microscaling standard <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__.
    This instruction calculates the required scales for each group of 32 values in ``src``, divides them by the calculated scale,
    and casts to the target MXFP8 datatype. The output layout is suitable for direct consumption by the
    ``nisa.nc_matmul_mx`` API running on Tensor Engine.

    **Memory types.**

    All input ``src`` and output tiles (``dst`` and ``dst_scale``) must be in SBUF.

    **Data types.**

    The input ``src`` tile must be float16 or bfloat16. The output ``dst`` tile must be float8_e5m2_x4 or
    float8_e4m3fn_x4 (4-packed FP8 data types). The ``dst_scale`` tile must be uint8.

    The 4-packed data types (float8_e5m2_x4/float8_e4m3fn_x4) are 32-bit data types that pack four 8-bit
    float8_e5m2/float8_e4m3fn values.

    **Layout.**

    The quantization operates on groups of 32 elements from the input ``src`` tile, where each group consists of
    8 partitions × 4 elements per partition. For each 32-element group, the instruction produces:

    - Quantized FP8 data in ``dst``
    - One shared scale value in ``dst_scale`` per group

    Logically, ``dst`` should have the same shape as ``src`` if ``dst`` is interpreted as a pure FP8 data type.
    However, in NKI, ``dst`` uses a custom 4-packed data type that packs four contiguous
    FP8 elements into a single float8_e5m2_x4/float8_e4m3fn_x4 element. Therefore, ``dst`` has one quarter of
    the element count per partition compared to that of ``src``.

    Logically, ``dst_scale`` should have 1/32 the element count of ``src`` due to the microscaling group size of 32.
    Physically, the ``dst_scale`` tensor follows a special SBUF quadrant (32 partitions) distribution pattern
    where scale values are distributed across multiple SBUF quadrants while maintaining the same
    partition offset at each quadrant.
    Within each SBUF quadrant, a 32-partition slice of ``src`` tile produces 32//8 = 4 partitions worth of scale,
    where 8 is due to each group consisted of 8 partitions × 4 elements per partition. The number of scales per
    partition is 1/4 of the free dimension size of the ``src`` tile.
    Different SBUF quadrants of scales are produced in parallel, with the scales written to the first
    (or second) 8 partitions of each SBUF quadrant.
    In other words, the ``dst_scale`` must be placed in the first 16 partitions of each SBUF quadrant.
    The ``dst_scale`` tile declaration must always occupy a multiple 32 partitions, even though not all partitions
    can be filled with scale values by ``nisa.quantize_mx``.

    **Tile size.**

    - The partition dimension size of ``src`` must be a multiple of 32 and must not exceed 128.
    - The free dimension size of ``src`` must be a multiple of 4 and must not exceed the physical size of each SBUF
      partition.
    - The ``dst`` tile has the same partition dimension size as ``src`` but a free dimension size
      that is 1/4 of ``src`` free dimension size due to the special 4-packed FP8 data types.
    - The ``dst_scale`` tile partition dimension depends on whether ``src`` spans multiple SBUF quadrants.
        - If ``src`` occupies only 32 partitions, ``dst_scale`` will occupy 4 partitions.
        - Otherwise, ``dst_scale`` will occupy the same number of partitions as ``src``.


    :param dst: the quantized MXFP8 output tile
    :param src: the input FP16/BF16 tile to be quantized
    :param dst_scale: the output scale tile

    """
    ...

def range_select(dst, on_true_tile, comp_op0, comp_op1, bound0, bound1, reduce_cmd=reduce_cmd.idle, reduce_res=None, reduce_op=np.amax, range_start=0.0, on_false_value=0.0, name=None):
    r"""

    Select elements from ``on_true_tile`` based on comparison with bounds using Vector Engine.

    .. note::

      Available only on NeuronCore-v3 and newer.

    For each element in ``on_true_tile``, compares its free dimension index + ``range_start`` against ``bound0`` and ``bound1``
    using the specified comparison operators (``comp_op0`` and ``comp_op1``). If both comparisons
    evaluate to True, copies the element to the output; otherwise uses  ``on_false_value``.

    Additionally performs a reduction operation specified by ``reduce_op`` on the results,
    storing the reduction result in ``reduce_res``.

    **Note on numerical stability:**

    In self-attention, we often have this instruction sequence: ``range_select`` (VectorE) -> ``reduce_res`` -> ``activation`` (ScalarE).
    When ``range_select`` outputs a full row of ``fill_value``, caution is needed to avoid NaN in the
    activation instruction that subtracts the output of ``range_select`` by ``reduce_res`` (max value):

    - If ``dtype`` and ``reduce_res`` are both FP32, we should not hit any NaN issue
      since ``FP32_MIN - FP32_MIN = 0``. Exponentiation on 0 is stable (1.0 exactly).

    - If ``dtype`` is FP16/BF16/FP8, the fill_value in the output tile will become ``-INF``
      since HW performs a downcast from FP32_MIN to a smaller dtype.
      In this case, you must make sure reduce_res uses FP32 ``dtype`` to avoid NaN in ``activation``.
      NaN can be avoided because ``activation`` always upcasts input tiles to FP32 to perform math operations: ``-INF - FP32_MIN = -INF``.
      Exponentiation on ``-INF`` is stable (0.0 exactly).

    **Constraints:**

    The comparison operators must be one of:

    - np.equal
    - np.less
    - np.less_equal
    - np.greater
    - np.greater_equal

    Partition dim sizes must match across ``on_true_tile``, ``bound0``, and ``bound1``:

    - ``bound0`` and ``bound1`` must have one element per partition
    - ``on_true_tile`` must be one of the FP dtypes, and ``bound0/bound1`` must be FP32 types.

    The comparison with ``bound0``, ``bound1``, and free dimension index is done in FP32.
    Make sure ``range_start`` + free dimension index is within 2^24 range.


    **Numpy equivalent:**

    .. code-block:: python

        indices = np.zeros_like(on_true_tile, dtype=np.float32)
        indices[:] = range_start + np.arange(on_true_tile[0].size)

        mask = comp_op0(indices, bound0) & comp_op1(indices, bound1)
        select_out_tile = np.where(mask, on_true_tile, on_false_value)
        reduce_tile = reduce_op(select_out_tile, axis=1, keepdims=True)

    :param dst: output tile with selected elements
    :param on_true_tile: input tile containing elements to select from
    :param on_false_value: constant value to use when selection condition is False.
      Due to HW constraints, this must be FP32_MIN FP32 bit pattern
    :param comp_op0: first comparison operator
    :param comp_op1: second comparison operator
    :param bound0: tile with one element per partition for first comparison
    :param bound1: tile with one element per partition for second comparison
    :param reduce_op: reduction operator to apply on across the selected output. Currently only ``np.max`` is supported.
    :param reduce_res: optional tile to store reduction results.
    :param range_start: starting base offset for index array for the free dimension of ``on_true_tile``
        Defaults to 0, and must be a compiler time integer.




    """
    ...

def reciprocal(dst, data, name=None):
    r"""
    Compute element-wise reciprocal (1.0/x) of the input ``data`` tile using Vector Engine.

    **Memory types.**

    Both the input ``data`` and output ``dst`` tiles can be in SBUF or PSUM.

    **Data types.**

    The input ``data`` tile can be any valid NKI data type (see :ref:`nki-dtype` for more information).
    The Vector Engine automatically casts the input data type to float32 and performs the reciprocal
    computation in float32 math. The float32 results are cast to the data type of ``dst``.

    **Layout.**

    The partition dimension of the input ``data`` is considered the parallel compute dimension.

    **Tile size.**

    The partition dimension size of input ``data`` and output ``dst`` tiles must be the same
    and must not exceed 128. The number of elements per partition of ``dst`` must match
    that of ``data`` and must not exceed the physical size of each SBUF partition.


    :param dst: the output tile
    :param data: the input tile

    """
    ...

def register_alloc(x=None):
    r"""
    Allocate a virtual register and optionally initialize it with an integer value ``x``.

    Each engine sequencer (Tensor/Scalar/Vector/GpSimd/Sync Engine) within a NeuronCore maintains its own set of
    physical registers for scalar operations (64x 32-bit registers per engine sequencer in NeuronCore v2-v4).
    The ``nisa.register_alloc`` API conceptually allocates a register within a virtual register space.
    Users do not need to expliclity free a register through nisa APIs. The NKI compiler
    handles physical register allocation (and deallocation) across the appropriate engine sequencers
    based on the dynamic program flow.


    NKI provides the following APIs to manipulate allocated registers:

    - ``nisa.register_move``: Move a constant value into a register
    - ``nisa.register_load``: Load a scalar (32-bit) value from HBM/SBUF into a register
    - ``nisa.register_store``: Store register contents to HBM/SBUF

    In the current NKI release, these registers are primarily used to specify dynamic loop boundaries and
    while loop conditions. The NKI compiler compiles such dynamic looping constructs to branching instructions
    executed by engine sequencers. For additional details, see ``nl.dynamic_range``. For more information
    on engine sequencer and its capabilities, see
    `Trainium/Inferentia2 architecture guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium_inferentia2_arch.html>`_.

    :param dst: a virtual register object
    :param x: optional integer value to initialize the register with

    """
    ...

def register_load(dst, src):
    r"""
    Load a scalar value from memory (HBM or SBUF) into a virtual register.

    This instruction reads a single scalar value (up to 32-bit) from a memory location (HBM or SBUF)
    and stores it in the specified virtual register. The source must be a NKI tensor with exactly
    one element (shape [1] or [1, 1]). This enables dynamic loading of values computed at
    runtime into registers for use in control flow operations.

    The virtual register system allows the NKI compiler to allocate physical registers across
    different engine sequencers as needed. See ``nisa.register_alloc`` for more details on
    virtual register allocation.


    :param dst: the destination virtual register (allocated via ``nisa.register_alloc``)
    :param src: the source tensor containing a single scalar value to load

    Example:

    .. code-block:: python

        # Load a computed value into a register
        computed_bound = nl.ones([1], dtype=nl.int32, buffer=nl.sbuf)  # bound of 1 in SBUF
        loop_reg = nisa.register_alloc()
        nisa.register_load(loop_reg, computed_bound)

    """
    ...

def register_move(dst, imm):
    r"""
    Move a compile-time constant integer value into a virtual register.

    This instruction loads an immediate (compile-time constant) integer value into the specified
    virtual register. The immediate value must be known at compile time and cannot be a runtime variable.
    This is typically used to initialize registers with known constants for loop bounds, counters,
    or other control flow operations.

    The virtual register system allows the NKI compiler to allocate physical registers across
    different engine sequencers as needed. See ``nisa.register_alloc`` for more details on
    virtual register allocation.

    This instruction operates on virtual registers only and does not access SBUF, PSUM, or HBM.

    :param dst: the destination virtual register (allocated via ``nisa.register_alloc``)
    :param imm: a compile-time constant integer value to load into the register

    Example:

    .. code-block:: python

        # Allocate a register and initialize it with a constant
        loop_count = nisa.register_alloc()
        nisa.register_move(loop_count, 10)  # Set register to 10

    """
    ...

def register_store(dst, src):
    r"""
    Store the value from a virtual register into memory (HBM/SBUF).

    This instruction writes the scalar value (up to 32-bit) stored in a virtual register to a memory location
    (HBM or SBUF). The destination must be a tensor with exactly one element (shape [1] or [1, 1]).
    This enables saving register values back to memory for later use or for output purposes.

    The virtual register system allows the NKI compiler to allocate physical registers across
    different engine sequencers as needed. See ``nisa.register_alloc`` for more details on
    virtual register allocation.

    :param dst: the destination tensor with a single element to store the register value
    :param src: the source virtual register (allocated via ``nisa.register_alloc``)

    Example:

    .. code-block:: python

        # Store a register value back to memory
        counter_reg = nisa.register_alloc(0)
        # ... perform operations that modify counter_reg ...
        result_tensor = nl.ndarray([1], dtype=nl.int32, buffer=nl.sbuf)
        nisa.register_store(result_tensor, counter_reg)

    """
    ...

def scalar_tensor_tensor(dst, data, op0, operand0, op1, operand1, reverse0=False, reverse1=False, name=None):
    r"""
    Apply two math operators in sequence using Vector Engine: ``(data <op0> operand0) <op1> operand1``.

    This instruction is equivalent to running two operations back-to-back:
    1. ``temp_result = tensor_scalar(data, op0, operand0)`` - broadcast ``operand0`` and apply ``op0``
    2. ``dst = tensor_tensor(temp_result, op1, operand1)`` - element-wise operation with ``operand1``

    The ``operand0`` can be either a compile-time
    constant scalar for broadcast across all elements of ``data`` or
    a tile of shape ``(data.shape[0], 1)`` for broadcast along the free dimension.
    The ``operand1`` tile must have the same shape as ``data`` for element-wise operation.

    The scalar broadcasting in the first operation is performed at no additional performance cost,
    making this instruction have approximately the same latency as a regular ``tensor_tensor`` instruction.

    Both ``op0`` and ``op1`` must be arithmetic operators (see :ref:`nki-aluop` for supported operators).
    Bitvec operators are not supported. When the operators are non-commutative (e.g., subtract),
    operand ordering can be reversed using ``reverse0`` and ``reverse1`` flags.

    **Memory types.**

    The input ``data`` tile can be an SBUF or PSUM tile. The ``operand0`` can be an SBUF or PSUM tile
    or a compile-time constant scalar. The ``operand1`` must be an SBUF or PSUM tile.
    However, ``data`` and ``operand1`` cannot both reside in PSUM. The output ``dst`` tile can be
    written to either SBUF or PSUM.

    **Data types.**

    All input tiles can be any supported NKI data type (see :ref:`nki-dtype` for more information).
    The Vector Engine automatically casts input data types to float32 and performs all computations
    in float32 math. The float32 results are cast to the data type of output ``dst``.

    **Layout.**

    The parallel computation dimension of ``nisa.scalar_tensor_tensor`` is along the partition dimension.

    **Tile size.**

    The partition dimension size of input ``data``, ``operand1``, and output ``dst`` tiles must be
    the same and must not exceed 128. The total number of elements per partition of input ``data``, ``operand1``,
    and output ``dst`` tiles must be the same and must not exceed the
    physical size of each SBUF partition.
    If operand0 is not a scalar, the partition dimension size of ``operand0`` must be the same as that of ``data``
    and the number of elements per partition of ``operand0`` must be 1.



    :param dst: the output tile
    :param data: the input tile
    :param op0: the first math operator used with operand0 (see :ref:`nki-aluop` for supported operators)
    :param operand0: a scalar constant or a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                    is the partition axis size of the input ``data`` tile
    :param reverse0: reverse ordering of inputs to ``op0``; if false, ``operand0`` is the rhs of ``op0``;
                     if true, ``operand0`` is the lhs of ``op0``
    :param op1: the second math operator used with operand1 (see :ref:`nki-aluop` for supported operators)
    :param operand1: a tile with the same size as ``data`` for element-wise operation
    :param reverse1: reverse ordering of inputs to ``op1``; if false, ``operand1`` is the rhs of ``op1``;
                     if true, ``operand1`` is the lhs of ``op1``

    """
    ...

def select_reduce(dst, predicate, on_true, on_false, reduce_res=None, reduce_cmd=reduce_cmd.idle, reduce_op=np.amax, reverse_pred=False, name=None):
    r"""
    Selectively copy elements from either ``on_true`` or ``on_false`` to the destination tile
    based on a ``predicate`` using Vector Engine, with optional reduction (max).

    The operation can be expressed in NumPy as:

    .. code-block:: python

        # Select:
        predicate = ~predicate if reverse_pred else predicate
        result = np.where(predicate, on_true, on_false)

        # With Reduce:
        reduction_result = np.max(result, axis=1, keepdims=True)

    **Memory constraints:**

    - Both ``on_true`` and ``predicate`` are permitted to be in SBUF
    - Either ``on_true`` or ``predicate`` may be in PSUM, but not both simultaneously
    - The destination ``dst`` can be in either SBUF or PSUM

    **Shape and data type constraints:**

    - ``on_true``, ``dst``, and ``predicate`` must have identical shapes (same number of partitions and elements per partition)
    - ``on_true`` can be any supported dtype except ``tfloat32``, ``int32``, ``uint32``
    - ``on_false`` dtype must be ``float32`` if ``on_false`` is a scalar.
    - ``on_false`` has to be either scalar or vector of shape ``(on_true.shape[0], 1)``
    - ``predicate`` dtype can be any supported integer type ``int8``, ``uint8``, ``int16``, ``uint16``
    - ``reduce_res`` must be a vector of shape ``(on_true.shape[0], 1)``
    - ``reduce_res`` dtype must of float type
    - ``reduce_op`` only supports ``max``

    **Behavior:**

    - Where predicate is True: The corresponding elements from ``on_true`` are copied to ``dst``
    - Where predicate is False: The corresponding elements from ``on_false`` are copied to ``dst``
    - When reduction is enabled, the max value from each partition of the ``result`` is computed and stored in ``reduce_res``

    **Accumulator behavior:**

    The Vector Engine maintains internal accumulator registers that can be controlled via the ``reduce_cmd`` parameter:

    - ``nisa.reduce_cmd.reset_reduce``: Reset accumulators to -inf, then accumulate the current results
    - ``nisa.reduce_cmd.reduce``: Continue accumulating without resetting (useful for multi-step reductions)
    - ``nisa.reduce_cmd.idle``: No accumulation performed (default)

    .. note::
      Even when ``reduce_cmd`` is set to ``idle``, the accumulator state may still be modified.
      Always use ``reset_reduce`` after any operations that ran with ``idle`` mode to ensure
      consistent behavior.

    .. note::
      The accumulator registers are shared for other Vector Engine accumulation instructions such :doc:`nki.isa.range_select <nki.isa.range_select>`

    :param dst: The destination tile to write the selected values to
    :param predicate: Tile that determines which value to select (on_true or on_false)
    :param on_true: Tile to select from when predicate is True
    :param on_false: Value to use when predicate is False, can be a scalar value or a vector tile of ``(on_true.shape[0], 1)``
    :param reduce_res: (optional) Tile to store reduction results, must have shape ``(on_true.shape[0], 1)``
    :param reduce_cmd: (optional) Control accumulator behavior using ``nisa.reduce_cmd`` values, defaults to idle
    :param reduce_op: (optional) Reduction operator to apply (only ``np.max`` is supported)
    :param reverse_pred: (optional) Reverse the meaning of the predicate condition, defaults to False

    """
    ...

def sendrecv(src, dst, send_to_rank, recv_from_rank, pipe_id, name=None):
    r"""
    Perform point-to-point communication between NeuronCores by sending and receiving data
    simultaneously using DMA engines.

    .. note::
      Available only on NeuronCore-v3 or newer.

    This instruction enables bidirectional data exchange between two NeuronCores within a
    Logical NeuronCore (LNC) configuration.
    The current NeuronCore sends its ``src`` tile to the ``dst`` location of the target
    NeuronCore specified by ``send_to_rank``,
    while simultaneously receiving data from ``recv_from_rank`` into its own ``dst`` tile.

    The use case is when NeuronCores need to exchange data for distributed computation patterns,
    such as all-gather communication or other collective operations where cores need to
    coordinate their computations by exchanging tiles.

    This instruction is only allowed in NeuronCore-v3 or newer when
    `LNC (Logical NeuronCore) <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html>`_
    is enabled. The communication occurs between NeuronCores that share the same HBM stack within the LNC configuration.
    Therefore, ``send_to_rank`` and ``recv_from_rank`` must be either 0 or 1.


    The ``pipe_id`` parameter provides synchronization control by grouping sendrecv operations. Operations with the same
    ``pipe_id`` form a logical group where all operations in the group must complete before any can proceed. Operations
    with different ``pipe_id`` values can progress independently without blocking each other.

    **Memory types.**

    Both ``src`` and ``dst`` tiles must be in SBUF.

    **Data types.**

    ``src`` and ``dst`` must have the same data type, but they can be any supported data types in NKI.

    **Layout.**

    ``src`` and ``dst`` must have the same shape and layout.

    **Tile size.**

    ``src`` and ``dst`` must have the same partition dimension size and the same number of elements per partition.

    :param src: the source tile on the current NeuronCore to be sent to the target NeuronCore
    :param dst: the destination tile on the current NeuronCore where received data will be stored
    :param send_to_rank: rank ID of the target NeuronCore to send data to
    :param recv_from_rank: rank ID of the source NeuronCore to receive data from
    :param pipe_id: synchronization identifier that groups sendrecv operations; operations with the same pipe_id are synchronized

    Example:

    .. code-block:: python

        # Exchange data between two cores in a ring pattern
        num_cores = 2
        current_rank = nl.program_id()
        next_rank = (current_rank + 1) % num_cores
        prev_rank = (current_rank - 1) % num_cores

        # Data to send and buffer to receive
        send_data = nl.ndarray((batch_size, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
        recv_buffer = nl.ndarray((batch_size, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)

        # Perform bidirectional exchange
        sendrecv(
            src=send_data,
            dst=recv_buffer,
            send_to_rank=next_rank,
            recv_from_rank=prev_rank,
            pipe_id=0
        )

        # Now recv_buffer contains data from the previous core

    """
    ...

def sequence_bounds(dst, segment_ids, name=None):
    r"""
    Compute the sequence bounds for a given set of segment IDs using GpSIMD Engine.

    Given a tile of segment IDs, this function identifies where each segment begins and ends.
    For each element, it returns a pair of values: [start_index, end_index] indicating
    the boundaries of the segment that element belongs to. All segment IDs must be non-negative
    integers. Padding elements (with segment ID of zero) receive special boundary
    values: a start index of n and an end index of (-1), where n is the length
    of ``segment_ids``.

    The output tile contains two values per input element: the start index (first column)
    and end index (second column) of each segment. The partition dimension must always be 1.
    For example, with input shape (1, 512), the output shape becomes (1, 2, 512), where
    the additional dimension holds the start and end indices for each element.

    The input tile (``segment_ids``) must have data type np.float32 or np.int32.
    The output tile data type is specified using the ``dtype`` field (must be np.float32 or np.int32).
    If ``dtype`` is not specified, the output data type will be the same as the input
    data type of ``segment_ids``.

    **NumPy equivalent:**

    .. nki_example:: ../../test/test_nki_isa_sequence_bounds.py
     :language: python
     :marker: NKI_EXAMPLE_1

    :param dst: tile containing the sequence bounds.
    :param segment_ids: tile containing the segment IDs. Elements with ID=0 are treated as padding.


    """
    ...

def tensor_copy(dst, src, engine=engine.unknown, name=None):
    r"""
    Create a copy of ``src`` tile within NeuronCore on-chip SRAMs using Vector, Scalar or GpSimd Engine.

    The output tile has the same partition axis size and also the same number of elements per partition
    as the input tile ``src``.

    All three compute engines, Vector, Scalar and GpSimd Engine can perform tensor copy. However, their copy behavior
    is slightly different across engines:

    - Scalar Engine on NeuronCore-v2 performs copy by first casting the input tile to FP32 internally and then casting from
      FP32 to the output dtype (``dtype``, or src.dtype if ``dtype`` is not specified). Therefore, users should be
      cautious with assigning this instruction to Scalar Engine when the input data type cannot be precisely cast to FP32
      (e.g., INT32).
    - Both GpSimd and Vector Engine can operate in two modes: (1) bit-accurate copy when input and output data types are
      the same or (2) intermediate FP32 cast when input and output data types differ, similar to Scalar Engine.

    In addition, since GpSimd Engine cannot access PSUM in NeuronCore, Scalar or Vector Engine must be chosen when the input or
    output tile is in PSUM (see :ref:`arch_sec_neuron_core_engines` for details). By default, this API returns
    a tile in SBUF, unless the returned value is assigned to a pre-declared PSUM tile.


    :param dst: a tile with the same content and partition axis size as the ``src`` tile.
    :param src: the source of copy, must be a tile in SBUF or PSUM.
    :param engine: (optional) the engine to use for the operation: `nki.isa.vector_engine`, `nki.isa.scalar_engine`,
                  `nki.isa.gpsimd_engine` or `nki.isa.unknown_engine` (default, compiler selects best engine based on engine workload).



    """
    ...

def tensor_copy_dynamic_dst(dst, src, engine=engine.unknown, name=None):
    r"""
    Create a copy of ``src`` tile within NeuronCore on-chip SRAMs using Vector or Scalar or GpSimd Engine,
    with ``dst`` located at a dynamic offset within each partition.

    Both source and destination tiles can be in either SBUF or PSUM.

    The source and destination tiles must also have the same number of partitions and the same number of elements
    per partition.

    The dynamic offset must be a scalar value resided in SBUF. If you have a list of dynamic offsets
    for scattering tiles in SBUF/PSUM, you may loop over each offset and call ``tensor_copy_dynamic_dst``
    once per offset.


    :param dst: the destination of copy, must be a tile in SBUF of PSUM that is dynamically indexed within each dimension.
    :param src: the source of copy, must be a tile in SBUF or PSUM.
    :param engine: (optional) the engine to use for the operation: `nki.isa.vector_engine`, `nki.isa.gpsimd_engine`,
                   `nki.isa.scalar_engine` or `nki.isa.unknown_engine` (default, let compiler select best engine).

    """
    ...

def tensor_copy_dynamic_src(dst, src, engine=engine.unknown, name=None):
    r"""
    Create a copy of ``src`` tile within NeuronCore on-chip SRAMs using Vector or Scalar or GpSimd Engine,
    with ``src`` located at a dynamic offset within each partition.

    Both source and destination tiles can be in either SBUF or PSUM. By default, this API returns
    a tile in SBUF, unless the returned value is assigned to a pre-declared PSUM tile.

    The source and destination tiles must also have the same number of partitions and the same number of elements
    per partition.

    The dynamic offset must be a scalar value resided in SBUF. If you have a list of dynamic offsets
    for gathering tiles in SBUF/PSUM, you may loop over each offset and call ``tensor_copy_dynamic_src``
    once per offset.


    :param src: the source of copy, must be a tile in SBUF or PSUM that is dynamically indexed within each partition.
    :param engine: (optional) the engine to use for the operation: `nki.isa.vector_engine`, `nki.isa.gpsimd_engine`,
                   `nki.isa.scalar_engine` or `nki.isa.unknown_engine` (default, let compiler select best engine).
    :param return: the modified destination of copy.



    """
    ...

def tensor_copy_predicated(dst, src, predicate, reverse_pred=False, name=None):
    r"""
    Conditionally copy elements from the ``src`` tile to the destination tile on SBUF / PSUM
    based on a ``predicate`` using Vector Engine.

    This instruction provides low-level control over conditional data movement on NeuronCores,
    optimized for scenarios where only selective copying of elements is needed. Either ``src`` or
    ``predicate`` may be in PSUM, but not both simultaneously. Both ``src`` and ``predicate`` are permitted to be in SBUF.

    Shape and data type constraints:

    1. ``src`` (if it is a tensor), ``dst``, and ``predicate`` must occupy the same number of partitions and same number of elements per partition.
    2. ``predicate`` must be of type ``uint8``, ``uint16``, or ``uint32``.
    3. ``src`` and ``dst`` must share the same data type.

    **Behavior:**

    - Where predicate is True: The corresponding elements from `src` are copied to `dst` tile. If `src` is a scalar, the scalar is copied to the `dst` tile.
    - Where predicate is False: The corresponding values in `dst` tile are unmodified


    :param ``src``: The source tile or number to copy elements from when ``predicate`` is True
    :param ``dst``: The destination tile to copy elements to
    :param ``predicate``: A tile that determines which elements to copy
    :param reverse_pred: A boolean that reverses the effect of ``predicate``.



    """
    ...

def tensor_partition_reduce(dst, op, data, name=None):
    r"""
    Apply a reduction operation across partitions of an input ``data`` tile using GpSimd Engine.

    :param dst: output tile with reduced result
    :param op: the reduction operator (add, max, bitwise_or, bitwise_and)
    :param data: the input tile to be reduced

    """
    ...

def tensor_reduce(dst, op, data, axis, negate=False, keepdims=False, name=None):
    r"""
    Apply a reduction operation to the free axes of an input ``data`` tile using Vector Engine.

    The reduction operator is specified in the ``op`` input field
    (see :ref:`nki-aluop` for a list of supported reduction operators).
    ``nisa.tensor_reduce`` supports two types of reduction operators: 1) bitvec operators (e.g., bitwise_and, bitwise_or)
    and 2) arithmetic operators (e.g., add, subtract, multiply).

    The reduction axes are specified in the ``axis`` field using a list of integer(s) to indicate axis indices.
    The reduction axes can contain up to four free axes and must start at the most minor free axis.
    Since axis 0 is the partition axis in a tile, the reduction axes must contain axis 1 (most-minor). In addition,
    the reduction axes must be consecutive: e.g., [1, 2, 3, 4] is a legal ``axis`` field, but [1, 3, 4] is not.

    When the reduction ``op`` is an arithmetic operator, the instruction can also multiply the output reduction
    results by ``-1.0`` before writing into the output tile, at no additional performance cost. This behavior is
    controlled by the ``negate`` input field.

    **Memory types.**

    Both the input ``data`` and ``dst`` tiles can be in SBUF or PSUM.

    **Data types.**

    For bitvec operators, the input/output data types must be integer types and Vector Engine treats
    all input elements as bit patterns without any data type casting. For arithmetic operators,
    the input/output data types can be any supported NKI data types, but the engine automatically casts
    input data types to float32
    and performs the reduction operation in float32 math. The float32 reduction results are cast to the
    data type of ``dst``.

    **Layout.**

    ``nisa.tensor_reduce`` only supports free axes reduction. Therefore, the partition dimension of the input
    ``data`` is considered the parallel compute dimension. To perform a partition axis reduction, we can either:

    1. invoke a ``nisa.nc_transpose`` instruction on the input tile and then this ``nisa.tensor_reduce``
       on the transposed tile, or
    2. invoke ``nki.isa.nc_matmul`` instructions to multiply a ``nl.ones([128, 1], dtype=data.dtype)`` as a stationary
       tensor with the input tile as a moving tensor. See more discussion on Tensor Engine alternative usage in
       `Trainium architecture guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium2_arch.html>`_.

    **Tile size.**

    The partition dimension size of input ``data`` and output ``dst`` tiles must be the same and must not exceed 128.
    The number of elements per partition of ``data`` must not
    exceed the physical size of each SBUF partition. The number of elements per partition in ``dst`` must be consistent
    with the ``axis`` field. For example, if ``axis`` indicates all free dimensions of ``data`` are reduced,
    the number of elements per partition in ``dst`` must be 1.


    :param dst: output tile of the reduction result
    :param op: the reduction operator (see :ref:`nki-aluop` for supported reduction operators)
    :param data: the input tile to be reduced
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: ``[1], [1,2], [1,2,3], [1,2,3,4]``
    :param negate: if True, reduction result is multiplied by ``-1.0``;
                   only applicable when op is an arithmetic operator
    :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                     With this option, the result will broadcast correctly against the input array.

    """
    ...

def tensor_scalar(dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, engine=engine.unknown, name=None):
    r"""
    Apply up to two math operators to the input ``data`` tile by broadcasting scalar/vector operands
    in the free dimension using Vector or Scalar or GpSimd Engine: ``(data <op0> operand0) <op1> operand1``.

    The input ``data`` tile can be an SBUF or PSUM tile. Both ``operand0`` and ``operand1`` can be
    SBUF or PSUM tiles of shape ``(data.shape[0], 1)``, i.e., vectors,
    or compile-time constant scalars.

    ``op1`` and ``operand1`` are optional, but must be ``None`` (default values) when unused.
    Note, performing one operator has the same performance cost as performing two operators in the instruction.

    When the operators are non-commutative (e.g., subtract), we can reverse ordering of the inputs for each operator through:

      - ``reverse0 = True``: ``tmp_res = operand0 <op0> data``
      - ``reverse1 = True``: ``operand1 <op1> tmp_res``

    The ``tensor_scalar`` instruction supports two types of operators: 1) bitvec
    operators (e.g., bitwise_and) and 2) arithmetic operators (e.g., add).
    See :ref:`nki-aluop` for the full list of supported operators.
    The two operators, ``op0`` and ``op1``, in a ``tensor_scalar`` instruction must be of the same type
    (both bitvec or both arithmetic).
    If bitvec operators are used, the ``tensor_scalar`` instruction must run on Vector Engine. Also, the input/output
    data types must be integer types, and input elements are treated as bit patterns without any data type casting.

    If arithmetic operators are used, the ``tensor_scalar`` instruction can run on Vector or Scalar or GpSimd Engine.
    However, each engine supports limited arithmetic operators (see :ref:``tbl-aluop``). The Scalar Engine on trn2 only
    supports a subset of the operator combination:

      - ``op0=np.multiply`` and ``op1=np.add``
      - ``op0=np.multiply`` and ``op1=None``
      - ``op0=add`` and ``op1=None``

    Also, arithmetic operators impose no restriction on the input/output data types,
    but the engine automatically casts input data types to float32
    and performs the operators in float32 math. The float32 computation results are cast to the target
    data type specified in the ``dtype`` field before written into the output tile, at no additional performance cost.
    If the ``dtype`` field is not specified, it is default to be the same as input tile data type.


    :param dst: an output tile of ``(data <op0> operand0) <op1> operand1`` computation
    :param data: the input tile
    :param op0: the first math operator used with operand0 (see :ref:`nki-aluop` for supported operators)
    :param operand0: a scalar constant or a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                    is the partition axis size of the input ``data`` tile
    :param reverse0: reverse ordering of inputs to ``op0``; if false, ``operand0`` is the rhs of ``op0``;
                     if true, ``operand0`` is the lhs of ``op0``
    :param op1: the second math operator used with operand1 (see :ref:`nki-aluop` for supported operators);
                this operator is optional
    :param operand1: a scalar constant or a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                    is the partition axis size of the input ``data`` tile
    :param reverse1: reverse ordering of inputs to ``op1``; if false, ``operand1`` is the rhs of ``op1``;
                     if true, ``operand1`` is the lhs of ``op1``
    :param engine: (optional) the engine to use for the operation: `nki.isa.vector_engine`, `nki.isa.scalar_engine`,
                   `nki.isa.gpsimd_engine` (only allowed for rsqrt) or `nki.isa.unknown_engine` (default, let
                   compiler select best engine based on the input tile shape).

    """
    ...

def tensor_scalar_reduce(dst, data, op0, operand0, reduce_op, reduce_res, reverse0=False, name=None):
    r"""
    Perform the same computation as ``nisa.tensor_scalar`` with one math operator
    and also a reduction along the free dimension of the ``nisa.tensor_scalar`` result using Vector Engine.

    Refer to :doc:`nisa.tensor_scalar <nki.isa.tensor_scalar>` for semantics of ``data/op0/operand0``.
    Unlike regular ``nisa.tensor_scalar`` where two operators are supported, only one
    operator is supported in this API. Also, ``op0`` can only be arithmetic operation in :ref:`nki-aluop`.
    Bitvec operators are not supported in this API.

    In addition to :doc:`nisa.tensor_scalar <nki.isa.activation>` computation, this API also performs a reduction
    along the free dimension(s) of the :doc:`nisa.tensor_scalar <nki.isa.activation>` result, at a small additional
    performance cost. The reduction result is returned in ``reduce_res`` in-place, which must be a
    SBUF/PSUM tile with the same partition axis size as the input tile ``data`` and one element per partition.
    The ``reduce_op`` can be any of ``nl.add``, ``nl.subtract``, ``nl.multiply``, ``nl.max`` or ``nl.min``.

    Reduction axis is not configurable in this API. If the input tile has multiple free axis, the API will
    reduce across all of them.

    .. math::
      result = data <op0> operand0 \\
      reduce\_res = reduce\_op(dst, axis=<FreeAxis>)


    :param dst: an output tile of ``(data <op0> operand0)`` computation
    :param data: the input tile
    :param op0: the math operator used with operand0 (any arithmetic operator in :ref:`nki-aluop` is allowed)
    :param operand0: a scalar constant or a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                    is the partition axis size of the input ``data`` tile
    :param reverse0: `(not supported yet)` reverse ordering of inputs to ``op0``; if false, ``operand0`` is the rhs of ``op0``;
                     if true, ``operand0`` is the lhs of ``op0``. `<-- currently not supported yet.`
    :param reduce_op: the reduce operation to perform on the free dimension of ``data <op0> operand0``
    :param reduce_res: a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                    is the partition axis size of the input ``data`` tile. The result of ``reduce_op(data <op0> operand0)``
                    is written in-place into the tile.
    """
    ...

def tensor_tensor(dst, data1, data2, op, engine=engine.unknown, name=None):
    r"""
    Perform an element-wise operation of input two tiles using Vector Engine or GpSimd Engine.
    The two tiles must have the same partition axis size and the same number of elements per partition.

    The element-wise operator is specified using the ``op`` field and can be any *binary* operator
    supported by NKI (see :ref:`nki-aluop` for details) that runs on the Vector Engine,
    or can be ``np.power``/``nl.power``  that runs on the GpSimd Engine.
    For bitvec operators, the input/output data types must be integer types and Vector Engine treats
    all input elements as bit patterns without any data type casting. For arithmetic operators, there is no
    restriction on the input/output data types, but the engine automatically casts input data types to float32
    and performs the element-wise operation in float32 math. The float32 results are cast to the target
    data type specified in the ``dtype`` field before written into the
    output tile. If the ``dtype`` field is not specified, it is default to be the same as the data type of ``data1``
    or ``data2``, whichever has the higher precision.

    Since GpSimd Engine cannot access PSUM, the input or output tiles cannot be in PSUM
    if ``op`` is ``np.power``/``nl.power``
    (see :ref:`arch_sec_neuron_core_engines` for details).
    Otherwise, the output tile can be in either SBUF or PSUM.
    However, the two input tiles, ``data1`` and ``data2`` cannot both reside in PSUM.
    The three legal cases are:

    1. Both ``data1`` and ``data2`` are in SBUF.
    2. ``data1`` is in SBUF, while ``data2`` is in PSUM.
    3. ``data1`` is in PSUM, while ``data2`` is in SBUF.

    Note, if you need broadcasting capability in the free dimension for either input tile, you should consider
    using :doc:`nki.isa.tensor_scalar <nki.isa.tensor_scalar>` API instead,
    which has better performance than ``nki.isa.tensor_tensor`` in general.



    :param dst: an output tile of the element-wise operation
    :param data1: lhs input operand of the element-wise operation
    :param data2: rhs input operand of the element-wise operation
    :param op: a binary math operator (see :ref:`nki-aluop` for supported operators)
    :param engine: (optional) the engine to use for the operation: `nki.isa.vector_engine`, `nki.isa.gpsimd_engine`
                   or `nki.isa.unknown_engine` (default, let compiler select best engine based on the input tile shape).

    """
    ...

def tensor_tensor_scan(dst, data0, data1, initial, op0, op1, reverse0=False, reverse1=False, name=None):
    r"""
    Perform a scan operation of two input tiles using Vector Engine.

    Mathematically, the tensor_tensor_scan instruction on Vector Engine performs
    the following computation per partition:

    .. code-block:: python

        # Let's assume we work with numpy, and data0 and data1 are 2D (with shape[0] being the partition axis)
        import numpy as np

        result = np.ndarray(data0.shape, dtype=data0.dtype)
        result[:, 0] = op1(op0(data0[:. 0], initial), data1[:, 0])

        for i in range(1, data0.shape[1]):
            result[:, i] = op1(op0(data0[:, i], result[:, i-1]), data1[:, i])

    The two input tiles (``data0`` and ``data1``) must have the same
    partition axis size and the same number of elements per partition.
    The third input ``initial`` can either be a float32 compile-time scalar constant
    that will be broadcasted in the partition axis of ``data0``/``data1``, or a tile
    with the same partition axis size as ``data0``/``data1`` and one element per partition.

    The two input tiles, ``data0`` and ``data1`` cannot both reside in PSUM. The three legal cases are:

    1. Both ``data1`` and ``data2`` are in SBUF.
    2. ``data1`` is in SBUF, while ``data2`` is in PSUM.
    3. ``data1`` is in PSUM, while ``data2`` is in SBUF.

    The scan operation supported by this API has two programmable
    math operators in ``op0`` and ``op1`` fields.
    Both ``op0`` and ``op1`` can be any binary arithmetic operator
    supported by NKI (see :ref:`nki-aluop` for details).
    We can optionally reverse the input operands of ``op0`` by setting ``reverse0`` to True
    (or ``op1`` by setting ``reverse1``). Reversing operands is useful for non-commutative
    operators, such as subtract.

    Input/output data types can be any supported NKI data type (see :ref:`nki-dtype`),
    but the engine automatically casts input data types to float32
    and performs the computation in float32 math. The float32 results are cast to the target
    data type specified in the ``dtype`` field before written into the
    output tile. If the ``dtype`` field is not specified, it is default to be the
    same as the data type of ``data0``
    or ``data1``, whichever has the highest precision.


    :param dst: an output tile of the scan operation
    :param data0: lhs input operand of the scan operation
    :param data1: rhs input operand of the scan operation
    :param initial: starting state of the scan; can be a SBUF/PSUM tile with 1 element/partition or a scalar
                        compile-time constant
    :param op0: a binary arithmetic math operator (see :ref:`nki-aluop` for supported operators)
    :param op1: a binary arithmetic math operator (see :ref:`nki-aluop` for supported operators)
    :param reverse0: reverse ordering of inputs to ``op0``; if false, ``data0`` is the lhs of ``op0``;
                   if true, ``data0`` is the rhs of ``op0``
    :param reverse1: reverse ordering of inputs to ``op1``; if false, ``data1`` is the rhs of ``op1``;
                   if true, ``data1`` is the lhs of ``op1``

    """
    ...

tensor_engine = engine.tensor
"""Tensor engine constant (deprecated: use engine.tensor)"""

vector_engine = engine.vector
"""Vector engine constant (deprecated: use engine.vector)"""

scalar_engine = engine.scalar
"""Scalar engine constant (deprecated: use engine.scalar)"""

gpsimd_engine = engine.gpsimd
"""GPSIMD engine constant (deprecated: use engine.gpsimd)"""

dma_engine = engine.dma
"""DMA engine constant (deprecated: use engine.dma)"""

unknown_engine = engine.unknown
"""Unknown engine constant (deprecated: use engine.unknown)"""

