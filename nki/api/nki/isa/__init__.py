from dataclasses import dataclass
from enum import Enum
from typing import *

from nki.language import NKIObject
from nki.language import equal
from nki.language import maximum

class engine(Enum):
    r"""Neuron Device engines."""

    tensor = ...
    r"""Tensor Engine"""

    vector = ...
    r"""Vector Engine"""

    scalar = ...
    r"""Scalar Engine"""

    gpsimd = ...
    r"""GpSIMD Engine"""

    dma = ...
    r"""DMA Engine"""

    sync = ...
    r"""Sync Engine"""

    unknown = ...
    r"""Unknown Engine"""

    ...

class reduce_cmd(Enum):
    r"""Engine register reduce commands."""

    idle = ...
    r"""Not using the accumulator registers"""

    reset = ...
    r"""Resets the accumulator registers to its initial state"""

    reduce = ...
    r"""Keeps accumulating over the current value of the accumulator registers"""

    reset_reduce = ...
    r"""Resets the accumulator registers then immediately accumulate the results of the current instruction into the accumulators"""

    load_reduce = ...
    r"""Loads a value into the accumulator registers, then accumulate the results of the current instruction into the accumulators"""

    ...

class dge_mode(Enum):
    r"""Descriptor Generation Engine mode."""

    unknown = ...
    r"""Unknown DGE mode, i.e., let compiler decide the DGE mode"""

    swdge = ...
    r"""Software DGE"""

    hwdge = ...
    r"""Hardware DGE"""

    none = ...
    r"""Not using DGE"""

    ...

class oob_mode(Enum):
    r"""Out-of-bounds access mode."""

    error = ...
    r"""Raise a runtime error when an out-of-bounds access is detected."""

    skip = ...
    r"""Silently skip the runtime out-of-bounds access."""

    ...

class matmul_perf_mode(Enum):
    r"""Performance mode for matmul."""

    none = ...
    r"""Default mode, no performance optimization"""

    double_row = ...
    r"""Double FP8 mode, 2x matmul throughput by packing two FP8 weight/ifmap element pairs"""

    ...

class nc_version(Enum):
    r"""NeuronCore version."""

    gen2 = ...
    r"""Trn1/Inf2 target"""

    gen3 = ...
    r"""Trn2 target"""

    gen4 = ...
    r"""Trn3 target"""

    ...

class dma_engine(Enum):
    r"""DMA transfer engine.
        """

    dma = ...
    r"""Shared DMA with CoreBarrier synchronization (default). Can be triggered from any engine."""

    gpsimd_dma = ...
    r"""GPSIMD's internal DMA engine for low-latency SB-to-SB swaps in LNC=2.
    Implies GPSIMD as the trigger engine."""

    ...

class NkiValidationError(Exception):
    r"""Raised when hardware constraints are violated."""

    ...

class VirtualRegister(NKIObject):
    r"""A virtual register on engine.

    Allocated via ``nisa.register_alloc()`` and manipulated via
    ``nisa.register_move()``, ``nisa.register_load()``, ``nisa.register_store()``.

    Virtual registers represent registers on engine and are used for various APIs
    such as loading and storing constants from tensors, as the return value of
    ``nki.collective`` and ``nki.isa`` APIs, and for dynamic addressing.

    In addition to NKI APIs, virtual registers can be used to represent dynamic
    loop bounds for for loops using :doc:`dynamic_range <nki.language.dynamic_range>`,
    and while loops.

    .. code-block:: python

        import nki.language as nl
        import nki.isa as nisa

        # Using a register in a dynamic for loop.
        reg = nisa.register_alloc(5)
        for _ in nl.dynamic_range(reg):
            tile = nl.load(input_tensor[0:128, 0:512])
            result = nl.multiply(tile, tile)
            nl.store(out_tensor[0:128, 0:512], result)

    .. code-block:: python

        import nki.language as nl
        import nki.isa as nisa

        # Using a register in a dynamic while loop.
        cond_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cond_sb, src=...)

        # Load condition into register
        reg = nisa.register_alloc()
        nisa.register_load(reg, cond_sb)

        while reg:
            ...
            nisa.dma_copy(dst=cond_sb, src=...)
            nisa.register_load(reg, cond_sb)"""

    def __init__(self, state=None, frozen=False):
        ...

    ...

tensor_engine = ...

vector_engine = ...

scalar_engine = ...

gpsimd_engine = ...

unknown_engine = ...

def get_nc_version():
    r"""Returns the nc_version of the current target context."""
    ...

def dma_copy(dst, src, priority=None, oob_mode=oob_mode.error, dge_mode=dge_mode.unknown, engine=engine.unknown, name=None):
    r"""Copy data from ``src`` to ``dst`` using DMA engines.

    This instruction performs data movement between memory locations (SBUF or HBM) using DMA engines.
    The operation copies data from the source tensor to the destination tensor: ``dst = src``.

    ``nisa.dma_copy`` supports different modes of DMA descriptor generation (DGE):

    - ``nisa.dge_mode.none``: Neuron Runtime generates DMA descriptors and stores them into HBM before NEFF execution.
    - ``nisa.dge_mode.swdge``: Gpsimd Engine generates DMA descriptors as part of the ``nisa.dma_copy`` instruction
      during NEFF execution.
    - ``nisa.dge_mode.hwdge``: Sync Engine or Scalar Engine sequencers invoke DGE hardware block to generate DMA
      descriptors as part of the ``nisa.dma_copy`` instruction during NEFF execution.

    See `Trainium2 arch guide` and `Introduction to DMA with NKI` for more discussion.

    When either ``sw_dge`` or ``hw_dge`` mode is used, the ``src`` and ``dst`` tensors can have a dynamic start address
    which depends on a variable that cannot be resolved at compile time. When ``sw_dge`` is selected, ``nisa.dma_copy``
    can also perform a gather or scatter operation, using a list of dynamic indices from SBUF.
    In both of these dynamic modes, out-of-bound address checking is turned on automatically during execution.
    By default a runtime error is raised (``oob_mode=oob_mode.error`` as default setting).
    Developers can disable this error and make the ``nisa.dma_copy`` instruction skip the DMA transfer for a given dynamic
    address or index when it is out of bound using ``oob_mode=oob_mode.skip``.

    **Memory types.**

    Both ``src`` and ``dst`` tiles can be in HBM or SBUF. However, if both tiles are in SBUF, consider using an alternative
    for better performance:

    - :doc:`nisa.tensor_copy <nki.isa.tensor_copy>` for direct copies
    - :doc:`nisa.nc_n_gather <nki.isa.nc_n_gather>` to gather elements within each partition independently
    - :doc:`nisa.local_gather <nki.isa.local_gather>` to gather elements within groups of partitions

    **Data types.**

    Both ``src`` and ``dst`` tiles can be any supported NKI data types (see :ref:`nki-dtype` for more information).

    The DMA engines automatically handle data type conversion when ``src`` and ``dst`` have different data types.
    The conversion is performed through a two-step process: first casting from ``src.dtype`` to float32, then
    from float32 to ``dst.dtype``.

    **Tile size.**

    The total number of data elements in ``src`` must match that of ``dst``.

    **Indirect addressing (gather/scatter).**

    ``nisa.dma_copy`` supports indirect addressing for dynamic row selection at runtime. This enables
    gather (read from dynamic rows) and scatter (write to dynamic rows) patterns. Indirect addressing
    is activated by calling ``.ap()`` on ``src`` or ``dst`` with a ``vector_offset`` or ``scalar_offset``
    parameter.

    There are two types of indirect addressing:

    *Vector indirection* provides per-partition dynamic offsets. Each of the hardware partitions
    gets its own index, enabling gather/scatter where different partitions access different rows.
    Use ``.ap(pattern=..., vector_offset=idx_tensor, indirect_dim=0)`` where ``idx_tensor`` is an
    SBUF tensor of shape ``(P, 1)`` containing one row index per partition.
    The tensor being indexed (the one ``.ap()`` is called on) must be in HBM.

    *Scalar indirection* provides a single dynamic offset applied uniformly to all partitions.
    Use ``.ap(pattern=..., scalar_offset=reg_or_tensor, indirect_dim=N)`` where the offset is
    either a 1x1 SBUF tensor or a ``VirtualRegister`` from ``nisa.register_alloc()``.

    ``vector_offset`` and ``scalar_offset`` are mutually exclusive.

    **Indirect gather example** (``vector_offset`` on ``src``):

    .. code-block:: python

        import nki
        import nki.isa as nisa
        import nki.language as nl

        @nki.jit
        def indirect_gather_kernel(data, indices):
            P, F = indices.shape[0], data.shape[1]
            output = nl.ndarray((P, F), dtype=data.dtype, buffer=nl.shared_hbm)

            idx = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx, src=indices)

            dst = nl.ndarray((P, F), dtype=data.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=dst,
                src=data.ap(
                    pattern=[[F, P], [1, F]],
                    vector_offset=idx,
                    indirect_dim=0,
                ),
            )

            nisa.dma_copy(dst=output, src=dst)
            return output

    **Indirect scatter example** (``vector_offset`` on ``dst``):

    .. code-block:: python

        import nki
        import nki.isa as nisa
        import nki.language as nl

        @nki.jit
        def indirect_scatter_kernel(src_data, indices, output):
            P, F = src_data.shape

            src = nl.ndarray((P, F), dtype=src_data.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=src, src=src_data)

            idx = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx, src=indices)

            nisa.dma_copy(
                dst=output.ap(
                    pattern=[[F, P], [1, F]],
                    vector_offset=idx,
                    indirect_dim=0,
                ),
                src=src,
            )
            return output

    :param dst: the destination tensor to copy data into
    :param src: the source tensor to copy data from
    :param priority: (optional): DMA quality-of-service priority level 0-3 where lower is higher priority (NeuronCore-v4+ only). Currently not supported when DGE is turned off (``dge_mode=nki.isa.dge_mode.none``).
    :param dge_mode: (optional) specify which Descriptor Generation Engine (DGE) mode to use for DMA descriptor generation: ``nki.isa.dge_mode.none`` (turn off DGE) or ``nki.isa.dge_mode.swdge`` (software DGE) or ``nki.isa.dge_mode.hwdge`` (hardware DGE)  or ``nki.isa.dge_mode.unknown`` (by default, let compiler select the best DGE mode). Hardware based DGE is only supported for NeuronCore-v3 or newer. See `Trainium2 arch guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium2_arch.html>`__ for more information.
    :param oob_mode: (optional) Specifies how to handle out-of-bounds (oob) array indices during indirect access operations. Valid modes are:

        - ``oob_mode.error``: (Default) Raises an error when encountering out-of-bounds indices.
        - ``oob_mode.skip``: Silently skips any operations involving out-of-bounds indices.

        For example, when using indirect gather/scatter operations, out-of-bounds indices can occur if the index array contains values that exceed the dimensions of the target array.

    :param engine: (optional) the engine to use for HWDGE descriptor generation: ``nki.isa.engine.sync`` or ``nki.isa.engine.scalar``.
                   Only valid when ``dge_mode=nisa.dge_mode.hwdge``. ``nki.isa.engine.unknown`` by default."""
    ...

def tensor_copy(dst, src, engine=engine.unknown, name=None):
    r"""Create a copy of ``src`` tile within NeuronCore on-chip SRAMs using Vector, Scalar or GpSimd Engine.

    The output tile has the same partition axis size and also the same number of elements per partition
    as the input tile ``src``.

    All three compute engines, Vector, Scalar and GpSimd Engine can perform tensor copy. However, their copy behavior
    is slightly different across engines:

    - Scalar Engine on NeuronCore-v2 performs copy by first casting the input tile to FP32 internally and then casting from
      FP32 to ``dst.dtype``. Users should be cautious with assigning this instruction to Scalar Engine when the input data
      type cannot be precisely cast to FP32 (e.g., INT32).
    - Both GpSimd and Vector Engine can operate in two modes: (1) bit-accurate copy when input and output data types are
      the same or (2) intermediate FP32 cast when input and output data types differ, similar to Scalar Engine.

    In addition, since GpSimd Engine cannot access PSUM in NeuronCore, Scalar or Vector Engine must be chosen when the input or
    output tile is in PSUM (see :ref:`arch_sec_neuron_core_engines` for details). By default, this API returns
    a tile in SBUF, unless the returned value is assigned to a pre-declared PSUM tile.

    On NeuronCore v2, ``tensor_copy`` is not supported on the Scalar Engine. Instead, use :doc:`nisa.activation <nki.isa.activation>` with ``op=nl.copy``.

    :param dst: a tile with the same content and partition axis size as the ``src`` tile.
    :param src: the source of copy, must be a tile in SBUF or PSUM.
    :param engine: (optional) the engine to use for the operation: `nki.isa.engine.vector`, `nki.isa.engine.scalar`,
                  `nki.isa.engine.gpsimd` or `nki.isa.engine.unknown` (default, compiler selects best engine based on engine workload)."""
    ...

def memset(dst, value, engine=engine.unknown, name=None):
    r"""Initialize ``dst`` by filling it with a compile-time constant ``value``, using Vector or GpSimd Engine.
    The memset instruction supports all valid NKI dtypes (see :ref:`nki-dtype`).

    :param dst: destination tile to initialize.
    :param value: the constant value to initialize with
    :param engine: specify which engine to use for memset: ``nki.isa.engine.vector`` or ``nki.isa.engine.gpsimd`` ;
                   ``nki.isa.engine.unknown`` by default, lets compiler select the best engine for the given
                   input tile shape

    .. note::
        For x4 packed types (``float8_e4m3fn_x4``, ``float8_e5m2_x4``,
        ``float4_e2m1fn_x4``), only ``value=0`` is supported."""
    ...

def tensor_copy_predicated(dst, src, predicate, reverse_pred=False, name=None):
    r"""Conditionally copy elements from the ``src`` tile to the destination tile on SBUF / PSUM
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
    :param reverse_pred: A boolean that reverses the effect of ``predicate``."""
    ...

def nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=matmul_perf_mode.none, name=None):
    r"""Compute ``dst = stationary.T @ moving`` matrix multiplication using Tensor Engine.

    The figure below illustrates how to map a matrix multiplication from a mathematical definition
    to ``nisa.nc_matmul`` on Tensor Engine. The stationary tensor is loaded into the systolic array first and
    stays in place, while the moving tensor streams through the array during computation.
    For more detailed discussion of Tensor Engine capabilities, see
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

    **Accumulation mode.**

    The ``accumulate`` parameter controls whether the matmul result should overwrite or accumulate on top of
    the ``dst`` PSUM tile. When ``accumulate=False``, the result overwrites the existing content.
    When ``accumulate=True``, the result is added to the existing content.
    When ``accumulate=None`` (default), the behavior is auto-detected: the first write to a PSUM location
    overwrites, and subsequent writes to the same location accumulate. Multiple ``nc_matmul`` instructions
    with ``accumulate=True`` can form an accumulation group before the PSUM tile content is evicted back to SBUF.

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

    **Data types.**

    The input ``stationary`` and ``moving`` tiles can be one of these supported data types:
    ``float8_e4m3/float8_e5m2/bfloat16/float16/tfloat32/float32``. Starting NeuronCore-v4, ``float8_e4m3fn``
    (OCP FP8) is also supported; ``float8_e4m3`` (legacy) and ``float8_e4m3fn`` cannot be mixed in the same matmul.
    The ``stationary`` and ``moving`` tiles can have different data types, with one exception: if one of the input
    tiles is ``tfloat32/float32``, the other tile must also be ``tfloat32/float32``.
    On NeuronCore-v3 and NeuronCore-v4, when performance mode is ``double_row``, ``stationary`` and ``moving`` tiles
    must be one of ``float8_e4m3`` or ``float8_e5m2`` (plus ``float8_e4m3fn`` on NeuronCore-v4), but the two input
    tiles can have different float8 formats (except that legacy ``float8_e4m3`` and OCP ``float8_e4m3fn`` cannot
    appear together).

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

    **Profiler view syntax.**

    Each ``nc_matmul`` call lowers to two ISA instructions in the profiler: a load instruction
    (to load the stationary operand into the Tensor Engine) followed by a multiply instruction.
    Both instructions will appear in profiler output for a single ``nc_matmul`` call.

    The multiply instruction operands are displayed in a compact ISA syntax:

    .. code-block:: text

        src=<dtype>@<address>[<strides>][<num_elem>]
        dst=<dtype>@<address>[<strides>][<num_elem>]
        <M>*<K> acc_flags=<flags> psum_zero=<val>

    Where:

    - ``<dtype>``: data type (e.g., ``bfloat16``, ``fp8e4``, ``fp8e5``)
    - ``<address>``: hex memory address in SBUF (for src) or PSUM (for dst)
    - ``[<strides>]``: element strides per dimension (multi-dimensional)
    - ``[<num_elem>]``: number of elements per dimension (multi-dimensional)
    - ``<M>*<K>``: matmul dimensions (M rows × K contraction)
    - ``acc_flags``: accumulator control flags (e.g., ``2`` = reset accumulator)
    - ``psum_zero``: PSUM zero-initialization control value

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
    :param accumulate: if True, accumulate the matmul result into the existing ``dst`` PSUM tile content;
                       if False, overwrite the existing content;
                       if None (default), auto-detect based on whether this PSUM location was previously written.
                       Not exposed for ``nc_transpose``.
    :param tile_position: a 2D tuple (start_row, start_column) to control starting row in Tensor Engine tiling mode; start_column must be 0
    :param tile_size: a 2D tuple (row_size, column_size) to control row tile size in Tensor Engine tiling mode; column_size must be 128
    :param perf_mode: controls Tensor Engine FP8 double performance mode on/off starting NeuronCore-v3: ``matmul_perf_mode.none`` (default) disables double FP8 mode; ``matmul_perf_mode.double_row`` enables double FP8 mode which achieves 2x matmul throughput by packing two FP8 weight/ifmap element pairs and computing two multiplications in parallel per cycle; cannot be combined with column tiling mode. See the `Trainium2 arch guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium2_arch.html>`__ for more information."""
    ...

def nc_matmul_mx(dst, stationary, moving, stationary_scale, moving_scale, tile_position=None, tile_size=None, accumulate=None, name=None):
    r"""Compute matrix multiplication of MXFP8/MXFP4 quantized matrices with integrated dequantization using Tensor Engine.

    .. note::

      Available only on NeuronCore-v4 and newer.

    The NeuronCore-v4 Tensor Engine supports matrix multiplication of MXFP8/MXFP4 quantized matrices as defined in the
    `OCP Microscaling standard <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__.
    This instruction performs matrix multiplication between quantized ``stationary`` and ``moving`` matrices while
    applying dequantization scales during computation. The micro-scaling group size is 32 elements in groups of
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

    **Data types.**

    The input ``stationary`` and ``moving`` tiles must be float8_e5m2_x4, float8_e4m3fn_x4, or float4_e2m1fn_x4
    (4-packed quantized data types). The ``stationary_scale`` and ``moving_scale`` tiles must be uint8.
    The ``dst`` tile can be float32 or bfloat16.

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

    **Profiler view syntax.**

    ``nc_matmul_mx`` uses the same profiler output format as :doc:`nisa.nc_matmul <nki.isa.nc_matmul>`,
    except the source access pattern is interpreted as an MX-quantized tensor:
    ``src=<dtype>@$MX[<data_addr>,<scale_addr>,<start_scale_partition>]@[<step_elem>][<num_elem>]``.

    :param dst: the matrix multiplication output (PSUM tile)
    :param stationary: the stationary quantized matrix (SBUF tile)
    :param moving: the moving quantized matrix (SBUF tile)
    :param stationary_scale: the dequantization scales for stationary matrix
                             (SBUF tile)
    :param moving_scale: the dequantization scales for moving matrix (SBUF tile)
    :param tile_position: a 2D tuple (start_row, start_column) to control
                          starting row and column in Tensor Engine tiling mode
    :param tile_size: a 2D tuple (row_size, column_size) to control row and
                      column tile sizes in Tensor Engine tiling mode
    :param accumulate: if True, accumulate the matmul result into the existing
                       ``dst`` PSUM tile content; if False, overwrite the
                       existing content; if None (default), auto-detect based on
                       whether this PSUM location was previously written"""
    ...

def nc_transpose(dst, data, engine=engine.unknown, name=None):
    r"""Perform a 2D transpose between the partition axis and the free axis of input ``data`` using Tensor or Vector Engine.

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
    :param engine: specify which engine to use for transpose: ``nki.isa.engine.tensor`` or ``nki.isa.engine.vector``;
                   by default, the best engine will be selected for the given input tile shape"""
    ...

def dma_transpose(dst, src, axes=None, priority=None, dge_mode=dge_mode.unknown, oob_mode=oob_mode.error, name=None):
    r"""Perform a transpose on input ``src`` using DMA Engine.

    The permutation of transpose follow the rules described below:

    1. For 2-d input tile, the permutation will be [1, 0]
    2. For 3-d input tile, the permutation will be [2, 1, 0]
    3. For 4-d input tile, the permutation will be [3, 1, 2, 0]

    **DMA Direct Transpose Constraints**

    The only valid ``dge_mode`` s are ``unknown`` and ``hwdge``. If ``hwdge``, this instruction will be lowered
    to a Hardware DGE transpose. This has additional restrictions:

    1. ``src.shape[0] == 16``
    2. ``src.shape[-1] % 128 == 0``
    3. ``src.dtype`` is 2 bytes

    **DMA Indirect Transpose Constraints**

    The only valid ``dge_mode`` s are ``unknown`` and ``swdge``. This instruction will be lowered
    to a Software DGE transpose (``dma_gather_transpose``). This has additional restrictions:

    #. When ``src`` is 4D: ``len(src[1])`` or ``len(src[2])`` must be 1
    #. ``src.shape[-1] <= 128``
    #. ``src.dtype`` is 2 bytes
    #. ``src`` tensor must be on HBM
    #. ``indices`` must be 2-d
    #. ``indices.shape[0] * indices.shape[1]`` must be ``>=`` ``src.shape[0]``
    #. ``src.shape[0]`` must be divisible by 16
    #. ``indices.shape[0]`` must be in ``[16, 128]`` and divisible by 16
    #. When ``indices.shape[1] > 1``: ``indices.shape[0]`` must be exactly 128
    #. ``indices.dtype`` is ``np.uint32``
    #. ``indices`` tensor must be on SBUF
    #. TRN2+ only

    Indirect transpose effectively performs the following operation:
    ``flat_indices = indices.T.flatten()[:src.shape[0]]``
    ``gathered = src[flat_indices, :]``
    ``dst = gathered.T``

    **Indirect transpose example with 1D indices** (``indices.shape=[128, 1]``):

    .. code-block:: python

        import nki
        import nki.isa as nisa
        import nki.language as nl

        @nki.jit
        def gather_transpose_kernel(src_hbm, idx_hbm):
            P, F = 128, 128
            output = nl.ndarray((P, F), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

            idx_sb = nl.load(idx_hbm)

            dst_sb = nl.ndarray((P, F), dtype=src_hbm.dtype, buffer=nl.sbuf)
            nisa.memset(dst=dst_sb, value=0)

            src_ap = src_hbm.ap(
                pattern=[[P, F], [1, P]],
                vector_offset=idx_sb,
                indirect_dim=0,
            )
            nisa.dma_transpose(dst=dst_sb, src=src_ap, axes=(1, 0))

            nisa.dma_copy(dst=output, src=dst_sb)
            return output

    **Indirect transpose example with 2D indices** (``indices.shape=[128, N]`` where N > 1):

    .. code-block:: python

        @nki.jit
        def gather_transpose_2d_kernel(src_hbm, idx_hbm):
            N_COLS = 2  # Number of columns in index tensor
            P = 128  # Partition dimension (max 128)
            F = 128 * N_COLS  # Free dimension: 256

            output = nl.ndarray((P, F), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

            idx_sb = nl.load(idx_hbm)

            dst_sb = nl.ndarray((P, F), dtype=src_hbm.dtype, buffer=nl.sbuf)
            nisa.memset(dst=dst_sb, value=0)

            src_ap = src_hbm.ap(
                pattern=[[P, F], [1, P]],
                vector_offset=idx_sb,
                indirect_dim=0,
            )
            nisa.dma_transpose(dst=dst_sb, src=src_ap, axes=(1, 0))

            nisa.dma_copy(dst=output, src=dst_sb)
            return output

    **4D indirect transpose example with 2D indices**:

    .. code-block:: python

        @nki.jit
        def gather_transpose_4d_kernel(src_hbm, idx_hbm):
            T, d1, d2, d3 = src_hbm.shape
            _, N = idx_hbm.shape
            F = 128 * N

            idx_sb = nl.load(idx_hbm)

            dst_sb = nl.ndarray((d3, d1, d2, F), dtype=src_hbm.dtype, buffer=nl.sbuf)
            nisa.memset(dst=dst_sb, value=0)

            src_ap = src_hbm.ap(
                pattern=[[d1 * d2 * d3, F], [d2 * d3, d1], [d3, d2], [1, d3]],
                vector_offset=idx_sb,
                indirect_dim=0,
            )

            nisa.dma_transpose(dst=dst_sb, src=src_ap, axes=(3, 1, 2, 0))

            output = nl.ndarray((d3, d1, d2, F), dtype=src_hbm.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=output, src=dst_sb)

            return output

    :param dst: the destination of transpose, must be a tile in SBUF.
    :param src: the source of transpose, must be a tile in HBM or SBUF. ``src.dtype == dst.dtype``
    :param axes: transpose axes where the i-th axis of the transposed tile will correspond to the axes[i] of the source.
                 Supported axes are ``(1, 0)``, ``(2, 1, 0)``, and ``(3, 1, 2, 0)``.
    :param priority: (optional): DMA quality-of-service priority level 0-3 where lower is higher priority (NeuronCore-v4+ only). Currently not supported when DGE is turned off (``dge_mode=nki.isa.dge_mode.none``).
    :param dge_mode: (optional) specify which Descriptor Generation Engine (DGE) mode to use for DMA descriptor generation: ``nki.isa.dge_mode.none`` (turn off DGE) or ``nki.isa.dge_mode.swdge`` (software DGE) or ``nki.isa.dge_mode.hwdge`` (hardware DGE)  or ``nki.isa.dge_mode.unknown`` (by default, let compiler select the best DGE mode). Hardware based DGE is only supported for NeuronCore-v3 or newer. See `Trainium2 arch guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium2_arch.html>`__ for more information.
    :param oob_mode: (optional) Specifies how to handle runtime out-of-bounds (oob) array indices during indirect access operations. Valid modes are:

        - ``oob_mode.error``: (Default) Raises an error when encountering runtime out-of-bounds indices.

        - ``oob_mode.skip``: Silently skips any operations involving out-of-bounds indices. Only valid when ``src`` uses indirect indexing."""
    ...

def tensor_tensor(dst, data1, data2, op, engine=engine.unknown, name=None):
    r"""Perform an element-wise operation of input two tiles using Vector Engine or GpSimd Engine.
    The two tiles must have the same partition axis size and the same number of elements per partition.

    The element-wise operator is specified using the ``op`` field. Valid choices for ``op``:

    1. Any supported *binary* operator that runs on the Vector Engine. (See :ref:`nki-aluop` for details.)
    2. ``nl.power``. (Which runs on the GpSimd engine.)

    For bitvec operators, the input/output data types must be integer types and Vector Engine treats
    all input elements as bit patterns without any data type casting. For arithmetic operators, the behavior
    depends on the data types:

    - **Float types**: The engine casts input data types to float32 and performs the element-wise operation
      in float32 math. The float32 results are cast to ``dst.dtype`` at no additional performance cost.
    - **int32/uint32 types**: When all input/output tiles are int32 or uint32, the operation defaults to
      GpSimd Engine, which uses native integer arithmetic. This ensures exact results for all 32-bit integer
      values. You may override this by passing ``engine=nki.isa.engine.vector`` explicitly.

    Since GpSimd Engine cannot access PSUM, the input/output tiles cannot be in PSUM if ``op`` is ``nl.power``.
    Similarly, the automatic GpSimd dispatch for int32/uint32 falls back to Vector Engine when any operand
    resides in PSUM. (See :ref:`arch_sec_neuron_core_engines` for details.)

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
    :param engine: (optional) the engine to use for the operation: `nki.isa.engine.vector`, `nki.isa.engine.gpsimd`
                   or `nki.isa.engine.unknown` (default, let compiler select best engine based on the input tile shape)."""
    ...

def tensor_scalar(dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, engine=engine.unknown, name=None):
    r"""Apply up to two math operators to the input ``data`` tile by broadcasting scalar/vector operands
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
    supports some operator combinations:

      - ``op0=nl.multiply`` and ``op1=nl.add``
      - ``op0=nl.multiply`` and ``op1=None``
      - ``op0=nl.add`` and ``op1=None``

    Also, arithmetic operators impose no restriction on the data types of input tensor ``data`` and output tensor ``dst``,
    but the operand0 and operand1 (if used) must be float32.
    The compute engine automatically casts ``data.dtype`` to float32
    and performs the operators in float32 math.
    The float32 computation results are cast to ``dst.dtype`` at no additional performance cost.

    :param dst: an output tile of ``(data <op0> operand0) <op1> operand1`` computation
    :param data: the input tile
    :param op0: the first math operator used with operand0 (see :ref:`nki-aluop` for supported operators).
    :param operand0: a scalar constant or a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                    is the partition axis size of the input ``data`` tile.
                    Must be ``None`` or ``0`` when ``op0`` is a unary operator (e.g., ``nl.abs``).
    :param reverse0: reverse ordering of inputs to ``op0``; if false, ``operand0`` is the rhs of ``op0``;
                     if true, ``operand0`` is the lhs of ``op0``
    :param op1: the second math operator used with operand1 (see :ref:`nki-aluop` for supported operators);
                this operator is optional
    :param operand1: a scalar constant or a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                    is the partition axis size of the input ``data`` tile
    :param reverse1: reverse ordering of inputs to ``op1``; if false, ``operand1`` is the rhs of ``op1``;
                     if true, ``operand1`` is the lhs of ``op1``
    :param engine: (optional) the engine to use for the operation: `nki.isa.engine.vector`, `nki.isa.engine.scalar`,
                   `nki.isa.engine.gpsimd` (only allowed for rsqrt) or `nki.isa.engine.unknown` (default, let
                   compiler select best engine based on the input tile shape)."""
    ...

def tensor_reduce(dst, op, data, axis, negate=False, keepdims=False, name=None):
    r"""Apply a reduction operation to the free axes of an input ``data`` tile using Vector Engine.

    The reduction operator is specified in the ``op`` input field
    (see :ref:`nki-aluop` for a list of supported reduction operators).
    ``nisa.tensor_reduce`` supports two types of reduction operators: 1) bitvec operators (e.g., bitwise_and, bitwise_or)
    and 2) arithmetic operators (e.g., add, subtract, multiply).

    The reduction axes are specified in the ``axis`` field as an int or list of ints indicating
    which dimensions to reduce. The reduction axes must be the last contiguous free dimension(s)
    of the tile, ending at the final dimension. Axis 0 (partition axis) cannot be reduced.

    For example, given a 4D tile ``(P, D1, D2, D3)``:

    - ``axis=(3,)`` reduces only ``D3``
    - ``axis=(2, 3)`` reduces ``D2`` and ``D3``
    - ``axis=(1, 2, 3)`` reduces ``D1``, ``D2``, and ``D3``

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
    2. invoke ``nisa.nc_matmul`` instructions to multiply a ``nl.ones([128, 1], dtype=data.dtype)`` tile as a stationary
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
    :param axis: int or tuple/list of ints. The axis (or axes) along which to reduce;
                 must be the last contiguous free dimension(s) ending at the final dim.
                 For example, for a 4D tile ``(P, D1, D2, D3)``: valid values are
                 ``(3,)``, ``(2, 3)``, or ``(1, 2, 3)``. Axis 0 (partition dim) cannot be reduced.
    :param negate: if True, reduction result is multiplied by ``-1.0``;
                   only applicable when op is an arithmetic operator
    :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                     With this option, the result will broadcast correctly against the input array."""
    ...

def tensor_partition_reduce(dst, op, data, name=None):
    r"""Apply a reduction operation across partitions of an input ``data`` tile using GpSimd Engine.

    :param dst: output tile with reduced result
    :param op: the reduction operator (add, max, bitwise_or, bitwise_and)
    :param data: the input tile to be reduced"""
    ...

def tensor_scalar_cumulative(dst, src, op0, op1, imm0, imm1=None, reduce_cmd=reduce_cmd.reset_reduce, name=None):
    r"""Perform tensor-scalar arithmetic operation with cumulative reduction using Vector Engine.

    The operation applies a scalar operation to each tensor element, then performs a cumulative
    reduction, storing the cumulative results in the destination tensor.

    The operation can be expressed in pseudocode as:

    .. code-block:: python

        if reduce_cmd == reset_reduce:
            if op1 == add or op1 == subtract:
                reg = 0
            elif op1 == mult:
                reg = 1
            elif op1 == max:
                reg = -inf
            elif op1 == min:
                reg = +inf
        elif reduce_cmd == reduce:
            reg = reg
        elif reduce_cmd == load_reduce:
            reg = imm1

        for i in len(in_tensor):
            if not reverse0:
                reg = op1(op0(in_tensor[i], imm0), reg)
                out_tensor[i] = reg
            else:
                reg = op1(op0(imm0, in_tensor[i]), reg)
                out_tensor[i] = reg

    **Operation constraints:**

    - Scalar operation (``op0``) must be an arithmetic op (e.g., add, mult, max)
    - Reduction operation (``op1``) is limited to add, subtract, mult, max, min
    - Input / output dtypes are restricted to BF16, FP16, FP32, FP8, UINT8, UINT16, INT8, INT16
        - INT32/UINT32 are not supported as input/output dtypes (ISA limitation)

    **Accumulator behavior:**

    The Vector Engine maintains internal accumulator registers controlled via ``reduce_cmd``:

    - ``reset_reduce``: Reset accumulator based on reduction operation type
    - ``load_reduce``: Initialize accumulator with ``imm1`` value
    - ``reduce``: Continue with existing accumulator value

    .. note::
      The accumulator registers are shared across Vector Engine accumulation instructions including
      :doc:`nki.isa.exponential <nki.isa.exponential>`, :doc:`nki.isa.range_select <nki.isa.range_select>`,
      :doc:`nki.isa.select_reduce <nki.isa.select_reduce>`, and
      :doc:`nki.isa.tensor_scalar_reduce <nki.isa.tensor_scalar_reduce>`.

    :param dst: The destination tensor to write cumulative results to
    :param src: The source tensor to process
    :param op0: Scalar arithmetic operation to apply to each element
    :param op1: Cumulative arithmetic operation for cumulative computation
    :param imm0: Scalar or vector value for tensor-scalar operation. Must be FP32 datatype
    :param imm1: (optional) Initial scalar or vector value for the accumulator when ``load_reduce``
                            is specified as the ``reduce_cmd``. Must be FP32 datatype
    :param reduce_cmd: (optional) Control accumulator behavior using ``nisa.reduce_cmd`` values,
                                defaults to ``reset_reduce``"""
    ...

def activation(dst, op, data, bias=None, scale=1.0, reduce_op=None, reduce_res=None, reduce_cmd=reduce_cmd.idle, name=None):
    r"""Apply an activation function on every element of the input tile using Scalar Engine, with an optional scale/bias operation
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

    .. code-block:: python

        output = op(data * scale + bias)

        if reduce_cmd == nisa.reduce_cmd.reset or reduce_cmd == nisa.reduce_cmd.reset_reduce:
            reduce_regs = 0

        result = reduce_op(reduce_regs, reduce_op(output, axis=<FreeAxis>))

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
    have a float32 data type, while the ``bias`` parameter can be any supported dtype except tfloat32.

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
    :param reduce_cmd: an enum member from ``nisa.reduce_cmd`` to control the state of ``reduce_regs``."""
    ...

def activation_reduce(dst, op, data, reduce_op, reduce_res, bias=None, scale=1.0, name=None):
    r"""Perform the same computation as ``nisa.activation`` and also a reduction along the free dimension of the
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
    On NeuronCore-v2, the ``reduce_op`` must be ``nl.add``.

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

    .. code-block:: python

        output = op(data * scale + bias)
        reduce_res = reduce_op(output, axis=<FreeAxis>)

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
                  for broadcast multiply"""
    ...

def activate2(dst, op, data, imm0, imm1, op0, op1, relu_param=0.0, reverse0=False, reverse1=False, reduce_op=None, reduce_res=None, reduce_cmd=reduce_cmd.idle, name=None):
    r"""Perform tensor activation with configurable tensor-scalar operations and optional reduction
    using Scalar Engine.

    .. note::
        Available only on NeuronCore-v4 and newer.

    This instruction provides a three-stage pipeline per partition:

    1. Tensor-scalar operations: ``(data op0 imm0) op1 imm1``
    2. Activation function application via ``op``
    3. Optional internal reduction controlled by ``reduce_op`` and ``reduce_cmd``

    The tensor-scalar stage supports six ``(op0, op1)`` combinations:

    - ``(nl.multiply, nl.add)`` — scale and bias
    - ``(nl.multiply, nl.subtract)`` — scale and negative bias
    - ``(nl.multiply, nl.bypass)`` — scale only
    - ``(nl.add, nl.bypass)`` — bias only
    - ``(nl.subtract, nl.bypass)`` — subtract only
    - ``(nl.bypass, nl.bypass)`` — no tensor-scalar operation

    When ``reverse0=True``, the first operation computes ``imm0 <op0> data`` instead of
    ``data <op0> imm0``. Similarly, ``reverse1=True`` computes ``imm1 <op1> result``.

    The Scalar Engine always performs math in float32 precision, automatically casting
    input data to float32 before computation and casting results to the output dtype
    at no additional performance cost.

    **Constraints**

    - Supported engines: Scalar.
    - ``data`` and ``dst`` must have the same partition dimension size (at most 128).
    - ``data`` and ``dst`` must have the same number of elements in the free dimensions.
    - All immediates (``imm0``, ``imm1``) must have the same dtype when both are tensors.
    - ``op1`` requires ``op0`` to be set.
    - ``reverse0`` requires ``op0`` to be set; ``reverse1`` requires ``op1`` to be set.

    :param dst: the activation output tile. Supported buffers: SBUF, PSUM.
    :param op: an activation function (see :ref:`nki-act-func` for supported functions).
    :param data: the input tile; layout: (partition axis <= 128, free axis). Supported buffers: SBUF, PSUM.
    :param imm0: scalar or ``[N, 1]`` vector value for the first tensor-scalar operation.
        ``N`` must match the partition dimension size of ``data``.
    :param imm1: scalar or ``[N, 1]`` vector value for the second tensor-scalar operation.
        ``N`` must match the partition dimension size of ``data``.
    :param op0: first ALU operation in tensor-scalar pipeline. Must be an arithmetic operator
        (e.g., ``nl.multiply``, ``nl.add``, ``nl.subtract``) or ``nl.bypass`` for no operation.
    :param op1: second ALU operation in tensor-scalar pipeline. Must be an arithmetic operator
        (e.g., ``nl.add``, ``nl.subtract``) or ``nl.bypass`` for no operation.
    :param relu_param: scalar or vector parameter for parameterized activation functions (e.g., PReLU).
        Defaults to ``0.0``.
    :param reverse0: reverse operand order for ``op0``. When ``True``, computes
        ``imm0 <op0> data`` instead of ``data <op0> imm0``. Requires ``op0`` to be set.
    :param reverse1: reverse operand order for ``op1``. When ``True``, computes
        ``imm1 <op1> result`` instead of ``result <op1> imm1``. Requires ``op1`` to be set.
    :param reduce_op: the reduce operation to perform on the free dimension of the activated data.
        Supported: ``nl.add``, ``nl.maximum``, ``nl.minimum``, ``nl.abs_max``, ``nl.abs_min``.
    :param reduce_res: a tile of shape ``(data.shape[0], 1)`` to hold the final state of the
        reduction registers. Supported buffers: SBUF, PSUM.
    :param reduce_cmd: an enum member from ``nisa.reduce_cmd`` to control the state of the
        reduction registers.

    **Accumulator behavior:**

    The Scalar Engine maintains internal accumulator registers (one FP32 value per lane, 128 total)
    that can be controlled via the ``reduce_cmd`` parameter:

    - ``reduce_cmd.reset_reduce``: Reset accumulators to the identity value for ``reduce_op``, then
      reduce the current activation results into the accumulators.
    - ``reduce_cmd.reduce``: Continue accumulating on top of existing accumulator values.
    - ``reduce_cmd.reset``: Reset accumulators only, without reducing current elements.
    - ``reduce_cmd.idle``: (default) Do not modify accumulator state.

    When ``reduce_res`` is provided, an additional instruction is emitted to read the accumulator
    values into the output tile.

    .. note::
      The accumulator registers are shared across Scalar Engine accumulation instructions including
      :doc:`nki.isa.activation <nki.isa.activation>` and ``nki.isa.activate2``.

    **Example**

    .. code-block:: python

        import nki
        import nki.isa as nisa
        import nki.language as nl
        import numpy as np
        import pytest

        @nki.jit
        def activate2_scale_bias_kernel(data_tensor):
            out = nl.ndarray(data_tensor.shape, dtype=nl.float32, buffer=nl.shared_hbm)

            # Load input from HBM to SBUF
            x = nl.ndarray(data_tensor.shape, dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=data_tensor)

            # activate2: multiply by 2.0, add 0.5, then apply GELU
            result = nl.ndarray(data_tensor.shape, dtype=nl.float32, buffer=nl.sbuf)
            nisa.activate2(
                dst=result,
                op=nl.gelu,
                data=x,
                imm0=2.0,
                imm1=0.5,
                op0=nl.multiply,
                op1=nl.add,
            )

            nisa.dma_copy(dst=out, src=result)
            return out

    **Behavior**

    .. code-block:: python

        for i in range(num_elements_per_partition):
            # Stage 1: tensor-scalar operations
            val = data[i]
            if op0 is not bypass:
                val = op0(val, imm0)       # or op0(imm0, val) if reverse0
            if op1 is not bypass:
                val = op1(val, imm1)       # or op1(imm1, val) if reverse1

            # Stage 2: activation function
            dst[i] = op(val, relu_param=relu_param)

            # Stage 3: optional reduction
            if reduce_cmd in (reset_reduce, reduce):
                accumulator = reduce_op(accumulator, dst[i])"""
    ...

def iota(dst, pattern, offset=0, channel_multiplier=0, name=None):
    r"""Generate a constant literal pattern into SBUF using GpSimd Engine.

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
    :param channel_multiplier: an int32 multiplier to be applied to the channel (parition) ID"""
    ...

def affine_select(dst, pattern, channel_multiplier, on_true_tile, on_false_value, cmp_op=equal, offset=0, name=None):
    r"""Select elements between an input tile ``on_true_tile`` and a scalar value ``on_false_value``
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
    :param cmp_op: comparison operator to use for predicate evaluation (default: nl.equal)"""
    ...

def local_gather(dst, src_buffer, index, num_elem_per_idx=1, num_valid_indices=None, name=None):
    r"""Gather SBUF data in ``src_buffer`` using ``index`` on GpSimd Engine.

    Each of the eight GpSimd cores in GpSimd Engine connects to 16 contiguous SBUF partitions
    (e.g., core[0] connected to partition[0:16]) and performs gather from the connected 16
    SBUF partitions *independently* in parallel. The indices used for gather on each core should also
    come from the same 16 connected SBUF partitions. If you only need to gather elements within a partition,
    consider using :doc:`nisa.nc_n_gather <nki.isa.nc_n_gather>` instead, which supports gathering more indices.

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
    full NKI code example with equivalent numpy implementation."""
    ...

def nc_n_gather(dst, data, indices, name=None):
    r"""Gather elements from ``data`` according to ``indices`` using GpSimd Engine.

    This instruction performs a gather operation where elements are selected from the input ``data`` tile
    based on flattened indices specified in the ``indices`` tile. The free dimensions of ``data`` are
    treated as if they were flattened into a single dimension for indexing purposes, while the partition
    dimension defines the parallel compute boundary.

    The gather operation works independently within each partition. For each partition, the free dimensions
    of ``data`` are conceptually flattened, and elements are gathered according to the corresponding
    flattened indices from the same partition in ``indices``. If you need to gather elements across partitions
    (within groups of partitions), consider using :doc:`nisa.local_gather <nki.isa.local_gather>`.

    The ``n`` in ``nc_n_gather`` indicates that this instruction corresponds to ``n`` groups of instructions
    in the underlying ISA, where ``n = ceil(elems_per_partition / 512)``.

    Alternatively, we could gather elements by calling :doc:`nisa.dma_copy <nki.isa.dma_copy>` with an
    indirect access pattern derived from ``indices``. However, this is less efficient than ``nc_n_gather``,
    which uses GpSimd Engine to perform local data movement within SBUF, without using DMA engines.

    **Memory types.**

    All input and output tiles (``data``, ``indices``, and ``dst``) must be in SBUF.
    GpSimd Engine cannot access PSUM (see :ref:`arch_sec_neuron_core_engines` for details).

    **Data types.**

    The input ``data`` tile can be any valid NKI data type (see :ref:`nki-dtype` for more information).
    The output ``dst`` tile must have the same data type as ``data``.
    The ``indices`` tile must be uint32.

    **Layout.**

    The partition dimension of ``data``, ``indices``, and ``dst`` must be the same.
    Within each partition, the free dimensions of ``data`` are flattened for indexing.
    The free dimensions of ``indices`` determine the shape of the output ``dst``.

    **Tile size.**

    The partition dimension size of ``data``, ``indices``, and ``dst`` must be the same and must not exceed 128.
    The number of elements per partition in ``dst`` must match the number of elements per partition in ``indices``.
    The indices' values must be within the range ``[0, data.size / data.shape[0])``.

    :param dst: output tile containing the gathered elements
    :param data: the input tile to gather elements from
    :param indices: the indices tile (uint32) specifying which elements to gather"""
    ...

def bn_stats(dst, data, name=None):
    r"""Compute mean- and variance-related statistics for each partition of an input tile ``data``
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
    The engine automatically casts the input ``data`` to float32 before performing computation.
    The float32 computation results are cast to ``dst.dtype`` at no additional performance cost.

    :param dst: an output tile with 6-element statistics per partition
    :param data: the input tile (up to 512 elements per partition)"""
    ...

def bn_aggr(dst, data, name=None):
    r"""Aggregate one or multiple ``bn_stats`` outputs to generate
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
    The engine automatically casts the input ``data`` to float32 before performing computation.
    The float32 computation results are cast to ``dst.dtype`` at no additional performance cost.

    :param dst: an output tile with two elements per partition: a mean followed by a variance
    :param data: an input tile with results of one or more :doc:`bn_stats <nki.isa.bn_stats>`"""
    ...

def tensor_tensor_scan(dst, data0, data1, initial, op0, op1, reverse0=False, reverse1=False, name=None):
    r"""Perform a scan operation of two input tiles using Vector Engine.

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
    and performs the computation in float32 math. The float32 computation results are
    cast to ``dst.dtype`` at no additional performance cost.

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
                   if true, ``data1`` is the lhs of ``op1``"""
    ...

def nc_stream_shuffle(dst, src, shuffle_mask, name=None):
    r"""Apply cross-partition data movement within a quadrant of 32 partitions from source tile
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
    :param shuffle_mask: a 32-element list that specifies the shuffle source and destination partition"""
    ...

def rng(dst, engine=engine.unknown, name=None):
    r"""Generate pseudo random numbers using the Vector or GpSimd Engine.

    This instruction generates 32 random bits per element and writes them to the
    destination tensor. Depending on the size of the dtype, the instruction truncates
    each 32-bit random value to the specified data type, taking the least significant bits.

    Example use case:
    To generate random FP32 numbers between 0.0 and 1.0, follow the Rng instruction
    with a normalization instruction (e.g., write 16 random bits as UINT16, then
    divide by (2^16-1) to get a random FP32 number between 0.0 and 1.0).

    **Memory types.**

    The output ``dst`` tile can be in SBUF or PSUM.

    **Data types.**

    The output ``dst`` tile must be an integer type: int8, int16, int32, uint8, uint16, or uint32.

    **Tile size.**

    The partition dimension size of ``dst`` must not exceed 128. The number of
    elements per partition of ``dst`` must not exceed the physical size of each SBUF/PSUM partition.

    **Constraints.**

    - Supported arch versions: NeuronCore-v2+.
    - Supported engines: NeuronCore-v2: Vector. NeuronCore-v3+: GpSimd, Vector.
    - Since GpSimd Engine cannot access PSUM, ``dst`` must be in SBUF when using GpSimd Engine.

    :param dst: the destination tensor to write random values to
    :param engine: specify which engine to use: ``nki.isa.engine.vector``, ``nki.isa.engine.gpsimd``,
                   or ``nki.isa.engine.unknown`` (default, the best engine will be selected)"""
    ...

def dropout(dst, data, prob, name=None):
    r"""Randomly replace some elements of the input tile ``data`` with zeros
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

    The output data type ``dst.dtype`` must match the input data type ``data.dtype``.

    :param dst: an output tile of the dropout result
    :param data: the input tile
    :param prob: a scalar or a tile of shape ``(data.shape[0], 1)`` to indicate the
                 probability of replacing elements with zeros"""
    ...

def set_rng_seed(src_seeds, name=None):
    r"""Seed the pseudo random number generator (PRNG) inside the Vector Engine.

    The PRNG state is cached inside the engine as a persistent state during the rest of NEFF
    execution. However, the state cannot survive TPB resets or Runtime reload.

    Using the same seed will generate the same sequence of random numbers when used
    together with the ``nisa.rng()`` on the Vector Engine.

    **Memory types.**

    The input ``src_seeds`` must be in SBUF or PSUM.

    **Data types.**

    The input ``src_seeds`` must be a 32-bit value.

    **Tile size.**

    The input ``src_seeds`` must be a [1,1] tensor.

    :param src_seeds: a [1,1] tensor on SBUF or PSUM with a 32-bit value to be used as the seed"""
    ...

def rand_set_state(src_seeds, engine=engine.unknown, name=None):
    r"""Seed the pseudo random number generator (PRNG) inside the engine.

    This instruction initializes the PRNG state for future random number generation operations.
    Each partition in the source tensor seeds the PRNG states for the corresponding compute lane
    inside the engine.

    The PRNG state is cached inside the engine as a persistent state during the rest of NEFF
    execution. However, the state cannot survive TPB resets or Runtime reload.

    **Memory types.**

    The input ``src_seeds`` tile must be in SBUF.

    **Data types.**

    The input ``src_seeds`` tile must be uint32.

    **Tile size.**

    - src_seeds element count for XORWOW must be 6 elements (GpSimd) or 24 elements (Vector).

    **Constraints.**

    - Supported arch versions: NeuronCore-v3+.
    - Supported engines: NeuronCore-v3: GpSimd. NeuronCore-v4+: GpSimd, Vector.
    - ``src_seeds`` must be in SBUF.

    :param src_seeds: the source tensor containing seed values for the PRNG; must be a 2D uint32 tensor
                      with the partition dimension representing the compute lanes and the free dimension
                      containing the seed values
    :param engine: specify which engine to use: ``nki.isa.engine.vector``, ``nki.isa.engine.gpsimd``,
                   or ``nki.isa.engine.unknown`` (default, the best engine will be selected)"""
    ...

def rand_get_state(dst, engine=engine.unknown, name=None):
    r"""Store the current pseudo random number generator (PRNG) states from the engine.

    This instruction stores the current PRNG states cached inside the engine to SBUF/PSUM.
    Each partition in the output tensor holds the PRNG states for the corresponding compute lane
    inside the engine.

    **Memory types.**

    The output ``dst`` tile must be in SBUF (NeuronCore-v3) or SBUF/PSUM (NeuronCore-v4+).

    **Data types.**

    The output ``dst`` tile must be uint32.

    **Tile size.**

    - dst element count for XORWOW must be 6 elements (GpSimd) or 24 elements (Vector).

    **Constraints.**

    - Supported arch versions: NeuronCore-v3+.
    - Supported engines: NeuronCore-v3: GpSimd. NeuronCore-v4+: GpSimd, Vector.
    - Since GpSimd Engine cannot access PSUM, ``dst`` must be in SBUF when using GpSimd Engine.

    :param dst: the destination tensor to store PRNG state values; must be a 2D uint32 tensor
    :param engine: specify which engine to use: ``nki.isa.engine.vector``, ``nki.isa.engine.gpsimd``,
                   or ``nki.isa.engine.unknown`` (default, the best engine will be selected)"""
    ...

def rand2(dst, min, max, name=None):
    r"""Generate pseudo random numbers with uniform distribution using Vector Engine.

    .. note::

      Available only on NeuronCore-v4 and newer.

    This instruction generates pseudo random numbers and stores them into SBUF/PSUM.
    The generated values follow a uniform distribution within the specified [min, max] range.

    Key features:

    - Uses XORWOW PRNG algorithm for high-quality random number generation
    - Generates FP32 random values with uniform distribution
    - Supports output conversion to various data types

    **Memory types.**

    The output ``dst`` tile can be in SBUF or PSUM.

    **Data types.**

    The output ``dst`` tile can be any of: float8_e4m3, float8_e5m2, float16, bfloat16, float32,
    tfloat32, int8, int16, int32, uint8, uint16, or uint32.

    **Tile size.**

    The partition dimension size of ``dst`` must not exceed 128. The number of
    elements per partition of ``dst`` must not exceed the physical size of each SBUF/PSUM partition.

    **Constraints.**

    - Supported arch versions: NeuronCore-v4+.
    - Supported engines: Vector.
    - min < max for valid range.

    :param dst: the destination tensor to write random values to
    :param min: minimum value for uniform distribution range (FP32), can be a scalar or vector value
    :param max: maximum value for uniform distribution range (FP32), can be a scalar or vector value"""
    ...

def exponential(dst, src, max_value=0.0, reduce_res=None, reduce_cmd: reduce_cmd=reduce_cmd.idle, reduce_init=0.0, name=None):
    r"""Apply exponential function to each element after subtracting a max_value using Vector Engine.

    .. note::
        Available only on NeuronCore-v4 and newer.

    This instruction computes ``exp(src - max_value)`` for each element. The instruction can
    optionally maintain a running sum of the exponential values using shared internal reduction
    registers in the Vector Engine.

    The exponential operation is performed as:

    .. code-block::

        dst[i] = exp(src[i] - max_value)

    When accumulation is enabled through ``reduce_cmd``, the instruction also computes:

    .. code-block::

        reduce_res[i] = sum(dst[i])

    The Vector Engine performs the computation in float32 precision internally and can
    output results in various data types as specified by the ``dst`` dtype field.

    **Constraints**

    - Supported engines: Vector.
    - ``src``, ``dst`` must have the same number of elements in the partition dimension.
    - ``src``, ``dst`` must have the same number of elements in the free dimensions.
    - ``src``, ``dst`` can be up to 4D tensor.
    - ``reduce_init`` should be unset or set to ``0.0`` when ``reduce_cmd`` is not ``load_reduce``.

    :param dst: The output tile with exponential function applied. Supported buffers: SBUF, PSUM. Supported dtypes: float8_e4m3, float8_e5m2, float16, bfloat16, float32, tfloat32, int8, int16, int32, uint8, uint16.
    :param src: The input tile to apply exponential function on. Supported buffers: SBUF, PSUM. Supported dtypes: float8_e4m3, float8_e5m2, float16, bfloat16, float32, int8, int16, int32, uint8, uint16, uint32.
    :param max_value: The maximum value to subtract from each element before applying exponential (for numerical stability). Can be a scalar or vector of shape ``(src.shape[0], 1)``. Supported dtypes: float32.
    :param reduce_res: Optional tile to store reduction results (sum of exponentials). Must have shape ``(src.shape[0], 1)``. Supported buffers: SBUF, PSUM. Supported dtypes: float8_e4m3, float8_e5m2, float16, bfloat16, float32, tfloat32.
    :param reduce_cmd: Control the state of reduction registers for accumulating exponential results. Supported: ``idle``, ``reset_reduce``, ``reduce``, ``load_reduce``.
    :param reduce_init: Initial value for reduction when using ``reduce_cmd.load_reduce``. Supported dtypes: float32.

    **Accumulator behavior:**

    The Vector Engine maintains internal accumulator registers that can be controlled via the ``reduce_cmd`` parameter:

    - ``reduce_cmd.reset_reduce``: Reset accumulators to 0, then accumulate the current results.
    - ``reduce_cmd.reduce``: Continue accumulating without resetting (useful for multi-step reductions).
    - ``reduce_cmd.load_reduce``: Load the values from ``reduce_init`` into the accumulator, then accumulate the current result on top of it.
    - ``reduce_cmd.idle``: (default) No accumulation performed, accumulator state unknown.

    .. note::
      Even when ``reduce_cmd`` is set to ``idle``, the accumulator state may still be modified.
      Always use ``reset_reduce`` after any Vector Engine operation that ran with ``idle`` mode to ensure
      consistent behavior.

    .. note::
      The accumulator registers are shared across Vector Engine accumulation instructions including
      :doc:`nki.isa.range_select <nki.isa.range_select>`, :doc:`nki.isa.select_reduce <nki.isa.select_reduce>`,
      :doc:`nki.isa.tensor_scalar_reduce <nki.isa.tensor_scalar_reduce>`, and
      :doc:`nki.isa.tensor_scalar_cumulative <nki.isa.tensor_scalar_cumulative>`.

    **Behavior**

    .. code-block:: python

        # Initialize reduction if requested
        if reduce_cmd == reduce_cmd.reset_reduce:
            accumulator = 0
        elif reduce_cmd == reduce_cmd.load_reduce:
            accumulator = reduce_init
        elif reduce_cmd == reduce_cmd.idle:
            accumulator = undefined  # Not used

        # Process each element
        for i in range(num_elements):
            dst[i] = exp(src[i] - max_value)

            # Update reduction if active
            if reduce_cmd != reduce_cmd.idle:
                accumulator += dst[i]"""
    ...

def reciprocal(dst, data, name=None):
    r"""Compute element-wise reciprocal (1.0/x) of the input ``data`` tile using Vector Engine.

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
    :param data: the input tile"""
    ...

def max8(dst, src, name=None):
    r"""Find the 8 largest values in each partition of the source tile.

    This instruction reads the input elements, converts them to fp32 internally, and outputs
    the 8 largest values in descending order for each partition. Outputs are converted to
    ``dst.dtype`` automatically.

    The source tile can be up to 3-dimensional, while the output tile is always 2-dimensional.
    The number of elements read per partition must be between 8 and 16,384 inclusive.
    The output will always contain exactly 8 elements per partition.
    The source and output must have the same partition dimension size:

    - source: [par_dim, ...]
    - output: [par_dim, 8]

    :param dst: a 2D tile containing the 8 largest values per partition in descending order with shape [par_dim, 8]
    :param src: the source tile to find maximum values from"""
    ...

def sequence_bounds(dst, segment_ids, name=None):
    r"""Compute the sequence bounds for a given set of segment IDs using GpSIMD Engine.

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

    Both the input tile (``segment_ids``) and output tile (``dst``) must have data type ``nl.float32`` or ``nl.int32``.

    **NumPy equivalent:**

    :param dst: tile containing the sequence bounds.
    :param segment_ids: tile containing the segment IDs. Elements with ID=0 are treated as padding."""
    ...

def scalar_tensor_tensor(dst, data, op0, operand0, op1, operand1, reverse0=False, reverse1=False, name=None):
    r"""Apply two math operators in sequence using Vector Engine: ``(data <op0> operand0) <op1> operand1``.

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
                     if true, ``operand1`` is the lhs of ``op1``"""
    ...

def tensor_scalar_reduce(dst, data, op0, operand0, reduce_op, reduce_res, reverse0=False, reduce_cmd: reduce_cmd=reduce_cmd.reset_reduce, reduce_init=None, name=None):
    r"""Perform the same computation as ``nisa.tensor_scalar`` with one math operator
    and also a reduction along the free dimension of the ``nisa.tensor_scalar`` result using Vector Engine.

    Refer to :doc:`nisa.tensor_scalar <nki.isa.tensor_scalar>` for semantics of ``data/op0/operand0``.
    Unlike regular ``nisa.tensor_scalar`` where two operators are supported, only one
    operator is supported in this API. Also, ``op0`` can only be arithmetic operation in :ref:`nki-aluop`.
    Bitvec operators are not supported in this API.

    In addition to :doc:`nisa.tensor_scalar <nki.isa.tensor_scalar>` computation, this API also performs a reduction
    along the free dimension(s) of the :doc:`nisa.tensor_scalar <nki.isa.tensor_scalar>` result, at a small additional
    performance cost. The reduction result is returned in ``reduce_res`` in-place, which must be a
    SBUF/PSUM tile with the same partition axis size as the input tile ``data`` and one element per partition.
    The ``reduce_op`` can be any of ``nl.add``, ``nl.multiply``, ``nl.max`` or ``nl.min``.

    Reduction axis is not configurable in this API. If the input tile has multiple free axis, the API will
    reduce across all of them.

    .. math::
      result = data <op0> operand0 \\
      reduce\_res = reduce\_op(dst, axis=<FreeAxis>)

    **Accumulator behavior:**

    The Vector Engine maintains internal accumulator registers that can be controlled via the ``reduce_cmd`` parameter:

    - ``reduce_cmd.reset_reduce``: (default) Reset accumulators to 0, then accumulate the current results.
    - ``reduce_cmd.reduce``: Continue accumulating without resetting (useful for multi-step reductions across tiles).
    - ``reduce_cmd.load_reduce``: Load the values from ``reduce_init`` into the accumulator, then accumulate
      the current result on top of it.

    .. note::
      ``reduce_init`` should only be set when ``reduce_cmd`` is ``load_reduce``.

    .. note::
      The accumulator registers are shared across Vector Engine accumulation instructions including
      :doc:`nki.isa.exponential <nki.isa.exponential>`, :doc:`nki.isa.range_select <nki.isa.range_select>`,
      :doc:`nki.isa.select_reduce <nki.isa.select_reduce>`, and
      :doc:`nki.isa.tensor_scalar_cumulative <nki.isa.tensor_scalar_cumulative>`.

    :param dst: an output tile of ``(data <op0> operand0)`` computation
    :param data: the input tile
    :param op0: the math operator used with operand0 (any arithmetic operator in :ref:`nki-aluop` is allowed).
    :param operand0: a scalar constant or a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                    is the partition axis size of the input ``data`` tile.
                    Must be ``None`` or ``0`` when ``op0`` is a unary operator (e.g., ``nl.abs``).
    :param reverse0: `(not supported yet)` reverse ordering of inputs to ``op0``; if false, ``operand0`` is the rhs of ``op0``;
                     if true, ``operand0`` is the lhs of ``op0``. `<-- currently not supported yet.`
    :param reduce_op: the reduce operation to perform on the free dimension of ``data <op0> operand0``
    :param reduce_res: a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                    is the partition axis size of the input ``data`` tile. The result of ``reduce_op(data <op0> operand0)``
                    is written in-place into the tile.
    :param reduce_cmd: Control the state of reduction registers for accumulating reduction results.
                       Supported: ``reset_reduce`` (default), ``reduce``, ``load_reduce``.
    :param reduce_init: Initial value for reduction when using ``reduce_cmd.load_reduce``.
                        Must be provided when ``reduce_cmd`` is ``load_reduce``. Supported dtypes: float32."""
    ...

def core_barrier(data, cores, engine=engine.gpsimd, name=None):
    r"""Synchronize execution across multiple NeuronCores by implementing a barrier mechanism.

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
    instruction (that is, the remote semaphore update and local semaphore wait). The barrier will block
    execution on this engine, other engines will not be blocked.

    :param data: the shared data that all cores need to synchronize on; must be data in shared HBM
    :param cores: a tuple of core indices to synchronize; only ``(0, 1)`` is supported when LNC2 is enabled
    :param engine: the engine to execute the barrier instruction on; defaults to GpSimd Engine

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

        # Now both cores can safely read the complete tensor"""
    ...

def dma_compute(dst, srcs, reduce_op, scales=None, unique_indices=True, oob_mode=oob_mode.error, name=None):
    r"""Perform math operations using compute logic inside DMA engines with element-wise scaling and reduction.

    This instruction leverages the compute capabilities within DMA engines to perform scaled element-wise operations
    followed by reduction across multiple source tensors. The computation follows the pattern:
    ``dst = reduce_op(srcs[0] * scales[0], srcs[1] * scales[1], ...)``, where each source tensor is first
    multiplied by its corresponding scale factor, then all scaled results are combined using the specified
    reduction operation.
    Currently, only ``nl.add`` is supported for ``reduce_op``, and
    all values in ``scales`` must be ``1.0`` (or ``scales`` can be ``None``
    which defaults to all 1.0).

    The DMA engines perform all computations in float32 precision internally. Input tensors are automatically
    cast from their source data types to float32 before computation, and the final float32 result is cast
    to the output data type in a pipelined fashion.

    **Read-Modify-Write with vector_offset (scatter and gather).**

    When one of the source tensors has a ``vector_offset`` (indirect indexing),
    ``dma_compute`` performs read-modify-write with two modes:

    **Scatter RMW**: ``dst(HBM)[indices] = dst(HBM)[indices] + src(SB)``
      - ``dst`` is in HBM with indirect indexing
      - One source matches ``dst`` and has ``vector_offset``
      - The other source is data in SBUF

    **Gather RMW**: ``dst(SB) = dst(SB) + src(HBM)[indices]``
      - ``dst`` is in SBUF
      - One source is data in HBM with ``vector_offset``
      - The other source matches ``dst``

    Both modes require:
      - Exactly 2 source tensors
      - All ``scales`` must be ``1.0`` (or ``None``)
      - ``unique_indices`` must be ``True`` (non-unique indices not yet supported)

    **Memory types.**

    Both input ``srcs`` tensors and output ``dst`` tensor can be in HBM or SBUF.
    Both ``srcs`` and ``dst`` tensors must have compile-time known addresses (unless using vector_offset for indirect access).

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
    :param reduce_op: the reduction operation to apply (currently only ``nl.add`` is supported)
    :param scales: (optional) a list of scale factors corresponding to each
                   tensor in ``srcs``. Must be all 1.0 if provided.
                   Defaults to None (equivalent to [1.0, 1.0, ...]).
    :param unique_indices: (optional) Whether scatter indices are unique.
                          Must be True when using vector_offset (non-unique
                          not yet supported). Default: True.
    :param oob_mode: (optional) Specifies how to handle out-of-bounds (oob)
                     array indices during indirect access operations. Valid
                     modes are:

        - ``oob_mode.error``: (Default) Raises an error when encountering
          out-of-bounds indices.
        - ``oob_mode.skip``: Silently skips any operations involving
          out-of-bounds indices.

        For example, when using indirect gather/scatter operations with
        ``vector_offset``, out-of-bounds indices can occur if the index
        array contains values that exceed the dimensions of the target array."""
    ...

def quantize_mx(dst, src, dst_scale, name=None):
    r"""Quantize FP16/BF16 data to MXFP8 tensors (both data and scales) using Vector Engine.

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

    **Tile size.**

    - The partition dimension size of ``src`` must be a multiple of 32 and must not exceed 128.
    - The free dimension size of ``src`` must be a multiple of 4 and must not exceed the physical size of each SBUF
      partition.
    - The ``dst`` tile has the same partition dimension size as ``src`` but a free dimension size
      that is 1/4 of ``src`` free dimension size due to the special 4-packed FP8 data types.

    :param dst: the quantized MXFP8 output tile
    :param src: the input FP16/BF16 tile to be quantized
    :param dst_scale: the output scale tile"""
    ...

def range_select(dst, on_true_tile, comp_op0, comp_op1, bound0, bound1, reduce_cmd=reduce_cmd.reset_reduce, reduce_res=None, reduce_op=maximum, range_start=0, on_false_value=..., name=None):
    r"""Select elements from ``on_true_tile`` based on comparison with bounds using Vector Engine.

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

    - If ``dst.dtype`` and ``reduce_res.dtype`` are both FP32, we should not hit any NaN issue
      since ``FP32_MIN - FP32_MIN = 0``. Exponentiation on 0 is stable (1.0 exactly).

    - If ``dst.dtype`` is FP16/BF16/FP8, the fill_value in the output tile will become ``-INF``
      since HW performs a downcast from FP32_MIN to a smaller dtype.
      In this case, you must make sure ``reduce_res.dtype`` is FP32 to avoid NaN in ``activation``.
      NaN can be avoided because ``activation`` always upcasts input tiles to FP32 to perform math operations: ``-INF - FP32_MIN = -INF``.
      Exponentiation on ``-INF`` is stable (0.0 exactly).

    **Constraints:**

    The comparison operators must be one of:

    - nl.equal
    - nl.less
    - nl.less_equal
    - nl.greater
    - nl.greater_equal

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
      Due to hardware constraints, this must be ``FP32_MIN`` (``-3.4028235e+38``).
      See the numerical stability note above for guidance on output dtype selection.
    :param comp_op0: first comparison operator
    :param comp_op1: second comparison operator
    :param bound0: tile with one element per partition for first comparison
    :param bound1: tile with one element per partition for second comparison
    :param reduce_op: reduction operator to apply on across the selected output. Currently only ``nl.maximum`` is supported.
    :param reduce_cmd: controls the state of the Vector Engine accumulator registers.
      Defaults to ``reduce_cmd.reset_reduce``. See :class:`nki.isa.reduce_cmd` for supported values.
    :param reduce_res: optional tile to store reduction results.
    :param range_start: starting base offset for index array for the free dimension of ``on_true_tile``.
        Defaults to 0, and must be a compile-time integer."""
    ...

def select_reduce(dst, predicate, on_true, on_false, reduce_res=None, reduce_cmd=reduce_cmd.idle, reduce_op=maximum, reverse_pred=False, name=None):
    r"""Selectively copy elements from either ``on_true`` or ``on_false`` to the destination tile
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
      The accumulator registers are shared across Vector Engine accumulation instructions including
      :doc:`nki.isa.exponential <nki.isa.exponential>`, :doc:`nki.isa.range_select <nki.isa.range_select>`,
      :doc:`nki.isa.tensor_scalar_reduce <nki.isa.tensor_scalar_reduce>`, and
      :doc:`nki.isa.tensor_scalar_cumulative <nki.isa.tensor_scalar_cumulative>`.

    :param dst: The destination tile to write the selected values to
    :param predicate: Tile that determines which value to select (on_true or on_false)
    :param on_true: Tile to select from when predicate is True
    :param on_false: Value to use when predicate is False, can be a scalar value or a vector tile of ``(on_true.shape[0], 1)``
    :param reduce_res: (optional) Tile to store reduction results, must have shape ``(on_true.shape[0], 1)``
    :param reduce_cmd: (optional) Control accumulator behavior using ``nisa.reduce_cmd`` values, defaults to idle
    :param reduce_op: (optional) Reduction operator to apply (only ``nl.maximum`` is supported)
    :param reverse_pred: (optional) Reverse the meaning of the predicate condition, defaults to False"""
    ...

def sendrecv(src, dst, send_to_rank, recv_from_rank, pipe_id, dma_engine=dma_engine.dma, name=None):
    r"""Perform point-to-point communication between NeuronCores by sending and receiving data
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

    The ``dma_engine`` parameter specifies which DMA transfer mechanism to use:

    - ``nisa.dma_engine.dma`` (default): Uses the standard DMA engine with CoreBarrier synchronization.
      Can be triggered from any engine.
    - ``nisa.dma_engine.gpsimd_dma``: Uses the GPSIMD's internal DMA engine for low-latency
      SB-to-SB swaps in LNC=2. Implies GPSIMD as the trigger engine. This mode restricts the data size
      per partition to not exceed:

       - 1024 bytes for 32-bit types
       - 512 bytes for 16-bit types
       - 256 bytes for 8-bit types

    **Constraints.**

    - ``src`` and ``dst`` tiles must both be in SBUF.
    - ``src`` and ``dst`` must have the same data type, but they can be any supported data types in NKI.
    - ``src`` and ``dst`` must have the same shape and layout.
    - ``src`` and ``dst`` must have the same partition dimension size and the same number of elements per partition.

    :param src: the source tile on the current NeuronCore to be sent to the target NeuronCore
    :param dst: the destination tile on the current NeuronCore where received data will be stored
    :param send_to_rank: rank ID of the target NeuronCore to send data to
    :param recv_from_rank: rank ID of the source NeuronCore to receive data from
    :param pipe_id: synchronization identifier that groups sendrecv operations; operations with the same pipe_id are synchronized
    :param dma_engine: the DMA transfer mode; defaults to ``nisa.dma_engine.dma``

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

        # Now recv_buffer contains data from the previous core"""
    ...

def register_alloc(x: int | None=None):
    r"""Allocate a virtual register and optionally initialize it with a value.

    Each engine sequencer (Tensor/Scalar/Vector/GpSimd/Sync Engine) within a NeuronCore maintains its own set of
    physical registers for scalar operations (64x 32-bit registers per engine sequencer in NeuronCore v2-v4).
    This API conceptually allocates a register within a virtual register space.
    Users do not need to explicitly free a register through nisa APIs. The NKI compiler
    handles physical register allocation (and deallocation) across the appropriate engine sequencers
    based on the dynamic program flow.

    NKI provides the following APIs to manipulate allocated registers:

    - ``nisa.register_move``: Move a constant integer or another register's value into a register
    - ``nisa.register_load``: Load a scalar (32-bit) value from HBM/SBUF into a register
    - ``nisa.register_store``: Store register contents to HBM/SBUF

    In the current NKI release, these registers are primarily used to specify dynamic loop boundaries and
    while loop conditions. The NKI compiler compiles such dynamic looping constructs to branching instructions
    executed by engine sequencers. For additional details, see ``nl.dynamic_range``. For more information
    on engine sequencer and its capabilities, see
    `Trainium/Inferentia2 architecture guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/arch/trainium_inferentia2_arch.html>`_.

    :param x: optional initialization value. Can be one of:

              - ``None`` (default): allocate an uninitialized register
              - ``int``: allocate a register initialized with this immediate integer value

    Example:

    Three ways to allocate a register initialized to zero:

    .. code-block:: python

        # Approach 1: Using an immediate value
        reg1 = nisa.register_alloc(0)

        # Approach 2: Two-step with register_load
        zero_tensor = nl.zeros([1, 1], dtype=nl.int32, buffer=nl.sbuf)
        reg2 = nisa.register_alloc(None)
        nisa.register_load(reg2, zero_tensor)"""
    ...

def register_move(dst, src):
    r"""Move a value into a virtual register.

    This instruction loads a value into the specified virtual register. The source can be
    either a compile-time constant integer or another virtual register.

    The virtual register system allows the NKI compiler to allocate physical registers across
    different engine sequencers as needed. See ``nisa.register_alloc`` for more details on
    virtual register allocation.

    This instruction operates on virtual registers only and does not access SBUF, PSUM, or HBM.

    :param dst: the destination virtual register (allocated via ``nisa.register_alloc``)
    :param src: source value - either a compile-time constant integer or a VirtualRegister

    Example:

    .. code-block:: python

        # Allocate a register and initialize it with a constant
        loop_count = nisa.register_alloc()
        nisa.register_move(loop_count, 10)  # Set register to 10

        # Copy from another register
        reg2 = nisa.register_alloc()
        nisa.register_move(reg2, loop_count)  # Copy value from loop_count"""
    ...

def register_load(dst, src):
    r"""Load a scalar value from memory (HBM or SBUF) into a virtual register.

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
        nisa.register_load(loop_reg, computed_bound)"""
    ...

def register_store(dst, src):
    r"""Store the value from a virtual register into memory (HBM/SBUF).

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
        nisa.register_store(result_tensor, counter_reg)"""
    ...

def nc_find_index8(dst, data, vals, name=None):
    r"""Find indices of the 8 given vals in each partition of the data tensor.

    This instruction first loads the 8 values,
    then loads the data tensor and outputs the indices (starting at 0) of the first
    occurrence of each value in the data tensor, for each partition.

    The data tensor can be up to 3-dimensional, while the vals tensor must be up
    to 3-dimensional. The data tensor must have between 8 and 16,384 elements per
    partition. The vals tensor must have exactly 8 elements per partition.
    The output will contain exactly 8 elements per partition and will be uint16 or
    uint32 type. Default output type is uint32.

    Behavior is undefined if vals tensor contains values that are not in
    the data tensor.

    If provided, a mask is applied only to the data tensor.

    :param dst: a 2D tile containing indices (uint16 or uint32) of the 8 values in each partition with shape [par_dim, 8]
    :param data: the data tensor to find indices from
    :param vals: tensor containing the 8 values per partition whose indices will be found"""
    ...

def nc_match_replace8(dst, data, vals, imm: float, dst_idx=None, name=None):
    r"""Replace first occurrence of each value in ``vals`` with ``imm`` in ``data``
    using the Vector engine and return the replaced tensor. If ``dst_idx``
    tile is provided, the indices of the matched values are written to ``dst_idx``.

    :param dst: output tile with replaced values
    :param data: the data tensor to search and replace in
    :param vals: tensor containing the 8 values per partition to match
    :param imm: the immediate float value to replace matched values with
    :param dst_idx: optional tile to store indices of matched values"""
    ...

def nonzero_with_count(dst, src, index_offset=0, padding_val=-1, name=None):
    r"""Find indices of nonzero elements in an input tensor and their total count using GpSimd Engine.

    .. note::

      Available only on NeuronCore-v3 and newer.

    NOTE: this instruction only operates on partitions [0, 16, 32, ..., 112] of the input tile
    and writes to partitions [0, 16, 32, ..., 112] of the destination tile. The data in other
    partitions of the destination tile are not modified, including the last 'extra' slot for count.

    This behavior is due to the physical connectivity of GpSimd engine. Each of the eight GpSimd cores
    connects to 16 contiguous SBUF partitions (e.g., core[0] connects to partitions[0:16]).
    In nonzero_with_count, each GpSimd core reads from and writes to its 0-th partition only.

    This instruction takes an input array and produces an output array containing the indices of all
    nonzero elements, followed by padding values, and ending with the count of nonzero elements found.

    The output tensor has one more element in the free dimension than the input tensor:

    - **First N elements**: 0-indexed positions of nonzero elements, offset by ``index_offset``
    - **Next T-N elements**: Filled with ``padding_val``
    - **Last element**: Count ``N`` of nonzero elements found

    The ``index_offset`` parameter is useful when processing arrays in tiles, allowing
    indices to be relative to the original array position rather than the tile.

    Example for one partition of the tensor:

    .. code-block::

        Input array (T=8): [0, 1, 1, 0, 0, 1, 0, 0]
        index_offset = 16
        padding_val = -1

        Output (T+1=9): [17, 18, 21, -1, -1, -1, -1, -1, 3]

        Where:

        - 17, 18, 21 are the indices (1, 2, 5) plus offset 16
        - -1 is the padding value for unused slots
        - 3 is the count of nonzero elements

    **Constraints**

    - Supported arch versions: NeuronCore-v3+.
    - Supported engines: GpSimd.
    - Parameters ``src``, ``dst`` must have the same number of elements in the partition dimension.
    - Destination tensor must have exactly 1 more element than the source tensor in the free dimension.
    - Only accesses the 0-th partition for each GpSimd core (i.e., [0, 16, 32, ..., 112]).
    - ``src`` must be in SBUF with dtype float32 or int32.
    - ``dst`` must be in SBUF with dtype int32.
    - ``index_offset`` and ``padding_val`` must be int32.

    :param src: Input tensor to find nonzero indices from. Only partitions [0, 16, 32, ..., 112] are read from. Supported buffers: SBUF. Supported dtypes: float32, int32.
    :param dst: Output tensor containing nonzero indices, padding, and count. Only partitions [0, 16, 32, ..., 112] are written to. It must have one extra element than src in the free dimension. Supported buffers: SBUF. Supported dtypes: int32.
    :param index_offset: Offset to add to the found indices (useful for tiled processing). Supported dtypes: int32.
    :param padding_val: Value to use for padding unused output elements. Supported dtypes: int32.

    **Behavior**

    .. code-block:: python

        # Find all nonzero elements in input
        nonzero_indices = []
        for i in range(len(input_array)):
            if input_array[i] != 0:
                nonzero_indices.append(i + index_offset)

        # Build output array
        output = []
        # Add found indices
        for idx in nonzero_indices:
            output.append(idx)
        # Add padding for remaining slots
        for _ in range(len(input_array) - len(nonzero_indices)):
            output.append(padding_val)
        # Add count as last element
        output.append(len(nonzero_indices))

    **Example**

    .. code-block:: python

        def nonzero_with_count_kernel(in_tensor):
            in_shape = in_tensor.shape
            assert len(in_tensor.shape) == 2, "expected 2D tensor"

            in_tile = nl.ndarray(in_shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=in_tile, src=in_tensor)

            out_tile = nl.ndarray((in_shape[0], in_shape[1] + 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.nonzero_with_count(dst=out_tile, src=in_tile, index_offset=0, padding_val=-1)

            out_tensor = nl.ndarray(out_tile.shape, dtype=out_tile.dtype, buffer=nl.hbm)
            nisa.dma_copy(dst=out_tensor, src=out_tile)

            return out_tensor"""
    ...
