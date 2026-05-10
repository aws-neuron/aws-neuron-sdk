"""Stubs for nki.language"""
from __future__ import annotations


from enum import Enum
from typing import *

class MemoryRegion(Enum):
    r"""Memory region constants for NKI tensors."""

    sbuf = 'sbuf'
    psum = 'psum'
    private_hbm = 'private_hbm'
    shared_hbm = 'shared_hbm'


class NKIObject:
    r"""Base class for NKI kernel dataclasses and configuration objects."""
    ...


class tile_size:
    r"""Hardware tile size constants (pmax, psum_fmax, gemm_stationary_fmax, etc.)"""
    bn_stats_fmax = ...
    """Maximum free dimension of BN_STATS"""
    gemm_moving_fmax = ...
    """Maximum free dimension of the moving operand of General Matrix Multiplication on Tensor Engine"""
    gemm_stationary_fmax = ...
    """Maximum free dimension of the stationary operand of General Matrix Multiplication on Tensor Engine"""
    pmax = ...
    """Maximum partition dimension of a tile"""
    psum_fmax = ...
    """Maximum free dimension of a tile on PSUM buffer"""
    psum_min_align = ...
    """Minimum byte alignment requirement for PSUM free dimension address"""
    sbuf_min_align = ...
    """Minimum byte alignment requirement for SBUF free dimension address"""
    total_available_sbuf_size = ...
    """Usable SBUF size per partition (total minus reserved bytes)."""


class NkiTensor(NKIObject):
    """NKI tensor with shape-based view operations.

    ``NkiTensor`` is the core tensor type in NKI. It represents a multi-dimensional
    array allocated on a specific memory buffer (SBUF, PSUM, or HBM) with a dtype
    and shape. All view operations (``slice``, ``permute``, ``reshape``, etc.) return
    new ``NkiTensor`` objects that share the same underlying storage — no data is
    copied. Views are consumed by NKI ISA instructions
    (e.g., ``nisa.tensor_copy``, ``nisa.dma_copy``).

    Tensors are created via :func:`nl.ndarray`:

    .. code-block:: python

        sb = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
        hbm = nl.ndarray((4, 128, 64), dtype=nl.float32, buffer=nl.shared_hbm)

    **Partition dimension.**

    On-chip tensors (SBUF, PSUM) have a partition dimension at dim 0 that maps to
    the hardware's parallel partitions. Most view operations cannot modify this
    dimension — see individual method docs for constraints.

    Attributes:
        shape (tuple[int, ...]): The size of each dimension of the view, e.g. ``(128, 64)``.
        strides (tuple[int, ...]): Element step between consecutive indices along each dimension in the underlying storage. A stride of ``0`` means the dimension is broadcast.
        offset (int): Element offset (not byte offset) from the start of the underlying storage at which this view begins. A freshly allocated tensor has ``offset == 0``; slicing produces views with non-zero offsets.
        dtype (str): Element type, e.g. ``nl.float32`` or ``nl.bfloat16``. See :ref:`nki-dtype` for the full list.
        buffer (MemoryRegion): Memory region the storage lives in: ``nl.sbuf``, ``nl.psum``, ``nl.hbm``, ``nl.private_hbm``, or ``nl.shared_hbm``.
        ndim (int): Number of dimensions — equivalent to ``len(shape)``.
        size (int): Total number of elements — equivalent to ``prod(shape)``.
        name (str): Optional debug label propagated into compiler diagnostics and :ref:`scheduling <how-to-scheduling-apis>`."""

    def __init__(self, shape: tuple[int, ...], dtype: str, storage: Any, buffer: Any=MemoryRegion.sbuf, name: str=''):
        """Create an ``NkiTensor`` bound to an existing storage handle.

        Most code should use :func:`nl.ndarray` instead — it allocates fresh
        storage and wraps it in a tensor. This constructor is useful when you
        need to re-bind a storage handle you already have (e.g. from a
        framework integration or test fixture).

        :param shape: tuple of positive integers.
        :param dtype: NKI dtype string (e.g. ``nl.float32``).
        :param storage: backend-specific storage handle. May be ``None`` when
            the tensor is used as a signature placeholder and never materialized.
        :param buffer: memory region (``nl.sbuf``, ``nl.psum``, ``nl.hbm``,
            ``nl.shared_hbm``). Defaults to ``nl.sbuf``.
        :param name: optional debug name, used in compiler diagnostics and
            :ref:`scheduling <how-to-scheduling-apis>`."""
        ...

    def is_contiguous(self) -> bool:
        """Return True if the view covers storage contiguously (row-major order).

        Computed from the current strides: each non-size-1 dimension's stride
        must equal the product of the shape sizes of inner dimensions."""
        ...

    def get_pattern(self) -> list[list[int]]:
        """Return the view's access pattern as ``[[stride, count], ...]``.

        Useful as a starting point when constructing a new ``.ap()`` that
        keeps most of the current layout intact. The returned pattern
        pairs each of the view's dimensions with its current stride, in
        the same order as :attr:`shape` / :attr:`strides`."""
        ...

    def is_indirect(self) -> bool:
        """Return True if this view already uses indirect addressing.

        Indirect addressing is produced by dynamic :meth:`select`,
        :meth:`vector_select`, or :meth:`ap` with ``scalar_offset`` /
        ``vector_offset``. Indirect views cannot be re-indirected, and the
        dimension that participates in the indirection cannot be further
        sliced or selected — use this query to guard against those chains."""
        ...

    def permute(self, dims: tuple[int, ...]) -> NkiTensor:
        """Reorder tensor dimensions.

        Returns a new view with dimensions rearranged according to ``dims``.
        No data is copied.

        .. code-block:: python

            t = nl.ndarray((128, 4, 8), dtype=nl.float32, buffer=nl.sbuf)
            t.permute((0, 2, 1))  # shape becomes (128, 8, 4)

        **Constraints.**

        - ``dims`` must be a permutation of ``range(t.ndim)``
        - On-chip tensors: ``dims[0]`` must be 0 (partition dim stays outermost)
        - After ``vector_select``: ``dims[0]`` must be 0 (indirect partition dim
          stays outermost; see :meth:`is_indirect`)

        :param dims: tuple of dimension indices in the desired order
        :return: new ``NkiTensor`` view with reordered dimensions"""
        ...

    def broadcast(self, dim: int, size: int) -> NkiTensor:
        """Expand a size-1 dimension to ``size`` by repeating elements.

        The dimension must have size 1 before broadcasting. No data is copied.

        .. code-block:: python

            t = nl.ndarray((128, 1, 64), dtype=nl.float32, buffer=nl.sbuf)
            t.broadcast(1, 8)  # shape becomes (128, 8, 64)

        **Constraints.**

        - ``t.shape[dim]`` must be 1
        - On-chip tensors: ``dim`` must not be 0 (partition dim)

        :param dim: dimension to broadcast
        :param size: new size for the dimension
        :return: new ``NkiTensor`` view with the broadcasted dimension"""
        ...

    def expand_dim(self, dim: int) -> NkiTensor:
        """Insert a new dimension of size 1 at position ``dim``.

        .. code-block:: python

            t = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
            t.expand_dim(1)  # shape becomes (128, 1, 64)

        **Constraints.**

        - On-chip tensors: ``dim`` must not be 0
        - After ``vector_select``: ``dim`` must not be 0 (cannot insert before
          the indirect partition dim; see :meth:`is_indirect`)

        :param dim: position at which to insert the new dimension
        :return: new ``NkiTensor`` view with an additional size-1 dimension"""
        ...

    def squeeze_dim(self, dim: int) -> NkiTensor:
        """Remove a dimension of size 1.

        .. code-block:: python

            t = nl.ndarray((128, 1, 64), dtype=nl.float32, buffer=nl.sbuf)
            t.squeeze_dim(1)  # shape becomes (128, 64)

        **Constraints.**

        - ``t.shape[dim]`` must be 1
        - On-chip tensors: ``dim`` must not be 0
        - After ``vector_select``: ``dim`` must not be 0

        :param dim: dimension to remove (must have size 1)
        :return: new ``NkiTensor`` view with the dimension removed"""
        ...

    def reshape_dim(self, dim: int, shape: tuple[int, ...]) -> NkiTensor:
        """Split a single dimension into multiple dimensions.

        The product of ``shape`` must equal ``t.shape[dim]``. One element of
        ``shape`` may be -1, in which case its value is inferred.

        .. code-block:: python

            t = nl.ndarray((128, 24), dtype=nl.float32, buffer=nl.sbuf)
            t.reshape_dim(1, (4, 6))   # shape becomes (128, 4, 6)
            t.reshape_dim(1, (4, -1))  # same result, 6 is inferred

        **Constraints.**

        - ``prod(shape) == t.shape[dim]``
        - On-chip tensors: ``dim`` must not be 0 (unless ``shape`` is trivial, e.g., ``(128,)``)
        - After ``vector_select``: ``dim`` must not be 0

        :param dim: dimension to split
        :param shape: tuple of sizes for the new dimensions (may contain one -1)
        :return: new ``NkiTensor`` view with the dimension split"""
        ...

    def flatten_dims(self, start_dim: int, end_dim: int) -> NkiTensor:
        """Merge a contiguous range of dimensions into one.

        Dimensions ``start_dim`` through ``end_dim`` (inclusive) are merged into
        a single dimension. The dimensions must already be contiguous in memory
        (no ``permute`` or non-contiguous slicing across them) so the merged view
        is itself a valid view of storage.

        .. code-block:: python

            t = nl.ndarray((128, 2, 3, 4), dtype=nl.float32, buffer=nl.sbuf)
            t.flatten_dims(1, 2)  # shape becomes (128, 6, 4)

        **Constraints.**

        - Dimensions ``start_dim..end_dim`` must be contiguous in memory
        - On-chip tensors: ``start_dim`` must be > 0
        - After ``vector_select``: ``start_dim`` must be > 0

        :param start_dim: first dimension to merge (inclusive)
        :param end_dim: last dimension to merge (inclusive)
        :return: new ``NkiTensor`` view with the merged dimension"""
        ...

    def reshape(self, shape: tuple[int, ...]) -> NkiTensor:
        """Reshape the tensor to a new shape without copying data.

        The total number of elements must remain the same. Fails if the current
        memory layout is incompatible with the requested shape (e.g. after a
        non-contiguous slice or permute).

        .. code-block:: python

            t = nl.ndarray((128, 4, 6), dtype=nl.float32, buffer=nl.sbuf)
            t.reshape((128, 24))       # merge last two dims
            t.reshape((128, 2, 12))    # split differently

        **Constraints.**

        - ``prod(shape) == prod(t.shape)``
        - On-chip tensors: ``shape[0]`` must equal ``t.shape[0]`` (partition dim preserved)
        - After ``vector_select``: ``shape[0]`` must equal ``t.shape[0]`` (indirect
          partition dim preserved)
        - Fails if the current layout is incompatible with the requested shape

        :param shape: tuple of new dimension sizes
        :return: new ``NkiTensor`` view with the requested shape"""
        ...

    def rearrange(self, src_pattern: tuple, dst_pattern: tuple, fixed_sizes: dict[str, int] | None=None) -> NkiTensor:
        """Rearrange tensor dimensions using einops-style patterns.

        Combines splitting, reordering, and merging dimensions into a single
        named operation. Patterns are tuples of strings (dimension names) or
        tuples of strings (grouped dimensions that are split or merged).

        .. code-block:: python

            t = nl.ndarray((128, 24), dtype=nl.float32, buffer=nl.sbuf)
            # Split dim 1 into (h, w), then reorder to (b, w, h):
            t.rearrange(('b', ('h', 'w')), ('b', 'w', 'h'), {'h': 4})
            # Result shape: (128, 6, 4)

        :param src_pattern: source dimension pattern (tuple of str or tuple-of-str)
        :param dst_pattern: destination dimension pattern (same dimension names)
        :param fixed_sizes: dict mapping dimension names to known sizes (for -1 inference)
        :return: new ``NkiTensor`` view with rearranged dimensions"""
        ...

    def slice(self, dim: int, start: int, end: int, step: int=1) -> NkiTensor:
        """Slice along a single dimension.

        Returns a view selecting elements from ``start`` to ``end`` (exclusive)
        with the given ``step``. Equivalent to ``t[:, start:end:step, :]`` when
        ``dim=1``.

        .. code-block:: python

            t = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
            t.slice(1, 8, 24, 1)   # shape becomes (128, 16)
            t.slice(1, 0, 64, 2)   # shape becomes (128, 32)

        **Constraints.**

        - ``0 <= start < end <= t.shape[dim]``
        - ``step >= 1``
        - On an indirect view (see :meth:`is_indirect`), cannot slice a
          dimension that participates in the indirection.

        :param dim: dimension to slice
        :param start: start index (inclusive)
        :param end: end index (exclusive)
        :param step: step size (default 1)
        :return: new ``NkiTensor`` view with the sliced dimension"""
        ...

    def select(self, dim: int, index: Union[int, NkiTensor]) -> NkiTensor:
        """Select a single element along a dimension, removing it.

        When ``index`` is an integer, performs static selection (equivalent to
        ``t[:, index, :]`` when ``dim=1``). When ``index`` is an ``NkiTensor``
        (e.g., a scalar loaded into SBUF), performs dynamic indirect selection
        where the index is resolved at runtime.

        .. code-block:: python

            t = nl.ndarray((128, 8, 64), dtype=nl.float32, buffer=nl.sbuf)
            t.select(1, 3)          # static: shape becomes (128, 64)

            # Dynamic select (HBM tensor, index resolved at runtime):
            idx = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            hbm_t = nl.ndarray((4, 128, 8), dtype=nl.float32, buffer=nl.shared_hbm)
            hbm_t.select(0, idx)    # shape becomes (128, 8)

        **Constraints.**

        - Static: ``0 <= index < t.shape[dim]``
        - Dynamic: only one dynamic select per tensor (no chaining);
          check :meth:`is_indirect` to guard
        - Dynamic on-chip: ``dim`` must not be 0 (partition dim)
        - On an indirect view (see :meth:`is_indirect`), static selection
          cannot target a dimension that participates in the indirection.

        :param dim: dimension to select from
        :param index: integer index (static) or ``NkiTensor`` scalar (dynamic)
        :return: new ``NkiTensor`` view with the dimension removed"""
        ...

    def vector_select(self, dim: int, vector_offset: NkiTensor) -> NkiTensor:
        """Per-partition indirect addressing using a vector of offsets.

        Each partition uses its own index from ``vector_offset`` as the base
        address for the selected dimension. The output partition count is
        ``vector_offset.shape[0]``.

        This is used for gather-style operations where different partitions
        read from different locations in the source tensor.

        .. code-block:: python

            hbm_t = nl.ndarray((64, 128, 8), dtype=nl.float32, buffer=nl.shared_hbm)
            offsets = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
            # Each of 128 partitions reads from a different row of hbm_t
            hbm_t.vector_select(0, offsets)  # shape becomes (128, 128, 8)

        **Constraints.**

        - ``dim`` must be 0
        - Only supported on HBM tensors
        - Only one dynamic select per tensor (cannot combine with a prior
          dynamic :meth:`select` or :meth:`vector_select`); check
          :meth:`is_indirect` to guard
        - The result is an indirect view: the selected dimension cannot be
          further sliced or selected.

        :param dim: dimension to apply indirect addressing (must be 0)
        :param vector_offset: SBUF tensor with per-partition indices, shape ``(num_partitions, 1)``
        :return: new ``NkiTensor`` view with dim 0 size set to ``vector_offset.shape[0]``"""
        ...

    def ap(self, pattern: list[list[int]], offset: int | None=None, scalar_offset: NkiTensor | None=None, vector_offset: NkiTensor | None=None, indirect_dim: int=0, dtype: DType | None=None) -> NkiTensor:
        """Low-level access pattern override (escape hatch).

        Replaces shape and strides with an explicit
        ``[[stride, count], ...]`` pattern addressing the underlying
        storage directly. Analogous to ``torch.as_strided``.

        .. code-block:: python

            sb = nl.ndarray((128, 32), dtype=nl.float32, buffer=nl.sbuf)
            # Explicit 2D pattern: partition stride=32, free stride=1
            sb.ap(pattern=[[32, 128], [1, 32]])

            # With indirect access and dtype reinterpret:
            sb.ap(pattern=[[64, 128], [1, 16]], dtype=nl.bfloat16,
                  scalar_offset=idx, indirect_dim=1)

        :param pattern: list of ``[stride, count]`` pairs defining the access pattern
        :param offset: element offset into storage. When ``None`` (default), inherits
            the current view's storage offset — matching ``torch.as_strided``'s
            ``storage_offset=None`` behaviour. Pass an explicit integer to override.
        :param scalar_offset: dynamic scalar index tensor for indirect access
        :param vector_offset: per-partition index tensor for indirect access
        :param indirect_dim: dimension in ``self.shape`` whose stride scales
            the indirect scalar/vector offset (default 0)
        :param dtype: reinterpret storage as this dtype (default: tensor's dtype)
        :return: new ``NkiTensor`` with the explicit access pattern"""
        ...

    def view(self, dtype: DType) -> NkiTensor:
        """Reinterpret the tensor's data as a different dtype.

        No data is copied. Equivalent to ``numpy.ndarray.view(dtype)`` or
        C++ ``reinterpret_cast``. The last dimension's size is scaled by the
        ratio of dtype sizes: reinterpreting a float32 tensor as uint8
        multiplies the last-dim count by 4; reinterpreting uint8 as float32
        divides it by 4.

        .. code-block:: python

            t = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
            t.view(nl.uint8)     # shape becomes (128, 256), 4x expansion
            t.view(nl.int32)     # shape stays (128, 64), same-size reinterpret

            u = nl.ndarray((128, 256), dtype=nl.uint8, buffer=nl.sbuf)
            u.view(nl.float32)   # shape becomes (128, 64), 4x contraction

        **Constraints.**

        - The last dimension must be contiguous in memory
        - For contraction (larger dtype): last-dim size must be divisible by the ratio
        - Not supported after dynamic / vector select

        :param dtype: target NKI dtype to reinterpret as
        :return: new ``NkiTensor`` view with the adjusted dtype and shape"""
        ...

    reinterpret_cast = ...

    def indirect(self, index: NkiTensor, num_elem: int | None=None) -> NkiTensor:
        """Create an indirect tensor view for Tensor Indirection (TI).

        Available on NeuronCore-v4 (Trn3) and later.

        TI allows reading or writing a column of data at given free-dimension
        offsets across contiguous partition dimensions. Can be used as input
        (gather) or output (scatter) in nisa operations.

        Offsets are stored in a snake pattern across partition groups: offset i
        comes from ``index[i % G, i // G]`` where G is the group size (16 for
        vector/scalar/gpsimd engines, 32 for tensor engine).

        :param index: SBUF tensor containing free-dimension offsets, shape ``(P, K)``
            where ``P == self.shape[0]``.
        :param num_elem: number of offsets to use. Defaults to ``index.size``.
        :return: new ``NkiTensor`` view with TI attached. Output shape is
            ``(P, num_elem)``."""
        ...

    ...

def abs(x, dtype=None):
    r"""Absolute value of the input, element-wise.

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has absolute values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.abs
        a = nl.full((128, 512), -1.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.abs(a)
        expected = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(b, expected)

        # nki.language.abs with explicit dtype
        a = nl.full((128, 512), -1.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.abs(a, dtype=nl.float16)
        expected = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


def add(x, y, dtype=None):
    r"""Add the inputs, element-wise.

    ((Similar to `numpy.add <https://numpy.org/doc/stable/reference/generated/numpy.add.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile that has ``x + y``, element-wise.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.add -- element-wise addition of two tiles
        a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.full((128, 512), 2.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.add(a, b)

        expected = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)

        # nki.language.add -- adding a scalar to every element of a tile
        a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.add(a, 2.0)
        expected = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...


def affine_range(start, stop=None, step=1):
    r"""Create a sequence for fully unrolled loop iteration.

    Create a sequence of numbers for use as loop iterators in NKI, resulting in
    a fully unrolled loop. Prefer :doc:`static_range <nki.language.static_range>` instead.

    .. warning::
    
        This API is deprecated and will be removed in future releases.

    :param start: start value (or stop if ``stop`` is None).
    :param stop: stop value (exclusive).
    :param step: step size.
    :return: an iterator yielding integer values from start to stop.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.affine_range
        for i in nl.affine_range(input_tensor.shape[1] // 512):
            offset = i * 512
            tile = nl.load(input_tensor[0:128, offset:offset+512])
            result = nl.multiply(tile, tile)
            nl.store(out_tensor[0:128, offset:offset+512], result)"""
    ...


def all(x, axis, dtype=None):
    r"""Whether all elements along the specified axis (or axes) evaluate to True.

    ((Similar to `numpy.all <https://numpy.org/doc/stable/reference/generated/numpy.all.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate;
        must be free dimensions, not partition dimension (0); can only be the
        last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile with the logical AND reduction along the provided axis."""
    ...


def arctan(x, dtype=None):
    r"""Inverse tangent of the input, element-wise.

    ((Similar to `numpy.arctan <https://numpy.org/doc/stable/reference/generated/numpy.arctan.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has inverse tangent values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.arctan -- arctan(0.0) = 0.0
        a = nl.full((128, 512), 0.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.arctan(a)
        expected = nl.full((128, 512), 0.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


bfloat16 = 'bfloat16'
"""16-bit floating-point number (1S,8E,7M)"""


def bitwise_and(x, y, dtype=None):
    r"""Compute the bitwise AND of two tiles element-wise.

    ((Similar to `numpy.bitwise_and <https://numpy.org/doc/stable/reference/generated/numpy.bitwise_and.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs must be integer typed.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile with the bitwise AND result."""
    ...


def bitwise_or(x, y, dtype=None):
    r"""Compute the bitwise OR of two tiles element-wise.

    ((Similar to `numpy.bitwise_or <https://numpy.org/doc/stable/reference/generated/numpy.bitwise_or.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs must be integer typed.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile with the bitwise OR result."""
    ...


def bitwise_xor(x, y, dtype=None):
    r"""Compute the bitwise XOR of two tiles element-wise.

    ((Similar to `numpy.bitwise_xor <https://numpy.org/doc/stable/reference/generated/numpy.bitwise_xor.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs must be integer typed.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile with the bitwise XOR result."""
    ...


bool_ = 'bool'
"""Boolean (True or False) stored as a byte"""


def broadcast_to(x, shape, dtype=None):
    r"""Broadcast a tile to a new shape following numpy broadcasting rules.

    ((Similar to `numpy.broadcast_to <https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    If ``x.shape`` is already the same as ``shape``, returns ``x`` unchanged
    (or a dtype-cast copy if ``dtype`` differs).

    :param x: the source tile in SBUF or PSUM.
    :param shape: the target shape. Must have the same rank as ``x``.
        Each dimension must either match or be broadcast from size 1.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile with the target shape containing broadcast values from ``x``."""
    ...


def ceil(x, dtype=None):
    r"""Ceiling of the input, element-wise.

    ((Similar to `numpy.ceil <https://numpy.org/doc/stable/reference/generated/numpy.ceil.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    The ceil of the scalar x is the smallest integer i, such that i >= x.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has ceiling values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.ceil -- rounds 3.2 up to 4.0
        a = nl.full((128, 512), 3.2, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.ceil(a)
        expected = nl.full((128, 512), 4.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)

        # nki.language.ceil -- rounds -3.7 up to -3.0
        a = nl.full((128, 512), -3.7, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.ceil(a)
        expected = nl.full((128, 512), -3.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...


def copy(x, dtype=None):
    r"""Create a copy of the input tile.

    .. warning::

       This API is experimental and may change in future releases.

    Uses the Scalar Engine via ``activation(op=copy)``. Note that the Scalar Engine
    internally casts through FP32, which may be lossy for integer types with
    values exceeding FP32 precision (e.g. int32 values > 2^23).

    :param x: the source of copy, must be a tile in SBUF or PSUM.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a new tile with the same layout as ``x``, allocated on the same buffer
        as ``x`` (SBUF or PSUM)."""
    ...


def cos(x, dtype=None):
    r"""Cosine of the input, element-wise.

    ((Similar to `numpy.cos <https://numpy.org/doc/stable/reference/generated/numpy.cos.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has cosine values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.cos -- cos(0.0) = 1.0
        a = nl.full((128, 512), 0.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.cos(a)
        expected = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


def device_print(print_prefix, tensor):
    r"""Print a message with a string prefix followed by the value of a tile.

    During kernel execution on hardware, the Neuron Runtime (NRT) exports device-printed tensors
    via the NRT debug stream API. By default, setting the environment variable
    ``NEURON_RT_DEBUG_OUTPUT_DIR`` to a directory path enables the default stream consumer,
    which dumps tensor data to that directory. The output is organized as:
    ``<output_dir>/<print_prefix>/core_<logical_core_id>/<iteration>/``.

    In CPU simulation, this prints immediately to stdout.

    :param print_prefix: prefix of the print message. Evaluated at trace time; must be a constant string.
    :param tensor: tensor to print out. Can be in SBUF or HBM."""
    ...


def divide(x, y, dtype=None):
    r"""Divide the inputs, element-wise.

    ((Similar to `numpy.divide <https://numpy.org/doc/stable/reference/generated/numpy.divide.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile that has ``x / y``, element-wise.
    """
    ...


def dropout(x, rate, dtype=None):
    r"""Randomly zeroes some of the elements of the input tile given a probability rate.

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param rate: the probability of zeroing each element. Can be a scalar constant
        or a tile of shape ``(x.shape[0], 1)`` for per-partition drop probabilities.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile with randomly zeroed elements of ``x``."""
    ...


def ds(start, size):
    r"""Create a dynamic slice for tensor indexing.

    :param start: the start index of the slice.
    :param size: the size of the slice.
    :return: a dynamic slice object for use in tensor indexing."""
    ...


def dynamic_range(start, stop=None, step=1):
    r"""Create a sequence for **dynamic**  loop iteration.

    Create a sequence of numbers for use as **dynamic** loop iterators in NKI.
    The loop runs on device with dynamic bounds.

    :param start: start value (or stop if ``stop`` is None), can be VirtualRegister.
    :param stop: stop value (exclusive), can be VirtualRegister.
    :param step: step size, must be a compile-time positive integer (not VirtualRegister).
    :return: an iterator yielding integer values from start to stop.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.dynamic_range
        for _ in nl.dynamic_range(1):
            tile = nl.load(input_tensor[0:128, 0:512])
            result = nl.multiply(tile, tile)
            nl.store(out_tensor[0:128, 0:512], result)"""
    ...


def empty_like(x, dtype=None, buffer=None, name=''):
    r"""Create a new tensor with the same shape and type as a given tensor.

    ((Similar to `numpy.empty_like <https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: the tensor.
    :param dtype: the data type of the tensor (default: same as ``x``).
    :param buffer: the specific buffer (ie, sbuf, psum, hbm), (default: same as ``x``).
    :param name: the name of the tensor, used in :ref:`scheduling <how-to-scheduling-apis>`.
    :return: a new :class:`NkiTensor` with the same shape and type as ``x``."""
    ...


def equal(x, y, dtype=None):
    r"""Return (x == y) element-wise.

    ((Similar to `numpy.equal <https://numpy.org/doc/stable/reference/generated/numpy.equal.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype.
        Use ``dtype=nl.uint8`` for a boolean-like result.
    :return: a tile with 1 where equal, 0 otherwise."""
    ...


def erf(x, dtype=None):
    r"""Error function, element-wise."""
    ...


def erf_dx(x, dtype=None):
    r"""Derivative of error function, element-wise."""
    ...


def exp(x, dtype=None):
    r"""Exponential of the input, element-wise.

    ((Similar to `numpy.exp <https://numpy.org/doc/stable/reference/generated/numpy.exp.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    The ``exp(x)`` is ``e^x`` where ``e`` is the Euler's number = 2.718281...

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has exponential values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.exp -- exp(0.0) = 1.0
        a = nl.full((128, 512), 0.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.exp(a)
        expected = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


def expand_dims(x, axis):
    r"""Expand the shape of a tile.

    ((Similar to `numpy.expand_dims <https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Insert a new axis that will appear at the axis position in the expanded tile shape.

    :param x: a tile.
    :param axis: position in the expanded axes where the new axis is placed.
    :return: a tile with view of input data with the number of dimensions increased."""
    ...


float16 = 'float16'
"""16-bit floating-point number"""


float32 = 'float32'
"""32-bit floating-point number"""


float4_e2m1fn_x4 = 'float4_e2m1fn_x4'
"""4x packed float4_e2m1fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""


float8_e4m3 = 'float8_e4m3'
"""8-bit floating-point number (1S,4E,3M)"""


float8_e4m3fn = 'float8_e4m3fn'
"""8-bit floating-point number (1S,4E,3M), Extended range: no inf, NaN represented by 0bS111'1111"""


float8_e4m3fn_x4 = 'float8_e4m3fn_x4'
"""4x packed float8_e4m3fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""


float8_e5m2 = 'float8_e5m2'
"""8-bit floating-point number (1S,5E,2M)"""


float8_e5m2_x4 = 'float8_e5m2_x4'
"""4x packed float8_e5m2 elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""


def floor(x, dtype=None):
    r"""Floor of the input, element-wise.

    ((Similar to `numpy.floor <https://numpy.org/doc/stable/reference/generated/numpy.floor.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    The floor of the scalar x is the largest integer i, such that i <= x.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has floor values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.floor -- rounds 3.7 down to 3.0
        a = nl.full((128, 512), 3.7, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.floor(a)
        expected = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)

        # nki.language.floor -- rounds -3.2 down to -4.0
        a = nl.full((128, 512), -3.2, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.floor(a)
        expected = nl.full((128, 512), -4.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...


def fmod(x, y, dtype=None):
    r"""Floating-point remainder of ``x / y``, element-wise.

    The remainder has the same sign as the dividend x.
    It is equivalent to the Matlab(TM) rem function and should not be confused with the Python modulus operator x % y.

    ((Similar to `numpy.fmod <https://numpy.org/doc/stable/reference/generated/numpy.fmod.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile. If x is a scalar value it will be broadcast to the shape of y.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile that has values ``x fmod y``.
    """
    ...


def full(shape, fill_value, dtype, buffer=MemoryRegion.sbuf, name=''):
    r"""Create a new tensor of given shape and dtype on the specified buffer, filled with initial value.

    .. warning::

       This API is experimental and may change in future releases.

    :param shape: the shape of the tensor.
    :param fill_value: the value to fill the tensor with.
    :param dtype: the data type of the tensor.
    :param buffer: the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.
    :param name: the name of the tensor, used in :ref:`scheduling <how-to-scheduling-apis>`.
    :return: a new :class:`NkiTensor` allocated on the buffer."""
    ...


def gather_flattened(data, indices, axis=0, dtype=None):
    r"""Gather elements from data tensor using indices after flattening.

    This instruction gathers elements from the data tensor using integer indices
    provided in the indices tensor. For each element in the indices tensor, it
    retrieves the corresponding value from the data tensor using the index value
    to select from the free dimension of data.

    .. warning::

       This API is experimental and may change in future releases.

    :param data: input tensor to gather from.
    :param indices: indices to gather.
    :param axis: axis along which to gather.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: gathered tensor.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.gather_flattened -- gather elements by index
        data = nl.load(data_tensor[0:128, 0:512])
        indices = nl.load(indices_tensor[0:128, 0:512])
        result = nl.gather_flattened(data, indices)
        nl.store(actual_tensor[0:128, 0:512], result)"""
    ...


def gelu(x, dtype=None):
    r"""GELU activation, element-wise."""
    ...


def gelu_apprx_sigmoid(x, dtype=None):
    r"""GELU approximation using sigmoid, element-wise."""
    ...


def gelu_apprx_sigmoid_dx(x, dtype=None):
    r"""Derivative of sigmoid-approximated GELU, element-wise."""
    ...


def gelu_apprx_tanh(x, dtype=None):
    r"""GELU approximation using tanh, element-wise."""
    ...


def gelu_dx(x, dtype=None):
    r"""Derivative of GELU activation, element-wise."""
    ...


def greater(x, y, dtype=None):
    r"""Return (x > y) element-wise.

    ((Similar to `numpy.greater <https://numpy.org/doc/stable/reference/generated/numpy.greater.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype.
        Use ``dtype=nl.uint8`` for a boolean-like result.
    :return: a tile with 1 where x > y, 0 otherwise."""
    ...


def greater_equal(x, y, dtype=None):
    r"""Return (x >= y) element-wise.

    ((Similar to `numpy.greater_equal <https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype.
        Use ``dtype=nl.uint8`` for a boolean-like result.
    :return: a tile with 1 where x >= y, 0 otherwise."""
    ...


hbm = MemoryRegion.private_hbm


int16 = 'int16'
"""16-bit signed integer number"""


int32 = 'int32'
"""32-bit signed integer number"""


int8 = 'int8'
"""8-bit signed integer number"""


def invert(x, dtype=None):
    r"""Compute the bitwise NOT element-wise.

    ((Similar to `numpy.invert <https://numpy.org/doc/stable/reference/generated/numpy.invert.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Input must be integer typed. Implemented as XOR with all-ones.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile with the bitwise NOT result."""
    ...


def is_hbm(buffer):
    r"""Check if buffer is any HBM type."""
    ...


def is_on_chip(buffer):
    r"""Check if buffer is on-chip (SBUF or PSUM)."""
    ...


def is_psum(buffer):
    r"""Check if buffer is PSUM."""
    ...


def is_sbuf(buffer):
    r"""Check if buffer is SBUF."""
    ...


def left_shift(x, y, dtype=None):
    r"""Left shift the bits of x by y positions element-wise.

    ((Similar to `numpy.left_shift <https://numpy.org/doc/stable/reference/generated/numpy.left_shift.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs must be integer typed.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile with the left-shifted result."""
    ...


def less(x, y, dtype=None):
    r"""Return (x < y) element-wise.

    ((Similar to `numpy.less <https://numpy.org/doc/stable/reference/generated/numpy.less.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype.
        Use ``dtype=nl.uint8`` for a boolean-like result.
    :return: a tile with 1 where x < y, 0 otherwise."""
    ...


def less_equal(x, y, dtype=None):
    r"""Return (x <= y) element-wise.

    ((Similar to `numpy.less_equal <https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype.
        Use ``dtype=nl.uint8`` for a boolean-like result.
    :return: a tile with 1 where x <= y, 0 otherwise."""
    ...


def load(src, dtype=None):
    r"""Load a tensor from device memory (HBM) into on-chip memory (SBUF).

    .. warning::

       This API is experimental and may change in future releases.

    :param src: HBM tensor to load the data from.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a new tile on SBUF with values from ``src``."""
    ...


def load_transpose2d(src, dtype=None):
    r"""Load a tensor from device memory (HBM) and 2D-transpose the data before storing into on-chip memory (SBUF).

    .. warning::

       This API is experimental and may change in future releases.

    :param src: HBM tensor to load the data from.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a new tile on SBUF with values from ``src`` 2D-transposed."""
    ...


def log(x, dtype=None):
    r"""Natural logarithm of the input, element-wise.

    ((Similar to `numpy.log <https://numpy.org/doc/stable/reference/generated/numpy.log.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    It is the inverse of the exponential function, such that: ``log(exp(x)) = x`` .
    The natural logarithm base is ``e``.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has natural logarithm values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.log -- log(1.0) = 0.0
        a = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.log(a)
        expected = nl.full((128, 512), 0.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


def logical_and(x, y, dtype=None):
    r"""Compute the logical AND of two tiles element-wise.

    ((Similar to `numpy.logical_and <https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs should be boolean-like (0 or 1 values).

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile with the logical AND result."""
    ...


def logical_not(x, dtype=None):
    r"""Compute the logical NOT element-wise.

    ((Similar to `numpy.logical_not <https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Implemented as XOR with 1, so inputs should be boolean-like (0 or 1 values).
    For non-boolean inputs, use ``nl.equal(x, 0)`` instead.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile with the logical NOT result."""
    ...


def logical_or(x, y, dtype=None):
    r"""Compute the logical OR of two tiles element-wise.

    ((Similar to `numpy.logical_or <https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs should be boolean-like (0 or 1 values).

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile with the logical OR result."""
    ...


def logical_xor(x, y, dtype=None):
    r"""Compute the logical XOR of two tiles element-wise.

    ((Similar to `numpy.logical_xor <https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs should be boolean-like (0 or 1 values).

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile with the logical XOR result."""
    ...


def matmul(x, y, transpose_x=False):
    r"""x @ y matrix multiplication of x and y.

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile on SBUF (partition dimension <= 128, free dimension <= 128),
        x's free dimension must match y's partition dimension.
    :param y: a tile on SBUF (partition dimension <= 128, free dimension <= 512).
    :param transpose_x: defaults to False. If True, x is treated as already transposed.
        If False, an additional transpose will be inserted to make x's partition
        dimension the contract dimension of the matmul to align with the Tensor Engine.
    :return: x @ y or x.T @ y if transpose_x=True.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.matmul -- identity.T @ ones = ones
        x = nl.shared_identity_matrix(n=128, dtype=nl.float32)
        y = nl.full((128, 128), 1.0, dtype=nl.float32, buffer=nl.sbuf)
        result_psum = nl.matmul(x, y, transpose_x=True)
        result = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(result, result_psum)
        expected = nl.full((128, 128), 1.0, dtype=nl.float32,
                           buffer=nl.sbuf)
        assert nl.equal(result, expected)"""
    ...


def max(x, axis, dtype=None, keepdims=False):
    r"""Maximum of elements along the specified axis (or axes) of the input.

    ((Similar to `numpy.max <https://numpy.org/doc/stable/reference/generated/numpy.max.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate;
        must be free dimensions, not partition dimension (0); can only be the
        last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: if True, the reduced axes are kept as size-one dimensions.
    :return: a tile with the maximum along the provided axis."""
    ...


def maximum(x, y, dtype=None):
    r"""Maximum of the inputs, element-wise.

    ((Similar to `numpy.maximum <https://numpy.org/doc/stable/reference/generated/numpy.maximum.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile that has the maximum of each element from x and y.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.maximum -- max(3.0, 5.0) = 5.0
        a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.maximum(a, b)
        expected = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)

        # nki.language.maximum -- with a scalar operand
        a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.maximum(a, 5.0)
        expected = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...


def mean(x, axis, dtype=None, keepdims=False):
    r"""Arithmetic mean along the specified axis (or axes) of the input.

    ((Similar to `numpy.mean <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate;
        must be free dimensions, not partition dimension (0); can only be the
        last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: if True, the reduced axes are kept as size-one dimensions.
    :return: a tile with the average of elements along the provided axis. Float32
        intermediate values are used for the computation."""
    ...


def min(x, axis, dtype=None, keepdims=False):
    r"""Minimum of elements along the specified axis (or axes) of the input.

    ((Similar to `numpy.min <https://numpy.org/doc/stable/reference/generated/numpy.min.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate;
        must be free dimensions, not partition dimension (0); can only be the
        last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: if True, the reduced axes are kept as size-one dimensions.
    :return: a tile with the minimum along the provided axis."""
    ...


def minimum(x, y, dtype=None):
    r"""Minimum of the inputs, element-wise.

    ((Similar to `numpy.minimum <https://numpy.org/doc/stable/reference/generated/numpy.minimum.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile that has the minimum of each element from x and y.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.minimum -- min(3.0, 5.0) = 3.0
        a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.minimum(a, b)
        expected = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)

        # nki.language.minimum -- with a scalar operand
        a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.minimum(a, 5.0)
        expected = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...


def mish(x, dtype=None):
    r"""Mish activation, element-wise."""
    ...


def mod(x, y, dtype=None):
    r"""Remainder of ``x / y``, element-wise.

    Computes the remainder complementary to the floor_divide function.
    It is equivalent to the Python modulus x % y and has the same sign as the divisor y.

    ((Similar to `numpy.mod <https://numpy.org/doc/stable/reference/generated/numpy.mod.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile. If x is a scalar value it will be broadcast to the shape of y.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile that has values ``x mod y``.
    """
    ...


def multiply(x, y, dtype=None):
    r"""Multiply the inputs, element-wise.

    ((Similar to `numpy.multiply <https://numpy.org/doc/stable/reference/generated/numpy.multiply.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile that has ``x * y``, element-wise.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.multiply -- element-wise multiplication of two tiles
        a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.full((128, 512), 4.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.multiply(a, b)
        expected = nl.full((128, 512), 12.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)

        # nki.language.multiply -- scaling every element by a scalar
        a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.multiply(a, 4.0)
        expected = nl.full((128, 512), 12.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...


def ndarray(shape, dtype, buffer=MemoryRegion.sbuf, name='', address=None):
    r"""Create a new tensor of given shape and dtype on the specified buffer.

    :param shape: the shape of the tensor.
    :param dtype: the data type of the tensor.
    :param buffer: the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.
    :param name: the name of the tensor, used in :ref:`scheduling <how-to-scheduling-apis>`.
    :param address: optional memory address ``(partition_offset, free_offset)``.
    :return: a new :class:`NkiTensor` allocated on the buffer."""
    ...


def negative(x, dtype=None):
    r"""Numerical negative of the input, element-wise.

    ((Similar to `numpy.negative <https://numpy.org/doc/stable/reference/generated/numpy.negative.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has numerical negative values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.negative -- negates 5.0 to -5.0
        a = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.negative(a)
        expected = nl.full((128, 512), -5.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)

        # nki.language.negative -- negates -3.0 to 3.0
        a = nl.full((128, 512), -3.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.negative(a)
        expected = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...


def no_reorder():
    r"""Prevent the scheduler from reordering operations in this region.

    Use as a context manager (``with nl.no_reorder():``) to guarantee that
    operations inside the block execute in program order. Without this
    directive, the compiler scheduler is free to reorder independent
    operations for better hardware utilization.

    Dynamic loops (``nl.dynamic_range``) are not supported inside a
    ``no_reorder`` block. Static loops (``nl.affine_range``,
    ``nl.sequential_range``, ``nl.static_range``) are allowed because
    they are fully unrolled at compile time.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.no_reorder -- guarantee execution order
        with nl.no_reorder():
            a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
            b = nl.full((128, 512), 2.0, dtype=nl.float32, buffer=nl.sbuf)
            c = nl.add(a, b)
        expected = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...


def not_equal(x, y, dtype=None):
    r"""Return (x != y) element-wise.

    ((Similar to `numpy.not_equal <https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information); Defaults to the input tile dtype.
        Use ``dtype=nl.uint8`` for a boolean-like result.
    :return: a tile with 1 where not equal, 0 otherwise."""
    ...


def num_programs(axes=0):
    r"""Number of SPMD programs along the given axes in the launch grid.

    :param axes: the axes of the launch grid. If not provided, returns the total
        number of programs along the entire launch grid.
    :return: the number of SPMD programs along ``axes`` in the launch grid."""
    ...


def ones(shape, dtype, buffer=MemoryRegion.sbuf, name=''):
    r"""Create a new tensor of given shape and dtype on the specified buffer, filled with ones.

    ((Similar to `numpy.ones <https://numpy.org/doc/stable/reference/generated/numpy.ones.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param shape: the shape of the tensor.
    :param dtype: the data type of the tensor.
    :param buffer: the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.
    :param name: the name of the tensor, used in :ref:`scheduling <how-to-scheduling-apis>`.
    :return: a new :class:`NkiTensor` allocated on the buffer."""
    ...


def power(x, y, dtype=None):
    r"""Elements of x raised to powers of y, element-wise.

    ((Similar to `numpy.power <https://numpy.org/doc/stable/reference/generated/numpy.power.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile that has values ``x`` to the power of ``y``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.power -- element-wise exponentiation of two tiles
        a = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.full((128, 512), 2.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.power(a, b)
        expected = nl.full((128, 512), 9.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...


private_hbm = MemoryRegion.private_hbm


def prod(x, axis, dtype=None, keepdims=False):
    r"""Product of elements along the specified axis (or axes) of the input.

    ((Similar to `numpy.prod <https://numpy.org/doc/stable/reference/generated/numpy.prod.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate;
        must be free dimensions, not partition dimension (0); can only be the
        last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: if True, the reduced axes are kept as size-one dimensions.
    :return: a tile with the product along the provided axis."""
    ...


def program_id(axis=0):
    r"""Index of the current SPMD program along the given axis in the launch grid.

    :param axis: the axis of the launch grid.
    :return: the program id along ``axis``."""
    ...


def program_ndim():
    r"""Number of dimensions in the SPMD launch grid.

    :return: the number of dimensions in the launch grid, i.e. the number of axes. 0 if no grid."""
    ...


psum = MemoryRegion.psum


def rand(shape, dtype, buffer=MemoryRegion.sbuf, name=''):
    r"""Create a new tensor of given shape and dtype on the specified buffer, filled with random values.

    Values are sampled from a uniform distribution between 0 and 1.

    .. warning::

       This API is experimental and may change in future releases.

    :param shape: the shape of the tensor.
    :param dtype: the data type of the tensor (see :ref:`nki-dtype` for more information).
    :param buffer: the specific buffer (sbuf, psum, hbm), defaults to sbuf.
    :param name: the name of the tensor, used in :ref:`scheduling <how-to-scheduling-apis>`.
    :return: a new :class:`NkiTensor` allocated on the buffer with random values.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.rand -- generate random values in [0, 1)
        a = nl.rand((128, 512), dtype=nl.float32)"""
    ...


def random_seed(seed):
    r"""Set the random seed for random number generation.

    Using the same seed will generate the same sequence of random numbers
    when used with ``rand()``.

    .. warning::

       This API is experimental and may change in future releases.

    :param seed: a [1,1] tensor on SBUF or PSUM with a 32-bit seed value.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.random_seed -- set seed for reproducible random values
        seed = nl.full((1, 1), 42, dtype=nl.int32, buffer=nl.sbuf)
        nl.random_seed(seed)
        a = nl.rand((128, 512), dtype=nl.float32)

        # nki.language.random_seed -- same seed produces same values
        seed = nl.full((1, 1), 42, dtype=nl.int32, buffer=nl.sbuf)
        nl.random_seed(seed)
        a = nl.rand((128, 512), dtype=nl.float32)
        nl.random_seed(seed)
        b = nl.rand((128, 512), dtype=nl.float32)
        assert nl.equal(a, b)"""
    ...


def reciprocal(x, dtype=None):
    r"""Reciprocal of the input, element-wise.

    ((Similar to `numpy.reciprocal <https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    ``reciprocal(x) = 1 / x``

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has reciprocal values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.reciprocal -- reciprocal(4.0) = 0.25
        a = nl.full((128, 512), 4.0, dtype=nl.float32,
                    buffer=nl.sbuf)
        b = nl.reciprocal(a)
        expected = nl.full((128, 512), 0.25, dtype=nl.float32,
                           buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


def relu(x, dtype=None):
    r"""ReLU activation, element-wise."""
    ...


def right_shift(x, y, dtype=None):
    r"""Right shift the bits of x by y positions element-wise.

    ((Similar to `numpy.right_shift <https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs must be integer typed.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile with the right-shifted result."""
    ...


def rms_norm(x, w, axis, n, epsilon=1e-06, dtype=None, compute_dtype=None):
    r"""Apply Root Mean Square Layer Normalization.

    .. warning::

       This API is experimental and may change in future releases.

    :param x: input tile.
    :param w: weight tile.
    :param axis: axis along which to compute the root mean square (rms) value.
    :param n: total number of values to calculate rms.
    :param epsilon: epsilon value used by rms calculation to avoid divide-by-zero.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param compute_dtype: (optional) dtype for the internal computation.
    :return: ``x / RMS(x) * w``

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.rms_norm -- normalize with unit weights
        x = nl.full((128, 512), 2.0, dtype=nl.float32, buffer=nl.sbuf)
        w = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
        result = nl.rms_norm(x, w, axis=1, n=512)"""
    ...


def rsqrt(x, dtype=None):
    r"""Reciprocal of the square-root of the input, element-wise.

    ((Similar to `torch.rsqrt <https://pytorch.org/docs/master/generated/torch.rsqrt.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    ``rsqrt(x) = 1 / sqrt(x)``

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has reciprocal square-root values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.rsqrt -- rsqrt(4.0) = 0.5
        a = nl.full((128, 512), 4.0, dtype=nl.float32,
                    buffer=nl.sbuf)
        b = nl.rsqrt(a)
        expected = nl.full((128, 512), 0.5, dtype=nl.float32,
                           buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


sbuf = MemoryRegion.sbuf


def sequential_range(start, stop=None, step=1):
    r"""Create a sequence for fully unrolled loop iteration.

    Create a sequence of numbers for use as loop iterators in NKI, resulting in
    a fully unrolled loop. Prefer :doc:`static_range <nki.language.static_range>` instead.

    .. warning::
    
        This API is deprecated and will be removed in future releases.

    :param start: start value (or stop if ``stop`` is None).
    :param stop: stop value (exclusive).
    :param step: step size.
    :return: an iterator yielding integer values from start to stop.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.sequential_range
        for i in nl.sequential_range(input_tensor.shape[1] // 512):
            offset = i * 512
            tile = nl.load(input_tensor[0:128, offset:offset+512])
            result = nl.multiply(tile, tile)
            nl.store(out_tensor[0:128, offset:offset+512], result)"""
    ...


shared_hbm = MemoryRegion.shared_hbm


def shared_identity_matrix(n, dtype='uint8', dst=None):
    r"""Create an identity matrix in SBUF with the specified data type.

    The compiler will reuse all identity matrices of the same
    dtype in the graph to save space.

    :param n: the number of rows (and columns) of the returned identity matrix
    :param dtype: the data type of the tensor, default to be ``nl.uint8`` (see :ref:`nki-dtype` for more information).
    :return: a new :class:`NkiTensor` which contains the identity tensor

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.shared_identity_matrix -- 128x128 identity matrix
        identity = nl.shared_identity_matrix(n=128, dtype=nl.float32)
        expected = nl.load(expected_tensor[0:128, 0:128])
        assert nl.equal(identity, expected)
        nl.store(actual_tensor[0:128, 0:128], identity)"""
    ...


def sigmoid(x, dtype=None):
    r"""Sigmoid activation, element-wise."""
    ...


def sign(x, dtype=None):
    r"""Sign of the numbers of the input, element-wise.

    ((Similar to `numpy.sign <https://numpy.org/doc/stable/reference/generated/numpy.sign.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    The sign function returns ``-1`` if ``x < 0``, ``0`` if ``x==0``, ``1`` if ``x > 0``.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has sign values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.sign -- sign(-5.0) = -1.0
        a = nl.full((128, 512), -5.0, dtype=nl.float32,
                    buffer=nl.sbuf)
        b = nl.sign(a)
        expected = nl.full((128, 512), -1.0, dtype=nl.float32,
                           buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


def silu(x, dtype=None):
    r"""SiLU (Swish) activation, element-wise."""
    ...


def silu_dx(x, dtype=None):
    r"""Derivative of SiLU activation, element-wise."""
    ...


def sin(x, dtype=None):
    r"""Sine of the input, element-wise.

    ((Similar to `numpy.sin <https://numpy.org/doc/stable/reference/generated/numpy.sin.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has sine values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.sin -- sin(0.0) = 0.0
        a = nl.full((128, 512), 0.0, dtype=nl.float32,
                    buffer=nl.sbuf)
        b = nl.sin(a)
        expected = nl.full((128, 512), 0.0, dtype=nl.float32,
                           buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


def softmax(x, axis=-1, dtype=None):
    r"""Softmax activation function on the input, element-wise.

    ((Similar to `torch.nn.functional.softmax <https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4]
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has softmax of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.softmax -- uniform input produces uniform output
        a = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
        result = nl.softmax(a, axis=1)"""
    ...


def softplus(x, dtype=None):
    r"""Softplus activation, element-wise."""
    ...


def sqrt(x, dtype=None):
    r"""Non-negative square-root of the input, element-wise.

    ((Similar to `numpy.sqrt <https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has square-root values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.sqrt -- sqrt(4.0) = 2.0
        a = nl.full((128, 512), 4.0, dtype=nl.float32,
                    buffer=nl.sbuf)
        b = nl.sqrt(a)
        expected = nl.full((128, 512), 2.0, dtype=nl.float32,
                           buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


def square(x, dtype=None):
    r"""Square of the input, element-wise.

    ((Similar to `numpy.square <https://numpy.org/doc/stable/reference/generated/numpy.square.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has square of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.square -- square(3.0) = 9.0
        a = nl.full((128, 512), 3.0, dtype=nl.float32,
                    buffer=nl.sbuf)
        b = nl.square(a)
        expected = nl.full((128, 512), 9.0, dtype=nl.float32,
                           buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


def static_range(start, stop=None, step=1):
    r"""Create a sequence for fully unrolled loop iteration.

    Create a sequence of numbers for use as loop iterators in NKI, resulting in
    a fully unrolled loop. Prefer this method over :doc:`affine_range <nki.language.affine_range>`
    and :doc:`sequential_range <nki.language.sequential_range>`

    :param start: start value (or stop if ``stop`` is None).
    :param stop: stop value (exclusive).
    :param step: step size.
    :return: an iterator yielding integer values from start to stop.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.static_range
        for i in nl.static_range(input_tensor.shape[1] // 512):
            offset = i * 512
            tile = nl.load(input_tensor[0:128, offset:offset+512])
            result = nl.multiply(tile, tile)
            nl.store(out_tensor[0:128, offset:offset+512], result)"""
    ...


def store(dst, value):
    r"""Store into a tensor on device memory (HBM) from on-chip memory (SBUF).

    .. warning::

       This API is experimental and may change in future releases.

    :param dst: HBM tensor to store the data into.
    :param value: an SBUF tile that contains the values to store."""
    ...


def subtract(x, y, dtype=None):
    r"""Subtract the inputs, element-wise.

    ((Similar to `numpy.subtract <https://numpy.org/doc/stable/reference/generated/numpy.subtract.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see NKI Type Promotion for more information);
    :return: a tile that has ``x - y``, element-wise.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.subtract -- element-wise subtraction of two tiles
        a = nl.full((128, 512), 10.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.subtract(a, b)
        expected = nl.full((128, 512), 7.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)

        # nki.language.subtract -- subtracting a scalar from every element
        a = nl.full((128, 512), 10.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.subtract(a, 3.0)
        expected = nl.full((128, 512), 7.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...


def sum(x, axis, dtype=None, keepdims=False):
    r"""Sum of elements along the specified axis (or axes) of the input.

    ((Similar to `numpy.sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate;
        must be free dimensions, not partition dimension (0); can only be the
        last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: if True, the reduced axes are kept as size-one dimensions.
    :return: a tile with the sum along the provided axis."""
    ...


def tan(x, dtype=None):
    r"""Tangent of the input, element-wise.

    ((Similar to `numpy.tan <https://numpy.org/doc/stable/reference/generated/numpy.tan.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has tangent values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.tan -- tan(0.0) = 0.0
        a = nl.full((128, 512), 0.0, dtype=nl.float32,
                    buffer=nl.sbuf)
        b = nl.tan(a)
        expected = nl.full((128, 512), 0.0, dtype=nl.float32,
                           buffer=nl.sbuf)
        assert nl.equal(b, expected)"""
    ...


def tanh(x, dtype=None):
    r"""Hyperbolic tangent, element-wise."""
    ...


tfloat32 = 'tfloat32'
"""32-bit floating-point number (1S,8E,10M)"""


def transpose(x, dtype=None):
    r"""Transposes a 2D tile between its partition and free dimension.

    .. warning::

       This API is experimental and may change in future releases.

    :param x: 2D input tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has the values of the input tile with its partition and free
        dimensions swapped.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.transpose -- transpose of identity is identity
        x = nl.shared_identity_matrix(n=128, dtype=nl.float32)
        result_psum = nl.transpose(x)
        result = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(result, result_psum)
        assert nl.equal(result, x)"""
    ...


def trunc(x, dtype=None):
    r"""Truncated value of the input, element-wise.

    ((Similar to `numpy.trunc <https://numpy.org/doc/stable/reference/generated/numpy.trunc.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    The truncated value of the scalar x is the nearest integer i which is closer to zero than x is.
    In short, the fractional part of the signed number x is discarded.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has truncated values of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.trunc -- truncates 3.7 toward zero to 3.0
        a = nl.full((128, 512), 3.7, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.trunc(a)
        expected = nl.full((128, 512), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)

        # nki.language.trunc -- truncates -3.7 toward zero to -3.0
        a = nl.full((128, 512), -3.7, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.trunc(a)
        expected = nl.full((128, 512), -3.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...


uint16 = 'uint16'
"""16-bit unsigned integer number"""


uint32 = 'uint32'
"""32-bit unsigned integer number"""


uint8 = 'uint8'
"""8-bit unsigned integer number"""


def var(x, axis, dtype=None, keepdims=False):
    r"""Variance along the specified axis (or axes) of the input.

    ((Similar to `numpy.var <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate;
        must be free dimensions, not partition dimension (0); can only be the
        last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: currently ignored; result always has keepdims=True shape.
    :return: a tile with the variance of the elements along the provided axis."""
    ...


def where(condition, x, y, dtype=None):
    r"""Return elements chosen from x or y depending on condition.

    ((Similar to `numpy.where <https://numpy.org/doc/stable/reference/generated/numpy.where.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param condition: condition tile with float values (1.0 for True, 0.0 for False).
    :param x: tensor from which to take elements where condition is True.
    :param y: tensor from which to take elements where condition is False.
    :param dtype: (optional) data type to cast the output type to (see Supported Data Types for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: tensor with elements from x or y based on condition.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.where -- select 10.0 where condition is 1, else 0.0
        cond = nl.full((128, 512), 1.0, dtype=nl.float32,
                       buffer=nl.sbuf)
        x = nl.full((128, 512), 10.0, dtype=nl.float32,
                    buffer=nl.sbuf)
        y = nl.full((128, 512), 0.0, dtype=nl.float32,
                    buffer=nl.sbuf)
        result = nl.where(cond, x, y)
        expected = nl.full((128, 512), 10.0, dtype=nl.float32,
                           buffer=nl.sbuf)
        assert nl.equal(result, expected)

        # nki.language.where -- select 5.0 where condition is 0
        cond = nl.full((128, 512), 0.0, dtype=nl.float32,
                       buffer=nl.sbuf)
        x = nl.full((128, 512), 10.0, dtype=nl.float32,
                    buffer=nl.sbuf)
        y = nl.full((128, 512), 5.0, dtype=nl.float32,
                    buffer=nl.sbuf)
        result = nl.where(cond, x, y)
        expected = nl.full((128, 512), 5.0, dtype=nl.float32,
                           buffer=nl.sbuf)
        assert nl.equal(result, expected)"""
    ...


def zeros(shape, dtype, buffer=MemoryRegion.sbuf, name=''):
    r"""Create a new tensor of given shape and dtype on the specified buffer, filled with zeros.

    ((Similar to `numpy.zeros <https://numpy.org/doc/stable/reference/generated/numpy.zeros.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param shape: the shape of the tensor.
    :param dtype: the data type of the tensor.
    :param buffer: the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.
    :param name: the name of the tensor, used in :ref:`scheduling <how-to-scheduling-apis>`.
    :return: a new :class:`NkiTensor` allocated on the buffer."""
    ...


def zeros_like(x, dtype=None, buffer=None, name=''):
    r"""Create a new tensor of zeros with the same shape and type as a given tensor.

    ((Similar to `numpy.zeros_like <https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: the tensor.
    :param dtype: the data type of the tensor.
    :param buffer: the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.
    :param name: the name of the tensor, used in :ref:`scheduling <how-to-scheduling-apis>`.
    :return: a new :class:`NkiTensor` of zeros with the same shape as ``x``."""
    ...