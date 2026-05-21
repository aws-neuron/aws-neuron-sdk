from dataclasses import dataclass
from enum import Enum
from typing import *

class MemoryRegion(Enum):
    r"""Memory region constants for NKI tensors."""

    sbuf = ...

    psum = ...

    private_hbm = ...

    shared_hbm = ...

    ...

class NKIObject:
    r"""Base class for NKI kernel dataclasses and configuration objects."""

    def __init__(self, **kwargs: Any) -> None:
        ...

    ...

class NkiTensor(NKIObject):
    r"""Tensor class with access pattern support.

    Attributes:
        shape: Tuple of dimension sizes
        dtype: NKI data type string
        buffer: Buffer location (sbuf, psum, hbm, etc.)
        _storage: Opaque storage handle, interpreted by the backend
        _pattern: Access pattern as list of [step, num] tuples (None = identity)
        offset: Element offset into storage"""

    def __init__(self, shape, dtype, storage, buffer=MemoryRegion.sbuf, name='', pattern=None, offset=0, scalar_offset=None, vector_offset=None, indirect_dim=0):
        ...

    def get_pattern(self):
        r"""Return the access pattern, resolving identity (None) to explicit form."""
        ...

    def is_identity(self):
        r"""Return True if tensor has identity pattern (no .ap() or slicing)."""
        ...

    @property
    def sim_value(self):
        r"""Simulator-computed value (current tensor data)."""
        ...

    @property
    def device_value(self):
        r"""Device dump value, or None if no device data."""
        ...

    def view(self, dtype):
        r"""Reinterpret cast tensor to a different dtype.

        The underlying storage bits are reinterpreted as the new dtype.
        Supports both upcasting (e.g. int8 -> int32) and downcasting (e.g. int32 -> int8).

        Args:
            dtype: Target dtype for reinterpretation

        Returns:
            New NkiTensor viewing same storage as different dtype

        Example:
            # Reinterpret int32 bits as float32
            int_tensor = nl.ndarray((128, 256), dtype=nl.int32, buffer=nl.sbuf)
            float_tensor = int_tensor.view(nl.float32)"""
        ...

    def ap(self, pattern, offset=0, scalar_offset=None, vector_offset=None, indirect_dim=0, dtype=None):
        r"""Create tensor with explicit access pattern sharing same storage.

        Args:
            pattern: List of [step, num] tuples defining access pattern
            offset: Element offset from start of storage
            scalar_offset: Dynamic offset applied on ``indirect_dim``.
                Can be an SBUF tensor (1×1) or a :class:`nisa.VirtualRegister`.
            vector_offset: SBUF location for per-partition dynamic offset
            indirect_dim: Dimension to apply scalar/vector offset
            dtype: Optional dtype for reinterpret casting

        Returns:
            New NkiTensor with specified access pattern

        Raises:
            NkiValidationError: If partition dimension access has holes (non-contiguous)
            IndexError: If access pattern exceeds tensor bounds"""
        ...

    def reshape(self, new_shape=None, shape=None):
        r"""Return a reshaped view of the tensor sharing storage."""
        ...

    ...

bool_ = ...
r"""Boolean (True or False) stored as a byte"""

int8 = ...
r"""8-bit signed integer number"""

int16 = ...
r"""16-bit signed integer number"""

int32 = ...
r"""32-bit signed integer number"""

uint8 = ...
r"""8-bit unsigned integer number"""

uint16 = ...
r"""16-bit unsigned integer number"""

uint32 = ...
r"""32-bit unsigned integer number"""

float16 = ...
r"""16-bit floating-point number"""

float32 = ...
r"""32-bit floating-point number"""

bfloat16 = ...
r"""16-bit floating-point number (1S,8E,7M)"""

tfloat32 = ...
r"""32-bit floating-point number (1S,8E,10M)"""

float8_e4m3 = ...
r"""8-bit floating-point number (1S,4E,3M)"""

float8_e4m3fn = ...
r"""8-bit floating-point number (1S,4E,3M), Extended range: no inf, NaN represented by 0bS111'1111"""

float8_e5m2 = ...
r"""8-bit floating-point number (1S,5E,2M)"""

float8_e5m2_x4 = ...
r"""4x packed float8_e5m2 elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""

float8_e4m3fn_x4 = ...
r"""4x packed float8_e4m3fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""

float4_e2m1fn_x4 = ...
r"""4x packed float4_e2m1fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""

sbuf = ...
r"""Memory region constants for NKI tensors."""

psum = ...
r"""Memory region constants for NKI tensors."""

hbm = ...
r"""Memory region constants for NKI tensors."""

private_hbm = ...
r"""Memory region constants for NKI tensors."""

shared_hbm = ...
r"""Memory region constants for NKI tensors."""

tile_size = ...
r"""Hardware tile size constants (pmax, psum_fmax, gemm_stationary_fmax, etc.)"""

prelu = ...
r"""Parametric ReLU activation function. Used as the ``op`` parameter in activation ISA instructions such as :doc:`nki.isa.activation </nki/api/generated/nki.isa.activation>`. The slope for negative inputs is controlled by the ``relu_param`` argument (see :ref:`nki-act-func`)."""

bypass = ...
r"""No-op operator that passes data through unchanged. Used as the ``op0`` or ``op1`` parameter in tensor-scalar ISA instructions (e.g., :doc:`nki.isa.activation </nki/api/generated/nki.isa.activation>`) to skip a computation stage."""

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

def add(x, y, dtype=None):
    r"""Add the inputs, element-wise.

    ((Similar to `numpy.add <https://numpy.org/doc/stable/reference/generated/numpy.add.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
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

def subtract(x, y, dtype=None):
    r"""Subtract the inputs, element-wise.

    ((Similar to `numpy.subtract <https://numpy.org/doc/stable/reference/generated/numpy.subtract.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
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

def multiply(x, y, dtype=None):
    r"""Multiply the inputs, element-wise.

    ((Similar to `numpy.multiply <https://numpy.org/doc/stable/reference/generated/numpy.multiply.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
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

def divide(x, y, dtype=None):
    r"""Divide the inputs, element-wise.

    ((Similar to `numpy.divide <https://numpy.org/doc/stable/reference/generated/numpy.divide.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
    :return: a tile that has ``x / y``, element-wise.
    """
    ...

def maximum(x, y, dtype=None):
    r"""Maximum of the inputs, element-wise.

    ((Similar to `numpy.maximum <https://numpy.org/doc/stable/reference/generated/numpy.maximum.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
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

def minimum(x, y, dtype=None):
    r"""Minimum of the inputs, element-wise.

    ((Similar to `numpy.minimum <https://numpy.org/doc/stable/reference/generated/numpy.minimum.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
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

def abs_max(x, y, dtype=None):
    r"""Maximum of the inputs compared by magnitude, element-wise.

    Compares ``abs(x)`` and ``abs(y)`` and returns the **original (signed) value**
    of whichever input has the larger absolute value.
    For example, ``abs_max(-5, 3)`` returns ``-5`` because ``|-5| > |3|``.

    .. note::
        Available only on NeuronCore-v4 (trn3) and newer.

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
    :return: a tile where each element is ``x`` if ``|x| > |y|``, else ``y``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.abs_max -- returns the input with the larger absolute value
        a = nl.full((128, 512), -5.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.full((128, 1), 3.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.abs_max(a, b)  # |−5| > |3| → returns -5.0 (original signed value)

        expected = nl.full((128, 512), -5.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)

    .. code-block:: python

        # nki.language.abs_max -- tie-breaking: returns y when |x| == |y|
        a = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.full((128, 1), -1.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.abs_max(a, b)  # |1| == |-1| → tie, returns y = -1.0

        expected = nl.full((128, 512), -1.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...

def abs_min(x, y, dtype=None):
    r"""Minimum of the inputs compared by magnitude, element-wise.

    Compares ``abs(x)`` and ``abs(y)`` and returns the **original (signed) value**
    of whichever input has the smaller absolute value.
    For example, ``abs_min(-5, 3)`` returns ``3`` because ``|3| < |-5|``.

    .. note::
        Available only on NeuronCore-v4 (trn3) and newer.

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
    :return: a tile where each element is ``x`` if ``|x| < |y|``, else ``y``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.abs_min -- element-wise absolute minimum of two tiles
        a = nl.full((128, 512), 5.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.full((128, 1), -3.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.abs_min(a, b)  # |-3| < |5|, so returns -3.0 (original signed value)

        expected = nl.full((128, 512), -3.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)

    .. code-block:: python

        # nki.language.abs_min -- tie-breaking: returns y when |x| == |y|
        a = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
        b = nl.full((128, 1), -1.0, dtype=nl.float32, buffer=nl.sbuf)
        c = nl.abs_min(a, b)  # |1| == |-1| → tie, returns y = -1.0

        expected = nl.full((128, 512), -1.0, dtype=nl.float32, buffer=nl.sbuf)
        assert nl.equal(c, expected)"""
    ...

def abs(x, dtype=None):
    r"""Absolute value of the input, element-wise.

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def square(x, dtype=None):
    r"""Square of the input, element-wise.

    ((Similar to `numpy.square <https://numpy.org/doc/stable/reference/generated/numpy.square.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def power(x, y, dtype=None):
    r"""Elements of x raised to powers of y, element-wise.

    ((Similar to `numpy.power <https://numpy.org/doc/stable/reference/generated/numpy.power.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
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

def equal(x, y, dtype=None):
    r"""Return (x == y) element-wise.

    ((Similar to `numpy.equal <https://numpy.org/doc/stable/reference/generated/numpy.equal.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); Defaults to the input tile dtype.
        Use ``dtype=nl.uint8`` for a boolean-like result.
    :return: a tile with 1 where equal, 0 otherwise."""
    ...

def not_equal(x, y, dtype=None):
    r"""Return (x != y) element-wise.

    ((Similar to `numpy.not_equal <https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); Defaults to the input tile dtype.
        Use ``dtype=nl.uint8`` for a boolean-like result.
    :return: a tile with 1 where not equal, 0 otherwise."""
    ...

def less(x, y, dtype=None):
    r"""Return (x < y) element-wise.

    ((Similar to `numpy.less <https://numpy.org/doc/stable/reference/generated/numpy.less.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); Defaults to the input tile dtype.
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); Defaults to the input tile dtype.
        Use ``dtype=nl.uint8`` for a boolean-like result.
    :return: a tile with 1 where x <= y, 0 otherwise."""
    ...

def greater(x, y, dtype=None):
    r"""Return (x > y) element-wise.

    ((Similar to `numpy.greater <https://numpy.org/doc/stable/reference/generated/numpy.greater.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); Defaults to the input tile dtype.
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); Defaults to the input tile dtype.
        Use ``dtype=nl.uint8`` for a boolean-like result.
    :return: a tile with 1 where x >= y, 0 otherwise."""
    ...

def bitwise_and(x, y, dtype=None):
    r"""Compute the bitwise AND of two tiles element-wise.

    ((Similar to `numpy.bitwise_and <https://numpy.org/doc/stable/reference/generated/numpy.bitwise_and.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs must be integer typed.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
    :return: a tile with the bitwise XOR result."""
    ...

def invert(x, dtype=None):
    r"""Compute the bitwise NOT element-wise.

    ((Similar to `numpy.invert <https://numpy.org/doc/stable/reference/generated/numpy.invert.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Input must be integer typed. Implemented as XOR with all-ones.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile with the bitwise NOT result."""
    ...

def left_shift(x, y, dtype=None):
    r"""Left shift the bits of x by y positions element-wise.

    ((Similar to `numpy.left_shift <https://numpy.org/doc/stable/reference/generated/numpy.left_shift.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs must be integer typed.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
    :return: a tile with the left-shifted result."""
    ...

def right_shift(x, y, dtype=None):
    r"""Right shift the bits of x by y positions element-wise.

    ((Similar to `numpy.right_shift <https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs must be integer typed.

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
    :return: a tile with the right-shifted result."""
    ...

def logical_and(x, y, dtype=None):
    r"""Compute the logical AND of two tiles element-wise.

    ((Similar to `numpy.logical_and <https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs should be boolean-like (0 or 1 values).

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
    :return: a tile with the logical AND result."""
    ...

def logical_or(x, y, dtype=None):
    r"""Compute the logical OR of two tiles element-wise.

    ((Similar to `numpy.logical_or <https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Inputs should be boolean-like (0 or 1 values).

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. At least one of x, y must be a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
    :return: a tile with the logical XOR result."""
    ...

def logical_not(x, dtype=None):
    r"""Compute the logical NOT element-wise.

    ((Similar to `numpy.logical_not <https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Implemented as XOR with 1, so inputs should be boolean-like (0 or 1 values).
    For non-boolean inputs, use ``nl.equal(x, 0)`` instead.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile with the logical NOT result."""
    ...

def exp(x, dtype=None):
    r"""Exponential of the input, element-wise.

    ((Similar to `numpy.exp <https://numpy.org/doc/stable/reference/generated/numpy.exp.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    The ``exp(x)`` is ``e^x`` where ``e`` is the Euler's number = 2.718281...

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def log(x, dtype=None):
    r"""Natural logarithm of the input, element-wise.

    ((Similar to `numpy.log <https://numpy.org/doc/stable/reference/generated/numpy.log.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    It is the inverse of the exponential function, such that: ``log(exp(x)) = x`` .
    The natural logarithm base is ``e``.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def sqrt(x, dtype=None):
    r"""Non-negative square-root of the input, element-wise.

    ((Similar to `numpy.sqrt <https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def rsqrt(x, dtype=None):
    r"""Reciprocal of the square-root of the input, element-wise.

    ((Similar to `torch.rsqrt <https://pytorch.org/docs/stable/generated/torch.rsqrt.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    ``rsqrt(x) = 1 / sqrt(x)``

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def tanh(x, dtype=None):
    r"""Hyperbolic tangent, element-wise."""
    ...

def sigmoid(x, dtype=None):
    r"""Sigmoid activation, element-wise."""
    ...

def relu(x, dtype=None):
    r"""ReLU activation, element-wise."""
    ...

def gelu(x, dtype=None):
    r"""GELU activation, element-wise."""
    ...

def gelu_apprx_sigmoid(x, dtype=None):
    r"""GELU approximation using sigmoid, element-wise."""
    ...

def gelu_apprx_tanh(x, dtype=None):
    r"""GELU approximation using tanh, element-wise."""
    ...

def gelu_dx(x, dtype=None):
    r"""Derivative of GELU activation, element-wise."""
    ...

def gelu_apprx_sigmoid_dx(x, dtype=None):
    r"""Derivative of sigmoid-approximated GELU, element-wise."""
    ...

def silu(x, dtype=None):
    r"""SiLU (Swish) activation, element-wise."""
    ...

def silu_dx(x, dtype=None):
    r"""Derivative of SiLU activation, element-wise."""
    ...

def softplus(x, dtype=None):
    r"""Softplus activation, element-wise."""
    ...

def mish(x, dtype=None):
    r"""Mish activation, element-wise."""
    ...

def erf(x, dtype=None):
    r"""Error function, element-wise."""
    ...

def erf_dx(x, dtype=None):
    r"""Derivative of error function, element-wise."""
    ...

def reciprocal(x, dtype=None):
    r"""Reciprocal of the input, element-wise.

    ((Similar to `numpy.reciprocal <https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    ``reciprocal(x) = 1 / x``

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def copy(x, dtype=None):
    r"""Create a copy of the input tile.

    .. warning::

       This API is experimental and may change in future releases.

    Uses the Scalar Engine via ``activation(op=copy)``. Note that the Scalar Engine
    internally casts through FP32, which may be lossy for integer types with
    values exceeding FP32 precision (e.g. int32 values > 2^23).

    :param x: the source of copy, must be a tile in SBUF or PSUM.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a new tile with the same layout as ``x``, allocated on the same buffer
        as ``x`` (SBUF or PSUM)."""
    ...

def sin(x, dtype=None):
    r"""Sine of the input, element-wise.

    ((Similar to `numpy.sin <https://numpy.org/doc/stable/reference/generated/numpy.sin.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def cos(x, dtype=None):
    r"""Cosine of the input, element-wise.

    ((Similar to `numpy.cos <https://numpy.org/doc/stable/reference/generated/numpy.cos.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def tan(x, dtype=None):
    r"""Tangent of the input, element-wise.

    ((Similar to `numpy.tan <https://numpy.org/doc/stable/reference/generated/numpy.tan.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def arctan(x, dtype=None):
    r"""Inverse tangent of the input, element-wise.

    ((Similar to `numpy.arctan <https://numpy.org/doc/stable/reference/generated/numpy.arctan.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def ceil(x, dtype=None):
    r"""Ceiling of the input, element-wise.

    ((Similar to `numpy.ceil <https://numpy.org/doc/stable/reference/generated/numpy.ceil.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    The ceil of the scalar x is the smallest integer i, such that i >= x.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def floor(x, dtype=None):
    r"""Floor of the input, element-wise.

    ((Similar to `numpy.floor <https://numpy.org/doc/stable/reference/generated/numpy.floor.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    The floor of the scalar x is the largest integer i, such that i <= x.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def trunc(x, dtype=None):
    r"""Truncated value of the input, element-wise.

    ((Similar to `numpy.trunc <https://numpy.org/doc/stable/reference/generated/numpy.trunc.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    The truncated value of the scalar x is the nearest integer i which is closer to zero than x is.
    In short, the fractional part of the signed number x is discarded.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def negative(x, dtype=None):
    r"""Numerical negative of the input, element-wise.

    ((Similar to `numpy.negative <https://numpy.org/doc/stable/reference/generated/numpy.negative.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def sign(x, dtype=None):
    r"""Sign of the numbers of the input, element-wise.

    ((Similar to `numpy.sign <https://numpy.org/doc/stable/reference/generated/numpy.sign.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    The sign function returns ``-1`` if ``x < 0``, ``0`` if ``x==0``, ``1`` if ``x > 0``.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def fmod(x, y, dtype=None):
    r"""Floating-point remainder of ``x / y``, element-wise.

    The remainder has the same sign as the dividend x.
    It is equivalent to the Matlab(TM) rem function and should not be confused with the Python modulus operator x % y.

    ((Similar to `numpy.fmod <https://numpy.org/doc/stable/reference/generated/numpy.fmod.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile. If x is a scalar value it will be broadcast to the shape of y.
    :param y: a tile or a scalar value.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
    :return: a tile that has values ``x fmod y``.
    """
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information);
    :return: a tile that has values ``x mod y``.
    """
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: if True, the reduced axes are kept as size-one dimensions.
    :return: a tile with the sum along the provided axis."""
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: if True, the reduced axes are kept as size-one dimensions.
    :return: a tile with the maximum along the provided axis."""
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: if True, the reduced axes are kept as size-one dimensions.
    :return: a tile with the minimum along the provided axis."""
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: if True, the reduced axes are kept as size-one dimensions.
    :return: a tile with the average of elements along the provided axis. Float32
        intermediate values are used for the computation."""
    ...

def prod(x, axis, dtype=None, keepdims=False):
    r"""Product of elements along the specified axis (or axes) of the input.

    ((Similar to `numpy.prod <https://numpy.org/doc/stable/reference/generated/numpy.prod.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate;
        must be free dimensions, not partition dimension (0); can only be the
        last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: if True, the reduced axes are kept as size-one dimensions.
    :return: a tile with the product along the provided axis."""
    ...

def var(x, axis, dtype=None, keepdims=False):
    r"""Variance along the specified axis (or axes) of the input.

    ((Similar to `numpy.var <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate;
        must be free dimensions, not partition dimension (0); can only be the
        last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4].
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param keepdims: currently ignored; result always has keepdims=True shape.
    :return: a tile with the variance of the elements along the provided axis."""
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile with the logical AND reduction along the provided axis."""
    ...

def softmax(x, axis=-1, dtype=None):
    r"""Softmax activation function on the input, element-wise.

    ((Similar to `torch.nn.functional.softmax <https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        x: A tile.
        axis: int or tuple/list of ints. The axis (or axes) along which to operate;
            must be free dimensions, not partition dimension (0); can only be the
            last contiguous dim(s) of the tile: [1], [1,2], [1,2,3], [1,2,3,4]
        dtype: (optional) data type to cast the output type to; if not specified,
            it will default to be the same as the data type of the input tile.

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: ``[1], [1,2], [1,2,3], [1,2,3,4]``
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has softmax of ``x``.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.softmax -- uniform input produces uniform output
        a = nl.full((128, 512), 1.0, dtype=nl.float32, buffer=nl.sbuf)
        result = nl.softmax(a, axis=1)"""
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def where(condition, x, y, dtype=None):
    r"""Return elements chosen from x or y depending on condition.

    ((Similar to `numpy.where <https://numpy.org/doc/stable/reference/generated/numpy.where.html>`_))

    .. warning::

       This API is experimental and may change in future releases.

    :param condition: condition tile with float values (1.0 for True, 0.0 for False).
    :param x: tensor from which to take elements where condition is True.
    :param y: tensor from which to take elements where condition is False.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def dropout(x, rate, dtype=None):
    r"""Randomly zeroes some of the elements of the input tile given a probability rate.

    .. warning::

       This API is experimental and may change in future releases.

    :param x: a tile.
    :param rate: the probability of zeroing each element. Can be a scalar constant
        or a tile of shape ``(x.shape[0], 1)`` for per-partition drop probabilities.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile with randomly zeroed elements of ``x``."""
    ...

def ones(shape, dtype, buffer=sbuf, name=''):
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

def full(shape, fill_value, dtype, buffer=sbuf, name=''):
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile with the target shape containing broadcast values from ``x``."""
    ...

def transpose(x, dtype=None):
    r"""Transposes a 2D tile between its partition and free dimension.

    .. warning::

       This API is experimental and may change in future releases.

    :param x: 2D input tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has the values of the input tile with its partition and free
        dimensions swapped.

    Examples:

    .. code-block:: python

        import nki.isa as nisa
        import nki.language as nl

        # nki.language.transpose -- transpose of identity is identity
        x = nl.shared_identity_matrix(n=128, dtype=nl.float32)
        result_psum = nl.transpose(x)
        result = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(result, result_psum)
        assert nl.equal(result, x)"""
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

        import nki.isa as nisa
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

def load_transpose2d(src, dtype=None):
    r"""Load a tensor from device memory (HBM) and 2D-transpose the data before storing into on-chip memory (SBUF).

    .. warning::

       This API is experimental and may change in future releases.

    :param src: HBM tensor to load the data from.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a new tile on SBUF with values from ``src`` 2D-transposed."""
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
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
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

def ds(start, size):
    r"""Create a dynamic slice for tensor indexing.

    :param start: the start index of the slice.
    :param size: the size of the slice.
    :return: a dynamic slice object for use in tensor indexing."""
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

def static_range(start, stop=None, step=1):
    r"""Create a sequence for fully unrolled loop iteration.

    Create a sequence of numbers for use as loop iterators in NKI, resulting in
    a fully unrolled loop. Prefer this method over :doc:`affine_range <nki.language.affine_range>`
    and :doc:`sequential_range <nki.language.sequential_range>`.

    :param start: start value (or stop if ``stop`` is None).
    :param stop: stop value (exclusive).
    :param step: step size.
    :return: an iterator yielding integer values from start to stop.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.static_range -- fully unrolled iteration over tiles
        for i in nl.static_range(input_tensor.shape[1] // 512):
            offset = i * 512
            tile = nl.load(input_tensor[0:128, offset:offset+512])
            result = nl.multiply(tile, tile)
            nl.store(out_tensor[0:128, offset:offset+512], result)"""
    ...

def dynamic_range(start, stop=None, step=1):
    r"""Create a sequence for **dynamic** loop iteration.

    Create a sequence of numbers for use as **dynamic** loop iterators in NKI.
    The loop runs on device with dynamic bounds.

    :param start: start value (or stop if ``stop`` is None), can be VirtualRegister.
    :param stop: stop value (exclusive), can be VirtualRegister.
    :param step: step size, must be a compile-time positive integer (not VirtualRegister).
    :return: an iterator yielding integer values from start to stop.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.dynamic_range -- dynamic iteration with runtime bounds
        for _ in nl.dynamic_range(1):
            tile = nl.load(input_tensor[0:128, 0:512])
            result = nl.multiply(tile, tile)
            nl.store(out_tensor[0:128, 0:512], result)"""
    ...

def ndarray(shape, dtype, buffer=sbuf, name='', address=None):
    r"""Create a new tensor of given shape and dtype on the specified buffer.

    :param shape: the shape of the tensor.
    :param dtype: the data type of the tensor.
    :param buffer: the specific buffer (ie, sbuf, psum, hbm), defaults to sbuf.
    :param name: the name of the tensor, used in :ref:`scheduling <how-to-scheduling-apis>`.
    :param address: optional memory address ``(partition_offset, free_offset)``.
    :return: a new :class:`NkiTensor` allocated on the buffer."""
    ...

def zeros(shape, dtype, buffer=sbuf, name=''):
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

def load(src, dtype=None):
    r"""Load a tensor from device memory (HBM) into on-chip memory (SBUF).

    .. warning::

       This API is experimental and may change in future releases.

    :param src: HBM tensor to load the data from.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a new tile on SBUF with values from ``src``."""
    ...

def store(dst, value):
    r"""Store into a tensor on device memory (HBM) from on-chip memory (SBUF).

    .. warning::

       This API is experimental and may change in future releases.

    :param dst: HBM tensor to store the data into.
    :param value: an SBUF tile that contains the values to store."""
    ...

def program_id(axis=0):
    r"""Index of the current SPMD program along the given axis in the launch grid.

    :param axis: the axis of the launch grid.
    :return: the program id along ``axis``."""
    ...

def num_programs(axes=0):
    r"""Number of SPMD programs along the given axes in the launch grid.

    :param axes: the axes of the launch grid. If not provided, returns the total
        number of programs along the entire launch grid.
    :return: the number of SPMD programs along ``axes`` in the launch grid."""
    ...

def program_ndim():
    r"""Number of dimensions in the SPMD launch grid.

    :return: the number of dimensions in the launch grid, i.e. the number of axes. 0 if no grid."""
    ...

def shared_constant(constant):
    r"""Create a tensor in shared HBM initialized with constant data.

    The constant is embedded in the compiled binary and loaded to HBM
    at model load time. With LNC=2, both cores share the same constant;
    the data must not diverge across cores.

    Supported element types: float32, float16, bfloat16, int32, int16,
    int8, uint32, uint16, uint8, float8_e4m3fn, float8_e5m2.
    Packed types (float8_e4m3fn_x4, float8_e5m2_x4, float4_e2m1fn_x4)
    and tfloat32 are supported at the MLIR level but not yet tested
    end-to-end on hardware.

    :param constant: the constant data. Can be a numpy array or a file path
        to a ``.npy`` file.
    :return: an NkiTensor in shared_hbm containing the constant data."""
    ...

def shared_identity_matrix(n, dtype=uint8, dst=None):
    r"""Create an identity matrix in SBUF with the specified data type.

    This function has the same behavior to :doc:`nki.language.shared_constant <nki.language.shared_constant>` but
    is preferred if the constant matrix is an identity matrix. The
    compiler will reuse all the identity matrices of the same
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

def rand(shape, dtype, buffer=sbuf, name=''):
    r"""Create a new tensor of given shape and dtype on the specified buffer, filled with random values.

    Values are sampled from a uniform distribution between 0 and 1.

    .. warning::

       This API is experimental and may change in future releases.

    :param shape: the shape of the tensor.
    :param dtype: the data type of the tensor (see :ref:`nki-dtype` for more information).
    :param buffer: the specific buffer (ie, :doc:`sbuf<nki.language.sbuf>`, :doc:`psum<nki.language.psum>`, :doc:`hbm<nki.language.hbm>`), defaults to :doc:`sbuf<nki.language.sbuf>`.
    :param name: the name of the tensor, used in :ref:`scheduling <how-to-scheduling-apis>`.
    :return: a new :class:`NkiTensor` allocated on the buffer with random values.

    Examples:

    .. code-block:: python

        import nki.language as nl

        # nki.language.rand -- generate random values in [0, 1)
        a = nl.rand((128, 512), dtype=nl.float32)"""
    ...
