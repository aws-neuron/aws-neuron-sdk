"""Auto-generated stub file"""
from enum import Enum
import numpy as np
import ml_dtypes

class NKIObject:

    ...

bfloat16 = 'bfloat16'
"""16-bit floating-point number (1S,8E,7M)"""

bool_ = 'bool'
"""Boolean (True or False) stored as a byte"""

float8_e4m3 = 'float8_e4m3'
"""8-bit floating-point number (1S,4E,3M)"""

float8_e5m2 = 'float8_e5m2'
"""8-bit floating-point number (1S,5E,2M)"""

float16 = 'float16'
"""16-bit floating-point number"""

float32 = 'float32'
"""32-bit floating-point number"""

int8 = 'int8'
"""8-bit signed integer number"""

int16 = 'int16'
"""16-bit signed integer number"""

int32 = 'int32'
"""32-bit signed integer number"""

tfloat32 = 'tfloat32'
"""32-bit floating-point number (1S,8E,10M)"""

uint8 = 'uint8'
"""8-bit unsigned integer number"""

uint16 = 'uint16'
"""16-bit unsigned integer number"""

uint32 = 'uint32'
"""32-bit unsigned integer number"""

float8_e4m3fn_x4 = 'float8_e4m3fn_x4'
"""4x packed float8_e4m3fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""

float8_e5m2_x4 = 'float8_e5m2_x4'
"""4x packed float8_e5m2 elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""

float4_e2m1fn_x4 = 'float4_e2m1fn_x4'
"""4x packed float4_e2m1fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""

def ndarray(shape, dtype, *, buffer=None, name=""):
    r"""
    Create a new tensor of given shape and dtype on the specified buffer.

    ((Similar to `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_))

    :param shape: the shape of the tensor.
    :param dtype: the data type of the tensor (see :ref:`nki-dtype` for more information).
    :param buffer: the specific buffer (ie, :doc:`sbuf<nki.language.sbuf>`, :doc:`psum<nki.language.psum>`, :doc:`hbm<nki.language.hbm>`), defaults to :doc:`sbuf<nki.language.sbuf>`.
    :param name: the name of the tensor.
    :return: a new tensor allocated on the buffer.
    """
    ...

def zeros(shape, dtype, *, buffer=None, name=""):
    r"""
    Create a new tensor of given shape and dtype on the specified buffer, filled with zeros.

    ((Similar to `numpy.zeros <https://numpy.org/doc/stable/reference/generated/numpy.zeros.html>`_))

    :param shape: the shape of the tensor.
    :param dtype: the data type of the tensor (see :ref:`nki-dtype` for more information).
    :param buffer: the specific buffer (ie, :doc:`sbuf<nki.language.sbuf>`, :doc:`psum<nki.language.psum>`, :doc:`hbm<nki.language.hbm>`), defaults to :doc:`sbuf<nki.language.sbuf>`.
    :param name: the name of the tensor.
    :return: a new tensor allocated on the buffer.
    """
    ...

def shared_constant(constant, dtype=None):
    r"""
    Create a new tensor filled with the data specified by data array.

    :param constant: the constant data to be filled into a tensor
    :return: a tensor which contains the constant data
    """
    ...

def shared_identity_matrix(n, dtype=np.uint8):
    r"""
    Create a new identity tensor with specified data type.

    This function has the same behavior to :doc:`nki.language.shared_constant <nki.language.shared_constant>` but
    is preferred if the constant matrix is an identity matrix. The
    compiler will reuse all the identity matrices of the same
    dtype in the graph to save space.

    :param n: the number of rows(and columns) of the returned identity matrix
    :param dtype: the data type of the tensor, default to be ``np.uint8`` (see :ref:`nki-dtype` for more information).
    :return: a tensor which contains the identity tensor
    """
    ...

def affine_range(start, stop=None, step=1):
    r"""
    Create a sequence of numbers for use as **parallel** loop iterators in NKI. ``affine_range`` should be the default
    loop iterator choice, when there is **no** loop carried dependency. Note, associative reductions are **not** considered
    loop carried dependencies in this context. A concrete example of associative reduction
    is :doc:`nisa.nc_matmul <nki.isa.nc_matmul>`, which acumulates into the same
    output buffer defined outside of this loop level (see code example #2 below).

    When the above conditions are not met, we recommend using :doc:`sequential_range <nki.language.sequential_range>`
    instead.

    Notes:

    - Using ``affine_range`` prevents Neuron compiler from unrolling the loops until entering compiler backend,
      which typically results in better compilation time compared to the fully unrolled iterator
      :doc:`static_range <nki.language.static_range>`.
    - Using ``affine_range`` also allows Neuron compiler to perform additional loop-level optimizations, such as
      loop vectorization in current release. The exact type of loop-level optimizations applied is subject
      to changes in future releases.
    - Since each kernel instance only runs on a single NeuronCore, `affine_range` does **not** parallelize
      different loop iterations across multiple NeuronCores. However, different iterations could be parallelized/pipelined
      on different compute engines within a NeuronCore depending on the invoked instructions (engines) and data dependency
      in the loop body.

    .. code-block::
      :linenos:

      import nki.language as nl

      #######################################################################
      # Example 1: No loop carried dependency
      # Input/Output tensor shape: [128, 2048]
      # Load one tile ([128, 512]) at a time, square the tensor element-wise,
      # and store it into output tile
      #######################################################################

      # Every loop instance works on an independent input/output tile.
      # No data dependency between loop instances.
      for i_input in nl.affine_range(input.shape[1] // 512):
        offset = i_input * 512
        input_sb = nl.load(input[0:input.shape[0], offset:offset+512])
        result = nl.multiply(input_sb, input_sb)
        nl.store(output[0:input.shape[0], offset:offset+512], result)

      #######################################################################
      # Example 2: Matmul output buffer accumulation, a type of associative reduction
      # Input tensor shapes for nl.matmul: xT[K=2048, M=128] and y[K=2048, N=128]
      # Load one tile ([128, 128]) from both xT and y at a time, matmul and
      # accumulate into the same output buffer
      #######################################################################

      result_psum = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.psum)
      for i_K in nl.affine_range(xT.shape[0] // 128):
        offset = i_K * 128
        xT_sbuf = nl.load(offset:offset+128, 0:xT.shape[1]])
        y_sbuf = nl.load(offset:offset+128, 0:y.shape[1]])

        result_psum += nl.matmul(xT_sbuf, y_sbuf, transpose_x=True)

    """
    ...

def ds(start, size):
    r"""
    Construct a dynamic slice for simple tensor indexing.

    .. nki_example:: ../../test/test_nki_nl_dslice.py
     :language: python
     :marker: NKI_EXAMPLE_1

    """
    ...

def sequential_range(start, stop, step):
    r"""
    Create a sequence of numbers for use as **sequential** loop iterators in NKI. ``sequential_range``
    should be used when there is a loop carried dependency. Note, associative reductions are **not** considered
    loop carried dependencies in this context. See :doc:`affine_range <nki.language.affine_range>` for
    an example of such associative reduction.

    Notes:

    - Inside a NKI kernel, any use of Python ``range(...)`` will be replaced with ``sequential_range(...)``
      by Neuron compiler.
    - Using ``sequential_range`` prevents Neuron compiler from unrolling the loops until entering compiler backend,
      which typically results in better compilation time compared to the fully unrolled iterator
      :doc:`static_range <nki.language.static_range>`.
    - Using ``sequential_range`` informs Neuron compiler to respect inter-loop dependency and perform
      much more conservative loop-level optimizations compared to ``affine_range``.
    - Using ``affine_range`` instead of ``sequential_range`` in case of loop carried dependency
      incorrectly is considered unsafe and could lead to numerical errors.


    .. code-block::
      :linenos:

      import nki.language as nl

      #######################################################################
      # Example 1: Loop carried dependency from tiling tensor_tensor_scan
      # Both sbuf tensor input0 and input1 shapes: [128, 2048]
      # Perform a scan operation between the two inputs using a tile size of [128, 512]
      # Store the scan output to another [128, 2048] tensor
      #######################################################################

      # Loop iterations communicate through this init tensor
      init = nl.zeros((128, 1), dtype=input0.dtype)

      # This loop will only produce correct results if the iterations are performed in order
      for i_input in nl.sequential_range(input0.shape[1] // 512):
        offset = i_input * 512

        # Depends on scan result from the previous loop iteration
        result = nisa.tensor_tensor_scan(input0[:, offset:offset+512],
                                         input1[:, offset:offset+512],
                                         initial=init,
                                         op0=nl.multiply, op1=nl.add)

        nl.store(output[0:input0.shape[0], offset:offset+512], result)

        # Prepare initial result for scan in the next loop iteration
        init[:, :] = result[:, 511]

    """
    ...

def static_range(start, stop=None, step=1):
    r"""
    Create a sequence of numbers for use as loop iterators in NKI, resulting in a fully unrolled loop.
    Unlike :doc:`affine_range <nki.language.affine_range>` or :doc:`sequential_range <nki.language.sequential_range>`,
    Neuron compiler will fully unroll the loop during NKI kernel tracing.

    Notes:

    - Due to loop unrolling, compilation time may go up significantly compared to
      :doc:`affine_range <nki.language.affine_range>` or :doc:`sequential_range <nki.language.sequential_range>`.
    - On-chip memory (SBUF) usage may also go up significantly compared to
      :doc:`affine_range <nki.language.affine_range>` or :doc:`sequential_range <nki.language.sequential_range>`.
    - No loop-level optimizations will be performed in the compiler.
    - ``static_range`` should only be used as a fall-back option for debugging purposes when
      :doc:`affine_range <nki.language.affine_range>` or :doc:`sequential_range <nki.language.sequential_range>`
      is giving functionally incorrect results or undesirable performance characteristics.


    """
    ...

class tile_size(NKIObject):
    r""" Tile size constants. """

    pmax: int = ...
    r"""Maximum partition dimension of a tile"""

    psum_fmax: int = ...
    r"""Maximum free dimension of a tile on PSUM buffer"""

    gemm_stationary_fmax: int = ...
    r"""Maximum free dimension of the stationary operand of General Matrix Multiplication on Tensor Engine"""

    gemm_moving_fmax: int = ...
    r"""Maximum free dimension of the moving operand of General Matrix Multiplication on Tensor Engine"""

    bn_stats_fmax: int = ...
    r"""Maximum free dimension of BN_STATS"""

    psum_min_align: int = ...
    r"""Minimum byte alignment requirement for PSUM free dimension address"""

    sbuf_min_align: int = ...
    r"""Minimum byte alignment requirement for SBUF free dimension address"""

    total_available_sbuf_size: int = ...
    r"""Total SBUF available size"""


def abs(x, *, dtype=None, mask=None):
    r"""
    Absolute value of the input, element-wise.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has absolute values of ``x``.
    """
    ...

def add(x, y, *, dtype=None, mask=None):
    r"""
    Add the inputs, element-wise.

    ((Similar to `numpy.add <https://numpy.org/doc/stable/reference/generated/numpy.add.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has ``x + y``, element-wise.

    Examples:

    .. nki_example:: ../../test/test_nki_nl_add.py
     :language: python
     :marker: NKI_EXAMPLE_20

    .. note::
      Broadcasting in the partition dimension is generally more expensive than broadcasting in free dimension. It is recommended to align your data to perform free dimension broadcast whenever possible.

    """
    ...

def all(x, axis, *, dtype=bool, mask=None):
    r"""
    Whether all elements along the specified axis (or axes) evaluate to True.

    ((Similar to `numpy.all <https://numpy.org/doc/stable/reference/generated/numpy.all.html>`_))

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: ``[1], [1,2], [1,2,3], [1,2,3,4]``
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a boolean tile with the result. This return tile will have a shape of the input tile's shape with the specified axes removed.
    """
    ...

def all_reduce(x, op, program_axes, *, dtype=None, mask=None, parallel_reduce=True, asynchronous=False):
    r"""
    Apply reduce operation over multiple SPMD programs.

    :param x: a tile.
    :param op: numpy ALU operator to use to reduce over the input tile.
    :param program_axes: a single axis or a tuple of axes along which the reduction operation is performed.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :param parallel_reduce: optional boolean parameter whether to turn on parallel reduction. Enable parallel
                 reduction consumes additional memory.
    :param asynchronous: Defaults to False. If `True`, caller should synchronize before
                         reading final result, e.g. using `nki.sync_thread`.
    :return: the reduced resulting tile
    """
    ...

def arctan(x, *, dtype=None, mask=None):
    r"""
    Inverse tangent of the input, element-wise.

    ((Similar to `numpy.arctan <https://numpy.org/doc/stable/reference/generated/numpy.arctan.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has inverse tangent values of ``x``.
    """
    ...

def bitwise_and(x, y, *, dtype=None, mask=None):
    r"""
    Bitwise AND of the two inputs, element-wise.

    ((Similar to `numpy.bitwise_and <https://numpy.org/doc/stable/reference/generated/numpy.bitwise_and.html>`_))

    Computes the bit-wise AND of the underlying binary representation of the integers
    in the input tiles. This function implements the C/Python operator ``&``

    :param x: a tile or a scalar value of integer type.
    :param y: a tile or a scalar value of integer type. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has values ``x & y``.
    """
    ...

def bitwise_or(x, y, *, dtype=None, mask=None):
    r"""
    Bitwise OR of the two inputs, element-wise.

    ((Similar to `numpy.bitwise_or <https://numpy.org/doc/stable/reference/generated/numpy.bitwise_or.html>`_))

    Computes the bit-wise OR of the underlying binary representation of the integers
    in the input tiles. This function implements the C/Python operator ``|``

    :param x: a tile or a scalar value of integer type.
    :param y: a tile or a scalar value of integer type. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has values ``x | y``.
    """
    ...

def bitwise_xor(x, y, *, dtype=None, mask=None):
    r"""
    Bitwise XOR of the two inputs, element-wise.

    ((Similar to `numpy.bitwise_xor <https://numpy.org/doc/stable/reference/generated/numpy.bitwise_xor.html>`_))

    Computes the bit-wise XOR of the underlying binary representation of the integers
    in the input tiles. This function implements the C/Python operator ``^``

    :param x: a tile or a scalar value of integer type.
    :param y: a tile or a scalar value of integer type. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has values ``x ^ y``.
    """
    ...

def ceil(x, *, dtype=None, mask=None):
    r"""
    Ceiling of the input, element-wise.

    ((Similar to `numpy.ceil <https://numpy.org/doc/stable/reference/generated/numpy.ceil.html>`_))

    The ceil of the scalar x is the smallest integer i, such that i >= x.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has ceiling values of ``x``.
    """
    ...

def cos(x, *, dtype=None, mask=None):
    r"""
    Cosine of the input, element-wise.

    ((Similar to `numpy.cos <https://numpy.org/doc/stable/reference/generated/numpy.cos.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has cosine values of ``x``.
    """
    ...

def copy(x):
    ...

def divide(x, y, *, dtype=None, mask=None):
    r"""
    Divide the inputs, element-wise.

    ((Similar to `numpy.divide <https://numpy.org/doc/stable/reference/generated/numpy.divide.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has ``x / y``, element-wise.
    """
    ...

def dropout(x, rate, *, dtype=None, mask=None):
    r"""
    Randomly zeroes some of the elements of the input tile given a probability rate.

    :param x: a tile.
    :param rate: a scalar value or a tile with 1 element, with the probability rate.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with randomly zeroed elements of ``x``.
    """
    ...

def equal(x, y, *, dtype=bool, mask=None):
    r"""
    Element-wise boolean result of x == y.

    ((Similar to `numpy.equal <https://numpy.org/doc/stable/reference/generated/numpy.equal.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with boolean result of ``x == y`` element-wise.
    """
    ...

def erf(x, *, dtype=None, mask=None):
    r"""
    Error function of the input, element-wise.

    ((Similar to `torch.erf <https://pytorch.org/docs/master/generated/torch.erf.html>`_))

    ``erf(x) = 2/sqrt(pi)*integral(exp(-t**2), t=0..x)`` .

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has erf of ``x``.
    """
    ...

def erf_dx(x, *, dtype=None, mask=None):
    r"""
    Derivative of the Error function (erf) on the input, element-wise.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has erf_dx of ``x``.
    """
    ...

def exp(x, *, dtype=None, mask=None):
    r"""
    Exponential of the input, element-wise.

    ((Similar to `numpy.exp <https://numpy.org/doc/stable/reference/generated/numpy.exp.html>`_))

    The ``exp(x)`` is ``e^x`` where ``e`` is the Euler's number = 2.718281...

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has exponential values of ``x``.
    """
    ...

def floor(x, *, dtype=None, mask=None):
    r"""
    Floor of the input, element-wise.

    ((Similar to `numpy.floor <https://numpy.org/doc/stable/reference/generated/numpy.floor.html>`_))

    The floor of the scalar x is the largest integer i, such that i <= x.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has floor values of ``x``.
    """
    ...

def fmod(x, y, dtype=None, mask=None):
    r"""
    Floor-mod of ``x / y``, element-wise.

    The remainder has the same sign as the dividend x.
    It is equivalent to the Matlab(TM) rem function and should not be confused with the Python modulus operator x % y.

    ((Similar to `numpy.fmod <https://numpy.org/doc/stable/reference/generated/numpy.fmod.html>`_))

    :param x: a tile. If x is a scalar value it will be broadcast to the shape of y. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has values ``x fmod y``.
    """
    ...

def gelu(x, *, dtype=None, mask=None):
    r"""
    Gaussian Error Linear Unit activation function on the input, element-wise.

    ((Similar to `torch.nn.functional.gelu <https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has gelu of ``x``.
    """
    ...

def gelu_apprx_tanh(x, *, dtype=None, mask=None):
    r"""
    Gaussian Error Linear Unit activation function on the input, element-wise, with tanh approximation.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has gelu of ``x``.
    """
    ...

def gelu_dx(x, *, dtype=None, mask=None):
    r"""
    Derivative of Gaussian Error Linear Unit (gelu) on the input, element-wise.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has gelu_dx of ``x``.
    """
    ...

def greater(x, y, *, dtype=bool, mask=None):
    r"""
    Element-wise boolean result of x > y.

    ((Similar to `numpy.greater <https://numpy.org/doc/stable/reference/generated/numpy.greater.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with boolean result of ``x > y`` element-wise.
    """
    ...

def greater_equal(x, y, *, dtype=bool, mask=None):
    r"""
    Element-wise boolean result of x >= y.

    ((Similar to `numpy.greater_equal <https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with boolean result of ``x >= y`` element-wise.
    """
    ...

def invert(x, *, dtype=None, mask=None):
    r"""
    Bitwise NOT of the input, element-wise.

    ((Similar to `numpy.invert <https://numpy.org/doc/stable/reference/generated/numpy.invert.html>`_))

    Computes the bit-wise NOT of the underlying binary representation of the integers
    in the input tile. This ufunc implements the C/Python operator ``~``

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with bitwise NOT ``x`` element-wise.
    """
    ...

def left_shift(x, y, *, dtype=None, mask=None):
    r"""
    Bitwise left-shift x by y, element-wise.

    ((Similar to `numpy.left_shift <https://numpy.org/doc/stable/reference/generated/numpy.left_shift.html>`_))

    Computes the bit-wise left shift of the underlying binary representation of the integers
    in the input tiles. This function implements the C/Python operator ``<<``

    :param x: a tile or a scalar value of integer type.
    :param y: a tile or a scalar value of integer type. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has values ``x << y``.
    """
    ...

def less(x, y, *, dtype=bool, mask=None):
    r"""
    Element-wise boolean result of x < y.

    ((Similar to `numpy.less <https://numpy.org/doc/stable/reference/generated/numpy.less.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with boolean result of ``x < y`` element-wise.
    """
    ...

def less_equal(x, y, *, dtype=bool, mask=None):
    r"""
    Element-wise boolean result of x <= y.

    ((Similar to `numpy.less_equal <https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with boolean result of ``x <= y`` element-wise.
    """
    ...

def log(x, *, dtype=None, mask=None):
    r"""
    Natural logarithm of the input, element-wise.

    ((Similar to `numpy.log <https://numpy.org/doc/stable/reference/generated/numpy.log.html>`_))

    It is the inverse of the exponential function, such that: ``log(exp(x)) = x`` .
    The natural logarithm base is ``e``.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has natural logarithm values of ``x``.
    """
    ...

def logical_and(x, y, *, dtype=bool, mask=None):
    r"""
    Element-wise boolean result of x AND y.

    ((Similar to `numpy.logical_and <https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with boolean result of ``x AND y`` element-wise.
    """
    ...

def logical_not(x, *, dtype=bool, mask=None):
    r"""
    Element-wise boolean result of NOT x.

    ((Similar to `numpy.logical_not <https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with boolean result of ``NOT x`` element-wise.
    """
    ...

def logical_or(x, y, *, dtype=bool, mask=None):
    r"""
    Element-wise boolean result of x OR y.

    ((Similar to `numpy.logical_or <https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with boolean result of ``x OR y`` element-wise.
    """
    ...

def logical_xor(x, y, *, dtype=bool, mask=None):
    r"""
    Element-wise boolean result of x XOR y.

    ((Similar to `numpy.logical_xor <https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with boolean result of ``x XOR y`` element-wise.
    """
    ...

def loop_reduce(x, op, loop_indices, *, dtype=None, mask=None):
    r"""
    Apply reduce operation over a loop. This is an ideal instruction to compute a
    high performance reduce_max or reduce_min.

    Note: The destination tile is also the rhs input to ``op``. For example,

    .. code-block:: python

      b = nl.zeros((N_TILE_SIZE, M_TILE_SIZE), dtype=float32, buffer=nl.sbuf)
      for k_i in affine_range(NUM_K_BLOCKS):

        # Skipping over multiple nested loops here.
        # a, is a psum tile from a matmul accumulation group.
        b = nl.loop_reduce(a, op=np.add, loop_indices=[k_i], dtype=nl.float32)

    is the same as:

    .. code-block:: python

      b = nl.zeros((N_TILE_SIZE, M_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
      for k_i in affine_range(NUM_K_BLOCKS):

        # Skipping over multiple nested loops here.
        # a, is a psum tile from a matmul accumulation group.
        b = nisa.tensor_tensor(data1=b, data2=a, op=np.add, dtype=nl.float32)

    If you are trying to use this instruction only for accumulating results on SBUF, consider
    simply using the ``+=`` operator instead.

    The ``loop_indices`` list enables the compiler to recognize which loops this reduction can be
    optimized across as part of any aggressive loop-level optimizations it may perform.

    :param x: a tile.
    :param op: numpy ALU operator to use to reduce over the input tile.
    :param loop_indices: a single loop index or a tuple of loop indices along which the reduction operation is performed.
                        Can be numbers or loop_index objects coming from ``nl.affine_range``.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: the reduced resulting tile
    """
    ...

def matmul(x, y, *, transpose_x=False, mask=None):
    r"""
    ``x @ y`` matrix multiplication of ``x`` and ``y``.

    ((Similar to `numpy.matmul <https://numpy.org/doc/stable/reference/generated/numpy.matmul.html>`_))

    .. note::
        For optimal performance on hardware, use :func:`nki.isa.nc_matmul`.
        You can also use ``nki.isa.nc_matmul`` also to access low-level features
        of the Tensor Engine.

    .. note::
        Implementation details:
        ``nc_matmul`` is neuron specific customized implementation of matmul that computes ``x.T @ y``,
        as a result, ``matmul(x, y)`` lowers to ``nc_matmul(transpose(x), y)``.
        To avoid this extra transpose instruction being inserted,
        use ``x.T`` and ``transpose_x=True`` inputs to this ``matmul``.

    :param x: a tile on SBUF (partition dimension ``<= 128``, free dimension ``<= 128``),
              ``x``'s free dimension must match ``y``'s partition dimension.
    :param y: a tile on SBUF (partition dimension ``<= 128``, free dimension ``<= 512``)
    :param transpose_x: Defaults to False. If ``True``, ``x`` is treated as already transposed.
                        If ``False``, an additional transpose will be inserted
                        to make ``x``'s partition dimension the contract dimension of the matmul
                        to align with the Tensor Engine.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)

    :return: ``x @ y`` or ``x.T @ y`` if ``transpose_x=True``
    """
    ...

def max(x, axis, *, dtype=None, mask=None, keepdims=False):
    r"""
    Maximum of elements along the specified axis (or axes) of the input.

    ((Similar to `numpy.max <https://numpy.org/doc/stable/reference/generated/numpy.max.html>`_))

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: ``[1], [1,2], [1,2,3], [1,2,3,4]``
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                     With this option, the result will broadcast correctly against the input array.
    :return: a tile with the maximum of elements along the provided axis. This return tile will have a shape of the input
             tile's shape with the specified axes removed.
    """
    ...

def maximum(x, y, *, dtype=None, mask=None):
    r"""
    Maximum of the inputs, element-wise.

    ((Similar to `numpy.maximum <https://numpy.org/doc/stable/reference/generated/numpy.maximum.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has the maximum of each elements from x and y.
    """
    ...

def mean(x, axis, *, dtype=None, mask=None, keepdims=False):
    r"""
    Arithmetic mean along the specified axis (or axes) of the input.

    ((Similar to `numpy.mean <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_))

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: ``[1], [1,2], [1,2,3], [1,2,3,4]``
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with the average of elements along the provided axis. This return tile will have a shape of the input
             tile's shape with the specified axes removed.
             ``float32`` intermediate and return values are used for integer inputs.
    """
    ...

def min(x, axis, *, dtype=None, mask=None, keepdims=False):
    r"""
    Minimum of elements along the specified axis (or axes) of the input.

    ((Similar to `numpy.min <https://numpy.org/doc/stable/reference/generated/numpy.min.html>`_))

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: ``[1], [1,2], [1,2,3], [1,2,3,4]``
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                     With this option, the result will broadcast correctly against the input array.
    :return: a tile with the minimum of elements along the provided axis. This return tile will have a shape of the input
             tile's shape with the specified axes removed.
    """
    ...

def minimum(x, y, *, dtype=None, mask=None):
    r"""
    Minimum of the inputs, element-wise.

    ((Similar to `numpy.minimum <https://numpy.org/doc/stable/reference/generated/numpy.minimum.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has the minimum of each elements from x and y.
    """
    ...

def mish(x, *, dtype=None, mask=None):
    r"""
    Mish activation function on the input, element-wise.

    Mish: A Self Regularized Non-Monotonic Neural Activation Function is defined as:

    .. math::
          mish(x) = x * tanh(softplus(x))

    see: https://arxiv.org/abs/1908.08681

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has mish of ``x``.
    """
    ...

def mod(x, y, dtype=None, mask=None):
    r"""
    Integer Mod of ``x / y``, element-wise

    Computes the remainder complementary to the floor_divide function.
    It is equivalent to the Python modulus x % y and has the same sign as the divisor y.

    ((Similar to `numpy.mod <https://numpy.org/doc/stable/reference/generated/numpy.mod.html>`_))

    :param x: a tile. If x is a scalar value it will be broadcast to the shape of y. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has values ``x mod y``.
    """
    ...

def multiply(x, y, *, dtype=None, mask=None):
    r"""
    Multiply the inputs, element-wise.

    ((Similar to `numpy.multiply <https://numpy.org/doc/stable/reference/generated/numpy.multiply.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has ``x * y``, element-wise.
    """
    ...

def negative(x, *, dtype=None, mask=None):
    r"""
    Numerical negative of the input, element-wise.

    ((Similar to `numpy.negative <https://numpy.org/doc/stable/reference/generated/numpy.negative.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has numerical negative values of ``x``.
    """
    ...

def not_equal(x, y, *, dtype=bool, mask=None):
    r"""
    Element-wise boolean result of x != y.

    ((Similar to `numpy.not_equal <https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with boolean result of ``x != y`` element-wise.
    """
    ...

def power(x, y, *, dtype=None, mask=None):
    r"""
    Elements of x raised to powers of y, element-wise.

    ((Similar to `numpy.power <https://numpy.org/doc/stable/reference/generated/numpy.power.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has values ``x`` to the power of ``y``.
    """
    ...

def prod(x, axis, *, dtype=None, mask=None, keepdims=False):
    r"""
    Product of elements along the specified axis (or axes) of the input.

    ((Similar to `numpy.prod <https://numpy.org/doc/stable/reference/generated/numpy.prod.html>`_))

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: ``[1], [1,2], [1,2,3], [1,2,3,4]``
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                     With this option, the result will broadcast correctly against the input array.
    :return: a tile with the product of elements along the provided axis. This return tile will have a shape of the input
             tile's shape with the specified axes removed.
    """
    ...

def reciprocal(x, *, dtype=None, mask=None):
    r"""
    Reciprocal of the the input, element-wise.

    ((Similar to `numpy.reciprocal <https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html>`_))

    ``reciprocal(x) = 1 / x``

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has reciprocal values of ``x``.
    """
    ...

def relu(x, *, dtype=None, mask=None):
    r"""
    Rectified Linear Unit activation function on the input, element-wise.

    ``relu(x) = (x)+ = max(0,x)``

    ((Similar to `torch.nn.functional.relu <https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has relu of ``x``.
    """
    ...

def right_shift(x, y, *, dtype=None, mask=None):
    r"""
    Bitwise right-shift x by y, element-wise.

    ((Similar to `numpy.right_shift <https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html>`_))

    Computes the bit-wise right shift of the underlying binary representation of the integers
    in the input tiles. This function implements the C/Python operator ``>>``

    :param x: a tile or a scalar value of integer type.
    :param y: a tile or a scalar value of integer type. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has values ``x >> y``.
    """
    ...

def rms_norm(x, w, axis, n, epsilon=1e-06, *, dtype=None, compute_dtype=None, mask=None):
    r"""
    Apply Root Mean Square Layer Normalization.

    :param x: input tile
    :param w: weight tile
    :param axis: axis along which to compute the root mean square (rms) value
    :param n: total number of values to calculate rms
    :param epsilon: epsilon value used by rms calculation to avoid divide-by-zero
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param compute_dtype: (optional) dtype for the internal computation -
                          *currently `dtype` and `compute_dtype` behave the same, both sets internal compute and return dtype.*
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: `` x / RMS(x) * w ``
    """
    ...

def rsqrt(x, *, dtype=None, mask=None):
    r"""
    Reciprocal of the square-root of the input, element-wise.

    ((Similar to `torch.rsqrt <https://pytorch.org/docs/master/generated/torch.rsqrt.html>`_))

    ``rsqrt(x) = 1 / sqrt(x)``

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has reciprocal square-root values of ``x``.
    """
    ...

def sigmoid(x, *, dtype=None, mask=None):
    r"""
    Logistic sigmoid activation function on the input, element-wise.

    ((Similar to `torch.nn.functional.sigmoid <https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html>`_))

    ``sigmoid(x) = 1/(1+exp(-x))``

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has sigmoid of ``x``.
    """
    ...

def sign(x, *, dtype=None, mask=None):
    r"""
    Sign of the numbers of the input, element-wise.

    ((Similar to `numpy.sign <https://numpy.org/doc/stable/reference/generated/numpy.sign.html>`_))

    The sign function returns ``-1`` if ``x < 0``, ``0`` if ``x==0``, ``1`` if ``x > 0``.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has sign values of ``x``.
    """
    ...

def silu(x, *, dtype=None, mask=None):
    r"""
    Sigmoid Linear Unit activation function on the input, element-wise.

    ((Similar to `torch.nn.functional.silu <https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has silu of ``x``.
    """
    ...

def silu_dx(x, *, dtype=None, mask=None):
    r"""
    Derivative of Sigmoid Linear Unit activation function on the input, element-wise.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has silu_dx of ``x``.
    """
    ...

def sin(x, *, dtype=None, mask=None):
    r"""
    Sine of the input, element-wise.

    ((Similar to `numpy.sin <https://numpy.org/doc/stable/reference/generated/numpy.sin.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has sine values of ``x``.
    """
    ...

def softmax(x, axis, *, dtype=None, compute_dtype=None, mask=None):
    r"""
    Softmax activation function on the input, element-wise.

    ((Similar to `torch.nn.functional.softmax <https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html>`_))

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: ``[1], [1,2], [1,2,3], [1,2,3,4]``
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param compute_dtype: (optional) dtype for the internal computation -
                          *currently `dtype` and `compute_dtype` behave the same, both sets internal compute and return dtype.*
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has softmax of ``x``.
    """
    ...

def softplus(x, *, dtype=None, mask=None):
    r"""
    Softplus activation function on the input, element-wise.

    Softplus is a smooth approximation to the ReLU activation, defined as:

    ``softplus(x) = log(1 + exp(x))``

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has softplus of ``x``.
    """
    ...

def sqrt(x, *, dtype=None, mask=None):
    r"""
    Non-negative square-root of the input, element-wise.

    ((Similar to `numpy.sqrt <https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has square-root values of ``x``.
    """
    ...

def square(x, *, dtype=None, mask=None):
    r"""
    Square of the input, element-wise.

    ((Similar to `numpy.square <https://numpy.org/doc/stable/reference/generated/numpy.square.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has square of ``x``.
    """
    ...

def subtract(x, y, *, dtype=None, mask=None):
    r"""
    Subtract the inputs, element-wise.

    ((Similar to `numpy.subtract <https://numpy.org/doc/stable/reference/generated/numpy.subtract.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has ``x - y``, element-wise.
    """
    ...

def sum(x, axis, *, dtype=None, mask=None, keepdims=False):
    r"""
    Sum of elements along the specified axis (or axes) of the input.

    ((Similar to `numpy.sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_))

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: ``[1], [1,2], [1,2,3], [1,2,3,4]``
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                     With this option, the result will broadcast correctly against the input array.
    :return: a tile with the sum of elements along the provided axis. This return tile will have a shape of the input
             tile's shape with the specified axes removed.
    """
    ...

def tan(x, *, dtype=None, mask=None):
    r"""
    Tangent of the input, element-wise.

    ((Similar to `numpy.tan <https://numpy.org/doc/stable/reference/generated/numpy.tan.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has tangent values of ``x``.
    """
    ...

def tanh(x, *, dtype=None, mask=None):
    r"""
    Hyperbolic tangent of the input, element-wise.

    ((Similar to `numpy.tanh <https://numpy.org/doc/stable/reference/generated/numpy.tanh.html>`_))

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has hyperbolic tangent values of ``x``.
    """
    ...

def transpose(x, *, dtype=None, mask=None):
    r"""
    Transposes a 2D tile between its partition and free dimension.

    :param x: 2D input tile
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has the values of the input tile with its partition and free dimensions swapped.
    """
    ...

def trunc(x, *, dtype=None, mask=None):
    r"""
    Truncated value of the input, element-wise.

    ((Similar to `numpy.trunc <https://numpy.org/doc/stable/reference/generated/numpy.trunc.html>`_))

    The truncated value of the scalar x is the nearest integer i which is closer to zero than x is.
    In short, the fractional part of the signed number x is discarded.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile that has truncated values of ``x``.
    """
    ...

def var(x, axis, *, dtype=None, mask=None):
    r"""
    Variance along the specified axis (or axes) of the input.

    ((Similar to `numpy.var <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_))

    :param x: a tile.
    :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: ``[1], [1,2], [1,2,3], [1,2,3,4]``
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :return: a tile with the variance of the elements along the provided axis. This return tile will have a shape of the input
             tile's shape with the specified axes removed.
    """
    ...

sbuf = 'sbuf'
"""State Buffer - Only visible to each individual kernel instance in the SPMD grid"""

psum = 'psum'
"""PSUM - Only visible to each individual kernel instance in the SPMD grid"""

hbm = 'hbm'
"""HBM - Alias of private_hbm"""

shared_hbm = 'shared_hbm'
"""Shared HBM - Visible to all kernel instances in the SPMD grid"""

private_hbm = 'private_hbm'
"""HBM - Only visible to each individual kernel instance in the SPMD grid"""

def num_programs(axes=None):
    r"""
    Number of SPMD programs along the given axes in the launch grid. If ``axes`` is not provided,
    returns the total number of programs.

    :param axes: The axes of the ND launch grid. If not provided, returns the total number of programs along the entire launch grid.
    :return:     The number of SPMD(single process multiple data) programs along ``axes`` in the launch grid
    """
    ...

def program_id(axis):
    r"""
    Index of the current SPMD program along the given axis in the launch grid.

    :param axis: The axis of the ND launch grid.
    :return:     The program id along ``axis`` in the launch grid
    """
    ...

def program_ndim():
    r"""
    Number of dimensions in the SPMD launch grid.

    :return:    The number of dimensions in the launch grid, i.e. the number of axes
    """
    ...

