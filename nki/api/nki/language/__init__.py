"""Auto-generated stub file"""
from enum import Enum
import nki.language as nl
import ml_dtypes

class NKIObject:

    ...

bool_ = 'bool'
"""Boolean (True or False) stored as a byte"""

int8 = 'int8'
"""8-bit signed integer number"""

int16 = 'int16'
"""16-bit signed integer number"""

int32 = 'int32'
"""32-bit signed integer number"""

uint8 = 'uint8'
"""8-bit unsigned integer number"""

uint16 = 'uint16'
"""16-bit unsigned integer number"""

uint32 = 'uint32'
"""32-bit unsigned integer number"""

float16 = 'float16'
"""16-bit floating-point number"""

float32 = 'float32'
"""32-bit floating-point number"""

bfloat16 = 'bfloat16'
"""16-bit floating-point number (1S,8E,7M)"""

tfloat32 = 'tfloat32'
"""32-bit floating-point number (1S,8E,10M)"""

float8_e4m3 = 'float8_e4m3'
"""8-bit floating-point number (1S,4E,3M)"""

float8_e5m2 = 'float8_e5m2'
"""8-bit floating-point number (1S,5E,2M)"""

float8_e4m3fn = 'float8_e4m3fn'
"""8-bit floating-point number (1S,4E,3M), Extended range: no inf, NaN represented by 0bS111'1111"""

float8_e5m2_x4 = 'float8_e5m2_x4'
"""4x packed float8_e5m2 elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""

float8_e4m3fn_x4 = 'float8_e4m3fn_x4'
"""4x packed float8_e4m3fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""

float4_e2m1fn_x4 = 'float4_e2m1fn_x4'
"""4x packed float4_e2m1fn elements, custom data type for nki.isa.nc_matmul_mx on NeuronCore-v4"""

def ndarray(shape, dtype, buffer=None, name=""):
    r"""
    Create a new tensor of given shape and dtype on the specified buffer.

    ((Similar to `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_))

    :param shape: the shape of the tensor.
    :param dtype: the data type of the tensor (see :ref:`nki-dtype` for more information).
    :param buffer: the specific buffer (ie, :doc:`sbuf<nki.language.sbuf>`, :doc:`psum<nki.language.psum>`, :doc:`hbm<nki.language.hbm>`), defaults to :doc:`sbuf<nki.language.sbuf>`.
    :param name: the name of the tensor. The ``name`` parameter has to be unique for tensors on each Physical NeuronCore(PNC) 
                within each Logical NeuronCore(LNC). It is optional for SRAM tensors, IO tensors, and any HBM tensors that are
                only visible to one Physical NeuronCore. 
                For ``shared_hbm`` tensors that are not used as kernel 
                inputs or outputs, ``name`` must be specified. In addition, the compiler uses the ``name``
                to link non-IO ``shared_hbm`` tensors among PNCs. In other word, ``shared_hbm`` tensors
                will point to the same underlying memory as long as they have the same name,
                even if the tensors appear in different control flow. 
    :return: a new tensor allocated on the buffer.

    """
    ...

def zeros(shape, dtype, buffer=None, name=""):
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
    Create a tensor filled with compile-time constant data.

    This function creates a tensor that contains constant data specified by a trace-time tensor.
    The constant data is evaluated at compile time and shared across all instances where the
    same constant is used, making it memory efficient for frequently used constant values.

    :param constant: A trace-time tensor containing the constant data to be filled into the output tensor.
                    This can be created using functions from :mod:`nki.tensor` such as 
                    :func:`nki.tensor.zeros`, :func:`nki.tensor.identity`, or :func:`nki.tensor.arange`.
    :type constant: nki.tensor.TraceTimeTensor
    :param dtype: The data type of the output tensor. Must be specified.
                 Only types that can be serialized to npy files are supported.
                 See :ref:`nki-dtype` for supported data types.
    :type dtype: nki.language dtype
    :return: A tensor containing the constant data with the specified dtype.
    :rtype: Tensor

    .. note::
       The constant tensor is shared across all uses of the same constant data and dtype,
       which helps reduce memory usage in the compiled kernel.

    **Examples:**

    Create a constant identity matrix::

        import nki.tensor as ntensor
        import nki.language as nl
        
        # Create a 128x128 identity matrix as a shared constant
        identity_matrix = nl.shared_constant(
            ntensor.identity(128, dtype=nl.int8), 
            dtype=nl.float16
        )

    Create a constant tensor with sequential values::

        # Create a constant tensor with values [0, 1, 2, ..., 31]
        sequential_values = nl.shared_constant(
            ntensor.arange(0, 32, 1, dtype=nl.int32),
            dtype=nl.float32
        )

    Create a constant tensor with arithmetic operations::

        # Create a constant tensor filled with ones
        ones_tensor = nl.shared_constant(
            ntensor.zeros((64, 64), dtype=nl.int8) + 1,
            dtype=nl.int16
        )
    """
    ...

def shared_identity_matrix(n, dtype="uint8"):
    r"""
    Create a new identity tensor with specified data type.

    This function has the same behavior to :doc:`nki.language.shared_constant <nki.language.shared_constant>` but
    is preferred if the constant matrix is an identity matrix. The
    compiler will reuse all the identity matrices of the same
    dtype in the graph to save space.

    :param n: the number of rows(and columns) of the returned identity matrix
    :param dtype: the data type of the tensor, default to be ``nl.uint8`` (see :ref:`nki-dtype` for more information).
    :return: a tensor which contains the identity tensor
    """
    ...

def affine_range(start, stop=None, step=1):
    r"""
    Create a sequence of numbers for use as **parallel** loop iterators in NKI. ``affine_range`` should be the default
    loop iterator choice, when there is **no** loop carried dependency. Note, associative reductions are **not** considered
    loop carried dependencies in this context. A concrete examplesof associative reduction
    is the set of :doc:`nisa.nc_matmul <nki.isa.nc_matmul>`calls accumulating into the same
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
    r"""Tile size constants."""

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


def abs(x, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
    ...

def add(x, y, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
    ...

def bitwise_and(x, y, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
    ...

def bitwise_or(x, y, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
    ...

def bitwise_xor(x, y, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
    ...

def divide(x, y, dtype=None):
    r"""
    Divide the inputs, element-wise.

    ((Similar to `numpy.divide <https://numpy.org/doc/stable/reference/generated/numpy.divide.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :return: a tile that has ``x / y``, element-wise.
    """
    ...

def equal(x, y, dtype=bool):
    r"""This operation is not supported in the current release of NKI."""
    ...

def gelu_apprx_sigmoid(x, dtype=None):
    r"""
    Gaussian Error Linear Unit activation function on the input, element-wise, with sigmoid approximation.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has gelu of ``x``.
    """
    ...

def gelu_apprx_sigmoid_dx(x, dtype=None):
    r"""
    Derivative of Gaussian Error Linear Unit activation function on the input, element-wise, with sigmoid approximation.

    :param x: a tile.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: a tile that has gelu_dx of ``x``.
    """
    ...

def greater(x, y, dtype=bool):
    r"""This operation is not supported in the current release of NKI."""
    ...

def greater_equal(x, y, dtype=bool):
    r"""This operation is not supported in the current release of NKI."""
    ...

def invert(x, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
    ...

def left_shift(x, y, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
    ...

def less(x, y, dtype=bool):
    r"""This operation is not supported in the current release of NKI."""
    ...

def less_equal(x, y, dtype=bool):
    r"""This operation is not supported in the current release of NKI."""
    ...

def logical_and(x, y, dtype=bool):
    r"""This operation is not supported in the current release of NKI."""
    ...

def logical_not(x, dtype=bool):
    r"""This operation is not supported in the current release of NKI."""
    ...

def logical_or(x, y, dtype=bool):
    r"""This operation is not supported in the current release of NKI."""
    ...

def logical_xor(x, y, dtype=bool):
    r"""This operation is not supported in the current release of NKI."""
    ...

def maximum(x, y, dtype=None):
    r"""
    Maximum of the inputs, element-wise.

    ((Similar to `numpy.maximum <https://numpy.org/doc/stable/reference/generated/numpy.maximum.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :return: a tile that has the maximum of each elements from x and y.
    """
    ...

def minimum(x, y, dtype=None):
    r"""
    Minimum of the inputs, element-wise.

    ((Similar to `numpy.minimum <https://numpy.org/doc/stable/reference/generated/numpy.minimum.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :return: a tile that has the minimum of each elements from x and y.
    """
    ...

def multiply(x, y, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
    ...

def not_equal(x, y, dtype=bool):
    r"""This operation is not supported in the current release of NKI."""
    ...

def power(x, y, dtype=None):
    r"""
    Elements of x raised to powers of y, element-wise.

    ((Similar to `numpy.power <https://numpy.org/doc/stable/reference/generated/numpy.power.html>`_))

    :param x: a tile or a scalar value.
    :param y: a tile or a scalar value. ``x.shape`` and ``y.shape`` must be `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ to a common shape, that will become the shape of the output.
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
    :return: a tile that has values ``x`` to the power of ``y``.
    """
    ...

def reciprocal(x, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
    ...

def right_shift(x, y, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
    ...

def rsqrt(x, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
    ...

def subtract(x, y, dtype=None):
    r"""This operation is not supported in the current release of NKI."""
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

def device_print(print_prefix, tensor):
    r"""
    Print a message with a string ``print_prefix`` followed by the value of a tile ``tensor``.

    By default, using this function will not result in your tensors being printed out. When running your kernel,
    you need to define the environment variable ``NEURON_RT_DEBUG_OUTPUT_DIR`` and point it to a directory that will
    store the tensor data grouped by prefix each time the device_print instruction is executed.

    The structure of the directory will be ``<print_prefix>/core_<logical core id>/<iteration>/...``.

    .. code-block:: python
        :caption: Example usage
        :emphasize-lines: 7

        import nki.isa as nisa
        import nki.language as nl

        def my_nki_kernel(input_tensor):
            a_tile = sbuf.view(input_tensor.dtype, input_tensor.shape)
            nisa.dma_copy(a_tile, input_tensor)
            nl.device_print("a_tile", a_tile)

            ...

    .. warning::
        This feature is only available when using the NxD Inference library.

    :param print_prefix:  prefix of the print message. This string is evaluated at trace time and must be a constant expression.
    :type print_prefix:   str
    :param tensor:        tensor to print out. Can be in SBUF or HBM.
    :return:              None
    """
    ...

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

