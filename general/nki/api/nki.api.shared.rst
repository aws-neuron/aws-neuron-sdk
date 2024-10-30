=======================
NKI API Common Fields
=======================

.. _nki-dtype:

Supported Data Types
========================

:ref:`tbl-dtype` below lists all supported data types by NKI.
Almost all the NKI APIs accept a data type field, ``dtype``, which can either be
a ``NumPy`` equivalent type or a ``nki.language`` data type.

.. _tbl-dtype:

.. table:: Table 1: Supported Data Types by NKI

  +------------------------+----------------------------+-------------------------------------------------+
  |                        | Data Type                  | Accepted ``dtype`` Field by NKI APIs            |
  +========================+============================+=================================================+
  |                        | 8-bit unsigned integer     | ``nki.language.uint8`` or ``numpy.uint8``       |
  |                        +----------------------------+-------------------------------------------------+
  |                        | 8-bit signed integer       | ``nki.language.int8`` or ``numpy.int8``         |
  |                        +----------------------------+-------------------------------------------------+
  | Integer                | 16-bit unsigned integer    | ``nki.language.uint16`` or ``numpy.uint16``     |
  |                        +----------------------------+-------------------------------------------------+
  |                        | 16-bit signed integer      | ``nki.language.int16`` or ``numpy.int16``       |
  |                        +----------------------------+-------------------------------------------------+
  |                        | 32-bit unsigned integer    | ``nki.language.uint32`` or ``numpy.uint32``     |
  |                        +----------------------------+-------------------------------------------------+
  |                        | 32-bit signed integer      | ``nki.language.int32`` or ``numpy.int32``       |
  +------------------------+----------------------------+-------------------------------------------------+
  |                        | float8 (1S,4E,3M) [#1]_    | ``nki.language.float8_e4m3``                    |
  |                        +----------------------------+-------------------------------------------------+
  |                        | float16 (1S,5E,10M)        | ``nki.language.float16`` or ``numpy.float16``   |
  |                        +----------------------------+-------------------------------------------------+
  | Float                  | bfloat16 (1S,8E,7M)        | ``nki.language.bfloat16``                       |
  |                        +----------------------------+-------------------------------------------------+
  |                        | tfloat32 (1S,8E,10M)       | ``nki.language.tfloat32``                       |
  |                        +----------------------------+-------------------------------------------------+
  |                        | float32 (1S,8E,23M)        | ``nki.language.float32`` or ``numpy.float32``   |
  +------------------------+----------------------------+-------------------------------------------------+
  | Boolean                | boolean stored as uint8    | ``nki.language.bool_`` or ``numpy.bool``        |
  +------------------------+----------------------------+-------------------------------------------------+

.. _nki-aluop:

Supported Math Operators
============================

:ref:`tbl-aluop` below lists all the mathematical operator primitives supported by NKI.
Many :ref:`nki.isa <nki-isa>` APIs (instructions) supported by Vector Engine
allow programmable operators through the ``op`` field. The supported operators fall into two categories:
*bitvec* and *arithmetic*. In general, instructions using *bitvec* operators expect integer data types
and treat input elements as bit patterns. On the hand, instructions using *arithmetic* operators
accept any valid NKI data types and convert input elements into float32 before performing the operators.

.. _tbl-aluop:
.. table:: Table 2: Supported Math Operators by NKI

  +------------------------+----------------------------+--------------------------------------+------------------------+
  |                        | Operator                   | Accepted ``op`` by Vector Engine     | Legal Reduction ``op`` |
  +========================+============================+======================================+========================+
  |                        | Bitwise Not                | ``numpy.invert``                     | N                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Bitwise And                | ``numpy.bitwise_and``                | Y                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Bitwise Or                 | ``numpy.bitwise_or``                 | Y                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  | Bitvec                 | Bitwise Xor                | ``numpy.bitwise_xor``                | Y                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Arithmetic Shift Left      | ``numpy.left_shift``                 | N                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Arithmetic Shift Right     |  Support coming soon.                | N                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Logical Shift Left         | ``numpy.left_shift``                 | N                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Logical Shift Right        | ``numpy.right_shift``                | N                      |
  +------------------------+----------------------------+--------------------------------------+------------------------+
  |                        | Add                        | ``numpy.add``                        | Y                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Subtract                   | ``numpy.subtract``                   | Y                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Multiply                   | ``numpy.multiply``                   | Y                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Max                        | ``numpy.maximum``                    | Y                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Min                        | ``numpy.minimum``                    | Y                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Is Equal to                | ``numpy.equal``                      | N                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Is Not Equal to            | ``numpy.not_equal``                  | N                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  | Arithmetic             | Is Greater than or Equal to| ``numpy.greater_equal``              | N                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Is Greater than to         | ``numpy.greater``                    | N                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Is Less than or Equal to   | ``numpy.less_equal``                 | N                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Is Less than               | ``numpy.less``                       | N                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Logical Not                | ``numpy.logical_not``                | N                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Logical And                | ``numpy.logical_and``                | Y                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Logical Or                 | ``numpy.logical_or``                 | Y                      |
  |                        +----------------------------+--------------------------------------+------------------------+
  |                        | Logical Xor                | ``numpy.logical_xor``                | Y                      |
  +------------------------+----------------------------+--------------------------------------+------------------------+

.. _nki-act-func:

Supported Activation Functions
==============================
:ref:`tbl-act-func` below lists all the activation function supported by the ``nki.isa.activation`` API. These
activation functions are approximated with piece-wise polynomials on Scalar Engine.


.. _tbl-act-func:
.. list-table:: Table 3: List of Supported Activation Functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Accepted ``op`` by Scalar Engine
   * - Identity
     - Support coming soon.
   * - Square
     - ``nki.language.square`` or ``numpy.square``
   * - Sigmoid
     - ``nki.language.sigmoid``
   * - Relu
     - ``nki.language.relu``
   * - Gelu
     - ``nki.language.gelu``
   * - Gelu Derivative
     - ``nki.language.gelu_dx``
   * - Gelu with Tanh Approximation
     - ``nki.language.gelu_apprx_tanh``
   * - Silu
     - Support coming soon.
   * - Silu Derivative
     - Support coming soon.
   * - Tanh
     - ``nki.language.tanh`` or ``numpy.tanh``
   * - Square Root
     - ``nki.language.sqrt`` or ``numpy.sqrt``
   * - Reverse Square Root
     - ``nki.language.rsqrt``
   * - Exponential
     - ``nki.language.exp`` or ``numpy.exp``
   * - Softplus
     - ``nki.language.softplus``
   * - Mish
     - ``nki.language.mish``
   * - Natural Log
     - ``nki.language.log`` or ``numpy.log``
   * - Erf
     - ``nki.language.erf``
   * - Erf Derivative
     - ``nki.language.erf_dx``
   * - Sine
     - ``nki.language.sin`` or ``numpy.sin``
   * - Cosine
     - ``nki.language.cos`` or ``numpy.cos``
   * - Arctan
     - ``nki.language.arctan`` or ``numpy.arctan``
   * - Sign
     - ``nki.language.sign`` or ``numpy.sign``


.. _nki-mask:

NKI API Masking
===============

All :ref:`nki.language <nki-language>` and :ref:`nki.isa <nki-isa>` APIs accept
an optional input field, ``mask``.
The ``mask`` field is an execution predicate known at compile-time, which informs the
compiler to skip generating the instruction or generate the instruction with a smaller
input tile shape. Masking is handled completely by Neuron compiler and hence does not incur
any performance overhead in the generated instructions.

The ``mask`` can be created using comparison expressions (e.g., ``a < b``) or multiple
comparison expressions concatenated with ``&`` (e.g., ``(a < b) & (c > d)``).
The left- or right-handside expression
of each comparator must be an affine expression of ``nki.language.arange()``,
``nki.language.affine_range()`` or ``nki.language.program_id()`` .
Each comparison expression should indicate which range of
indices along one of the input tile axes should be valid for the computation. For example,
assume we have an input tile ``in_tile`` of shape ``(128, 512)``, and we would like to perform a square
operation on this tile for elements in ``[0:64, 0:256]``, we can invoke the ``nki.language.square()``
API using the following:


.. literalinclude:: ../test/test_nki_mask.py
  :language: python
  :lines: 9, 22-24, 26

The above example will be lowered into a hardware ISA instruction that only processes
64x256 elements by Neuron Compiler.

The above ``mask`` definition works for most APIs where there is only one input tile or both input tiles
share the same axes. One exception is the ``nki.language.matmul`` and similarly ``nki.isa.nc_matmul``
API, where the two input tiles ``lhs`` and ``rhs`` contain three unique axes:

1. The contraction axis: both ``lhs`` and ``rhs`` partition axis (``lhs_rhs_p``)
2. The first axis of matmul output: ``lhs`` free axis (``lhs_f``)
3. The second axis of matmul output: ``rhs`` free axis (``rhs_f``)

As an example, let's assume we have ``lhs`` tile of shape ``(sz_p, sz_m)``
and ``rhs`` tile of shape ``(sz_p, sz_n)``,
and we call ``nki.language.matmul`` to calculate an output tile of shape ``(sz_m, sz_n)``:

.. code-block:: python

  import neuronxcc.nki.language as nl

  i_p = nl.arange(sz_p)[:, None]

  i_lhs_f = nl.arange(sz_m)[None, :]
  i_rhs_f = nl.arange(sz_n)[None, :] # same as `i_rhs_f = i_lhs_f`

  result = nl.matmul(lhs[i_p, i_lhs_f], rhs[i_p, i_rhs_f], transpose_x=True)

Since both ``i_lhs_f`` and ``i_rhs_f`` are identical to the Neuron Compiler, the Neuron Compiler
cannot distinguish the two input axes if they were to be passed into the ``mask`` field directly.

Therefore, we introduce "operand masking" syntax for matmult APIs to let users to precisely define
the masking on the inputs to the matmult APIs (currently only matmult APIs support operand masking,
subject to changes in future releases). Let's assume we need to constraint ``sz_m <= 64`` and
``sz_n <= 256``:

.. code-block:: python

  import neuronxcc.nki.language as nl

  i_p = nl.arange(sz_p)[:, None]

  i_lhs_f = nl.arange(sz_m)[None, :]
  i_rhs_f = nl.arange(sz_n)[None, :] # same as `i_rhs_f = i_lhs_f`

  i_lhs_f_virtual = nl.arange(sz_m)[None, :, None]

  result = nl.matmul(lhs_T[i_lhs_f <= 64], rhs[i_rhs_f <= 256], transpose_x=True)

There are two notable use cases for masking:

1. When the tiling factor doesn't divide the tensor dimension sizes
2. Skip ineffectual instructions that compute known output values

We will present an example of the first use case below.
Let's assume we would like to evaluate the exponential function on an input tensor
of shape ``[sz_p, sz_f]`` from HBM. Since the input to
``nki.language.load/nki.language.store/nki.language.exp`` expects a tile with a
partition axis size not exceeding
``nki.language.tile_size.pmax == 128``, we should loop over the input tensor using a tile
size of ``[nki.language.tile_size.pmax, sz_f]``.

However, ``sz_p`` is not guaranteed to be an
integer multiple of ``nki.language.tile_size.pmax``. In this case, one option is to write a loop
with trip count of ``sz_p // nki.language.tile_size.pmax`` followed by a single invocation
of ``nki.language.exp`` with an input tile of shape ``[sz_p % nki.language.tile_size.pmax, sz_f]``.
This effectively "unrolls" the last instance of tile computation, which could lead to messy code
in a complex kernel. Using masking here will allow us to avoid such unrolling, as illustrated in
the example below:

.. code-block:: python

  import neuronxcc.nki.language as nl
  from torch_neuronx import nki_jit

  @nki_jit
  def tensor_exp_kernel_(in_tensor, out_tensor):

  sz_p, sz_f = in_tensor.shape

  i_f = nl.arange(sz_f)[None, :]

  trip_count = math.ceil(sz_p/nl.tile_size.pmax)

  for p in nl.affine_range(trip_count):
      # Generate tensor indices for the input/output tensors
      # pad index to pmax, for simplicity
      i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]

      # Load input data from external memory to on-chip memory
      # only read up to sz_p
      in_tile = nl.load(in_tensor[i_p, i_f], mask=(i_p < sz_p))

      # perform the computation
      out_tile = nl.exp(in_tile)

      # store the results back to external memory
      # only write up to sz_p
      nl.store(out_tensor[i_p, i_f], value=out_tile, mask=(i_p<sz_p))




.. _nki-type-promotion:

NKI Type Promotion
==================

When the data types (dtypes) of inputs to an arithmetic operation (i.e., add, multiply, tensor_tensor, etc.) differ, we promote the dtypes 
following the rules below:

**(float, integer)**: Pick the float type. Example:

- ``(np.int32, np.float16) -> np.float16``
- ``(np.uint16, nl.tfloat32) -> nl.tfloat32``

**(float, float)**: Pick the wider float type. Example:

- ``(np.float32, nl.tfloat32) -> np.float32``
- ``(np.float32, nl.bfloat16) -> np.float32``
- ``(np.float16, nl.bfloat16) -> np.float32``
- ``(nl.float8_e4m3, np.float16) -> np.float16``
- ``(nl.float8_e4m3, nl.bfloat16) -> nl.bfloat16``

**(int, int)**: Pick the wider type or a new widened type that fits the values range. Example:

- ``(np.int16, np.int32) -> np.int32``
- ``(np.uint8, np.uint16) -> np.uint16``
- ``(np.uint16, np.int32) -> np.int32``
- ``(np.int8, np.uint8) -> np.int16`` (new widened type)
- ``(np.int8, np.uint16) -> np.int32`` (new widened type)
- ``(np.int32, np.uint32) -> np.float32`` (new widened type is float32, since int64 isn't supported on the hardware)

The output of the arithmetic operation will get the promoted type by default.

**Note:** The Vector Engine internally performs most of the computation in FP32 (see :ref:`arch_guide_vector_engine`) and casts the output back to the specific type.


.. code-block:: python

  x = np.ndarray((N, M), dtype=nl.float8_e4m3) 
  y = np.ndarray((N, M), dtype=np.float16)
  z = nl.add(x, y) # calculation done in FP32, output cast to np.float16
  assert z.dtype == np.float16 

To prevent the compiler from automatically widening output dtype based on mismatching input dtypes, you may explicitly set the output dtype in the arithmetic operation API.
This would be useful if the output is passed into another operation that benefits from a smaller dtype.

.. code-block:: python

   x = np.ndarray((N, M), dtype=nl.bfloat16)
   y = np.ndarray((N, M), dtype=np.float16)
   z = nl.add(x, y, dtype=nl.bfloat16)  # without explicit `dtype`, `z.dtype` would have been np.float32
   assert z.dtype == nl.bfloat16


Weakly Typed Scalar Type Inference
----------------------------------

Weakly typed scalars (scalar values where the type wasn't explicitly specified) will be inferred as the widest dtype supported by hardware:

- ``bool --> uint8``
- ``integer --> int32``
- ``floating --> float32``

Doing an arithmetic operation with a scalar may result in a larger output type than expected, for example:

- ``(np.int8, 2) -> np.int32``
- ``(np.float16, 1.2) -> np.float32``

To prevent larger dtypes from being inferred from weak scalar types, do either of:

1. Explicitly set the datatype of the scalar, like ``np.int8(2)``, so that the output type is what you desire:

  .. code-block:: python
    
    x = np.ndarray((N, M), dtype=np.float16) 
    y = np.float16(2) 
    z = nl.add(x, y) 
    assert z.dtype == np.float16 

2. Explicitly set the output dtype of the arithmetic operation:

  .. code-block:: python

    x = np.ndarray((N, M), dtype=np.int16)
    y = 2
    z = nl.add(x, y, dtype=nl.bfloat16)
    assert z.dtype == nl.bfloat16

**Note:** The Vector Engine internally performs most of the computation in FP32 (see :ref:`arch_guide_vector_engine`) and casts the output back to the specific type.


.. rubric:: Footnotes

.. [#1] S: sign bits, E: exponent bits, M: mantissa bits
