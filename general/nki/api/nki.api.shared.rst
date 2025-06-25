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

.. table:: Supported Data Types by NKI

  +------------------------+------------------------------+-------------------------------------------------+
  |                        | Data Type                    | Accepted ``dtype`` Field by NKI APIs            |
  +========================+==============================+=================================================+
  |                        | 8-bit unsigned integer       | ``nki.language.uint8`` or ``numpy.uint8``       |
  |                        +------------------------------+-------------------------------------------------+
  |                        | 8-bit signed integer         | ``nki.language.int8`` or ``numpy.int8``         |
  |                        +------------------------------+-------------------------------------------------+
  | Integer                | 16-bit unsigned integer      | ``nki.language.uint16`` or ``numpy.uint16``     |
  |                        +------------------------------+-------------------------------------------------+
  |                        | 16-bit signed integer        | ``nki.language.int16`` or ``numpy.int16``       |
  |                        +------------------------------+-------------------------------------------------+
  |                        | 32-bit unsigned integer      | ``nki.language.uint32`` or ``numpy.uint32``     |
  |                        +------------------------------+-------------------------------------------------+
  |                        | 32-bit signed integer        | ``nki.language.int32`` or ``numpy.int32``       |
  +------------------------+------------------------------+-------------------------------------------------+
  |                        | float8_e4m3 (1S,4E,3M) [#1]_ | ``nki.language.float8_e4m3``                    |
  |                        +------------------------------+-------------------------------------------------+
  |                        | float8_e5m2 (1S,5E,2M)       | ``nki.language.float8_e5m2``                    |
  |                        +------------------------------+-------------------------------------------------+
  |                        | float16 (1S,5E,10M)          | ``nki.language.float16`` or ``numpy.float16``   |
  |                        +------------------------------+-------------------------------------------------+
  | Float                  | bfloat16 (1S,8E,7M)          | ``nki.language.bfloat16``                       |
  |                        +------------------------------+-------------------------------------------------+
  |                        | tfloat32 (1S,8E,10M)         | ``nki.language.tfloat32``                       |
  |                        +------------------------------+-------------------------------------------------+
  |                        | float32 (1S,8E,23M)          | ``nki.language.float32`` or ``numpy.float32``   |
  +------------------------+------------------------------+-------------------------------------------------+
  | Boolean                | boolean stored as uint8      | ``nki.language.bool_`` or ``numpy.bool``        |
  +------------------------+------------------------------+-------------------------------------------------+

.. _nki-aluop:

Supported Math Operators for NKI ISA
====================================

:ref:`tbl-aluop` below lists all the mathematical operator primitives supported by NKI.
Many :ref:`nki.isa <nki-isa>` APIs (instructions) allow programmable operators through the ``op`` field. 
The supported operators fall into two categories: *bitvec* and *arithmetic*. In general, instructions 
using *bitvec* operators expect integer data types and treat input elements as bit patterns. On the other 
hand, instructions using *arithmetic* operators accept any valid NKI data types and convert input elements 
into float32 before performing the operators.

.. _tbl-aluop:
.. table:: Supported Math Operators by NKI ISA

  +------------------------+----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Operator                   | ``op``                                      | Legal Reduction ``op`` | Supported Engine     |
  +========================+============================+=============================================+========================+======================+
  |                        | Bitwise Not                | ``nki.language.invert``                     | N                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Bitwise And                | ``nki.language.bitwise_and``                | Y                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Bitwise Or                 | ``nki.language.bitwise_or``                 | Y                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  | Bitvec                 | Bitwise Xor                | ``nki.language.bitwise_xor``                | Y                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Arithmetic Shift Left      | ``nki.language.left_shift``                 | N                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Arithmetic Shift Right     |  Not supported                              | N                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Logical Shift Left         | ``nki.language.left_shift``                 | N                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Logical Shift Right        | ``nki.language.right_shift``                | N                      | Vector               |
  +------------------------+----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Add                        | ``nki.language.add``                        | Y                      | Vector/GpSIMD/Scalar |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Subtract                   | ``nki.language.subtract``                   | Y                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Multiply                   | ``nki.language.multiply``                   | Y                      | Vector/GpSIMD/Scalar |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Max                        | ``nki.language.maximum``                    | Y                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Min                        | ``nki.language.minimum``                    | Y                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Is Equal to                | ``nki.language.equal``                      | N                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Is Not Equal to            | ``nki.language.not_equal``                  | N                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  | Arithmetic             | Is Greater than or Equal to| ``nki.language.greater_equal``              | N                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Is Greater than to         | ``nki.language.greater``                    | N                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Is Less than or Equal to   | ``nki.language.less_equal``                 | N                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Is Less than               | ``nki.language.less``                       | N                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Logical Not                | ``nki.language.logical_not``                | N                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Logical And                | ``nki.language.logical_and``                | Y                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Logical Or                 | ``nki.language.logical_or``                 | Y                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Logical Xor                | ``nki.language.logical_xor``                | Y                      | Vector               |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Reverse Square Root        | ``nki.language.rsqrt``                      | N                      | GpSIMD/Scalar        |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Reciprocal                 | ``nki.language.reciprocal``                 | N                      | Vector/Scalar        |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Absolute                   | ``nki.language.abs``                        | N                      | Vector/Scalar        |
  |                        +----------------------------+---------------------------------------------+------------------------+----------------------+
  |                        | Power                      | ``nki.language.power``                      | N                      | GpSIMD               |
  +------------------------+----------------------------+---------------------------------------------+------------------------+----------------------+

**Note** Add and Multiply are supported on Scalar Engine only from NeuronCore-v3. 32-bit integer Add and Multiply are only supported on GpSIMD Engine.

.. _nki-act-func:

Supported Activation Functions for NKI ISA
==========================================
:ref:`tbl-act-func` below lists all the activation function supported by the ``nki.isa.activation`` API. These
activation functions are approximated with piece-wise polynomials on Scalar Engine.
*NOTE*: if input values fall outside the supported **Valid Input Range** listed below, 
the Scalar Engine will generate invalid output results.


.. _tbl-act-func:
.. table:: Supported Activation Functions by NKI ISA
   :widths: 25 25 25

   +--------------------------------+-----------------------------------------------------+---------------------+
   | Function Name                  | Accepted ``op`` by Scalar Engine                    | Valid Input Range   |
   +================================+=====================================================+=====================+
   | Identity                       | ``nki.language.copy`` or ``numpy.copy``             | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Square                         | ``nki.language.square`` or ``numpy.square``         | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Sigmoid                        | ``nki.language.sigmoid``                            | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Relu                           | ``nki.language.relu``                               | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Gelu                           | ``nki.language.gelu``                               | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Gelu Derivative                | ``nki.language.gelu_dx``                            | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Gelu with Tanh Approximation   | ``nki.language.gelu_apprx_tanh``                    | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Silu                           | ``nki.language.silu``                               | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Silu Derivative                | ``nki.language.silu_dx``                            | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Tanh                           | ``nki.language.tanh`` or ``numpy.tanh``             | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Softplus                       | ``nki.language.softplus``                           | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Mish                           | ``nki.language.mish``                               | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Erf                            | ``nki.language.erf``                                | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Erf Derivative                 | ``nki.language.erf_dx``                             | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Exponential                    | ``nki.language.exp`` or ``numpy.exp``               | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Natural Log                    | ``nki.language.log`` or ``numpy.log``               | ``[2^-64, 2^64]``   |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Sine                           | ``nki.language.sin`` or ``numpy.sin``               | ``[-PI, PI]``       |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Arctan                         | ``nki.language.arctan`` or ``numpy.arctan``         | ``[-PI/2, PI/2]``   |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Square Root                    | ``nki.language.sqrt`` or ``numpy.sqrt``             | ``[2^-116, 2^118]`` |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Reverse Square Root            | ``nki.language.rsqrt``                              | ``[2^-87, 2^97]``   |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Reciprocal                     | ``nki.language.reciprocal`` or ``numpy.reciprocal`` | ``±[2^-42, 2^42]``  |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Sign                           | ``nki.language.sign`` or ``numpy.sign``             | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+
   | Absolute                       | ``nki.language.abs`` or ``numpy.abs``               | ``[-inf, inf]``     |
   +--------------------------------+-----------------------------------------------------+---------------------+

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
The left- or right-hand side expression
of each comparator must be an affine expression of ``nki.language.arange()``,
``nki.language.affine_range()`` or ``nki.language.program_id()`` .
Each comparison expression should indicate which range of
indices along one of the input tile axes should be valid for the computation. For example,
assume we have an input tile ``in_tile`` of shape ``(128, 512)``, and we would like to perform a square
operation on this tile for elements in ``[0:64, 0:256]``, we can invoke the ``nki.language.square()``
API using the following:


.. nki_example:: ../test/test_nki_mask.py
  :language: python
  :marker: NKI_EXAMPLE_15

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
      out_tile = nl.exp(in_tile, mask=(i_p < sz_p))

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

**(float, float)**: Pick the wider float type or a new widened type that fits the values range. Example:

- ``(np.float32, nl.tfloat32) -> np.float32``
- ``(np.float32, nl.bfloat16) -> np.float32``
- ``(np.float16, nl.bfloat16) -> np.float32`` (new widened type)
- ``(nl.float8_e4m3, np.float16) -> np.float16``
- ``(nl.float8_e4m3, nl.bfloat16) -> nl.bfloat16``
- ``(nl.float8_e4m3, nl.float8_e5m2) -> nl.bfloat16`` (new widened type)

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


.. _nki-engine-sel:

NKI Engine Selection for Operators Supported on Multiple Engines
================================================================
There is a tradeoff between precision and speed on different engines for operators with multiple engine options. Users can select which engine to map to based on 
their needs. We take reciprocal and reverse square root as two examples and explain the tradeoff below.

1. Reciprocal can run on Scalar Engine or Vector Engine:

  Reciprocal can run on Vector Engine with ``nki.isa.reciprocal`` or on Scalar Engine with ``nki.isa.activation(nl.reciprocal)``. Vector Engine performs reciprocal 
  at a higher precision compared to Scalar Engine; however, the computation throughput of reciprocal on Vector Engine is about 8x lower than Scalar Engine for large 
  input tiles. For input tiles with a small number of elements per partition (less than 64, processed one per cycle), instruction initiation interval (roughly 64 
  cycles) dominates performance so Scalar Engine and Vector Engine have comparable performance. In this case, we suggest using Vector Engine to achieve better precision.

  **Estimated cycles on different engines:**

  .. list-table::
    :widths: 40 60
    :header-rows: 1

    * - Cost `(Engine Cycles)`
      - Condition
    * - ``max(MIN_II, N)``
      - mapped to Scalar Engine ``nki.isa.scalar_engine``
    * - ``max(MIN_II, 8*N)``
      - mapped to Vector Engine ``nki.isa.vector_engine``

  where,

  - ``N`` is the number of elements per partition in the input tile.
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles.
    ``MIN_II`` is roughly 64 engine cycles.

  **Note** ``nki.isa.activation(op=nl.reciprocal)`` doesn't support setting bias on NeuronCore-v2.

2. Reverse square root can run on GpSIMD Engine or Scalar Engine:

  Reverse square root can run on GpSIMD Engine with ``nki.isa.tensor_scalar(op0=nl.rsqrt, operand0=0.0)`` or on Scalar Engine with ``nki.isa.activation(nl.rsqrt)``. 
  GpSIMD Engine performs reverse square root at a higher precision compared to Scalar Engine; however, the computation throughput of reverse square root on GpSIMD 
  Engine is 4x lower than Scalar Engine. 


.. rubric:: Footnotes

.. [#1] S: sign bits, E: exponent bits, M: mantissa bits
