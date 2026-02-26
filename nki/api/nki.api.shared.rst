=======================
NKI API Common Fields
=======================

.. _nki-dtype:

Supported Data Types
========================

:ref:`tbl-dtype` below lists all supported data types by NKI.
Almost all of the NKI APIs accept a data type field, `dtype`, 
which must be a `nki.language` data type.

.. _tbl-dtype:

.. table:: Supported Data Types by NKI

  +------------------------+------------------------------+-------------------------------------------------+
  |                        | Data Type                    | Accepted ``dtype`` Field by NKI APIs            |
  +========================+==============================+=================================================+
  |                        | 8-bit unsigned integer       | ``nki.language.uint8``                          |
  |                        +------------------------------+-------------------------------------------------+
  |                        | 8-bit signed integer         | ``nki.language.int8``                           |
  |                        +------------------------------+-------------------------------------------------+
  | Integer                | 16-bit unsigned integer      | ``nki.language.uint16``                         |
  |                        +------------------------------+-------------------------------------------------+
  |                        | 16-bit signed integer        | ``nki.language.int16``                          |
  |                        +------------------------------+-------------------------------------------------+
  |                        | 32-bit unsigned integer      | ``nki.language.uint32``                         |
  |                        +------------------------------+-------------------------------------------------+
  |                        | 32-bit signed integer        | ``nki.language.int32``                          |
  +------------------------+------------------------------+-------------------------------------------------+
  |                        | float8_e4m3 (1S,4E,3M) [#1]_ | ``nki.language.float8_e4m3``                    |
  |                        +------------------------------+-------------------------------------------------+
  |                        | float8_e5m2 (1S,5E,2M)       | ``nki.language.float8_e5m2``                    |
  |                        +------------------------------+-------------------------------------------------+
  |                        | float16 (1S,5E,10M)          | ``nki.language.float16``                        |
  |                        +------------------------------+-------------------------------------------------+
  | Float                  | bfloat16 (1S,8E,7M)          | ``nki.language.bfloat16``                       |
  |                        +------------------------------+-------------------------------------------------+
  |                        | tfloat32 (1S,8E,10M)         | ``nki.language.tfloat32``                       |
  |                        +------------------------------+-------------------------------------------------+
  |                        | float32 (1S,8E,23M)          | ``nki.language.float32``                        |
  +------------------------+------------------------------+-------------------------------------------------+
  | Boolean                | boolean stored as uint8      | ``nki.language.bool_``                          |
  +------------------------+------------------------------+-------------------------------------------------+

.. _nki-aluop:

Supported Math Operators for NKI ISA
====================================

:ref:`tbl-aluop` below lists all the mathematical operator primitives supported by NKI.
Many :ref:`nki.isa <nki-isa>` APIs (instructions) allow programmable operators through the ``op`` field.
The supported operators fall into two categories: *bitvec* and *arithmetic*. In general, instructions
using *bitvec* operators expect integer data types and treat input elements as bit patterns. On the other
hand, instructions using *arithmetic* operators accept any valid NKI data type and convert input elements
into float32 before performing the operators.

.. _tbl-aluop:
.. table:: Supported Math Operators by NKI ISA

  +------------------------+----------------------------+---------------------------------------------+------------------------+
  |                        | Operator                   | ``op``                                      | Legal Reduction ``op`` |
  +========================+============================+=============================================+========================+
  |                        | Bitwise Not                | ``nki.language.invert``                     | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Bitwise And                | ``nki.language.bitwise_and``                | Y                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Bitwise Or                 | ``nki.language.bitwise_or``                 | Y                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  | Bitvec                 | Bitwise Xor                | ``nki.language.bitwise_xor``                | Y                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Arithmetic Shift Left      | ``nki.language.left_shift``                 | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Arithmetic Shift Right     |  Not supported                              | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Logical Shift Left         | ``nki.language.left_shift``                 | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Logical Shift Right        | ``nki.language.right_shift``                | N                      |
  +------------------------+----------------------------+---------------------------------------------+------------------------+
  |                        | Add                        | ``nki.language.add``                        | Y                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Subtract                   | ``nki.language.subtract``                   | Y                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Multiply                   | ``nki.language.multiply``                   | Y                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Max                        | ``nki.language.maximum``                    | Y                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Min                        | ``nki.language.minimum``                    | Y                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Is Equal to                | ``nki.language.equal``                      | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Is Not Equal to            | ``nki.language.not_equal``                  | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  | Arithmetic             | Is Greater than or Equal to| ``nki.language.greater_equal``              | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Is Greater than to         | ``nki.language.greater``                    | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Is Less than or Equal to   | ``nki.language.less_equal``                 | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Is Less than               | ``nki.language.less``                       | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Logical And                | ``nki.language.logical_and``                | Y                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Logical Or                 | ``nki.language.logical_or``                 | Y                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Logical Xor                | ``nki.language.logical_xor``                | Y                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Reverse Square Root        | ``nki.language.rsqrt``                      | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Reciprocal                 | ``nki.language.reciprocal``                 | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Absolute                   | ``nki.language.abs``                        | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Power                      | ``nki.language.power``                      | N                      |
  +------------------------+----------------------------+---------------------------------------------+------------------------+

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

   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Function Name                              | Accepted ``op`` by Scalar Engine                    | Valid Input Range   |
   +============================================+=====================================================+=====================+
   | Identity                                   | ``nki.language.copy``                               | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Square                                     | ``nki.language.square``                             | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Sigmoid                                    | ``nki.language.sigmoid``                            | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Relu                                       | ``nki.language.relu``                               | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Gelu                                       | ``nki.language.gelu``                               | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Gelu Derivative                            | ``nki.language.gelu_dx``                            | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Gelu with Tanh Approximation               | ``nki.language.gelu_apprx_tanh``                    | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Gelu with Sigmoid Approximation            | ``nki.language.gelu_apprx_sigmoid``                 | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Gelu with Sigmoid Approximation Derivative | ``nki.language.gelu_apprx_sigmoid_dx``              | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Silu                                       | ``nki.language.silu``                               | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Silu Derivative                            | ``nki.language.silu_dx``                            | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Tanh                                       | ``nki.language.tanh``                               | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Softplus                                   | ``nki.language.softplus``                           | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Mish                                       | ``nki.language.mish``                               | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Erf                                        | ``nki.language.erf``                                | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Erf Derivative                             | ``nki.language.erf_dx``                             | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Exponential                                | ``nki.language.exp``                                | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Natural Log                                | ``nki.language.log``                                  ``[2^-64, 2^64]``   |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Sine                                       | ``nki.language.sin``                                | ``[-PI, PI]``       |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Arctan                                     | ``nki.language.arctan``                             | ``[-PI/2, PI/2]``   |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Square Root                                | ``nki.language.sqrt``                               | ``[2^-116, 2^118]`` |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Reverse Square Root                        | ``nki.language.rsqrt``                              | ``[2^-87, 2^97]``   |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Reciprocal                                 | ``nki.language.reciprocal``                         | ``±[2^-42, 2^42]``  |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Sign                                       | ``nki.language.sign``                               | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+
   | Absolute                                   | ``nki.language.abs``                                | ``[-inf, inf]``     |
   +--------------------------------------------+-----------------------------------------------------+---------------------+

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
