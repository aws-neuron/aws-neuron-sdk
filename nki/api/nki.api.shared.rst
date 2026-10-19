=========================
NKI ISA API Common Fields
=========================

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
  |                        | float8_e4m3fn (1S,4E,3M)     | ``nki.language.float8_e4m3fn``                  |
  |                        +------------------------------+-------------------------------------------------+
  |                        | float8_e8m0fnu (0S,8E,0M)    | ``nki.language.float8_e8m0fnu``                 |
  |                        +------------------------------+-------------------------------------------------+
  |                        | float16 (1S,5E,10M)          | ``nki.language.float16``                        |
  |                        +------------------------------+-------------------------------------------------+
  | Float                  | bfloat16 (1S,8E,7M)          | ``nki.language.bfloat16``                       |
  |                        +------------------------------+-------------------------------------------------+
  |                        | tfloat32 (1S,8E,10M)         | ``nki.language.tfloat32``                       |
  |                        +------------------------------+-------------------------------------------------+
  |                        | float32 (1S,8E,23M)          | ``nki.language.float32``                        |
  +------------------------+------------------------------+-------------------------------------------------+
  |                        | float8_e5m2_x4               | ``nki.language.float8_e5m2_x4``                 |
  |                        +------------------------------+-------------------------------------------------+
  | Packed [#2]_           | float8_e4m3fn_x4             | ``nki.language.float8_e4m3fn_x4``               |
  |                        +------------------------------+-------------------------------------------------+
  |                        | float4_e2m1fn_x4             | ``nki.language.float4_e2m1fn_x4``               |
  +------------------------+------------------------------+-------------------------------------------------+
  | Boolean                | boolean stored as uint8      | ``nki.language.bool_``                          |
  +------------------------+------------------------------+-------------------------------------------------+

.. _nki-type-casting:

Type Casting
============

Several NKI ISA operations convert input values through an intermediate ``float32`` representation before writing them in the destination data type.
Whether this conversion occurs generally depends on the specific combination of opcode, source and destination data types, and, in some cases, the selected engine.
See individual NKI ISA API documentation for details on whether this conversion occurs.

Integer-to-Integer Conversion Through FP32
------------------------------------------

The source integer is first converted to ``float32``, and the resulting intermediate value is then converted to the destination integer data type.
As a result, consider the potential precision loss and saturation behavior described below.

Integer-to-FP32 Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~

For this conversion, integer values that can be exactly represented by ``float32`` are preserved, while other
values are rounded to the nearest representable ``float32`` value, with ties
rounded to even (RNE).

.. list-table::
   :header-rows: 1

   * - Source data type
     - Conversion behavior
   * - ``int8``, ``uint8``, ``int16``, ``uint16``
     - All values are represented exactly.
   * - ``int32``, ``uint32``
     - Values may lose precision. Every integer with magnitude at most
       :math:`2^{24}` is represented exactly, but not every larger integer is.

For example, :math:`2^{24} + 1` lies halfway between the representable values
:math:`2^{24}` and :math:`2^{24} + 2`, so RNE converts it to :math:`2^{24}`.

FP32-to-Integer Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``float32`` intermediate produced from the source integer is integer-valued. Values within the
destination data type's range are preserved, while out-of-range values are
saturated to the destination minimum or maximum.

.. list-table::
   :header-rows: 1

   * - Intermediate ``float32`` value
     - Destination data type
     - Result
   * - ``42.0``
     - ``int32``
     - ``42`` (preserved)
   * - ``2147483648.0``
     - ``uint8``
     - ``255`` (destination maximum)
   * - ``-300.0``
     - ``int8``
     - ``-128`` (destination minimum)
   * - ``-1.0``
     - ``uint32``
     - ``0`` (destination minimum)

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
  |                        | Power                      | ``nki.language.power``                      | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Absolute Maximum           | ``nki.language.abs_max``                    | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Absolute Minimum           | ``nki.language.abs_min``                    | N                      |
  +------------------------+----------------------------+---------------------------------------------+------------------------+
  | Arithmetic Unary       | Absolute                   | ``nki.language.abs``                        | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Square                     | ``nki.language.square``                     | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Relu                       | ``nki.language.relu``                       | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Reverse Square Root        | ``nki.language.rsqrt``                      | N                      |
  |                        +----------------------------+---------------------------------------------+------------------------+
  |                        | Reciprocal                 | ``nki.language.reciprocal``                 | N                      |
  +------------------------+----------------------------+---------------------------------------------+------------------------+

.. _nki-act-func:

Supported Activation Functions for NKI ISA
==========================================
:ref:`tbl-act-func` below lists all the activation functions supported by the
:func:`nki.isa.activation` and :func:`nki.isa.activate2` APIs, along with the
valid input range and ULP on each engine that implements them.

On the **Scalar Engine** (default), activation functions are approximated with
piece-wise polynomials from an on-engine table. If an input falls outside the
**Scalar Engine Valid Range** listed below, the Scalar Engine will generate
invalid output results.

On the **GpSIMD Engine** (available on NeuronCore-v4+), activation functions
are computed accurately over the full FP32 domain.

.. _tbl-act-func:
.. table:: Supported Activation Functions by NKI ISA
   :widths: 24 20 18 8 15 8

   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Function Name                              | Accepted ``op`` by           | | ScalarE:          | | ScalarE: | | GpSimdE:          | | GpSimdE: |
   |                                            | activation instructions      | | Valid Range       | | ULP      | | Valid Range       | | ULP      |
   +============================================+==============================+=====================+============+=====================+============+
   | Identity                                   | ``nl.copy``                  | ``[-inf, inf]``     | ``1``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Square                                     | ``nl.square``                | ``[-inf, inf]``     | ``1``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Sigmoid                                    | ``nl.sigmoid``               | ``[-inf, inf]``     | ``40``     | ``[-inf, inf]``     | ``3``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Relu                                       | ``nl.relu``                  | ``[-inf, inf]``     | ``1``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | PReLU                                      | ``nl.prelu``                 | ``[-inf, inf]``     | ``1``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Gelu                                       | ``nl.gelu``                  | ``[-inf, inf]``     | ``4``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Gelu Derivative                            | ``nl.gelu_dx``               | ``[-inf, inf]``     | ``40``     | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Gelu with Tanh Approximation               | ``nl.gelu_apprx_tanh``       | ``[-inf, inf]``     | ``40``     | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Gelu with Sigmoid Approximation            | ``nl.gelu_apprx_sigmoid``    | ``[-inf, inf]``     | ``40``     | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Gelu with Sigmoid Approximation Derivative | ``nl.gelu_apprx_sigmoid_dx`` | ``[-inf, inf]``     | ``4096``   | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Silu                                       | ``nl.silu``                  | ``[-inf, inf]``     | ``32``     | ``[-inf, inf]``     | ``4``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Silu Derivative                            | ``nl.silu_dx``               | ``[-inf, inf]``     | ``32``     | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Tanh                                       | ``nl.tanh``                  | ``[-inf, inf]``     | ``4``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Softplus                                   | ``nl.softplus``              | ``[-inf, inf]``     | ``40``     | ``[-inf, inf]``     | ``3``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Mish                                       | ``nl.mish``                  | ``[-inf, inf]``     | ``4``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Erf                                        | ``nl.erf``                   | ``[-inf, inf]``     | ``4``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Erf Derivative                             | ``nl.erf_dx``                | ``[-inf, inf]``     | ``400``    | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Exponential                                | ``nl.exp``                   | ``[-inf, inf]``     | ``400``    | ``[-inf, inf]``     | ``1``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Natural Log                                | ``nl.log``                   | ``[2^-64, 2^64]``   | ``40``     | ``[-inf, inf]``     | ``2``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Natural Log (1+x)                          | ``nl.log1p``                 | ``-``               | ``-``      | ``[-inf, inf]``     | ``2``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Sine                                       | ``nl.sin``                   | ``[-PI, PI]``       | ``4``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Arctan                                     | ``nl.arctan``                | ``[-PI/2, PI/2]``   | ``4``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Square Root                                | ``nl.sqrt``                  | ``[2^-116, 2^118]`` | ``65536``  | ``[-inf, inf]``     | ``0``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Reverse Square Root                        | ``nl.rsqrt``                 | ``[2^-87, 2^97]``   | ``40000``  | ``[-inf, inf]``     | ``1``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Reciprocal                                 | ``nl.reciprocal``            | ``±[2^-42, 2^42]``  | ``400``    | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Sign                                       | ``nl.sign``                  | ``[-inf, inf]``     | ``1``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+
   | Absolute                                   | ``nl.abs``                   | ``[-inf, inf]``     | ``1``      | ``-``               | ``-``      |
   +--------------------------------------------+------------------------------+---------------------+------------+---------------------+------------+

A dash (``-``) means the function is not supported on that engine.

.. note::

   **Scalar Engine.** Functions whose Scalar Engine range is narrower than
   ``[-inf, inf]`` (``sin``, ``arctan``, ``log``, ``sqrt``, ``rsqrt``,
   ``reciprocal``) may produce invalid results outside that range. For
   ``sin`` specifically, the Scalar Engine performs **no argument range
   reduction or clamping**: it evaluates the fitting polynomial directly, so
   ``nki.isa.activation(op=nl.sin)`` extrapolates a cubic outside
   ``[-PI, PI]`` and ``sin(9.0)`` returns ~27.65 (not ~0.41). The Python
   simulator (``nki.simulate``) reproduces this hardware behavior for
   ``sin`` only.

   :func:`nki.language.sin` high-level API applies additional instructions to perform range
   reduction *before* the activation, so it stays accurate over the full input
   domain at a performance cost. Call :func:`nki.language.sin` instead of
   :func:`nki.isa.activation` / :func:`nki.isa.activate2`
   instructions when you need correct results for inputs outside ``[-PI, PI]``.


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
.. [#2] 4x packed elements, custom data type for ``nki.isa.nc_matmul_mx`` and ``nki.isa.quantize_mx`` on NeuronCore-v4
