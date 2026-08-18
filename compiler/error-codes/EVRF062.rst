.. _error-code-evrf062:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF062.

NCC_EVRF062
===========

**Error message**: Opcode (power) does not support 64-bit integer operands on this target; they are downcast to 32-bit integer, which loses precision for values larger than 2^31.

This error occurs when the ``power`` operation is applied to ``int64`` operands. ``power`` has no native 64-bit integer datapath, so it is emulated by downcasting the operands to ``int32``. Because ``int32`` can only represent signed values up to 2^31 - 1, this downcast loses precision for larger magnitudes, so the compiler rejects the operation by default rather than silently losing precision.

Why this happens
-----------------

Unlike the operations that fall back to floating-point compute (see :ref:`NCC_EVRF035 <error-code-evrf035>`, :ref:`NCC_EVRF060 <error-code-evrf060>`, :ref:`NCC_EVRF061 <error-code-evrf061>`, and :ref:`NCC_EVRF063 <error-code-evrf063>`), ``power`` runs in the integer domain by narrowing its ``int64`` operands to ``int32``. ``int32`` can only represent integers exactly up to 2^31, so any result above that limit would lose precision.

Common causes
--------------

- A model raises an ``int64`` tensor to an integer power.

Resolution
-----------

1. **Explicitly cast the operands to a supported type** so the operation does not require an ``int64`` datapath.
2. **Allow the int64 to int32 downcast explicitly** by passing ``--implicit-integer-downcast=pow`` (or ``--implicit-integer-downcast=all``). This downgrades the error to a warning and lets the operation run through the ``int64`` to ``int32`` downcast.

   .. note:: Turning this on might cause numerical inaccuracy for values larger than 2^31.
