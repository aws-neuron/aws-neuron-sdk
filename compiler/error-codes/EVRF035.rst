.. _error-code-evrf035:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF035.

NCC_EVRF035
===========

**Error message**: Opcode (dot) does not support 32-bit or 64-bit integer operands on this target.

This error occurs when a matrix multiplication (``dot``) is applied to integer operands. The matrix-multiplication engine computes in floating point (f32) and has no native integer datapath, so integer operands would have to be converted to f32, which loses precision for values larger than 2^24. The compiler rejects the operation by default rather than silently losing precision.

Why this happens
-----------------

On Trainium2 and later (NeuronCore-v3 and above) the matrix-multiplication engine has no native integer datapath. To evaluate an integer matrix multiplication the compiler would convert the operands to f32, multiply them, and convert the result back. f32 can only represent integers exactly up to 2^24, so any result above that limit would lose precision.

.. note:: On Trainium1 (NeuronCore-v2) and earlier, 32-bit integer matrix multiplication is supported and only emits a warning. This error applies to ``int32`` operands on Trainium2 and later, and to ``int64`` operands on all targets.

Common causes
--------------

- A model performs an integer matrix multiplication on ``int32`` or ``int64`` tensors.

Resolution
-----------

1. **Explicitly cast the operands to a floating-point type** (for example ``bf16`` or ``f32``) before the matrix multiplication.
2. **Allow the f32 downcast explicitly** by passing ``--implicit-integer-downcast=dot`` (or ``--implicit-integer-downcast=all``). This downgrades the error to a warning and lets the operation run through the f32 conversion.

   .. note:: Turning this on might cause numerical inaccuracy for values larger than 2^24.
