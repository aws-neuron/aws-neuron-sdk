.. _error-code-evrf063:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF063.

NCC_EVRF063
===========

**Error message**: ReduceScatter performs its reduction compute in floating point (f32) and does not support 32-bit or 64-bit integer operands on this target; integer values lose precision for magnitudes larger than 2^24.

This error occurs when a ``reduce-scatter`` collective operation is applied to 32-bit or 64-bit integer operands. The collective compute engine performs its reduction (``add``) in floating point (f32). Because f32 has a 24-bit mantissa, it cannot exactly represent integer values whose magnitude exceeds 2^24 (16,777,216), so the compiler rejects the operation by default rather than silently losing precision.

Why this happens
-----------------

The Neuron collective compute engine has no native integer reduction datapath, so an integer ``reduce-scatter`` runs in floating point: the compiler converts the operands to f32, reduces them, and converts the result back. f32 can only represent integers exactly up to 2^24, so any result above that limit would lose precision.

Common causes
--------------

- A distributed model performs an integer ``reduce-scatter`` on ``int32`` or ``int64`` tensors.

Resolution
-----------

1. **Explicitly cast the operands to a floating-point type** (for example ``bf16`` or ``f32``) before the ``reduce-scatter``.
2. **Allow the f32 downcast explicitly** by passing ``--implicit-integer-downcast=reduce_scatter`` (or ``--implicit-integer-downcast=all``). This downgrades the error to a warning and lets the reduction run in f32.

   .. note:: Turning this on might cause numerical inaccuracy for values larger than 2^24.
