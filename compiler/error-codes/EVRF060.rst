.. _error-code-evrf060:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF060.

NCC_EVRF060
===========

**Error message**: Scatter does not support N-bit integer operands on this target; they are converted to f32, which loses precision for values larger than 2^24.

This error occurs when a *scatter-with-compute* is applied to integer operands. A scatter-with-compute is a scatter whose update computation combines each scattered value into the destination with an arithmetic reduction (``add``, ``multiply``, ``minimum``, or ``maximum``), rather than simply overwriting it. In PyTorch these correspond to operations such as ``scatter_add``, ``index_add``, ``embedding_dense_backward``, and ``nll_loss_backward``. (A plain scatter that only overwrites values is not affected.)

Scatter-with-compute is emulated on an engine that computes in floating point (f32). Because f32 has a 24-bit mantissa, it cannot exactly represent 32-bit or 64-bit integer values whose magnitude exceeds 2^24 (16,777,216), so the compiler rejects the operation by default rather than silently losing precision.

Why this happens
-----------------

The Neuron device has no native integer datapath for scatter-with-compute, so the reduction runs in floating point: the compiler converts the integer operands to f32, reduces them, and converts the result back. f32 can only represent integers exactly up to 2^24, so any result above that limit would lose precision.

Common causes
--------------

- A model performs an integer ``scatter_add`` (or other scatter-with-compute) on ``int32`` or ``int64`` tensors.
- Index/embedding update logic that accumulates integer counts through a scatter-with-compute.

Resolution
-----------

1. **Explicitly cast the operands to a supported type** (for example ``bf16`` or ``f32``) before the scatter, or restructure the model so the scatter does not reduce integer data.
2. **Allow the f32 downcast explicitly** by passing ``--implicit-integer-downcast=scatter`` (or ``--implicit-integer-downcast=all``). This downgrades the error to a warning and lets the operation run through the f32 conversion.

   .. note:: Turning this on might cause numerical inaccuracy for values larger than 2^24.
