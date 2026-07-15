.. _error-code-evrf043:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF043.

NCC_EVRF043
===========

**Error message**: ScaledMatmul custom call must have exactly 4 operands.

The ``__op$block_scaled_dot`` custom call performs matrix multiplication on MXFP8-quantized tensors produced by ``QuantizeMX``. It requires exactly 4 operands in order: lhs (left-hand side matrix), rhs (right-hand side matrix), lhs_scale (per-block scales for lhs), and rhs_scale (per-block scales for rhs). The compiler raises this error when a different number of operands is provided.

To fix this error, pass all 4 operands (lhs, rhs, lhs_scale, rhs_scale).
