.. _error-code-evrf044:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF044.

NCC_EVRF044
===========

**Error message**: ScaledMatmul custom call LHS input type is unsupported.

The ``__op$block_scaled_dot`` custom call performs matrix multiplication on MXFP8-quantized tensors. The LHS (left-hand side) operand must be an FP8 tensor (``f8E5M2`` or ``f8E4M3FN``) produced by the ``QuantizeMX`` custom call. The compiler raises this error when the LHS operand has a different element type.

To fix this error, ensure the LHS operand is the FP8 quantized data tensor returned by ``QuantizeMX`` (or any equivalent FP8 ``f8E5M2`` / ``f8E4M3FN`` tensor).
