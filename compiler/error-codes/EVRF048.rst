.. _error-code-evrf048:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF048.

NCC_EVRF048
===========

**Error message**: ScaledMatmul custom call batch dimension mismatch.

The ``__op$block_scaled_dot`` custom call performs batched matrix multiplication on MXFP8-quantized tensors. The product of LHS batch dimension sizes must equal the product of RHS batch dimension sizes. The compiler raises this error when the total batch sizes do not match.

To fix this error, ensure the product of LHS and RHS batch dimension sizes match.
