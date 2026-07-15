.. _error-code-evrf048:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF048.

NCC_EVRF048
===========

**Error message**: ScaledMatmul custom call batch dimension mismatch.

The ``__op$block_scaled_dot`` custom call performs batched matrix multiplication
on MXFP8-quantized tensors. When the LHS and RHS specify the same number of
batch dimensions, the compiler compares the products of their batch dimension
sizes and raises this error if the products differ.

To fix this error, ensure the product of LHS and RHS batch dimension sizes match.
