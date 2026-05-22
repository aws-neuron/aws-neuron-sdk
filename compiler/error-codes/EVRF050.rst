.. _error-code-evrf050:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF050.

NCC_EVRF050
===========

**Error message**: ScaledMatmul custom call contracting dimension sizes mismatch.

The ``__op$block_scaled_dot`` custom call performs matrix multiplication on MXFP8-quantized tensors. In matrix multiplication, the contracting dimensions (the dimensions that are summed over) must have equal sizes in both operands. For example, in C = A @ B where A is [M, K] and B is [K, N], the contracting dimension K must match. The compiler raises this error when LHS and RHS contracting dimensions have different sizes.

To fix this error, ensure the LHS and RHS contracting dimension sizes match.
