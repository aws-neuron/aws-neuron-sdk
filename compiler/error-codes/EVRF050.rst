.. _error-code-evrf050:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF050.

NCC_EVRF050
===========

**Error message**: ScaledMatmul custom call contracting dimension sizes mismatch.

The ``__op$block_scaled_dot`` custom call performs matrix multiplication on
MXFP8-quantized tensors. The compiler compares the first configured LHS and RHS
contracting dimensions and raises this error when their sizes differ. If a
contracting dimension is omitted, the compiler uses the last dimension.

To fix this error, ensure the LHS and RHS contracting dimension sizes match.
