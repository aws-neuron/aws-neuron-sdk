.. _error-code-evrf055:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF055.

NCC_EVRF055
===========

**Error message**: ScaledMatmul custom call contracting dimension index out of bounds.

The ``__op$block_scaled_dot`` custom call performs matrix multiplication on
MXFP8-quantized tensors. The compiler checks the first configured contracting
dimension for each operand, or the last dimension when none is configured, and
raises this error when that index is negative or greater than or equal to the
operand rank.

To fix this error, use dimension indices within the valid range (0 <= dim < rank).
