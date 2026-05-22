.. _error-code-evrf055:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF055.

NCC_EVRF055
===========

**Error message**: ScaledMatmul custom call contracting dimension index out of bounds.

The ``__op$block_scaled_dot`` custom call performs matrix multiplication on MXFP8-quantized tensors. Contracting dimension indices specified in the backend_config must be valid (0 <= dim < rank). The compiler raises this error when a contracting dimension index is negative or exceeds the tensor rank.

To fix this error, use dimension indices within the valid range (0 <= dim < rank).
