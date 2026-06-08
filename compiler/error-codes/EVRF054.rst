.. _error-code-evrf054:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF054.

NCC_EVRF054
===========

**Error message**: ScaledMatmul custom call batch dimension index out of bounds.

The ``__op$block_scaled_dot`` custom call performs batched matrix multiplication on MXFP8-quantized tensors. Batch dimension indices specified in the backend_config must be valid (0 <= dim < rank). The compiler raises this error when a batch dimension index is negative or exceeds the tensor rank.

To fix this error, use dimension indices within the valid range (0 <= dim < rank).
