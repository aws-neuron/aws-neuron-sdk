.. _error-code-evrf047:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF047.

NCC_EVRF047
===========

**Error message**: ScaledMatmul custom call RHS tensor must have rank >= 2.

The ``__op$block_scaled_dot`` custom call performs matrix multiplication on MXFP8-quantized tensors. Matrix operations require at least 2-dimensional tensors (matrices). The RHS (right-hand side) operand must have rank >= 2. The compiler raises this error when the RHS tensor is a scalar or 1-D vector.

To fix this error, reshape the RHS to have at least 2 dimensions.
