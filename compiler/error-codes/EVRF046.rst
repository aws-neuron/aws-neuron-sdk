.. _error-code-evrf046:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF046.

NCC_EVRF046
===========

**Error message**: ScaledMatmul custom call LHS tensor must have rank >= 2.

The ``__op$block_scaled_dot`` custom call performs matrix multiplication on MXFP8-quantized tensors. Matrix operations require at least 2-dimensional tensors (matrices). The LHS (left-hand side) operand must have rank >= 2. The compiler raises this error when the LHS tensor is a scalar or 1-D vector.

To fix this error, reshape the LHS to have at least 2 dimensions.
