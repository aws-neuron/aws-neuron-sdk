.. _error-code-evrf044:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF044.

NCC_EVRF044
===========

**Error message**: ScaledMatmul custom call LHS input type is unsupported.

The ``__op$block_scaled_dot`` custom call performs matrix multiplication on
MXFP8-quantized tensors. The LHS (left-hand side) operand must be a ``U32``
tensor containing four packed FP8 values per element, as produced by the
``QuantizeMX`` custom call. The compiler raises this error when the LHS operand
has a different element type.

To fix this error, use the packed ``U32`` quantized data tensor returned by
``QuantizeMX`` as the LHS operand.
