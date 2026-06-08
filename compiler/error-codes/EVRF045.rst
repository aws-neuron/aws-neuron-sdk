.. _error-code-evrf045:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF045.

NCC_EVRF045
===========

**Error message**: ScaledMatmul custom call output type is unsupported.

The ``__op$block_scaled_dot`` custom call performs matrix multiplication on MXFP8-quantized tensors and dequantizes the result. Only F32 and BF16 output types are supported. The output dtype is controlled by the ``dequantize_type`` field in the backend_config. The compiler raises this error when the result tensor has a different element type.

To fix this error, declare the result as F32 or BF16.
