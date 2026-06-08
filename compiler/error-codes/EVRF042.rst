.. _error-code-evrf042:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF042.

NCC_EVRF042
===========

**Error message**: QuantizeMX custom call validation failed.

The ``QuantizeMX`` custom call implements OCP MXFP-8 microscaling quantization (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) and produces a tuple of (quantized_data, scale). This error aggregates several output type and shape validation failures. The compiler raises this error when:

1. The backend_config ``dtype`` field is not ``float8_e5m2`` or ``float8_e4m3fn``
2. The quantized_data element type does not match the expected FP8 dtype
3. The quantized_data shape does not match the expected dimensions

The specific failure is one of:

- the ``backend_config`` ``dtype`` is not ``float8_e5m2`` or ``float8_e4m3fn``
- the ``quantized_data`` element type does not match the ``dtype`` declared in ``backend_config``
- the ``quantized_data`` shape does not match the input tensor shape

To fix this error, ensure the ``QuantizeMX`` result tuple uses the FP8 element type that matches the ``backend_config`` ``dtype`` field, with a shape that matches the input tensor.
