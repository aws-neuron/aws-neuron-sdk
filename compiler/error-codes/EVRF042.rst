.. _error-code-evrf042:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF042.

NCC_EVRF042
===========

**Error message**: QuantizeMX custom call is malformed.

The ``QuantizeMX`` custom call implements OCP MXFP-8 microscaling quantization (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
and produces packed FP8 data and block scale factors. The logical FP8 format is
selected by the ``dtype`` field, but the ``quantized_data`` output is physically
represented as ``U32``, with four FP8 values packed into each element.

The compiler raises this error when:

1. ``backend_config`` omits ``dtype`` or specifies a value other than
   ``"float8_e5m2"`` or ``"float8_e4m3fn"``.
2. The ``quantized_data`` output element type is not ``U32``.
3. The ``quantized_data`` rank matches the input rank, but a dimension has the
   wrong size. Each dimension must match the input, except the quantization
   dimension, which is divided by four for FP8 x4 packing.

To fix this error, use a supported logical ``dtype`` and declare
``quantized_data`` as a ``U32`` tensor whose shape matches the input except
that the quantization dimension is one fourth of the input size.
