.. _error-code-evrf038:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF038.

NCC_EVRF038
===========

**Error message**: QuantizeMX custom call dim is invalid for input tensor rank.

The ``QuantizeMX`` custom call implements OCP MXFP-8 microscaling quantization (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) and produces a tuple of (quantized_data, scale). The ``dim`` parameter specifies which dimension to quantize along. Only the last dimension (dim=-1) or second-to-last dimension (dim=-2) are supported.

To fix this error, use dim=-1 (last dimension) or dim=-2 (second-to-last dimension).
