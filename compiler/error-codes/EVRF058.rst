.. _error-code-evrf058:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF058.

NCC_EVRF058
===========

**Error message**: QuantizeMX custom call input dimension must be divisible by 4.

The ``QuantizeMX`` custom call implements OCP MXFP-8 microscaling quantization (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). The quantization dimension size must be divisible by 4. The compiler raises this error when the input tensor's quantization dimension size is not a multiple of 4.

To fix this error, pad or reshape the input tensor so the quantization dimension size is a multiple of 4.
