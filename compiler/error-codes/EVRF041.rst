.. _error-code-evrf041:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF041.

NCC_EVRF041
===========

**Error message**: QuantizeMX custom call input type is unsupported.

The ``QuantizeMX`` custom call implements OCP MXFP-8 microscaling quantization (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). Only BF16 and F16 input tensors are supported for quantization. The compiler raises this error when the input tensor has a different element type.

To fix this error, cast the input tensor to BF16 or F16 before quantization.
