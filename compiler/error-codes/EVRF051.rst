.. _error-code-evrf051:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF051.

NCC_EVRF051
===========

**Error message**: Data type F8E4M3FN is not supported on TRN1/TRN2.

The F8E4M3FN (8-bit floating point with 4-bit exponent and 3-bit mantissa) data type is only supported on Trainium3 (Trn3) and later hardware generations. The compiler raises this error when a model uses F8E4M3FN quantization but targets Trn1 or Trn2 architectures.

To fix this error, either target Trn3 or use the experimental flag to cast F8E4M3FN to F8E4M3 (``--experimental-unsafe-fp8e4m3fn-as-fp8e4m3``).

* More information on supported data types: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html
