.. _error-code-evrf057:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF057.

NCC_EVRF057
===========

**Error message**: QuantizeMX custom call must return a tuple with exactly 2 outputs.

The ``QuantizeMX`` custom call implements OCP MXFP-8 microscaling quantization (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) and must produce a 2-element tuple: the quantized data tensor and the per-block scale tensor. The compiler raises this error when the result type is not a 2-element tuple.

To fix this error, declare a 2-element tuple result type ``(quantized_data, scale)``.
