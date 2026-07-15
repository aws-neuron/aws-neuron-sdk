.. _error-code-evrf052:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF052.

NCC_EVRF052
===========

**Error message**: Data type F8E4M3 is not supported on hardware newer than Trn3.

The compiler raises this error when an operation has an F8E4M3 operand and
targets a hardware generation later than Trn3. Later hardware generations
support the F8E4M3FN format instead.

To fix this error, use F8E4M3FN instead of F8E4M3.

* More information on supported data types: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html
