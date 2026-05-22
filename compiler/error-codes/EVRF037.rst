.. _error-code-evrf037:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF037.

NCC_EVRF037
===========

**Error message**: QuantizeMX custom call operand count must be exactly 1 (input tensor).

The ``QuantizeMX`` custom call takes a single input tensor and produces a
tuple of two outputs (the quantized data and the per-block scale). The
compiler raises this error when the call has any number of operands other
than one.

To fix this error, pass exactly one input tensor as the operand.
