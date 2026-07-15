.. _error-code-evrf038:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF038.

NCC_EVRF038
===========

**Error message**: QuantizeMX custom call dim is invalid for input tensor rank.

The ``dim`` parameter specifies the input dimension to quantize. The compiler
supports the last dimension for inputs with rank 1 or greater and the
second-to-last dimension for inputs with rank 2 or greater. Specify the
dimension as ``-1`` or ``-2``, or use the corresponding non-negative index.

To fix this error, use ``dim=-1`` or ``rank - 1``. For inputs with rank 2 or
greater, you can also use ``dim=-2`` or ``rank - 2``.
