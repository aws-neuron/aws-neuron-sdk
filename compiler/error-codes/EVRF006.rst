.. _error-code-evrf006:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF006.

NCC_EVRF006
===========

The compiler encountered a RNGBitGenerator operation using a random number generation algorithm other than RNG_DEFAULT.
-----------------------------------------------------------------------------------------------------------------------

Ensure that you are using standard JAX/PyTorch random APIs and not explicity specifying an RNG algorithm.
