.. _error-code-evrf036:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF036.

NCC_EVRF036
===========

**Error message**: QuantizeMX custom call has invalid backend_config JSON.

The ``QuantizeMX`` custom call requires a ``backend_config`` attribute that
contains valid JSON. The compiler raises this error when the attribute is
missing or cannot be parsed as JSON.

The ``dim``, ``block_size``, and ``scale_method`` fields are optional and
default to ``0``, ``32``, and ``"EMAX"``, respectively. A missing or unsupported
``dtype`` is reported as :ref:`NCC_EVRF042 <error-code-evrf042>`.

To fix this error, provide a valid JSON object in ``backend_config``.
