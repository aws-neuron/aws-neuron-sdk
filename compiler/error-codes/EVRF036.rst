.. _error-code-evrf036:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF036.

NCC_EVRF036
===========

**Error message**: QuantizeMX custom call has invalid backend_config JSON.

The ``QuantizeMX`` custom call requires a ``backend_config`` attribute that
is a valid JSON string with the fields ``dtype``, ``dim``, ``block_size`` and
``scale_method``. The compiler raises this error when the attribute string
cannot be parsed as JSON.

To fix this error, ensure ``backend_config`` is valid JSON. The logical fields validated by separate downstream errors (EVRF038/039/040/041) are ``dtype``, ``dim``, ``block_size``, ``scale_method``.
