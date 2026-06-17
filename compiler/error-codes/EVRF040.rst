.. _error-code-evrf040:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF040.

NCC_EVRF040
===========

**Error message**: QuantizeMX custom call scale_method is unsupported.

The ``QuantizeMX`` custom call implements OCP MXFP-8 microscaling quantization (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). Currently, only the 'EMAX' scale computation method is supported. The compiler raises this error when a different scale_method value is provided in the backend_config.

To fix this error, use ``scale_method='EMAX'``.
