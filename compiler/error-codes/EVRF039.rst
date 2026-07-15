.. _error-code-evrf039:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF039.

NCC_EVRF039
===========

**Error message**: QuantizeMX custom call block_size must be 32.

The ``QuantizeMX`` custom call implements OCP MXFP-8 microscaling quantization (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). The OCP MXFP specification requires a block size of 32 elements per scaling factor. The compiler raises this error when a different block_size value is provided in the backend_config.

To fix this error, use ``block_size=32``.
