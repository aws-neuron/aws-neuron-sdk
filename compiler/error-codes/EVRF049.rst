.. _error-code-evrf049:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF049.

NCC_EVRF049
===========

**Error message**: ScaledMatmul custom call could not parse backend_config.

The ``__op$block_scaled_dot`` custom call performs matrix multiplication on MXFP8-quantized tensors. It requires a ``backend_config`` attribute containing valid JSON with a ``scaled_dot_backend_config`` object. The compiler raises this error when the backend_config JSON is malformed.

EVRF049 fires only when the ``backend_config`` JSON itself is malformed; missing fields default rather than triggering this error. Output dtype is determined by the result-type element type, not by a JSON field.

The optional fields in ``scaled_dot_backend_config`` with their defaults are:

- ``lhs_batch_dimensions``: array of batch dimension indices for LHS (default ``[]``)
- ``rhs_batch_dimensions``: array of batch dimension indices for RHS (default ``[]``)
- ``lhs_contracting_dimensions``: array of contracting dimension indices for LHS (default ``[]``)
- ``rhs_contracting_dimensions``: array of contracting dimension indices for RHS (default ``[]``)
- ``element_dtype``: FP8 element dtype ``"float8_e5m2"`` or ``"float8_e4m3fn"`` if specified (default empty string)

To fix this error, provide properly-formatted JSON for the ``scaled_dot_backend_config`` object.
