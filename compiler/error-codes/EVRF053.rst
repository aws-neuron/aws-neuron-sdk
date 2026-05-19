.. _error-code-evrf053:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF053.

NCC_EVRF053
===========

**Error message**: ScaledMatmul custom call contracting dimension overlaps with batch dimension.

The ``__op$block_scaled_dot`` custom call performs batched matrix multiplication on MXFP8-quantized tensors. Batch dimensions and contracting dimensions must be disjoint sets. A dimension cannot be both a batch dimension and a contracting dimension. The compiler raises this error when a contracting dimension index also appears in the batch dimensions list.

To fix this error, ensure batch dimensions and contracting dimensions are disjoint.
