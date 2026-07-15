.. _error-code-evrf053:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF053.

NCC_EVRF053
===========

**Error message**: ScaledMatmul custom call contracting dimension overlaps with batch dimension.

The ``__op$block_scaled_dot`` custom call performs batched matrix multiplication
on MXFP8-quantized tensors. Batch dimensions and contracting dimensions must be
disjoint. The compiler checks the first configured contracting dimension for
each operand, or the last dimension when none is configured, and raises this
error when that dimension also appears in the operand's batch dimension list.

To fix this error, ensure batch dimensions and contracting dimensions are disjoint.
