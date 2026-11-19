.. _error-code-edvr004:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EDVR004.

NCC_EDVR004
===========

**Error message**: Linking subgraphs without DRAM allocation is not supported.

When the compiler links several compiled subgraphs into one program, the tensors that one subgraph passes to the next are kept in device memory (HBM), at addresses the compiler's DRAM allocation assigns. Without DRAM allocation those tensors have nowhere to live between subgraphs, so the compiler stops with this error instead of producing a program it cannot link.

This error occurs only when the internal compiler option ``--internal-disable-dram-allocation`` is set and the model has more than one subgraph to link.

Resolution
----------

Remove ``--internal-disable-dram-allocation`` from your compiler flags, for example from ``NEURON_CC_FLAGS``. DRAM allocation is on by default.
