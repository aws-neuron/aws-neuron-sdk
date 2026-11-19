.. _error-code-edvr003:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EDVR003.

NCC_EDVR003
===========

**Error message**: Linking subgraphs without a call graph is not supported.

When the compiler splits a model into several subgraphs, it compiles each one separately and then links them into one program. It links them through a call graph: a main function that calls each compiled subgraph in turn. The other way of linking, which merges the subgraphs into a single function, is no longer supported, so the compiler stops with this error instead of using it.

This error occurs only when the internal compiler option ``--disable-internal-call-graph`` is set and the model has more than one subgraph to link.

Resolution
----------

Remove ``--disable-internal-call-graph`` from your compiler flags, for example from ``NEURON_CC_FLAGS``. Linking through a call graph is the default.
