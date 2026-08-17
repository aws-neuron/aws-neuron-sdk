.. _error-code-evrf064:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF064.

NCC_EVRF064
===========

**Error message**: HLO verification failed since the input HLO graph contains an illegal instruction that violates XLA shape constraints.

This error occurs when the compiler's internal HLO verifier detects that an instruction in the graph has operands or output shapes that violate XLA's shape rules. This typically means the graph was constructed incorrectly by the framework before reaching the compiler.

Common causes
--------------

- Operand shapes that are incompatible with the operation (such as mismatched dimensions in element-wise operations).
- Incorrect shape inference from custom graph transformations or manual HLO construction.

Resolution
-----------

1. **Check the neuronx-cc logs** for the specific XLA verification error message, which will identify the exact instruction and the shape mismatch.
2. **Inspect the reported instruction** and ensure all operand shapes are valid for the given operation.
3. If you are constructing HLO directly (e.g., via ``stablehlo`` or XLA builder APIs), verify that all shape constraints are satisfied for each operation.
4. If the error occurs from a standard framework (PyTorch, JAX), ensure you are using a compatible version of the framework and ``neuronx-cc``.
