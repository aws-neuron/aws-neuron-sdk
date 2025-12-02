.. meta::
    :description: API reference for the Output Projection CTE kernel included in the NKI Library .
    :date-modified: 11/28/2025

.. currentmodule:: nkilib.core.output_projection.output_projection_cte

Output Projection CTE Kernel API Reference
===========================================

This topic provides the API reference for the ``Output Projection CTE`` kernel. The kernel computes the output projection operation typically used after an attention block in transformer models, optimized for Context Encoding (Prefill) use cases.

The kernel supports:

* Efficient projection of attention outputs
* Optional bias addition
* LNC sharding for distributed computation
* Optimized memory access patterns
* Head dimension packing for improved performance

Background
--------------

The ``Output Projection CTE`` kernel computes the operation ``out = attention @ weight + bias``, which is commonly used to project the output scores after an attention block in transformer models. This kernel is specifically optimized for Context Encoding (Prefill) use cases, where the sequence length can be large (typically ``S`` ≥ 512).

The kernel employs efficient tiling strategies and memory access patterns to maximize performance on Neuron hardware, with support for sharding across multiple Logical Neuron Cores (LNCs) to handle large hidden dimensions. When ``LNC>1``, the ``H`` dimension is sharded across the cores, which avoids the need for any inter-core collective operations as each core produces part of the output tensor.

API Reference
----------------

**Source code for this kernel API can be found at**: https://github.com/aws-neuron/nki-library

output_projection_cte
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: output_projection_cte(attention, weight, bias=None)

   Output Projection Kernel optimized for Context Encoding (Prefill) use cases.

   This kernel computes ``out = attention @ weight + bias``, typically used to project the output scores after an attention block in transformer models.

   This kernel is optimized for Context Encoding (aka Prefill) use cases where sequence length ``S`` is large. Using this kernel with ``S < 512`` may result in degraded performance.

   This kernel uses a layout also used by other Context Encoding kernels to avoid need for transposes.

   :param attention: Input tensor in HBM, typically the scores output from an attention block. Shape: ``[B, N, D, S]``, where ``B`` is batch size, ``N`` is number of heads, ``D`` is head dimension, and ``S`` is sequence length. Indexing: ``[b, n, d, s]``.
   :type attention: ``nl.ndarray``
   :param weight: Weight tensor in HBM. Shape: ``[N*D, H]``, where ``H`` is hidden dimension size. Indexing: ``[n * D + d, h]``.
   :type weight: ``nl.ndarray``
   :param bias: Optional bias tensor in HBM. Shape: ``[1, H]``. Indexing: ``[1, h]``.
   :type bias: ``nl.ndarray``, optional
   :return: Output tensor in HBM. Shape: ``[B, S, H]``. Indexing: ``[b, s, h]``.
   :rtype: ``nl.ndarray``

   **Data Types**:
     This kernel supports ``nl.float32``, ``nl.float16`` and ``nl.bfloat16`` data types.
     However, for ``nl.float32``, large inputs may not fit in SBUF.

   **Dimensions**:
     * ``B``: Batch size
     * ``N``: Number of heads
     * ``S``: Sequence length
     * ``H``: Hidden dimension size
     * ``D``: Head dimension size

   **Restrictions**:

   * The contract dimension of input and weight tensors must match (``N*D == weight.shape[0]``)
   * Output projection kernel currently only supports ``H`` to be no more than 32768
   * Hidden dimension (``H``) needs to be divisible by LNC size since LNC sharding is on the weight hidden dimension
   * Head dimension (``D``) must be <= 128
   * Maximum validated ``H`` size is 20705
   * Maximum validated ``B*S`` size is 131072
   * Maximum validated ``N`` size is 17

Implementation Details
-------------------------

The kernel implementation includes several key optimizations:

1. **Dimension Packing**: Optimizes the contraction dimension by folding ``N`` (number of heads) into ``D`` (head dimension) when beneficial, improving computational efficiency.

2. **Efficient Tiling Strategy**: Uses carefully chosen tile sizes for processing batches and sequences to maximize hardware utilization.

3. **LNC Sharding**: Supports sharding across multiple Logical Neuron Cores (LNCs) by dividing the hidden dimension, enabling processing of larger models.

4. **Memory Access Optimization**: Employs optimized memory access patterns to maximize bandwidth utilization and minimize data movement.

5. **PSUM Bank Utilization**: Efficiently utilizes PSUM banks for accumulating partial results during matrix multiplication operations.

6. **Stream Shuffle Broadcast**: Uses stream shuffle broadcast for bias tensors to efficiently distribute them across processing elements.

7. **Specialized Engine Selection**: Alternates between scalar and vector engines for tensor copy operations to balance workload and improve performance.

See Also
-----------

* :doc:`Output Projection TKG Kernel API Reference </nki/library/api/output-projection-tkg>`
* :doc:`QKV Kernel API Reference </nki/library/api/qkv>`
