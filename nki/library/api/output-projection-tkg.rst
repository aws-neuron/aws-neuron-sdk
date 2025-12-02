.. meta::
    :description: API reference for the Output Projection TKG kernel included in the NKI Library .
    :date-modified: 11/28/2025

.. currentmodule:: nkilib.core.output_projection.output_projection_tkg

Output Projection TKG Kernel API Reference
===========================================

This topic provides the API reference for the ``Output Projection TKG`` kernel. The kernel computes the output projection operation typically used after an attention block in transformer models, optimized for Token Generation (Decode) use cases.

The kernel supports:

* Efficient projection of attention outputs
* Optional bias addition
* LNC sharding for distributed computation
* Optimized memory access patterns
* Head dimension packing for improved performance
* Flexible output tensor layouts
* SBUF output option for kernel fusion

Background
--------------

The ``Output Projection TKG`` kernel computes the operation ``out = attention @ weight + bias``, which is commonly used to project the output scores after an attention block in transformer models. This kernel is specifically optimized for Token Generation (Decode) use cases, where the sequence length ``S`` is small (often 1 or a small number for speculative decoding).

The kernel employs efficient tiling strategies and memory access patterns to maximize performance on Neuron hardware, with support for sharding across multiple Logical Neuron Cores (LNCs) to handle large hidden dimensions. When ``LNC>1``, the ``H`` dimension is sharded across the cores, which avoids the need for any inter-core collective operations as each core produces part of the output tensor.

The input layouts expected for this kernel are different from those for the CTE kernel. In TKG workloads, the ``S`` dimension is small, so placing the ``N`` dimension next to it allows more efficient GQA implementations by loading multiple heads at once.

API Reference
----------------

**Source code for this kernel API can be found at**: https://github.com/aws-neuron/nki-library

output_projection_tkg
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: output_projection_tkg(attention, weight, bias, TRANSPOSE_OUT=False, OUT_IN_SB=False)

   Output Projection Kernel optimized for Token Generation (Decode) use cases.

   This kernel computes ``out = attention @ weight + bias``, typically used to project the output scores after an attention block in transformer models.

   This kernel is optimized for Token Generation (aka Decode) use cases where sequence length ``S`` is small.

   :param attention: Input tensor in HBM or SBUF, typically the scores output from an attention block. Shape: ``[D, B, N, S]``, where ``D`` is head dimension, ``B`` is batch size, ``N`` is number of heads, and ``S`` is sequence length. Indexing: ``[d, b, n, s]``.
   :type attention: ``nl.ndarray``
   :param weight: Weight tensor in HBM. Shape: ``[N*D, H]``, where ``H`` is hidden dimension size. Indexing: ``[n * D + d, h]``.
   :type weight: ``nl.ndarray``
   :param bias: Optional bias tensor in HBM. Shape: ``[1, H]``. Indexing: ``[1, h]``.
   :type bias: ``nl.ndarray``
   :param TRANSPOSE_OUT: Whether to store the output in transposed shape. If ``False``, output shape is ``[B*S, H]`` with indexing ``[b*S+s, h]``. If ``True``, output shape is ``[H_1, H_0, H_2, B*S]`` with indexing ``[h_1, h_0, h_2, b*S+s]``, where ``H_0 = logical core size (LNC)``, ``H_1 = 128``, ``H_2 = H/(H_0*H_1)``, such that ``h = h_0*H_1*H_2 + h_1*H_2 + h_2``.
   :type TRANSPOSE_OUT: ``bool``
   :param OUT_IN_SB: If ``True``, output is in SBUF. Else, it is written out to HBM.
   :type OUT_IN_SB: ``bool``
   :return: Output tensor in HBM or SBUF. Shape depends on ``TRANSPOSE_OUT`` parameter.
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
   * Hidden dimension (``H``) needs to be divisible by LNC size since LNC sharding is on the weight hidden dimension
   * ``B*S`` must be <= 128
   * Head dimension (``D``) must be <= 128
   * When ``TRANSPOSE_OUT`` is ``False``, ``H`` must be a multiple of ``512*LNC``
   * When ``TRANSPOSE_OUT`` is ``True``, ``H`` must be a multiple of ``128*LNC``
   * When ``TRANSPOSE_OUT`` is ``True`` and using 32-bit floats, ``N*H`` must be <= 81920
   * When ``TRANSPOSE_OUT`` is ``True`` and using 16-bit floats, ``N*H`` must be <= 163840

Implementation Details
-------------------------

The kernel implementation includes several key optimizations:

1. **Dimension Packing**: Optimizes the contraction dimension by folding ``N`` (number of heads) into ``D`` (head dimension) when beneficial, improving computational efficiency.

2. **Efficient Tiling Strategy**: Uses carefully chosen tile sizes for processing batches and sequences to maximize hardware utilization.

3. **LNC Sharding**: Supports sharding across multiple Logical Neuron Cores (LNCs) by dividing the hidden dimension, enabling processing of larger models.

4. **Memory Access Optimization**: Employs optimized memory access patterns to maximize bandwidth utilization and minimize data movement.

5. **PSUM Bank Utilization**: Efficiently utilizes PSUM banks for accumulating partial results during matrix multiplication operations.

6. **Stream Shuffle Broadcast**: Uses stream shuffle broadcast for bias tensors to efficiently distribute them across processing elements.

7. **Flexible Output Layouts**: Supports both standard and transposed output layouts to accommodate different downstream kernel requirements.

8. **SBUF Output Option**: Provides the option to keep output in SBUF for fusion with subsequent operations.

9. **Block-based Weight Loading**: Uses block-based loading of weights to encourage prefetching and improve memory access patterns.

See Also
-----------

* :doc:`Output Projection CTE Kernel API Reference </nki/library/api/output-projection-cte>`
* :doc:`QKV Kernel API Reference </nki/library/api/qkv>`
