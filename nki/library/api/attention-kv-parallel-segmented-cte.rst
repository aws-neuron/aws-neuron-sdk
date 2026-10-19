.. meta::
    :description: KV-parallel segmented prefill attention.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.core.attention

Attention KV-Parallel Segmented CTE Kernel API Reference
========================================================

KV-parallel segmented prefill attention.

Distributes attention computation across ranks, where each rank holds a shard of the KV cache. Uses online softmax to merge partial results.

Background
-----------

The ``attention_kv_parallel_segmented_cte`` kernel enables context parallelism for prefill by distributing the KV cache across ranks. Each rank computes attention over its local KV shard in segments, then the partial results are merged across ranks using online softmax. It supports paged (block-based) KV cache, sliding-window masking, and both contiguous and interleaved (round-robin) KV distribution. This kernel replaces the earlier ``kv_parallel_segmented_prefill`` kernel.

API Reference
--------------

**Source code for this kernel API can be found at**: `attention_kv_parallel_segmented_cte.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/core/attention/attention_kv_parallel_segmented_cte.py>`_

attention_kv_parallel_segmented_cte
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: attention_kv_parallel_segmented_cte(q: nl.NkiTensor, k_cache: nl.NkiTensor, v_cache: nl.NkiTensor, block_tables: nl.NkiTensor, kvp_q_offset: nl.NkiTensor, replica_groups: ncc.ReplicaGroup, group_size: int, block_size: int, seg_size: int, scale: float = 1.0, global_q_offset: int = 0, tp_out: bool = False, sliding_window: int = 0, kvp_rank_id: Optional[nl.NkiTensor] = None, kvp_group_size: int = 0, apc_mode: bool = False, valid_num_prior_tokens: nl.NkiTensor = None, fp8_packed: bool = False, k_scale: Optional[nl.NkiTensor] = None, v_scale: Optional[nl.NkiTensor] = None) -> nl.NkiTensor

   KV-parallel segmented prefill attention.

   :param q: [q_heads_per_rank, S, D], This rank's Q heads.
   :type q: ``nl.NkiTensor``
   :param k_cache: Local KV cache (K). Shape depends on fp8_packed: - False: [num_blocks, num_kv_heads, block_size, D] - True: [num_blocks, num_kv_heads, block_size // 2, D, 2]
   :type k_cache: ``nl.NkiTensor``
   :param v_cache: [num_blocks, num_kv_heads, block_size, D], Local KV cache (V).
   :type v_cache: ``nl.NkiTensor``
   :param block_tables: [1, max_blocks] int32, Block indices for paged KV.
   :type block_tables: ``nl.NkiTensor``
   :param kvp_q_offset: [1, 1] int32, Causal mask offset. For contiguous KV sharding: -rank_id * local_kv_len + global_q_offset. For round-robin KV distribution: the actual global Q position (runtime), since the kernel uses kvp_rank_id to handle K-side positioning internally.
   :type kvp_q_offset: ``nl.NkiTensor``
   :param replica_groups: ReplicaGroup for collective operations.
   :type replica_groups: ``ncc.ReplicaGroup``
   :param group_size: Number of logical ranks in the replica group (= len(replica_groups)).
   :type group_size: ``int``
   :param block_size: KV cache block size.
   :type block_size: ``int``
   :param seg_size: Segment size for attention iteration.
   :type seg_size: ``int``
   :param scale: Attention scale factor (default 1.0).
   :type scale: ``float``
   :param global_q_offset: Compile-time global position of Q token 0 (default 0). In non-APC mode, this is required for correctness — it determines how many prior KV segments are loaded per chunk. In APC mode, set to 0 since the prior token count is derived at runtime from kvp_q_offset instead.
   :type global_q_offset: ``int``
   :param tp_out: If True, output is transposed to [q_heads_per_rank, D, S] (default False).
   :type tp_out: ``bool``
   :param sliding_window: Sliding window size for attention (0 = disabled).
   :type sliding_window: ``int``
   :param kvp_rank_id: [1, 1] int32, This rank's index within the KV-parallel group. Required for interleaved (round-robin) KV distribution to convert global K positions to segment-local positions.
   :type kvp_rank_id: ``Optional[nl.NkiTensor]``
   :param kvp_group_size: Number of ranks sharing the KV cache in round-robin fashion. When > 0, enables interleaved KV mode where rank r holds global blocks r, r+R, r+2R, ... (R = kvp_group_size).
   :type kvp_group_size: ``int``
   :param apc_mode: Automated prefix caching mode (default False). When True, prior token count is taken from valid_num_prior_tokens (required). Compile-time tile-visibility optimizations are disabled since global_q_offset may not reflect the actual prefix length. Set global_q_offset=0.
   :type apc_mode: ``bool``
   :param valid_num_prior_tokens: [1, 1] int32. Required when apc_mode is True. The number of fully-visible local prior tokens for Q chunk 0 on this rank. Must be a multiple of block_size. The kernel increments this per chunk by seg_size // kvp_group_size as later chunks see more local KV.
   :type valid_num_prior_tokens: ``nl.NkiTensor``
   :param fp8_packed: If True, K uses the packed FP8 layout described above.
   :type fp8_packed: ``bool``
   :param k_scale: Optional K-cache dequantization scale, shape [128, 1].
   :type k_scale: ``Optional[nl.NkiTensor]``
   :param v_scale: Optional V-cache dequantization scale, shape [128, 1]. All entries must repeat one scalar; the kernel applies this tensor along the query partition, not as per-head-dimension values.
   :type v_scale: ``Optional[nl.NkiTensor]``
   :return: [q_heads_per_rank, S, D], Merged attention output for this rank's Q heads.
   :rtype: ``nl.NkiTensor``

   **Dimensions**:

   * BS: Batch size (lnc_degree = Q heads per physical rank)
   * S: Sequence length
   * D: Head dimension
   * G: Group size (number of ranks per replica group)

