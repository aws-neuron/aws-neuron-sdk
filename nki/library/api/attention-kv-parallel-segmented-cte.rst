.. meta::
    :description: KV-parallel segmented prefill attention.
    :date-modified: 06/11/2026

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

**Source code for this kernel API can be found at**: `attention_kv_parallel_segmented_cte.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_kv_parallel_segmented_cte.py>`_

attention_kv_parallel_segmented_cte
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: attention_kv_parallel_segmented_cte(q: nl.ndarray, k_cache: nl.ndarray, v_cache: nl.ndarray, block_tables: nl.ndarray, kvp_q_offset: nl.ndarray, replica_groups: ReplicaGroup, group_size: int, block_size: int, seg_size: int, scale: float = 1.0, global_q_offset: int = 0, tp_out: bool = False, sliding_window: int = 0, kvp_rank_id: Optional[nl.ndarray] = None, kvp_group_size: int = 0) -> nl.ndarray

   KV-parallel segmented prefill attention.

   :param q: [BS, S, D], This rank's Q heads (BS = lnc_degree).
   :type q: ``nl.ndarray``
   :param k_cache: [num_blocks, num_kv_heads, block_size, D], Local KV cache (K).
   :type k_cache: ``nl.ndarray``
   :param v_cache: [num_blocks, num_kv_heads, block_size, D], Local KV cache (V).
   :type v_cache: ``nl.ndarray``
   :param block_tables: [1, max_blocks] int32, Block indices for paged KV.
   :type block_tables: ``nl.ndarray``
   :param kvp_q_offset: [1, 1] int32, Causal mask offset = -rank_id * local_kv_len + global_q_offset. For round-robin KV distribution, set to just global_q_offset since the global-to-local bound conversion handles K-side positioning.
   :type kvp_q_offset: ``nl.ndarray``
   :param replica_groups: ReplicaGroup for collective operations.
   :type replica_groups: ``ReplicaGroup``
   :param group_size: Number of ranks in the replica group.
   :type group_size: ``int``
   :param block_size: KV cache block size.
   :type block_size: ``int``
   :param seg_size: Segment size for attention iteration.
   :type seg_size: ``int``
   :param scale: Attention scale factor (default 1.0).
   :type scale: ``float``
   :param global_q_offset: Global token position of Q token 0 (default 0). Used to compute how many prior KV tokens exist within this rank's shard for each Q chunk.
   :type global_q_offset: ``int``
   :param tp_out: If True, output is transposed to [BS, D, S] (default False).
   :type tp_out: ``bool``
   :param sliding_window: Sliding window size for attention (0 = disabled).
   :type sliding_window: ``int``
   :param kvp_rank_id: [1, 1] int32, This rank's index within the KV-parallel group. Required for interleaved (round-robin) KV distribution to convert global K positions to segment-local positions.
   :type kvp_rank_id: ``Optional[nl.ndarray]``
   :param kvp_group_size: Number of ranks sharing the KV cache in round-robin fashion. When > 0, enables interleaved KV mode where rank r holds global blocks r, r+R, r+2R, ... (R = kvp_group_size).
   :type kvp_group_size: ``int``
   :return: [BS, S, D], Merged attention output for this rank's Q heads.
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * BS: Batch size (lnc_degree = Q heads per physical rank)
   * S: Sequence length
   * D: Head dimension
   * G: Group size (number of ranks per replica group)

