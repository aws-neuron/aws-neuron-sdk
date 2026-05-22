.. meta::
    :description: KV-parallel segmented prefill attention.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.core.attention

KV Parallel Segmented Prefill Kernel API Reference
===================================================

KV-parallel segmented prefill attention.

Distributes attention computation across ranks, where each rank holds a shard of the KV cache. Uses online softmax to merge partial results.

Background
-----------

The ``kv_parallel_segmented_prefill`` kernel implements KV-parallel segmented prefill attention, distributing the attention computation across ranks where each rank holds a shard of the KV cache and using online softmax to merge partial results.

API Reference
--------------

**Source code for this kernel API can be found at**: `kv_parallel_segmented_prefill.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/kv_parallel_segmented_prefill.py>`_

kv_parallel_segmented_prefill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: kv_parallel_segmented_prefill(q: nl.ndarray, k_cache: nl.ndarray, v_cache: nl.ndarray, block_tables: nl.ndarray, kvp_offset: nl.ndarray, replica_groups: ReplicaGroup, group_size: int, block_size: int, seg_size: int, scale: float = 1.0, global_q_offset: int = 0, tp_out: bool = False) -> nl.ndarray

   KV-parallel segmented prefill attention.

   :param q: [BS, S, D], This rank's Q heads (BS = lnc_degree).
   :type q: ``nl.ndarray``
   :param k_cache: [num_blocks, num_kv_heads, block_size, D], Local KV cache (K).
   :type k_cache: ``nl.ndarray``
   :param v_cache: [num_blocks, num_kv_heads, block_size, D], Local KV cache (V).
   :type v_cache: ``nl.ndarray``
   :param block_tables: [1, max_blocks] int32, Block indices for paged KV.
   :type block_tables: ``nl.ndarray``
   :param kvp_offset: [1, 1] int32, Causal mask offset = -rank_id * local_kv_len + global_q_offset.
   :type kvp_offset: ``nl.ndarray``
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
   :return: [BS, S, D], Merged attention output for this rank's Q heads.
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * BS: Batch size (lnc_degree = Q heads per physical rank)
   * S: Sequence length
   * D: Head dimension
   * G: Group size (number of ranks per replica group)

