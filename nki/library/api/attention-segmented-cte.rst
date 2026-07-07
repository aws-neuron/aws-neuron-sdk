.. meta::
    :description: Segmented attention computation with block-based KV cache and prefix caching.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.core.attention

Attention Segmented CTE Kernel API Reference
============================================

Segmented attention computation with block-based KV cache and prefix caching support optimized for Context Encoding.

Background
-----------

The ``attention_segmented_cte`` kernel implements segmented attention that processes the KV cache in configurable segments, supporting block-based KV cache layout and prefix caching. It includes helper kernels for floor/ceil operations needed for position-based computations and a KV cache loader that reads blocks from page tables into SBUF.

API Reference
--------------

**Source code for this kernel API can be found at**: `attention_segmented_cte.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_segmented_cte.py>`_

floor_nisa_kernel
^^^^^^^^^^^^^^^^^

.. py:function:: floor_nisa_kernel(src_t: nl.ndarray, dst_t: nl.ndarray, p_size: int, f_size: int, allocator: ModularAllocator)

   NISA implementation for floor operation using integer casting.

   :param src_t: Source tensor to compute floor of (dtype: fp32)
   :type src_t: ``nl.ndarray``
   :param dst_t: Destination tensor for floor result (dtype: int32)
   :type dst_t: ``nl.ndarray``
   :param p_size: First dimension size
   :type p_size: ``int``
   :param f_size: Second dimension size
   :type f_size: ``int``
   :param allocator: SBUF allocator for temporary tensors
   :type allocator: ``ModularAllocator``

ceil_nisa_kernel
^^^^^^^^^^^^^^^^

.. py:function:: ceil_nisa_kernel(src_t: nl.ndarray, dst_t: nl.ndarray, p_size: int, f_size: int, allocator: ModularAllocator)

   NISA implementation for ceil operation using floor.

   :param src_t: Source tensor to compute ceil of (dtype: fp32)
   :type src_t: ``nl.ndarray``
   :param dst_t: Destination tensor for ceil result (dtype: int32)
   :type dst_t: ``nl.ndarray``
   :param p_size: First dimension size
   :type p_size: ``int``
   :param f_size: Second dimension size
   :type f_size: ``int``
   :param allocator: SBUF allocator for temporary tensors
   :type allocator: ``ModularAllocator``

load_kv_cache
^^^^^^^^^^^^^

.. py:function:: load_kv_cache(k_cache, v_cache, block_tables, k_sbuf, v_sbuf, b_i, h_i, block_table_offset, num_blocks, allocator: ModularAllocator, k_pre_transposed: bool = False, fp8_packed: bool = False)

   Load KV cache from block tables to SBUF for a single KV head.

   :param k_cache: K cache in HBM. Shape depends on k_pre_transposed: - False: (num_blocks_total, num_kv_head, block_size, head_dim) - True:  (num_blocks_total * num_kv_head, head_dim, block_size)
   :param v_cache: V cache in HBM with shape (num_blocks_total, num_kv_head, block_size, head_dim)
   :param block_tables: Block table tensor with shape (batch_size, max_blocks_per_seq)
   :param k_sbuf: K SBUF tiles to load into
   :param v_sbuf: V SBUF tiles to load into
   :param b_i: Current sequence index in batch
   :param h_i: Current KV head index
   :param block_table_offset: SBUF tensor (1, 1) indicating the block offset for the current segment
   :param num_blocks: Number of blocks to load
   :param allocator: SBUF allocator for temporary tensor allocation
   :type allocator: ``ModularAllocator``
   :param k_pre_transposed: If True, K cache is already stored in transposed layout (head_dim, block_size) per block, so no transpose is needed during loading.
   :type k_pre_transposed: ``bool``

attention_segmented_cte
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: attention_segmented_cte(q: nl.ndarray, k_cache: nl.ndarray, v_cache: nl.ndarray, block_tables: nl.ndarray, prior_tokens: nl.ndarray, block_size: int, prior_seg_size: int, scale: float = 1.0, tp_q: bool = True, tp_out: bool = False, sliding_window: Optional[int] = None, sink: Optional[nl.ndarray] = None, num_q_heads: int = 1, k_pre_transposed: bool = False, fp8_packed: bool = False, k_scale: Optional[nl.ndarray] = None, v_scale: Optional[nl.ndarray] = None, kvp_q_offset: Optional[nl.ndarray] = None, kvp_rank_id: Optional[nl.ndarray] = None, kvp_group_size: int = 0, kvp_cp_offset_int: int = 0, kvp_seg_block_offset_int: int = 0, kvp_prior_load_blocks: int = 0, kvp_prior_fully_visible: bool = False)

   Segmented attention computation with block-based KV cache and prefix caching.

   :param q: Query tensor with shape (batch_size, seqlen_q, d) when tp_q=True
   :type q: ``nl.ndarray``
   :param k_cache: K cache in HBM. Shape depends on k_pre_transposed: - False: (num_blocks, num_kv_head, block_size, head_dim) - True:  (num_blocks * num_kv_head, head_dim, block_size)
   :type k_cache: ``nl.ndarray``
   :param v_cache: V cache in HBM with shape (num_blocks, num_kv_head, block_size, head_dim)
   :type v_cache: ``nl.ndarray``
   :param block_tables: Block table tensor with shape (batch_size, max_blocks_per_seq). May contain -1 values for padding (triggers DMA skipping). If prior_last_segment_tokens < prior_seg_size, caller should prepend -1 padding.
   :type block_tables: ``nl.ndarray``
   :param prior_tokens: Total number of prior (cached) tokens, shape (1, 1). Must be multiple of block_size.
   :type prior_tokens: ``nl.ndarray``
   :param block_size: Size of each block in the KV cache
   :type block_size: ``int``
   :param prior_seg_size: Size of each KV segment to process iteratively
   :type prior_seg_size: ``int``
   :param scale: Scaling factor for attention scores (default 1.0)
   :type scale: ``float``
   :param tp_q: Query tensor transpose flag (default True)
   :type tp_q: ``bool``
   :param tp_out: Output tensor transpose flag (default False)
   :type tp_out: ``bool``
   :param k_pre_transposed: If True, K cache is already stored in transposed layout (head_dim, block_size) per block, written by _quantize_and_store_k_transposed in qkv_cte.
   :type k_pre_transposed: ``bool``
   :param fp8_packed: If True, the KV cache is stored in FP8-packed layout. Default: ``False``.
   :type fp8_packed: ``bool``
   :param k_scale: Optional per-head-dim dequantization scale for K cache, shape (128, 1). When provided, Q is scaled by k_scale before QK^T matmul (delayed dequant).
   :type k_scale: ``Optional[nl.ndarray]``
   :param v_scale: Optional per-head-dim dequantization scale for V cache, shape (128, 1). When provided, the output is scaled by v_scale after PV matmul normalization.
   :type v_scale: ``Optional[nl.ndarray]``
   :param kvp_q_offset: (KV-parallel) Causal-mask Q offset for this rank's KV shard. Replaces the previous ``kvp_offset`` parameter. Default: ``None``.
   :type kvp_q_offset: ``Optional[nl.ndarray]``
   :param kvp_rank_id: (KV-parallel) This rank's index within the KV-parallel group, used for interleaved (round-robin) KV distribution. Default: ``None``.
   :type kvp_rank_id: ``Optional[nl.ndarray]``
   :param kvp_group_size: (KV-parallel) Number of ranks sharing the KV cache in round-robin fashion (0 = disabled). Default: ``0``.
   :type kvp_group_size: ``int``
   :param kvp_cp_offset_int: (KV-parallel) Static context-parallel offset for this rank's KV shard. Default: ``0``.
   :type kvp_cp_offset_int: ``int``
   :param kvp_seg_block_offset_int: (KV-parallel) Static segment-block offset for this rank's KV shard. Default: ``0``.
   :type kvp_seg_block_offset_int: ``int``
   :param kvp_prior_load_blocks: (KV-parallel) Number of prior blocks to load for this rank. Default: ``0``.
   :type kvp_prior_load_blocks: ``int``
   :param kvp_prior_fully_visible: (KV-parallel) Whether the prior segment is fully visible to all queries on this rank. Default: ``False``.
   :type kvp_prior_fully_visible: ``bool``

