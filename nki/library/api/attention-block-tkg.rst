.. meta::
    :description: Attention Block TKG kernel implements fused attention block optimized for Token Generation.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.core.attention_block_tkg.attention_block_tkg

.. _nki_library_attention_block_tkg:

Attention Block TKG Kernel API Reference
=========================================

**[Experimental]** Implements a fully fused attention block optimized for Token Generation (autoregressive decoding), keeping all intermediate tensors in SBUF to minimize HBM traffic.

The kernel supports:

* Fused multi-stage computation: pre-normalization, QKV projection, RoPE, post-normalization, attention, KV cache update, and output projection
* Multiple KV cache layouts: flat (transposed/non-transposed) and block-based
* Grouped-Query Attention (GQA) with configurable Q/KV head ratios
* Optional RMSNorm at multiple stages (pre-projection, post-projection per-head)
* Optional Rotary Position Embedding (RoPE) with configurable layouts
* Flexible quantization support (FP8, FP16, BF16)
* FP8 KV cache quantization support
* Configurable softmax scaling factor
* Batch processing with per-batch cache indexing
* Single program multiple data (SPMD) sharding for distributed computation

Background
----------

The ``attention_block_tkg`` kernel combines multiple stages of transformer attention computation into a single fused operation that minimizes data movement between HBM and on-chip memory (SBUF).

**Fused Operations:**

The kernel fuses the following stages in SBUF to avoid HBM round-trips:

1. **Pre-normalization**: Optional RMSNorm on input hidden states
2. **QKV Projection**: Linear projection to Query, Key, Value tensors
3. **RoPE**: Optional Rotary Position Embedding on Q and K
4. **Post-normalization**: Optional per-head RMSNorm on Q and K
5. **Attention Computation**: Scaled dot-product attention with KV cache
6. **KV Cache Update**: Write new K/V tokens to cache
7. **Output Projection**: Linear projection of attention output

**Performance Benefits:**

By keeping intermediate tensors in SBUF throughout the computation, this kernel achieves:

* Reduced HBM bandwidth consumption
* Lower latency for token generation
* Better hardware utilization through operation fusion

API Reference
-------------

**Source code for this kernel API can be found at**: `attention_block_tkg.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/transformer/attention_block_tkg.py>`_

attention_block_tkg
^^^^^^^^^^^^^^^^^^^

.. py:function:: attention_block_tkg(X: nl.NkiTensor, X_hidden_dim_actual: Optional[int], rmsnorm_X_enabled: bool, rmsnorm_X_eps: Optional[float], rmsnorm_X_gamma: Optional[nl.NkiTensor], W_qkv: nl.NkiTensor, bias_qkv: Optional[nl.NkiTensor], quantization_type_qkv: QuantizationType, weight_dequant_scale_qkv: Optional[nl.NkiTensor], input_dequant_scale_qkv: Optional[nl.NkiTensor], rmsnorm_QK_pre_rope_enabled: bool, rmsnorm_QK_pre_rope_eps: float, rmsnorm_QK_pre_rope_W_Q: Optional[nl.NkiTensor], rmsnorm_QK_pre_rope_W_K: Optional[nl.NkiTensor], cos: Optional[nl.NkiTensor], sin: Optional[nl.NkiTensor], rope_contiguous_layout: bool, rmsnorm_QK_post_rope_enabled: bool, rmsnorm_QK_post_rope_eps: float, rmsnorm_QK_post_rope_W_Q: Optional[nl.NkiTensor], rmsnorm_QK_post_rope_W_K: Optional[nl.NkiTensor], K_cache_transposed: bool, active_blocks_table: Optional[nl.NkiTensor], K_cache: nl.NkiTensor, V_cache: nl.NkiTensor, attention_mask: Optional[nl.NkiTensor], sink: Optional[nl.NkiTensor], update_cache: bool, kv_cache_update_idx: Optional[nl.NkiTensor], W_out: Optional[nl.NkiTensor], bias_out: Optional[nl.NkiTensor], quantization_type_out: QuantizationType, weight_dequant_scale_out: Optional[nl.NkiTensor], input_dequant_scale_out: Optional[nl.NkiTensor], transposed_out: bool, out_in_sb: bool, transposed_in: bool = False, softmax_scale: Optional[float] = None, enable_fa_s_prior_tiling: bool = True, fp8_packed: bool = False, k_scale: Optional[nl.NkiTensor] = None, v_scale: Optional[nl.NkiTensor] = None, sbm: Optional[SbufManager] = None, skip_attention: bool = False, is_h_transposed_by_4: bool = False, KVDP: int = 1, KVDP_replica_group: Optional[ReplicaGroup] = None, KVDP_collective_mode: Optional[KVDPCollectiveMode] = None, KVDP_rank: Optional[nl.NkiTensor] = None, pos_ids: Optional[nl.NkiTensor] = None, swa_start_pos_ids: Optional[nl.NkiTensor] = None, S_ctx: Optional[int] = None, max_context_len: Optional[nl.NkiTensor] = None, dtype_mode: DtypeMode = DtypeMode.NON_OCP, CP: int = 1, CP_replica_group: Optional[ReplicaGroup] = None, CP_collective_mode: Optional[CPCollectiveMode] = None, seq_packed_attention_mask: Optional[nl.NkiTensor] = None, seq_packed_active_blocks_table: Optional[nl.NkiTensor] = None, seq_packed_q_index_table: Optional[nl.NkiTensor] = None, seq_packed_seq_id_table: Optional[nl.NkiTensor] = None, seq_packed_accumulator_partial_route_table: Optional[nl.NkiTensor] = None, seq_packed_accumulator_group_state_route_table: Optional[nl.NkiTensor] = None)

   Fused Attention Block for Token Generation (TKG).

   Performs end-to-end attention block computation optimized for autoregressive decoding:
   X → [RMSNorm] → QKV Projection → [RMSNorm Q/K] → [RoPE] → [RMSNorm Q/K] →
   Attention → KV Cache Update → [Output Projection] → Output

   All intermediate tensors remain in SBUF to minimize HBM traffic.

   :param X: Input hidden states
   :type X: ``nl.NkiTensor``
   :param X_hidden_dim_actual: Actual hidden dim if X is padded
   :type X_hidden_dim_actual: ``Optional[int]``
   :param rmsnorm_X_enabled: Apply RMSNorm to X before QKV projection
   :type rmsnorm_X_enabled: ``bool``
   :param rmsnorm_X_eps: RMSNorm epsilon (default 1e-3)
   :type rmsnorm_X_eps: ``Optional[float]``
   :param rmsnorm_X_gamma: [1, H] @ HBM, RMSNorm weights
   :type rmsnorm_X_gamma: ``Optional[nl.NkiTensor]``
   :param W_qkv: [H, d_head*(q_heads + 2*kv_heads)] @ HBM, QKV projection weights. Packs Q (q_heads heads), K (kv_heads heads), V (kv_heads heads) per token.
   :type W_qkv: ``nl.NkiTensor``
   :param bias_qkv: [1, d_head*(q_heads + 2*kv_heads)] @ HBM, QKV bias.
   :type bias_qkv: ``Optional[nl.NkiTensor]``
   :param quantization_type_qkv: Type of quantization for QKV projection (NONE, STATIC, ROW).
   :type quantization_type_qkv: ``QuantizationType``
   :param weight_dequant_scale_qkv: Weight dequantization scale for QKV projection.
   :type weight_dequant_scale_qkv: ``Optional[nl.NkiTensor]``
   :param input_dequant_scale_qkv: Input dequantization scale for QKV projection.
   :type input_dequant_scale_qkv: ``Optional[nl.NkiTensor]``
   :param rmsnorm_QK_pre_rope_enabled: Apply RMSNorm to Q/K before RoPE
   :type rmsnorm_QK_pre_rope_enabled: ``bool``
   :param rmsnorm_QK_pre_rope_eps: Pre-RoPE RMSNorm epsilon
   :type rmsnorm_QK_pre_rope_eps: ``float``
   :param rmsnorm_QK_pre_rope_W_Q: [1, d_head] @ HBM, Pre-RoPE Q gamma weights
   :type rmsnorm_QK_pre_rope_W_Q: ``Optional[nl.NkiTensor]``
   :param rmsnorm_QK_pre_rope_W_K: [1, d_head] @ HBM, Pre-RoPE K gamma weights
   :type rmsnorm_QK_pre_rope_W_K: ``Optional[nl.NkiTensor]``
   :param cos: [d_head//2, B, S_tkg] @ HBM, RoPE cosine embeddings (None = skip RoPE)
   :type cos: ``Optional[nl.NkiTensor]``
   :param sin: [d_head//2, B, S_tkg] @ HBM, RoPE sine embeddings (None = skip RoPE)
   :type sin: ``Optional[nl.NkiTensor]``
   :param rope_contiguous_layout: True for contiguous halves, False for interleaved
   :type rope_contiguous_layout: ``bool``
   :param rmsnorm_QK_post_rope_enabled: Apply RMSNorm to Q/K after RoPE
   :type rmsnorm_QK_post_rope_enabled: ``bool``
   :param rmsnorm_QK_post_rope_eps: Post-RoPE RMSNorm epsilon
   :type rmsnorm_QK_post_rope_eps: ``float``
   :param rmsnorm_QK_post_rope_W_Q: [1, d_head] @ HBM, Post-RoPE Q weights
   :type rmsnorm_QK_post_rope_W_Q: ``Optional[nl.NkiTensor]``
   :param rmsnorm_QK_post_rope_W_K: [1, d_head] @ HBM, Post-RoPE K weights
   :type rmsnorm_QK_post_rope_W_K: ``Optional[nl.NkiTensor]``
   :param K_cache_transposed: Whether K cache is stored transposed in HBM. If True: K cache is [B, d_head, S_max_ctx]. If False: K cache is [B, S_max_ctx, d_head]. Must be False for block KV cache.
   :type K_cache_transposed: ``bool``
   :param active_blocks_table: @ HBM, Block indices for block KV cache. kv_heads == 1: [B, num_blocks]. kv_heads >  1: [B, kv_heads, num_blocks] (per-head block tables; each (batch, head) pair owns its own slot range so writes never collide). Block indices must use ``num_blocks * kv_heads`` addressing.
   :type active_blocks_table: ``Optional[nl.NkiTensor]``
   :param K_cache: Key cache @ HBM. Flat KV: [B, kv_heads, d_head, S_max_ctx] if K_cache_transposed else [B, kv_heads, S_max_ctx, d_head]. Block KV: kv_heads == 1: [num_blocks, block_len, d_head] or [num_blocks, 1, block_len, d_head]. kv_heads >  1: [num_blocks, kv_heads, block_len, d_head]. Block KV with fp8_packed: kv_heads == 1: [num_blocks, block_len // 2, d_head, 2] or [num_blocks, 1, block_len // 2, d_head, 2]. kv_heads >  1: [num_blocks, kv_heads, block_len // 2, d_head, 2].
   :type K_cache: ``nl.NkiTensor``
   :param V_cache: Value cache @ HBM. Flat KV: [B, kv_heads, S_max_ctx, d_head]. Block KV: kv_heads == 1: [num_blocks, block_len, d_head] or [num_blocks, 1, block_len, d_head]. kv_heads >  1: [num_blocks, kv_heads, block_len, d_head].
   :type V_cache: ``nl.NkiTensor``
   :param attention_mask: Attention mask @ HBM. Required for ordinary attention. Sequence-packed attention uses ``seq_packed_attention_mask`` instead, so this may be None. When pos_ids is provided: [S_tkg, B, q_heads_attn, S_tkg], active-only portion of the mask (the prior mask is generated in-kernel; layout is the same for swap and default). When pos_ids is None (full pre-generated mask): the layout MUST match the QK-swap decision. Determine it with is_qk_swapped(...) first (see attention_tkg / attention_tkg_utils for the full mask-shape guidance), then supply the matching layout: - default (K-stationary): [S_ctx, B, q_heads_attn, S_tkg] - QK-swap (Q-stationary): [B, q_heads_attn, S_tkg, S_ctx] gen_mask_tkg_hbm produces the correct layout when given the same transposed_out flag.
   :type attention_mask: ``Optional[nl.NkiTensor]``
   :param sink: [q_heads_attn, 1] @ HBM, one value per attention (Q) head. ``q_heads_attn = q_heads * KVDP`` (= ``kv_heads * q_per_group``).
   :type sink: ``Optional[nl.NkiTensor]``
   :param update_cache: Update KV cache with new tokens
   :type update_cache: ``bool``
   :param kv_cache_update_idx: Cache write positions (uint32_max = skip). Flat KV: [B, 1] — the shared start position; consecutive tokens are assumed. Block KV: per-token physical slots. kv_heads == 1: [B, S_tkg]. kv_heads >  1: [B, kv_heads, S_tkg].
   :type kv_cache_update_idx: ``Optional[nl.NkiTensor]``
   :param W_out: [q_heads*d_head, H] @ HBM, Output projection weights. When using FP8 KV cache (k_scale/v_scale provided), the attention output has magnitude scaled by v_scale. The caller must absorb v_scale into W_out or its dequant scale to compensate: ``W_out = W_out_original / v_scale`` (for NONE quantization) or ``weight_dequant_scale_out = scale / v_scale`` (for ROW/STATIC).
   :type W_out: ``Optional[nl.NkiTensor]``
   :param bias_out: [1, H] @ HBM, Output projection bias
   :type bias_out: ``Optional[nl.NkiTensor]``
   :param quantization_type_out: Type of quantization for output projection (NONE, STATIC, ROW).
   :type quantization_type_out: ``QuantizationType``
   :param weight_dequant_scale_out: Weight dequantization scale for output projection.
   :type weight_dequant_scale_out: ``Optional[nl.NkiTensor]``
   :param input_dequant_scale_out: Input dequantization scale for output projection.
   :type input_dequant_scale_out: ``Optional[nl.NkiTensor]``
   :param transposed_out: Transpose output layout (requires W_out)
   :type transposed_out: ``bool``
   :param out_in_sb: Return output in SBUF instead of HBM
   :type out_in_sb: ``bool``
   :param transposed_in: When True, X is in transposed HBM layout [H0=pmax, n_prgs, H1_shard, BxS] from a preceding layer's transposed_out. (Note that using [H0, n_prgs, BxS, H1_shard] instead would require an extra tensor_copy upstream so all-in-all it doesn't save us from doing the equivalent in attention.) Requires B*S_tkg <= pmax (small batch path). Default: False.
   :type transposed_in: ``bool``
   :param softmax_scale: Scaling factor for attention scores. If None, defaults to (1/√D) / k_scale. When using FP8 KV cache (k_scale/v_scale provided) and softmax_scale is None, the kernel automatically divides by k_scale to dequantize the KV cache values in the QK matmul, effectively setting ``softmax_scale = (1/√D) / k_scale``. When softmax_scale is explicitly provided, the caller is responsible for incorporating k_scale (e.g., ``softmax_scale = scaling / k_scale``).
   :type softmax_scale: ``Optional[float]``
   :param enable_fa_s_prior_tiling: bool: Whether to enable flash attention (FA) for attention computation. When enabled, the attention computation is tiled along the context (s_prior) to reduce peak memory usage.
   :type enable_fa_s_prior_tiling: ``bool``
   :param fp8_packed: When True with block KV, K_cache is fp8 [num_blocks, block_len // 2, d_head, 2] where the last dimension packs two consecutive sequence positions. Enables DMA transpose for FP8 block KV, avoiding the slower PE transpose fallback.
   :type fp8_packed: ``bool``
   :param k_scale: Scale for K quantization to FP8. Shape (PMAX, 1) or (1, 1) @ HBM. Must contain a single scalar value (replicated or scalar). When provided with v_scale, enables FP8 KV cache quantization. Supported dtypes: float32, float16, bfloat16.
   :type k_scale: ``Optional[nl.NkiTensor]``
   :param v_scale: Scale for V quantization to FP8. Shape (PMAX, 1) or (1, 1) @ HBM. Must contain a single scalar value (replicated or scalar). When provided with k_scale, enables FP8 KV cache quantization. Supported dtypes: float32, float16, bfloat16.
   :type v_scale: ``Optional[nl.NkiTensor]``
   :param sbm: SBUF memory manager (otherwise auto-allocated)
   :type sbm: ``Optional[SbufManager]``
   :param skip_attention: Skip attention computation (for testing)
   :type skip_attention: ``bool``
   :param is_h_transposed_by_4: Whether input X and RMSNorm gamma have been pre-shuffled along the H dimension for MXFP quantization. Required to be True when quantization_type_qkv is MX, STATIC_MX, or ROW_MX. The pre-shuffle reorders [B, S, H//512, 128, 4] → [B, S, 4, H//512, 128] so the data matches the layout expected by the hardware quantize_mx instruction. Default: False.
   :type is_h_transposed_by_4: ``bool``
   :param KVDP: KV cache data parallelism degree - number of ranks that shard the KV cache across the batch dimension (1 = disabled). Each rank processes B/KVDP batches.
   :type KVDP: ``int``
   :param KVDP_replica_group: Replica group for collective ops
   :type KVDP_replica_group: ``Optional[ReplicaGroup]``
   :param KVDP_rank: [1] @ HBM, uint32. This rank's position within its KVDP replica group (0 to KVDP-1). Used for batch/head slicing. Required when KVDP > 1.
   :type KVDP_rank: ``Optional[nl.NkiTensor]``
   :param pos_ids: [B, S_tkg] @ HBM, Absolute sequence position of each active token. When provided, the kernel generates the prior causal mask on-chip (attending to KV positions < pos_id) instead of loading a full pre-generated mask from HBM. In this mode, attention_mask carries only the active portion.
   :type pos_ids: ``Optional[nl.NkiTensor]``
   :param swa_start_pos_ids: [B, S_tkg] @ HBM, Per-query SWA window start positions. When provided together with pos_ids, generates a banded sliding window attention mask where each token attends to positions in [start_pos, pos_id). When None, standard causal attention mask is generated (token attends to all positions < pos_id).
   :type swa_start_pos_ids: ``Optional[nl.NkiTensor]``
   :param S_ctx: Explicit context length for flat KV with pos_ids. Required when pos_ids is provided and block KV is not used, since S_ctx cannot be derived from the attention mask. Must be None otherwise.
   :type S_ctx: ``Optional[int]``
   :param max_context_len: [1] @ HBM, int32 scalar for dynamic FA early exit. When provided, the FA loop exits early after processing ceil(max_context_len / tile_size) tiles. Requires block KV cache and pos_ids.
   :type max_context_len: ``Optional[nl.NkiTensor]``
   :param dtype_mode: Quantization dtype policy forwarded to all FP8-aware subkernels and to the FP8 KV cache when ``K_cache`` / ``V_cache`` are allocated with the opaque ``"float8e4"`` sentinel. Concrete dtypes are honored as-is. - ``DtypeMode.NON_OCP`` (default): ``nl.float8_e4m3`` (max=240). - ``DtypeMode.OCP``: ``nl.float8_e4m3fn`` (max=448). TRN3 only. - ``DtypeMode.AUTO``: ``nl.float8_e4m3fn`` on TRN3, else ``nl.float8_e4m3``.
   :type dtype_mode: ``DtypeMode``
   :param CP: Context parallelism degree - number of ranks that shard the KV cache along the sequence (context) dimension (1 = disabled). Each rank holds an S_ctx/CP slice of the prior KV cache and runs local flash attention; the distributed softmax results are combined across ranks by the CP collectives.
   :type CP: ``int``
   :param CP_replica_group: Replica group for CP collective ops. Required when CP > 1.
   :type CP_replica_group: ``Optional[ReplicaGroup]``
   :param CP_collective_mode: Collective mode for CP output redistribution (ALL_TO_ALL or REDUCE_SCATTER). Defaults to ALL_TO_ALL when None. KV Data Parallelism (KVDP > 1): KV-DP partitions the KV cache across ranks along the batch dimension. Each rank holds B/KVDP batches of the KV cache. Before attention: all_gather Q heads, slice Q/K/V batch. After attention: all_gather output batch, slice heads. When KV data parallelism is enabled, input/output shapes change: - B_attn = B / KVDP (batches per rank for attention) - q_heads_attn = q_heads * KVDP (query heads per rank after gather) Input shape changes: - K_cache, V_cache: [B_attn, ...] instead of [B, ...] - attention_mask: [S_ctx, B_attn, q_heads_attn, S_tkg] - pos_ids: [B_attn, S_tkg] instead of [B, S_tkg] - swa_start_pos_ids: [B_attn, S_tkg] instead of [B, S_tkg] - kv_cache_update_idx: [B_attn, 1] (caller must slice per rank) Output shape changes (when update_cache=False): - K_out: kv_heads == 1: [d_head, B_attn, S_tkg] kv_heads >  1: [d_head, B_attn, kv_heads, S_tkg] - V_out: [B_attn, kv_heads, S_tkg, d_head] Context Parallelism (CP > 1): CP partitions the prior KV cache across ranks along the sequence (context) dimension. Each rank holds an S_ctx/CP slice and runs local flash attention, returning unnormalized output plus per-rank softmax stats (running max/sum). Before attention: all_gather Q heads (q_heads -> q_heads*CP, no K/V slicing since each rank owns its context shard). After
   :type CP_collective_mode: ``Optional[CPCollectiveMode]``
   :param seq_packed_attention_mask: Packed schedule-row mask. When provided, the active-block, Q-index, and sequence-ID tables are required and the kernel dispatches to attention_tkg_seq_packed. The two accumulator route tables are an optional all-or-none pair. This is the only attention mask consumed on the packed path. See attention_tkg_sequence_packing_design_spec.md in core/attention for the schedule contract.
   :type seq_packed_attention_mask: ``Optional[nl.NkiTensor]``
   :param seq_packed_active_blocks_table: Packed physical block routes.
   :type seq_packed_active_blocks_table: ``Optional[nl.NkiTensor]``
   :param seq_packed_q_index_table: Packed Q tensor-indirection routes.
   :type seq_packed_q_index_table: ``Optional[nl.NkiTensor]``
   :param seq_packed_seq_id_table: Packed absolute sequence IDs.
   :type seq_packed_seq_id_table: ``Optional[nl.NkiTensor]``
   :param seq_packed_accumulator_partial_route_table: Tensor Indirection routes that scatter each row's slot partials into sequence groups. Must be supplied with the group-state route.
   :type seq_packed_accumulator_partial_route_table: ``Optional[nl.NkiTensor]``
   :param seq_packed_accumulator_group_state_route_table: Tensor Indirection routes that gather and scatter each group's persistent sequence state. Must be supplied with the partial route.
   :type seq_packed_accumulator_group_state_route_table: ``Optional[nl.NkiTensor]``
   :return: Output tensor with shape depending on projection and output location: - Without projection (W_out=None): - out_in_sb=False: [B, q_heads, d_head, S_tkg] @ HBM - out_in_sb=True: [d_head, B*q_heads*S_tkg] @ SBUF - With projection (W_out provided): - transposed_out=False, out_in_sb=False: [B*S_tkg, H] @ HBM - transposed_out=False, out_in_sb=True: [B*S_tkg, H//lnc] @ SBUF - transposed_out=True, out_in_sb=False: [128, lnc, H//lnc//128, B*S_tkg] @ HBM - transposed_out=True, out_in_sb=True: [128, H//lnc//128, B*S_tkg] @ SBUF
   :return: - If update_cache=True: Updated K cache (shape matches K_cache input) - If update_cache=False: New K tokens @ HBM with shape: kv_heads == 1: [d_head, B_attn, S_tkg] kv_heads >  1: [d_head, B_attn, kv_heads, S_tkg]
   :return: - If update_cache=True: Updated V cache (shape matches V_cache input) - If update_cache=False: New V tokens [B_attn, kv_heads, S_tkg, d_head] @ HBM
   :rtype: ``nl.NkiTensor``

   **Dimensions**:

   * B: batch size
   * S_tkg: number of new tokens to generate
   * S_ctx: KV cache sequence length in current bucket
   * S_max_ctx: maximum KV cache capacity of current bucket
   * H: hidden dimension
   * d_head: head dimension (must be even)
   * q_heads: number of query heads
   * kv_heads: 1 (GQA with single KV head)

   **Supported Data Types**:

   * Supports nl.float16 and nl.bfloat16

   **Constraints**:

   * Requires NeuronCore v3+
   * d_head must be even
   * H must be multiple of 128
   * Requires ``batch * sequence_tkg * q_heads <= pmax (=128)``


Implementation Details
----------------------

**Computation Flow:**

The kernel executes the following stages in sequence:

1. **Input Pre-normalization** (optional):
   
   - Apply RMSNorm to input hidden states: ``X_norm = RMSNorm(X, rmsnorm_pre_W, rmsnorm_pre_eps)``
   - Computed in FP32, result cast back to input dtype

2. **QKV Projection**:
   
   - Compute ``QKV = X_norm @ W_qkv.T`` using matrix multiplication
   - Result shape: ``[B, S_tkg, (q_heads + 2) * d_head]``
   - Supports FP8 quantization with dequantization scales

3. **Q/K Processing** (per head group):
   
   - Extract Q heads: ``Q = QKV[:, :, :q_heads * d_head]``
   - Extract K head: ``K = QKV[:, :, q_heads * d_head : (q_heads + 1) * d_head]``
   - Apply RoPE if enabled: ``Q, K = RoPE(Q, K, cos, sin, position_ids)``
   - Apply per-head RMSNorm if enabled: ``Q = RMSNorm(Q, rmsnorm_post_W_Q)``, ``K = RMSNorm(K, rmsnorm_post_W_K)``

4. **V Processing**:
   
   - Extract V head: ``V = QKV[:, :, (q_heads + 1) * d_head :]``

5. **KV Cache Update**:
   
   - Write new K/V tokens to cache at positions specified by ``kv_cache_update_idx``
   - Supports multiple cache layouts (flat, transposed, block-based)
   - Uses indirect addressing for efficient batch processing

6. **Attention Computation**:
   
   - Compute scaled dot-product attention: ``Attn = softmax(Q @ K_cache.T / scale) @ V_cache``
   - Apply causal masking based on ``S_ctx`` (context lengths)
   - Use FP32 accumulation if ``mixed_precision=True``
   - Supports Grouped-Query Attention by replicating KV heads

7. **Output Projection**:
   
   - Reshape attention output: ``Attn_flat = Attn.reshape([B, S_tkg, q_heads * d_head])``
   - Compute ``out = Attn_flat @ W_o.T``
   - Supports FP8 quantization with dequantization scales

**Memory Management:**

The kernel uses a custom SBUF memory manager (``SbufManager``) to efficiently allocate and reuse on-chip memory:

- Stack-based allocation for temporary tensors
- Automatic memory reuse after tensor lifetime ends
- Minimizes SBUF fragmentation

**Parallelization:**

The kernel supports data parallelism across multiple Neuron Cores:

- Batch dimension (``B``) can be sharded across cores
- Each core processes a subset of batch elements independently
- KV cache updates use per-core indexing

**Cache Layout Support:**

1. **Flat Cache** (``is_block_kv=False``):
   
   - K cache: ``[B, S_max_ctx, d_head]`` or ``[B, d_head, S_max_ctx]`` (transposed)
   - V cache: ``[B, S_max_ctx, d_head]``
   - Direct indexing by batch and sequence position

2. **Block Cache** (``is_block_kv=True``):
   
   - K/V cache: ``[num_blocks, block_len, d_head]``
   - Indirect indexing via block slot mapping
   - Efficient for variable-length sequences

**Quantization Support:**

- FP8 weights: Provide ``qkv_scale`` and ``o_scale`` for dequantization
- Mixed precision: FP32 accumulation with FP16/BF16 inputs
- Automatic dtype handling throughout the pipeline

**Key Implementation Notes:**

1. **Grouped-Query Attention**: The kernel processes Q heads in groups, where each group shares a single K/V head. This reduces KV cache memory by a factor of ``q_heads / kv_heads``.

2. **RoPE Application**: Rotary embeddings are applied using position indices derived from ``S_ctx`` (current context length). Supports both contiguous and interleaved layouts.

3. **Causal Masking**: Attention scores are masked such that token at position ``i`` can only attend to positions ``0`` to ``i`` in the context. Implemented by adding ``-inf`` to masked positions before softmax.

4. **Cache Update Optimization**: 
   
   - For ``S_tkg=1``: Uses batched vector DMA with ``vector_offset`` for all batches in one operation
   - For ``S_tkg>1``: Uses per-batch scalar DMA with ``scalar_offset``
   - Block cache uses indirect addressing via block slot indices

5. **Memory Efficiency**: All intermediate tensors (QKV, Q, K, V, attention scores, attention output) remain in SBUF. Only input ``X``, weights, caches, and final output ``out`` reside in HBM.
