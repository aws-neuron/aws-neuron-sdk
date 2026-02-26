.. meta::
    :description: Attention Block TKG kernel implements fused attention block optimized for Token Generation.
    :date-modified: 02/13/2026

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

**Source code for this kernel API can be found at**: `attention_block_tkg.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/transformer/attention_block_tkg.py>`_

attention_block_tkg
^^^^^^^^^^^^^^^^^^^

.. py:function:: attention_block_tkg(X: nl.ndarray, X_hidden_dim_actual: Optional[int], rmsnorm_X_enabled: bool, rmsnorm_X_eps: Optional[float], rmsnorm_X_gamma: Optional[nl.ndarray], W_qkv: nl.ndarray, bias_qkv: Optional[nl.ndarray], quantization_type_qkv: QuantizationType, weight_dequant_scale_qkv: Optional[nl.ndarray], input_dequant_scale_qkv: Optional[nl.ndarray], rmsnorm_QK_pre_rope_enabled: bool, rmsnorm_QK_pre_rope_eps: float, cos: Optional[nl.ndarray], sin: Optional[nl.ndarray], rope_contiguous_layout: bool, rmsnorm_QK_post_rope_enabled: bool, rmsnorm_QK_post_rope_eps: float, rmsnorm_QK_post_rope_W_Q: Optional[nl.ndarray], rmsnorm_QK_post_rope_W_K: Optional[nl.ndarray], K_cache_transposed: bool, active_blocks_table: Optional[nl.ndarray], K_cache: nl.ndarray, V_cache: nl.ndarray, attention_mask: nl.ndarray, sink: Optional[nl.ndarray], softmax_scale: Optional[float] = None, update_cache: bool, kv_cache_update_idx: Optional[nl.ndarray], k_scale: Optional[nl.ndarray] = None, v_scale: Optional[nl.ndarray] = None, W_out: Optional[nl.ndarray], bias_out: Optional[nl.ndarray], quantization_type_out: QuantizationType, weight_dequant_scale_out: Optional[nl.ndarray], input_dequant_scale_out: Optional[nl.ndarray], transposed_out: bool, out_in_sb: bool, sbm: Optional[SbufManager] = None, skip_attention: bool = False)

   Fused Attention Block for Token Generation (TKG).

   Performs end-to-end attention block computation optimized for autoregressive decoding:
   X → [RMSNorm] → QKV Projection → [RMSNorm Q/K] → [RoPE] → [RMSNorm Q/K] →
   Attention → KV Cache Update → [Output Projection] → Output

   All intermediate tensors remain in SBUF to minimize HBM traffic.

   :param X: Input hidden states ``[B, S_tkg, H]`` @ HBM or ``[pmax, B*S_tkg, H//pmax]`` @ SBUF
   :type X: ``nl.ndarray``
   :param X_hidden_dim_actual: Actual hidden dim if X is padded
   :type X_hidden_dim_actual: ``int``, optional
   :param rmsnorm_X_enabled: Apply RMSNorm to X before QKV projection
   :type rmsnorm_X_enabled: ``bool``
   :param rmsnorm_X_eps: RMSNorm epsilon (default 1e-3)
   :type rmsnorm_X_eps: ``float``, optional
   :param rmsnorm_X_gamma: RMSNorm weights ``[1, H]`` @ HBM
   :type rmsnorm_X_gamma: ``nl.ndarray``, optional
   :param W_qkv: QKV projection weights ``[H, d_head*(q_heads+2)]`` @ HBM
   :type W_qkv: ``nl.ndarray``
   :param bias_qkv: QKV bias ``[1, d_head*(q_heads+2)]`` @ HBM
   :type bias_qkv: ``nl.ndarray``, optional
   :param quantization_type_qkv: Quantization type for QKV projection
   :type quantization_type_qkv: ``QuantizationType``
   :param weight_dequant_scale_qkv: Weight dequantization scale for QKV projection
   :type weight_dequant_scale_qkv: ``nl.ndarray``, optional
   :param input_dequant_scale_qkv: Input dequantization scale for QKV projection
   :type input_dequant_scale_qkv: ``nl.ndarray``, optional
   :param rmsnorm_QK_pre_rope_enabled: Apply RMSNorm to Q/K before RoPE
   :type rmsnorm_QK_pre_rope_enabled: ``bool``
   :param rmsnorm_QK_pre_rope_eps: Pre-RoPE RMSNorm epsilon
   :type rmsnorm_QK_pre_rope_eps: ``float``
   :param cos: RoPE cosine embeddings ``[d_head//2, B, S_tkg]`` @ HBM (None = skip RoPE)
   :type cos: ``nl.ndarray``, optional
   :param sin: RoPE sine embeddings ``[d_head//2, B, S_tkg]`` @ HBM (None = skip RoPE)
   :type sin: ``nl.ndarray``, optional
   :param rope_contiguous_layout: True for contiguous halves, False for interleaved
   :type rope_contiguous_layout: ``bool``
   :param rmsnorm_QK_post_rope_enabled: Apply RMSNorm to Q/K after RoPE
   :type rmsnorm_QK_post_rope_enabled: ``bool``
   :param rmsnorm_QK_post_rope_eps: Post-RoPE RMSNorm epsilon
   :type rmsnorm_QK_post_rope_eps: ``float``
   :param rmsnorm_QK_post_rope_W_Q: Post-RoPE Q weights ``[1, d_head]`` @ HBM
   :type rmsnorm_QK_post_rope_W_Q: ``nl.ndarray``, optional
   :param rmsnorm_QK_post_rope_W_K: Post-RoPE K weights ``[1, d_head]`` @ HBM
   :type rmsnorm_QK_post_rope_W_K: ``nl.ndarray``, optional
   :param K_cache_transposed: K cache layout flag
   :type K_cache_transposed: ``bool``
   :param active_blocks_table: Block indices for block KV cache ``[B, num_blocks]`` @ HBM
   :type active_blocks_table: ``nl.ndarray``, optional
   :param K_cache: Key cache @ HBM
   :type K_cache: ``nl.ndarray``
   :param V_cache: Value cache @ HBM
   :type V_cache: ``nl.ndarray``
   :param attention_mask: Attention mask ``[S_ctx, B, q_heads, S_tkg]`` @ HBM
   :type attention_mask: ``nl.ndarray``
   :param sink: Attention sink tokens ``[H, 1]`` @ HBM
   :type sink: ``nl.ndarray``, optional
   :param softmax_scale: Scaling factor for attention scores (``Q @ K^T * softmax_scale``). If ``None``, defaults to ``1.0 / sqrt(d_head)``.
   :type softmax_scale: ``float``, optional
   :param update_cache: Update KV cache with new tokens
   :type update_cache: ``bool``
   :param kv_cache_update_idx: Cache write positions ``[B, 1]`` (uint32_max = skip)
   :type kv_cache_update_idx: ``nl.ndarray``, optional
   :param k_scale: Key quantization scale for FP8 KV cache. Enables FP8 quantization of K values written to cache.
   :type k_scale: ``nl.ndarray``, optional
   :param v_scale: Value quantization scale for FP8 KV cache. Enables FP8 quantization of V values written to cache.
   :type v_scale: ``nl.ndarray``, optional
   :param W_out: Output projection weights ``[q_heads*d_head, H]`` @ HBM
   :type W_out: ``nl.ndarray``, optional
   :param bias_out: Output projection bias ``[1, H]`` @ HBM
   :type bias_out: ``nl.ndarray``, optional
   :param quantization_type_out: Quantization type for output projection
   :type quantization_type_out: ``QuantizationType``
   :param weight_dequant_scale_out: Weight dequantization scale for output projection
   :type weight_dequant_scale_out: ``nl.ndarray``, optional
   :param input_dequant_scale_out: Input dequantization scale for output projection
   :type input_dequant_scale_out: ``nl.ndarray``, optional
   :param transposed_out: Transpose output layout (requires W_out)
   :type transposed_out: ``bool``
   :param out_in_sb: Return output in SBUF instead of HBM
   :type out_in_sb: ``bool``
   :param sbm: SBUF memory manager (otherwise auto-allocated)
   :type sbm: ``SbufManager``, optional
   :param skip_attention: Skip attention computation (for testing). Default: False.
   :type skip_attention: ``bool``
   :return: Tuple of (out, K_out, V_out) - Output tensor, updated K cache or new K tokens, updated V cache or new V tokens
   :rtype: ``tuple``

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
