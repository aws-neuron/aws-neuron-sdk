.. meta::
    :description: DeepSeek MLA QKV projection with MX quantization for Context Encoding.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.qkv

QKV CTE MLA Kernel API Reference
================================

DeepSeek MLA QKV projection with MX quantization for Context Encoding.

Implements the full QKV projection pipeline for Multi-head Latent Attention (MLA) with MX (fp8) quantization. Includes two-stage low-rank projections for both Q and KV paths, fused RMSNorm, and Rotary Position Embedding (RoPE). Supports LNC sharding on the sequence dimension. Sequence length is tiled at 128, and batch size is 1 (CTE processes a single context).

Background
-----------

The ``qkv_mla_mx`` kernel implements the full QKV projection pipeline for DeepSeek Multi-head Latent Attention (MLA) with MX (fp8) quantization during Context Encoding. It performs two-stage low-rank projections for both the Q and KV paths, fused RMSNorm, and RoPE. The ``qkv_mla_mx_deepseek_v4`` entry point provides the DeepSeek v4 variant of the same projection.

API Reference
--------------

**Source code for this kernel API can be found at**: `qkv_cte_mla.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/qkv/qkv_cte_mla.py>`_

qkv_mla_mx
^^^^^^^^^^

.. py:function:: qkv_mla_mx(x_hbm: nl.ndarray, wqkv_a_hbm: nl.ndarray, wqkv_a_scale_hbm: nl.ndarray, wq_b_hbm: nl.ndarray, wq_b_scale_hbm: nl.ndarray, q_norm_gamma_hbm: nl.ndarray, wkv_b_hbm: nl.ndarray, wkv_b_scale_hbm: nl.ndarray, kv_norm_gamma_hbm: nl.ndarray, cos_cache_hbm: nl.ndarray, sin_cache_hbm: nl.ndarray, n_heads: int, qk_nope_head_dim: int, qk_rope_head_dim: int, v_head_dim: int, kv_lora_rank: int, qk_lora_rank: int, norm_eps: float = 1e-06) -> Tuple[nl.ndarray, nl.ndarray, nl.ndarray]

   DeepSeek MLA QKV projection with MX quantization for Context Encoding.

   :param x_hbm: [B, S, H] bf16, Input hidden states
   :type x_hbm: ``nl.ndarray``
   :param wqkv_a_hbm: [H//4, qk_lora_rank + kv_lora_rank + qk_rope_head_dim] fp8x4, First combined Q/KV projection weights
   :type wqkv_a_hbm: ``nl.ndarray``
   :param wqkv_a_scale_hbm: [H//128, ceil((qk_lora_rank + kv_lora_rank + qk_rope_head_dim)/128)] uint8, DeepSeek block-128 compact scales for wqkv_a
   :type wqkv_a_scale_hbm: ``nl.ndarray``
   :param wq_b_hbm: [qk_lora_rank//4, n_heads * qk_head_dim] fp8x4, Second Q projection weights
   :type wq_b_hbm: ``nl.ndarray``
   :param wq_b_scale_hbm: [qk_lora_rank//128, ceil(n_heads * qk_head_dim / 128)] uint8, DeepSeek block-128 compact scales for wq_b
   :type wq_b_scale_hbm: ``nl.ndarray``
   :param q_norm_gamma_hbm: [1, qk_lora_rank] bf16, RMSNorm gamma for Q intermediate
   :type q_norm_gamma_hbm: ``nl.ndarray``
   :param wkv_b_hbm: [kv_lora_rank//4, n_heads * (qk_nope_head_dim + v_head_dim)] fp8x4, Second KV projection weights
   :type wkv_b_hbm: ``nl.ndarray``
   :param wkv_b_scale_hbm: [kv_lora_rank//128, ceil(n_heads * (qk_nope_head_dim + v_head_dim) / 128)] uint8, MX scales for wkv_b
   :type wkv_b_scale_hbm: ``nl.ndarray``
   :param kv_norm_gamma_hbm: [1, kv_lora_rank] bf16, RMSNorm gamma for KV intermediate
   :type kv_norm_gamma_hbm: ``nl.ndarray``
   :param cos_cache_hbm: [B, S, qk_rope_head_dim] bf16, Cosine RoPE frequencies
   :type cos_cache_hbm: ``nl.ndarray``
   :param sin_cache_hbm: [B, S, qk_rope_head_dim] bf16, Sine RoPE frequencies
   :type sin_cache_hbm: ``nl.ndarray``
   :param n_heads: Number of attention heads
   :type n_heads: ``int``
   :param qk_nope_head_dim: Non-RoPE portion of Q/K head dimension
   :type qk_nope_head_dim: ``int``
   :param qk_rope_head_dim: RoPE portion of Q/K head dimension
   :type qk_rope_head_dim: ``int``
   :param v_head_dim: Value head dimension
   :type v_head_dim: ``int``
   :param kv_lora_rank: Latent dimension for KV compression
   :type kv_lora_rank: ``int``
   :param qk_lora_rank: Latent dimension for Q compression
   :type qk_lora_rank: ``int``
   :param norm_eps: RMSNorm epsilon. Defaults to 1e-6
   :type norm_eps: ``float``
   :return: [B, S, n_heads, qk_head_dim] bf16, Query projections with RoPE applied
   :rtype: ``nl.ndarray``
   :return: [B, S, n_heads, qk_head_dim] bf16, Key projections with RoPE applied
   :rtype: ``nl.ndarray``
   :return: [B, S, n_heads, v_head_dim] bf16, Value projections
   :rtype: ``nl.ndarray``

   **Notes**:

   * Matmul shapes: Combined Q/KV Path Stage 1: x[B,S,H] @ wqkv_a[H, qk_lora_rank + kv_lora_rank + qk_rope_head_dim] -> qkv_a_out[B,S,qk_lora_rank + kv_lora_rank + qk_rope_head_dim] Split: qr[B,S,qk_lora_rank], kv[B,S,kv_lora_rank], k_pe[B,S,qk_rope_head_dim] Q Path Stage 2: norm(qr)[B,S,qk_lora_rank] @ wq_b[qk_lora_rank, n_heads*qk_head_dim] -> q[B,S,n_heads*qk_head_dim] KV Path Stage 2: norm(kv)[B,S,kv_lora_rank] @ wkv_b[kv_lora_rank, n_heads*(qk_nope_head_dim+v_head_dim)] -> kv_out[B,S,n_heads*(qk_nope_head_dim+v_head_dim)] Split: k_nope[B,S,n_heads,qk_nope_head_dim], v[B,S,n_heads,v_head_dim] Final assembly: q_pe = q[..., qk_nope_head_dim:] -> RoPE -> q[..., qk_nope_head_dim:] k_pe -> RoPE -> broadcast to all heads -> concat with k_nope -> K

   **Dimensions**:

   * B: Batch size
   * S: Sequence length
   * H: Hidden dimension (input)
   * n_heads: Number of attention heads
   * qk_lora_rank: Q latent dimension (e.g., 1536)
   * kv_lora_rank: KV latent dimension (e.g., 512)
   * qk_head_dim: qk_nope_head_dim + qk_rope_head_dim (e.g., 128 + 64 = 192)
   * qk_nope_head_dim: Non-rotary part of Q/K (e.g., 128)
   * qk_rope_head_dim: Rotary part of Q/K (e.g., 64)
   * v_head_dim: Value head dimension (e.g., 128)

qkv_mla_mx_deepseek_v4
^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: qkv_mla_mx_deepseek_v4(x_hbm: nl.ndarray, wqkv_hbm: nl.ndarray, wqkv_scale_hbm: nl.ndarray, wq_b_hbm: nl.ndarray, wq_b_scale_hbm: nl.ndarray, q_norm_gamma_hbm: nl.ndarray, kv_norm_gamma_hbm: nl.ndarray, cos_cache_hbm: nl.ndarray, sin_cache_hbm: nl.ndarray, n_heads: int, head_dim: int, qk_rope_head_dim: int, kv_lora_rank: int, qk_lora_rank: int, norm_eps: float = 1e-06) -> Tuple[nl.ndarray, nl.ndarray]

   DeepSeek v4 MLA QKV projection with MX quantization.

   :param x_hbm: [B, S, H] bf16 input hidden states
   :type x_hbm: ``nl.ndarray``
   :param wqkv_hbm: [H//4, qk_lora_rank + kv_dim] fp8x4 fused Q/KV first projection
   :type wqkv_hbm: ``nl.ndarray``
   :param wqkv_scale_hbm: [H//128, ceil((qk_lora_rank + kv_dim)/128)] uint8 DeepSeek block-128 compact scales
   :type wqkv_scale_hbm: ``nl.ndarray``
   :param wq_b_hbm: [qk_lora_rank//4, n_heads * head_dim] fp8x4 second Q projection
   :type wq_b_hbm: ``nl.ndarray``
   :param wq_b_scale_hbm: [qk_lora_rank//128, ceil(n_heads * head_dim / 128)] uint8 DeepSeek block-128 compact scales
   :type wq_b_scale_hbm: ``nl.ndarray``
   :param q_norm_gamma_hbm: [1, qk_lora_rank] bf16 RMSNorm gamma for Q intermediate
   :type q_norm_gamma_hbm: ``nl.ndarray``
   :param kv_norm_gamma_hbm: [1, kv_dim] bf16 RMSNorm gamma for KV
   :type kv_norm_gamma_hbm: ``nl.ndarray``
   :param cos_cache_hbm: [B, S, qk_rope_head_dim] bf16
   :type cos_cache_hbm: ``nl.ndarray``
   :param sin_cache_hbm: [B, S, qk_rope_head_dim] bf16
   :type sin_cache_hbm: ``nl.ndarray``
   :param n_heads: number of attention heads
   :type n_heads: ``int``
   :param head_dim: full Q/K head dimension (nope + rope)
   :type head_dim: ``int``
   :param qk_rope_head_dim: RoPE dimension
   :type qk_rope_head_dim: ``int``
   :param kv_lora_rank: KV latent dimension
   :type kv_lora_rank: ``int``
   :param qk_lora_rank: Q latent dimension
   :type qk_lora_rank: ``int``
   :param norm_eps: RMSNorm epsilon
   :type norm_eps: ``float``
   :return: [B, S, n_heads, head_dim] bf16 with RoPE on last rope_dim
   :rtype: ``nl.ndarray``
   :return: [B, S, kv_dim] bf16 with RoPE on last rope_dim
   :rtype: ``nl.ndarray``

   **Notes**:

   * Matmul shapes: Fused first projection: x[B,S,H] @ wqkv[H, qk_lora_rank + kv_dim] -> qkv_a_out[B,S,qk_lora_rank + kv_dim] Split: qr[B,S,qk_lora_rank], kv[B,S,kv_dim] Q Path Stage 2: norm(qr)[B,S,qk_lora_rank] @ wq_b[qk_lora_rank, n_heads*head_dim] -> q[B,S,n_heads*head_dim] Final assembly: q -> per-head rsqrt norm -> RoPE on q[..., -rope_dim:] kv -> RoPE on kv[..., -rope_dim:] (kv latent returned directly)

   **Dimensions**:

   * B: Batch size
   * S: Sequence length
   * H: Hidden dimension (input)
   * n_heads: Number of attention heads
   * qk_lora_rank: Q latent dimension
   * kv_lora_rank: KV latent dimension
   * kv_dim: kv_lora_rank + qk_rope_head_dim (full kv output width)
   * head_dim: full Q/K head dimension (nope + rope)
   * qk_rope_head_dim: RoPE portion of Q/K head dimension

