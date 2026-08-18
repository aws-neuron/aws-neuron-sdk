.. meta::
    :description: DeepSeek MLA QKV projection (MX), emitting absorbed latents.
    :date-modified: 08/17/2026

.. currentmodule:: nkilib.experimental.mla.deepseek

MLA QKV CTE Kernel API Reference
================================

DeepSeek MLA QKV projection (MX), emitting absorbed latents.

Consumes the packed MX activation from rmsnorm_mx_prefill (pack_scales=True) directly — no in-kernel quantize. Intended for Context Encoding (prefill) with DeepSeek-V3.2 dimensions: absorption requires qk_nope_head_dim == 128, and the kernel is tuned for n_heads up to 128 and hidden dim H up to 7168 with LNC sharding over the sequence dimension. Best used when the packed MX activation is produced upstream so no re-quantization is needed.

Background
-----------

The ``mla_qkv_cte_kernel`` kernel performs DeepSeek MLA QKV projection (MX), emitting absorbed latents.

API Reference
--------------

**Source code for this kernel API can be found at**: `mla_qkv_cte.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/mla/deepseek/mla_qkv_cte.py>`_

mla_qkv_cte_kernel
^^^^^^^^^^^^^^^^^^

.. py:function:: mla_qkv_cte_kernel(x_hbm_mx: nl.NkiTensor, wqkv_a_hbm: nl.NkiTensor, wqkv_a_scale_hbm: nl.NkiTensor, wq_b_hbm: nl.NkiTensor, wq_b_scale_hbm: nl.NkiTensor, q_norm_gamma_hbm: nl.NkiTensor, kv_norm_gamma_hbm: nl.NkiTensor, wuk_hbm: nl.NkiTensor, cos_cache_hbm: nl.NkiTensor, sin_cache_hbm: nl.NkiTensor, n_heads: int, qk_nope_head_dim: int, qk_rope_head_dim: int, kv_lora_rank: int, qk_lora_rank: int, norm_eps: float = 1e-06, qr_qtz_hbm: nl.NkiTensor = None, qr_scale_hbm: nl.NkiTensor = None) -> Tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]

   DeepSeek MLA QKV projection (MX), emitting absorbed latents.

   :param x_hbm_mx: ``[B, S, H + scale_region]`` fp8 packed MX activation (see the param comment).
   :type x_hbm_mx: ``nl.NkiTensor``
   :param wqkv_a_hbm: ``[H // 4, qk_lora_rank + kv_lora_rank + qk_rope_head_dim]`` fp8x4.
   :type wqkv_a_hbm: ``nl.NkiTensor``
   :param wqkv_a_scale_hbm: compact block-128 scales for ``wqkv_a``.
   :type wqkv_a_scale_hbm: ``nl.NkiTensor``
   :param wq_b_hbm: ``[qk_lora_rank // 4, n_heads * (qk_nope_head_dim + qk_rope_head_dim)]`` fp8x4.
   :type wq_b_hbm: ``nl.NkiTensor``
   :param wq_b_scale_hbm: compact block-128 scales for ``wq_b``.
   :type wq_b_scale_hbm: ``nl.NkiTensor``
   :param q_norm_gamma_hbm: ``[1, qk_lora_rank]`` bf16 RMSNorm gamma for the Q intermediate.
   :type q_norm_gamma_hbm: ``nl.NkiTensor``
   :param kv_norm_gamma_hbm: ``[1, kv_lora_rank]`` bf16 RMSNorm gamma for the KV latent.
   :type kv_norm_gamma_hbm: ``nl.NkiTensor``
   :param wuk_hbm: ``[qk_nope_head_dim, n_heads * kv_lora_rank]`` bf16 absorption weight (``W_uk[h] = [nope, kv_lora]``, contraction = nope; head ``h`` owns columns ``[h * kv_lora_rank, (h + 1) * kv_lora_rank)``).
   :type wuk_hbm: ``nl.NkiTensor``
   :param cos_cache_hbm: ``[B, S, qk_rope_head_dim]`` bf16 RoPE cosine cache.
   :type cos_cache_hbm: ``nl.NkiTensor``
   :param sin_cache_hbm: ``[B, S, qk_rope_head_dim]`` bf16 RoPE sine cache.
   :type sin_cache_hbm: ``nl.NkiTensor``
   :param n_heads: Number of attention heads.
   :type n_heads: ``int``
   :param qk_nope_head_dim: Non-RoPE per-head Q/K dimension (must be 128).
   :type qk_nope_head_dim: ``int``
   :param qk_rope_head_dim: RoPE per-head Q/K dimension.
   :type qk_rope_head_dim: ``int``
   :param kv_lora_rank: Latent KV LoRA rank.
   :type kv_lora_rank: ``int``
   :param qk_lora_rank: Q LoRA rank.
   :type qk_lora_rank: ``int``
   :param norm_eps: RMSNorm epsilon (default 1e-6).
   :type norm_eps: ``float``
   :param qr_qtz_hbm: Optional pre-allocated buffer for the exported MX-quantized qr latent (SAI indexer fast path). None (default) means no qr export.
   :type qr_qtz_hbm: ``nl.NkiTensor``
   :param qr_scale_hbm: Optional pre-allocated scale buffer paired with ``qr_qtz_hbm``.
   :type qr_scale_hbm: ``nl.NkiTensor``
   :return: ``[B, S, n_heads, kv_lora_rank]`` bf16, per-head absorbed Q latent.
   :rtype: ``nl.ndarray``
   :return: ``[B, S, n_heads, qk_rope_head_dim]`` bf16, per-head RoPE queries.
   :rtype: ``nl.ndarray``
   :return: ``[B, S, kv_lora_rank]`` bf16, shared latent KV (RMSNorm(kv) * gamma).
   :rtype: ``nl.ndarray``
   :return: ``[B, S, qk_rope_head_dim]`` bf16, shared RoPE key.
   :rtype: ``nl.ndarray``

   **Notes**:

   * Absorption requires ``qk_nope_head_dim == 128`` (full-partition bf16 contraction).
   * The absorption matmul (q_nope @ W_uk) runs in bf16, not MX.

   **Dimensions**:

   * B: Batch size
   * S: Sequence length (tokens)
   * H: Hidden dimension size
   * n_heads: Number of attention heads
   * qk_nope_head_dim: Non-RoPE portion of the per-head Q/K dimension (must be 128)
   * qk_rope_head_dim: RoPE portion of the per-head Q/K dimension (R)
   * kv_lora_rank: Latent KV LoRA rank (L)

