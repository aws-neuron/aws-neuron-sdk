.. meta::
    :description: Fused GPT-OSS SWA block. Returns (out [B,S,H], k_cache, v_cache) with caches updated in place.
    :date-modified: 08/17/2026

.. currentmodule:: nkilib.experimental.attention

SWA Fused CTE Kernel API Reference
==================================

Fused GPT-OSS SWA block. Returns (out [B,S,H], k_cache, v_cache) with caches updated in place.

Packed-FP8 KV cache: when k_cache/v_cache are an FP8 dtype, the K cache is stored PACKED as (num_blocks, num_kv_heads, block_size // 2, d_head, 2) -- two consecutive tokens in the trailing length-2 axis so the K cache views as bf16 (2 fp8 = 1 bf16 width) for DMA. The V cache is UNPACKED, token-major (num_blocks, num_kv_heads, block_size, d_head) fp8 -- same layout as the bf16 V cache, only the dtype differs (so V is loaded straight as fp8 with no implicit cast, then dequantized). The prior window is dequantized to bf16 on load (carry stays bf16, attention math unchanged); freshly-computed K is quantized + packed and V is quantized (token-major, no pack) on the write-back scatter. Dequant/quant use the per-tensor static k_scale/v_scale.

Background
-----------

The ``swa_fused_cte`` kernel is a fused GPT-OSS sliding-window-attention (SWA) block. It returns ``(out [B,S,H], k_cache, v_cache)`` with the KV caches updated in place.

API Reference
--------------

**Source code for this kernel API can be found at**: `swa_fused_cte.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/attention/swa_fused_cte.py>`_

swa_fused_cte
^^^^^^^^^^^^^

.. py:function:: swa_fused_cte(hidden_states: nl.NkiTensor, qkv_weight: nl.NkiTensor, op_weight: nl.NkiTensor, k_cache: nl.NkiTensor, v_cache: nl.NkiTensor, block_tables: nl.NkiTensor, cos_cache: nl.NkiTensor, sin_cache: nl.NkiTensor, sink: nl.NkiTensor, prior_tokens: nl.NkiTensor, qkv_bias: nl.NkiTensor, op_bias: nl.NkiTensor, scale: float = 1.0, sliding_window: int = 128, block_size: int = 128, num_q_heads: int = 16, num_kv_heads: int = 2, d_head: int = 64, k_scale: nl.NkiTensor = None, v_scale: nl.NkiTensor = None)

   Fused GPT-OSS SWA block. Returns (out [B,S,H], k_cache, v_cache) with caches updated in place.


