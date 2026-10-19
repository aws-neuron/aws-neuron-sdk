.. meta::
    :description: Computes the DeepSeek-V3.2 MLA query and key/value projections for one decode step, applies interleaved RoPE, absorbs th
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.deepseek_v32.mla_decode_qkv

MLA Decode QKV Kernel API Reference
===================================

Computes the DeepSeek-V3.2 MLA query and key/value projections for one decode step, applies interleaved RoPE, absorbs the query into the cache's latent space, and scatters this step's KV row into the paged cache.

This is the front-fold of an MLA decode layer. It is designed to be traced inline into an attention core rather than run standalone: the two query tiles are returned **in SBUF**, already feature-major in the layout the core's score matmuls want. Returning them through HBM was measured as pure latency, an 18 KB round trip into an identical SBUF layout. A standalone caller must therefore evict them itself. Both RMSNorms are fused into the projections that feed them, and the KV row is built directly in cache-row order so that no transpose is needed before the scatter.

Background
-----------

The ``mla_decode_qkv`` kernel computes the DeepSeek-V3.2 MLA query and key/value projections for one decode step, applies interleaved RoPE, absorbs the query into the latent space, and writes the step's KV row into the paged cache. It returns the query tiles in SBUF for an attention kernel such as ``mla_decode_attention``.

API Reference
--------------

**Source code for this kernel API can be found at**: `mla_decode_qkv.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/deepseek_v32/mla_decode_qkv/mla_decode_qkv.py>`_

mla_decode_qkv
^^^^^^^^^^^^^^

.. py:function:: mla_decode_qkv(hidden_states: nl.NkiTensor, wq_a: nl.NkiTensor, q_norm_w: nl.NkiTensor, wq_b: nl.NkiTensor, wkv_a: nl.NkiTensor, kv_norm_w: nl.NkiTensor, cos: nl.NkiTensor, sin: nl.NkiTensor, q_absorb_w: nl.NkiTensor, kv_cache: nl.NkiTensor, slot_mapping: nl.NkiTensor, n_heads: int, qk_nope_head_dim: int, qk_rope_head_dim: int, kv_lora_rank: int, norm_eps: float = 1e-06, input_norm_w: nl.NkiTensor | None = None) -> MlaDecodeQkvResult

   Computes the DeepSeek-V3.2 MLA query and key/value projections for one decode step, applies interleaved RoPE, absorbs the query into the cache's latent space, and scatters this step's KV row into the paged cache.

   :param hidden_states: ``[n_tokens, hidden]``. This step's hidden states.
   :type hidden_states: ``nl.NkiTensor``
   :param wq_a: ``[hidden, q_lora_rank]``. Query down-projection.
   :type wq_a: ``nl.NkiTensor``
   :param q_norm_w: ``[1, q_lora_rank]`` fp32. RMSNorm gamma for the query latent.
   :type q_norm_w: ``nl.NkiTensor``
   :param wq_b: ``[q_lora_rank, qk_packed_dim]``. Query up-projection, emitting every head's ``[q_nope | q_pe]`` concatenated.
   :type wq_b: ``nl.NkiTensor``
   :param wkv_a: ``[hidden, kv_row_dim]``. KV down-projection, emitting one cache row per token.
   :type wkv_a: ``nl.NkiTensor``
   :param kv_norm_w: ``[1, kv_lora_rank]`` fp32. RMSNorm gamma for the KV latent.
   :type kv_norm_w: ``nl.NkiTensor``
   :param cos: ``[n_tokens, qk_rope_dim // 2]``. Cosine table for interleaved RoPE.
   :type cos: ``nl.NkiTensor``
   :param sin: ``[n_tokens, qk_rope_dim // 2]``. Sine table for interleaved RoPE.
   :type sin: ``nl.NkiTensor``
   :param q_absorb_w: ``[n_heads, qk_nope_dim, kv_lora_rank]``. The nope half of ``wkv_b``, which maps a head's nope query into the cache's latent space.
   :type q_absorb_w: ``nl.NkiTensor``
   :param kv_cache: ``[n_blocks, 1, block_size, kv_row_dim]``. Paged cache, **mutated in place**.
   :type kv_cache: ``nl.NkiTensor``
   :param slot_mapping: ``[n_tokens]`` int32. Destination cache row for each token.
   :type slot_mapping: ``nl.NkiTensor``
   :param n_heads: Attention heads.
   :type n_heads: ``int``
   :param qk_nope_head_dim: Per-head QK width RoPE does not rotate.
   :type qk_nope_head_dim: ``int``
   :param qk_rope_head_dim: Per-head QK width RoPE rotates.
   :type qk_rope_head_dim: ``int``
   :param kv_lora_rank: LoRA rank of the KV path.
   :type kv_lora_rank: ``int``
   :param norm_eps: Epsilon for every RMSNorm in this kernel.
   :type norm_eps: ``float``
   :param input_norm_w: ``[1, hidden]`` fp32, optional. When given, the layer's input RMSNorm is folded into the hidden load.
   :type input_norm_w: ``nl.NkiTensor | None``

   **Notes**:

   * The cache row is laid out ``[k_pe | kv_latent]``, which **inverts** DeepSeek's reference order of ``[k_latent | k_pe]``. Building it inverted costs nothing here, because the two fields land in separate PSUM banks anyway, and it saves a transpose later. ``F_MAX == kv_row_latent_dim`` is asserted so the PSUM n-tile boundary falls exactly on the field boundary, which is what lets the two fields be evicted independently. RoPE here is **interleaved**: element ``2i`` pairs with element ``2i + 1``. This differs from the sparse indexer, where feature ``i`` pairs with ``i + rope_dim_half``.

   **Dimensions**:

   * n_tokens: Decode tokens in the batch, one per sequence. Must be <= P_MAX.
   * hidden: Model hidden width, the contraction dimension of both down-projections.
   * q_lora_rank: LoRA rank of the query path, derived from ``wq_a``.
   * kv_lora_rank: LoRA rank of the KV path. Must be a multiple of P_MAX.
   * n_heads: Attention heads.
   * qk_nope_dim: Per-head QK width that RoPE does not rotate. Must be <= P_MAX.
   * qk_rope_dim: Per-head QK width that RoPE rotates. Must be even.

