.. meta::
    :description: Computes one DeepSeek-V3.2 MLA decode step's attention over the prior keys an indexer selected, then projects the result
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.deepseek_v32.mla_decode_attention

MLA Decode Attention Kernel API Reference
=========================================

Computes one DeepSeek-V3.2 MLA decode step's attention over the prior keys an indexer selected, then projects the result out of the cache's latent space and through the o-projection, returning this rank's attention-block output.

This is the back half of an MLA decode layer, and the counterpart of the QKV front-fold. It is designed to be traced inline after that front-fold rather than run standalone: the two query tiles arrive **in SBUF**, already feature-major in the layout the score matmuls index, so a standalone caller has to stage them itself. Selection has already happened, so there is no sparse score bias here. What the selection leaves behind is a per-token map from key slot to absolute position, and that map drives the causal mask. Rather than read a pre-gathered feature-major prior out of HBM, the kernel gathers each token's selected rows straight from ``kv_cache`` in the cache's native token-major layout, and then transposes once to feature-major for the score matmul. The weighted sum contracts over keys and so wants the token-major tile, which is what the gather already produced, and reads it directly. That deletes a 33.5 MB HBM round trip and one of two mirror-image transposes.

Background
-----------

The ``mla_decode_attention`` kernel computes DeepSeek-V3.2 MLA decode attention over the prior keys selected by an indexer, then applies the V up-projection and output projection. It is designed to be traced inline after ``mla_decode_qkv``.

API Reference
--------------

**Source code for this kernel API can be found at**: `mla_decode_attention.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/deepseek_v32/mla_decode_attention/mla_decode_attention.py>`_

mla_decode_attention
^^^^^^^^^^^^^^^^^^^^

.. py:function:: mla_decode_attention(q_absorbed_sbuf: nl.NkiTensor, q_pe_sbuf: nl.NkiTensor, kv_cache: nl.NkiTensor, kv_row_cur: nl.NkiTensor, prior_rows: nl.NkiTensor, prior_key_pos: nl.NkiTensor, prior_rows_natural: nl.NkiTensor, positions: nl.NkiTensor, out_absorb_w: nl.NkiTensor, o_proj_w: nl.NkiTensor, n_prior: int, softmax_scale: float) -> nl.NkiTensor

   Computes one DeepSeek-V3.2 MLA decode step's attention over the prior keys an indexer selected, then projects the result out of the cache's latent space and through the o-projection, returning this rank's attention-block output.

   :param q_absorbed_sbuf: ``[P_MAX, n_latent_tiles, n_tokens, n_heads]``. The query already mapped into the cache's latent space by the front-fold's absorb.
   :type q_absorbed_sbuf: ``nl.NkiTensor``
   :param q_pe_sbuf: ``[P_MAX, n_tokens, n_heads]``. The query's RoPE'd positional half.
   :type q_pe_sbuf: ``nl.NkiTensor``
   :param kv_cache: ``[n_blocks, 1, block_size, kv_row_dim]``. Paged cache, read only here. The current token's row must already have been scattered into it by the front-fold.
   :type kv_cache: ``nl.NkiTensor``
   :param kv_row_cur: ``[n_tokens, kv_row_dim]``. This step's cache row per token, laid out ``[k_pe | kv_latent]``.
   :type kv_row_cur: ``nl.NkiTensor``
   :param prior_rows: ``[P_MAX, n_tokens, n_prior_tiles]`` int32. Flat cache row per selected key, in the batched gather's consumption order. From ``build_prior_index``.
   :type prior_rows: ``nl.NkiTensor``
   :param prior_key_pos: ``[P_MAX, n_tokens, n_prior_tiles]`` fp32. Absolute position of each selected key, key-major. From ``build_prior_index``.
   :type prior_key_pos: ``nl.NkiTensor``
   :param prior_rows_natural: ``[P_MAX, n_tokens, n_prior_tiles]`` uint32. The same rows in natural key order, which is what the indirect ``dma_transpose`` needs. From ``build_prior_index``.
   :type prior_rows_natural: ``nl.NkiTensor``
   :param positions: ``[n_tokens]`` int32. This step's decode position per sequence.
   :type positions: ``nl.NkiTensor``
   :param out_absorb_w: ``[n_heads, v_head_dim, kv_lora_rank]``. The value half of ``wkv_b``, which maps a head's latent attention output back out into ``v_head_dim``.
   :type out_absorb_w: ``nl.NkiTensor``
   :param o_proj_w: ``[n_heads * v_head_dim, hidden]``. This rank's o-projection.
   :type o_proj_w: ``nl.NkiTensor``
   :param n_prior: Prior keys the indexer selected per token.
   :type n_prior: ``int``
   :param softmax_scale: Scale applied to the scores before masking and softmax.
   :type softmax_scale: ``float``

   **Notes**:

   * Scores and softmax weights are held **key-major**: key ``tile * P_MAX + p`` of head ``h`` sits at ``[p, tile, h]``. Swapping which matmul operand is stationary transposes the output for free, and putting keys on partitions pays off three times over. The softmax then runs 128 lanes wide instead of on the single partition a ``[n_heads, n_keys]`` layout leaves at ``n_heads == 1``; the causal mask becomes a plain per-partition column instead of a broadcast matmul; and the weighted sum's stationary operand is already oriented correctly, deleting a transpose. This was the single largest win in this kernel's history, 521.5 -> 402.2 us, and most of it came from lane utilization rather than from the transpose removal. The out-absorb and o-projection weights do not depend on the token, so they are loaded once above the token loop. Both tails are also batched across tokens: they are per-token GEMVs against shared weights, and a ``128x1`` matmul costs about 360 ns against 179 ns for a ``128x128``, because the cost is set by the moving operand and a width-1 stationary pays full price for 1/128 of the array. Emitting the weighted sum for the previous token, so the PE could run ahead of a softmax, was measured and is exactly neutral: the post-scheduler reorders globally, so NKI emission order does not control per-engine issue order. Do not expect a win from hand-scheduling here. Giving each latent tile its own 32-partition band via ``tile_position``, so the tiles issue concurrently instead of chaining through one PSUM bank, was measured at 21% slower (433.8 -> 525.7 us). It band-sums per token and key chunk rather than once per kernel, and each sum is three extra full-width Vector operations on what is already the bottleneck engine.

   **Dimensions**:

   * n_tokens: Decode tokens in the batch, one per sequence. Must be <= P_MAX.
   * n_heads: Attention heads on this rank. Must be <= P_MAX.
   * n_prior: Prior keys the indexer selected per token.
   * n_keys: Keys scored per token, ``n_prior + 1``, the selected prior plus the current token.
   * kv_lora_rank: LoRA rank of the KV path, and the width of a cache row's latent field. Must be a multiple of P_MAX and fit one PSUM bank.
   * qk_rope_dim: Width of a cache row's ``k_pe`` field. Must be <= P_MAX.
   * v_head_dim: Per-head width the out-absorb produces. Must be <= P_MAX.
   * hidden: Model hidden width, which the o-projection emits.

