.. meta::
    :description: Computes one DeepSeek-V3.2 MLA decode step end to end, from the residual stream to this rank's attention-block output, a
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.deepseek_v32.mla_attention_block_gather_decode

MLA Attention Block Gather Decode Kernel API Reference
======================================================

Computes one DeepSeek-V3.2 MLA decode step end to end, from the residual stream to this rank's attention-block output, attending over only the prior keys the Lightning Indexer selected.

This is the whole attention block: it traces the QKV front-fold and the attention core into a single NEFF rather than calling them as separate kernels. That is what makes the composition worth having, because the front-fold's two query tiles stay **in SBUF** and are handed straight to the core. Run standalone the tiles would have to round-trip through HBM, and the core's signature is built for the inline hand-off, not for a caller staging them itself. The block owns no numerics of its own. Every value it returns is computed by one of the two kernels it composes, so its correctness is theirs plus the wiring, and its reference composes their references for the same reason.

Background
-----------

The ``mla_attention_block_gather_decode`` kernel computes one DeepSeek-V3.2 MLA decode step end to end, from the residual stream to the rank's attention-block output, attending only over the prior keys selected by the lightning indexer. It composes ``mla_decode_qkv`` and ``mla_decode_attention`` in a single kernel so the query tiles stay in SBUF.

API Reference
--------------

**Source code for this kernel API can be found at**: `mla_attention_block_gather_decode.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/deepseek_v32/mla_attention_block_gather_decode/mla_attention_block_gather_decode.py>`_

mla_attention_block_gather_decode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: mla_attention_block_gather_decode(hidden_states: nl.NkiTensor, input_norm_w: nl.NkiTensor, wq_a: nl.NkiTensor, q_norm_w: nl.NkiTensor, wq_b: nl.NkiTensor, wkv_a: nl.NkiTensor, kv_norm_w: nl.NkiTensor, q_absorb_w: nl.NkiTensor, out_absorb_w: nl.NkiTensor, o_proj_w: nl.NkiTensor, cos: nl.NkiTensor, sin: nl.NkiTensor, kv_cache: nl.NkiTensor, slot_mapping: nl.NkiTensor, block_table: nl.NkiTensor, positions: nl.NkiTensor, gather_indices: nl.NkiTensor, n_heads: int, qk_nope_head_dim: int, qk_rope_head_dim: int, kv_lora_rank: int, softmax_scale: float, norm_eps: float = 1e-06) -> nl.NkiTensor

   Computes one DeepSeek-V3.2 MLA decode step end to end, from the residual stream to this rank's attention-block output, attending over only the prior keys the Lightning Indexer selected.

   :param hidden_states: ``[n_tokens, hidden]``. This step's residual-stream hidden states.
   :type hidden_states: ``nl.NkiTensor``
   :param input_norm_w: ``[1, hidden]`` fp32. RMSNorm gamma for the layer input, folded in by the front-fold rather than applied by the caller.
   :type input_norm_w: ``nl.NkiTensor``
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
   :param q_absorb_w: ``[n_heads, qk_nope_dim, kv_lora_rank]``. The nope half of ``wkv_b``, which maps a head's nope query into the cache's latent space.
   :type q_absorb_w: ``nl.NkiTensor``
   :param out_absorb_w: ``[n_heads, v_head_dim, kv_lora_rank]``. The value half of ``wkv_b``, which maps the latent attention output back out to a head's value width.
   :type out_absorb_w: ``nl.NkiTensor``
   :param o_proj_w: ``[n_heads * v_head_dim, hidden]``. This rank's shard of the o-projection.
   :type o_proj_w: ``nl.NkiTensor``
   :param cos: ``[n_tokens, qk_rope_dim // 2]``. Cosine table for interleaved RoPE.
   :type cos: ``nl.NkiTensor``
   :param sin: ``[n_tokens, qk_rope_dim // 2]``. Sine table for interleaved RoPE.
   :type sin: ``nl.NkiTensor``
   :param kv_cache: ``[n_blocks, 1, block_size, kv_row_dim]``. Paged cache, **mutated in place**: the front-fold writes this step's row, and the core reads the selected prior rows back.
   :type kv_cache: ``nl.NkiTensor``
   :param slot_mapping: ``[n_tokens]`` int32. Flat cache row each token's own key is written to.
   :type slot_mapping: ``nl.NkiTensor``
   :param block_table: ``[n_tokens, max_blocks_per_seq]`` int32. Per-sequence logical-to-physical block map, which resolves a selected position to a cache row.
   :type block_table: ``nl.NkiTensor``
   :param positions: ``[n_tokens]`` int32. This step's absolute position per token, which the causal mask compares each selected key against.
   :type positions: ``nl.NkiTensor``
   :param gather_indices: ``[n_tokens, n_prior]`` int32. The absolute prior positions the indexer selected, unsorted. Its width fixes ``n_prior``.
   :type gather_indices: ``nl.NkiTensor``
   :param n_heads: Attention heads on this rank.
   :type n_heads: ``int``
   :param qk_nope_head_dim: Per-head query/key width that RoPE does not rotate.
   :type qk_nope_head_dim: ``int``
   :param qk_rope_head_dim: Per-head query/key width that RoPE rotates.
   :type qk_rope_head_dim: ``int``
   :param kv_lora_rank: LoRA rank of the KV path, and the latent width of a cache row.
   :type kv_lora_rank: ``int``
   :param softmax_scale: Score scale, conventionally the inverse square root of the full per-head query/key width.
   :type softmax_scale: ``float``
   :param norm_eps: Epsilon for every RMSNorm in the front-fold.
   :type norm_eps: ``float``

   **Notes**:

   * The index build is issued **first**, ahead of the front-fold that does not consume it. Its two inputs are under 1 MB together, but queued behind the projection weights they sit behind roughly 31 MB on the same DMA queue and do not land until about 97 us, and no per-token gather in the core can start until they do. Fusing the two halves is worth about 45 us at the shipping shape, measured as 385.8 us for the two run separately against 340.2 us here. The saving is the HBM round trip the query tiles would otherwise make, which is why the core takes them in SBUF.

   **Dimensions**:

   * n_tokens: Decode tokens in the batch, one per sequence.
   * n_heads: Attention heads on this rank.
   * n_prior: Prior keys the indexer selected per token, taken from ``gather_indices``.

