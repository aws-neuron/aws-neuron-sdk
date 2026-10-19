.. meta::
    :description: Score every cached key slot against the current query and select the highest scoring ones.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.deepseek_v32.sparse_indexer

DeepSeek-V3.2 Sparse Indexer Kernel API Reference
=================================================

Score every cached key slot against the current query and select the highest scoring ones.

This is the DSA lightning indexer for one decode step, optimized for the token-generation regime where each sequence contributes a single token: ``n_tokens <= P_MAX``, with 16 at the shipping shape. The query projection is always MX-quantized, so it requires trn3 or later.

Background
-----------

The ``deepseek_v32_sparse_indexer`` kernel implements the DeepSeek-V3.2 sparse attention (DSA) lightning indexer for one decode step. It scores every cached key slot against the current query and selects the highest-scoring slots. It requires Trn3 or later.

API Reference
--------------

**Source code for this kernel API can be found at**: `sparse_indexer.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/deepseek_v32/sparse_indexer/sparse_indexer.py>`_

deepseek_v32_sparse_indexer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: deepseek_v32_sparse_indexer(hidden_states: nl.NkiTensor, qr: nl.NkiTensor, wq_b: nl.NkiTensor, wq_b_scale: nl.NkiTensor, wk: nl.NkiTensor, k_norm_weight: nl.NkiTensor, k_norm_bias: nl.NkiTensor, weights_proj: nl.NkiTensor, hadamard: nl.NkiTensor, cos: nl.NkiTensor, sin: nl.NkiTensor, key_cache: nl.NkiTensor, positions: nl.NkiTensor, index_topk: int, valid_len: int | None = None) -> DeepseekV32SparseIndexerOutput

   Score every cached key slot against the current query and select the highest scoring ones.

   :param hidden_states: [n_tokens, hidden], Hidden states for this step
   :type hidden_states: ``nl.NkiTensor``
   :param qr: [n_tokens, q_lora_rank], ``q_norm(wq_a(x))`` from the MLA block
   :type qr: ``nl.NkiTensor``
   :param wq_b: [q_lora_rank // K_PACK, n_heads * head_dim], Query projection weight, fp8x4 MX-packed along the contraction dimension
   :type wq_b: ``nl.NkiTensor``
   :param wq_b_scale: [q_lora_rank // MX_BLOCK, n_heads * head_dim], uint8 e8m0 MX scale paired with ``wq_b``
   :type wq_b_scale: ``nl.NkiTensor``
   :param wk: [hidden, head_dim], Key projection weight
   :type wk: ``nl.NkiTensor``
   :param k_norm_weight: [1, head_dim], fp32 LayerNorm gain for the key
   :type k_norm_weight: ``nl.NkiTensor``
   :param k_norm_bias: [1, head_dim], fp32 LayerNorm bias for the key
   :type k_norm_bias: ``nl.NkiTensor``
   :param weights_proj: [hidden, n_heads], Per-head weight projection
   :type weights_proj: ``nl.NkiTensor``
   :param hadamard: [head_dim, head_dim], Scaled Hadamard matrix with ``1/sqrt(head_dim)`` pre-baked
   :type hadamard: ``nl.NkiTensor``
   :param cos: [n_tokens, rope_dim // 2], RoPE cosine table
   :type cos: ``nl.NkiTensor``
   :param sin: [n_tokens, rope_dim // 2], RoPE sine table
   :type sin: ``nl.NkiTensor``
   :param key_cache: [n_tokens, max_seq_len, head_dim], Caller-owned key cache, written in place at each token's current position
   :type key_cache: ``nl.NkiTensor``
   :param positions: [n_tokens], int32 absolute position of each sequence's current token
   :type positions: ``nl.NkiTensor``
   :param shapes: Input shapes and derived tiling constants
   :return: [n_tokens, max_seq_len], fp32 per-slot relevance, with slots the token cannot attend to clamped to the mask bound
   :rtype: ``nl.NkiTensor``
   :return: [n_tokens, index_topk], int32 selected slot positions
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * ``topk_indices`` is an unordered set: column order carries no meaning.
   * The mask bound is finite rather than ``-inf`` because the top-k consumer reads bfloat16, where ``-inf`` makes the hardware max-select produce NaN.
   * ``q_lora_rank`` must be a multiple of ``P_MAX * K_PACK``, the MX K-tile.

   **Dimensions**:

   * n_tokens: Decode tokens in the batch, one per sequence
   * n_heads: Indexer heads
   * head_dim: Per-head feature width, which must equal P_MAX
   * hidden: Model hidden width, the contraction dimension of the key projection
   * q_lora_rank: LoRA rank of the query path, the contraction dimension of the query projection
   * rope_dim: Leading feature span that RoPE rotates
   * max_seq_len: Key-cache capacity per sequence, in slots

