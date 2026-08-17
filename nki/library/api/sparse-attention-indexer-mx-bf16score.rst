.. meta::
    :description: DeepSeek Sparse Attention Indexer — MX projections + BF16 score.
    :date-modified: 08/17/2026

.. currentmodule:: nkilib.experimental.sparse_attention_indexer

Sparse Attention Indexer MX BF16Score Kernel API Reference
==========================================================

DeepSeek Sparse Attention Indexer — MX projections + BF16 score.

Supported/optimal usage: head_dim == 128; dim and q_lora_rank multiples of 512; batch_size == 1 and S a multiple of P_MAX (=128) on the validated v3_long/h64 configs; LNC sharding degree up to 2.

Background
-----------

The ``sparse_attention_indexer_mx_bf16score`` kernel implements the DeepSeek Sparse Attention Indexer with MX projections and a BF16 score path.

API Reference
--------------

**Source code for this kernel API can be found at**: `sparse_attention_indexer_mx_bf16score.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/sparse_attention_indexer/sparse_attention_indexer_mx_bf16score.py>`_

sparse_attention_indexer_mx_bf16score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: sparse_attention_indexer_mx_bf16score(x: nl.NkiTensor, wq_b: nl.NkiTensor, wk: nl.NkiTensor, k_norm_gamma: nl.NkiTensor, k_norm_beta: nl.NkiTensor, weights_proj: nl.NkiTensor, cos: nl.NkiTensor, sin: nl.NkiTensor, k_cache: nl.NkiTensor, mask: nl.NkiTensor, n_heads: int, head_dim: int, rope_head_dim: int, index_topk: int, start_pos: int, use_hadamard: bool = False, batch_size: int = 1, wq_b_scale: Optional[nl.NkiTensor] = None, wk_scale: Optional[nl.NkiTensor] = None, k_scale_cache: Optional[nl.NkiTensor] = None, x_non_mx: Optional[nl.NkiTensor] = None, x_mx_data: Optional[nl.NkiTensor] = None, x_mx_scale: Optional[nl.NkiTensor] = None, qr_qtz_hbm: Optional[nl.NkiTensor] = None, qr_scale_hbm: Optional[nl.NkiTensor] = None, phase: str = 'all', k_seq_out_hbm: Optional[nl.NkiTensor] = None, end_pos_arg: Optional[int] = None, emit_flat_topk: bool = False, emit_tiled_topk: bool = False) -> tuple[nl.NkiTensor, nl.NkiTensor]

   DeepSeek Sparse Attention Indexer — MX projections + BF16 score.

   :param x: [B*S, dim] hidden states.
   :type x: ``nl.NkiTensor``
   :param qr_qtz_hbm: [num_s_tiles, P_MAX, q_lora_rank // (P_MAX*H_PACK), P_MAX] uint32 — pre-quantized (fp8x4-as-uint32) compressed query from the upstream QKV kernel.
   :type qr_qtz_hbm: ``Optional[nl.NkiTensor]``
   :param qr_scale_hbm: matching uint8 block-32 e8m0 scales for qr_qtz_hbm.
   :type qr_scale_hbm: ``Optional[nl.NkiTensor]``
   :param wq_b: [q_lora_rank // 4, n_heads * head_dim] fp8_e4m3fn_x4.
   :type wq_b: ``nl.NkiTensor``
   :param wq_b_scale: [q_lora_rank // 32, n_heads * head_dim] uint8 e8m0.
   :type wq_b_scale: ``Optional[nl.NkiTensor]``
   :param wk: [dim // 4, head_dim] fp8_e4m3fn_x4 (contraction dim on partition).
   :type wk: ``nl.NkiTensor``
   :param wk_scale: [dim // 32, head_dim] uint8 e8m0. k_norm_gamma, k_norm_beta: [head_dim] LayerNorm gamma/beta.
   :type wk_scale: ``Optional[nl.NkiTensor]``
   :param weights_proj: [n_heads, dim] per-head weights projection (always bf16). cos, sin: [S, rope_head_dim // 2] RoPE caches.
   :type weights_proj: ``nl.NkiTensor``
   :param k_cache: [B, head_dim, max_seq_len] bf16 (mutable).
   :type k_cache: ``nl.NkiTensor``
   :param k_scale_cache: ignored (no MX scales in bf16 path).
   :type k_scale_cache: ``Optional[nl.NkiTensor]``
   :param mask: [B*S, end_pos] attention mask.
   :type mask: ``nl.NkiTensor``
   :param index_topk: number of top positions to select per query (topk k).
   :type index_topk: ``int``
   :param x_non_mx: pre-cast bf16 view of x for W-projection (skips the f32->bf16 cast). x_mx_data, x_mx_scale: pre-quantized x for K-projection (skips the in-kernel swizzle + quantize_mx). qr_qtz_hbm, qr_scale_hbm: pre-quantized qr latent from an upstream QKV kernel (indexer skips its own qr transpose+norm+quantize). This is the only Q input path — qr is never quantized in-kernel.
   :type x_non_mx: ``Optional[nl.NkiTensor]``
   :param phase: context-parallel phased split. "all" (default): fused self-attention (queries == keys == S). "kproj": project K for this rank's S shard only; write seq-major [S, head_dim] to k_seq_out_hbm, return (that, None), no score. "score": skip K-proj; score this rank's S queries vs the pre-gathered full k_cache [B, head_dim, end_pos_arg=S_kv] over [0, S_kv).
   :type phase: ``str``
   :param k_seq_out_hbm: [S, head_dim] bf16 K output for phase="kproj".
   :type k_seq_out_hbm: ``Optional[nl.NkiTensor]``
   :param end_pos_arg: key range for phase="score" (defaults to start_pos + S).
   :type end_pos_arg: ``Optional[int]``
   :param emit_flat_topk: decode the snake topk to per-query positions in-kernel and return flat [B*S, index_topk] int32 (the sparse_mla_latent_attn topk_indices contract). Fuses the out-of-kernel snake->flat decode.
   :type emit_flat_topk: ``bool``
   :param emit_tiled_topk: split-fix output — skip the per-query scatter and return positions in the natural partition-tiled layout [num_s_tiles, NUM_TOPK_BATCHES, P_MAX, index_topk // 16] int32, read per-query by the paired attention kernel. Mutually exclusive with emit_flat_topk; both default False (snake output, decoded by the consumer, e.g. the fused mega kernel).
   :type emit_tiled_topk: ``bool``
   :return: [B*S, end_pos] f32 — pre-topk per-position scores.
   :rtype: ``nl.ndarray``
   :return: [num_S_tiles_total, NUM_TOPK_BATCHES, P_MAX, index_topk] uint32 — hardware-topk output. ``topk_idx[s_tile, t, 16g + j%16, j//16]`` holds the position (0-based in ``[0, end_pos)``) of query ``8t + g``'s j-th largest score (ascending). ``num_S_tiles_total = batch_size * ceil(S / P_MAX)``.
   :rtype: ``nl.ndarray``

   **Notes**:

   * Q/K/W projections still use ``nc_matmul_mx`` for speed, but the score matmul uses bf16 nc_matmul (no Q/K quantization, no Hadamard rotation). This removes the Q swizzle + quantize_mx + Hadamard chain from the per-S-tile critical path.
   * K cache stores bf16 instead of fp8x4 (no need for MX format if downstream consumers use this same kernel; otherwise downstream must coordinate).
   * Algorithm equivalent to the MX variant via Hadamard's orthogonality: ``q_H @ (k_H).T = q @ H @ H.T @ k.T = q @ k.T`` so dropping Hadamard is mathematically a no-op. The MX quantization that Hadamard exists to support is also dropped on the score path. Score precision improves (bf16 score matmul vs MX-rounded score matmul).

   **Dimensions**:

   * B: batch size (= batch_size).
   * S: query sequence length per batch.
   * M: B * S (total query rows, = x.shape[0]).
   * dim: hidden size (= x.shape[1]).
   * q_lora_rank: compressed-query rank (derived from qr_qtz_hbm layout).
   * n_heads: number of indexer heads.
   * head_dim: per-head dimension (== 128).
   * rope_head_dim: RoPE-rotated slice of head_dim.
   * end_pos: start_pos + S (KV positions scored per query).
   * P_MAX: partition tile size (= 128).

