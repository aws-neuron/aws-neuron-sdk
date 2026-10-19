.. meta::
    :description: Standalone sparse latent + RoPE attention (S-sharded across cores).
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.mla.deepseek

MLA Sparse Attention CTE Kernel API Reference
=============================================

Standalone sparse latent + RoPE attention (S-sharded across cores).

KERNEL A of the split DeepSeek-V3.2 sparse-MLA forward: the un-projected latent value path. Computes, per query, absorbed-latent sparse attention over the topk-selected cache rows. Intended for Context Encoding with DeepSeek-V3.2 dims (L == 512 kv_lora_rank, R == 64, up to 128 heads, topk K a multiple of 128 up to ~2048); pair with the o_proj kernel B for V-up + o_proj. Requires B == 1 and S divisible by the number of cores.

Background
-----------

The ``mla_sparse_attention_cte_kernel`` kernel performs standalone sparse latent + RoPE attention (S-sharded across cores).

API Reference
--------------

**Source code for this kernel API can be found at**: `mla_sparse_attention_cte.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/mla/deepseek/mla_sparse_attention_cte.py>`_

mla_sparse_attention_cte_kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: mla_sparse_attention_cte_kernel(q_lift_hbm: nl.NkiTensor, q_pe_hbm: nl.NkiTensor, c_kv_hbm: nl.NkiTensor, k_pe_hbm: nl.NkiTensor, softmax_scale: float, topk_indices_hbm: nl.NkiTensor = None, topk_tiled: bool = False, dense: bool = False, q_pos_offset_hbm: nl.NkiTensor = None, precision: MlaPrecision = MlaPrecision.MX, use_sbuf_indirect: bool = False) -> nl.NkiTensor

   Standalone sparse latent + RoPE attention (S-sharded across cores).

   :param q_lift_hbm: [B, S, H, L] bf16, per-head absorbed Q latent.
   :type q_lift_hbm: ``nl.NkiTensor``
   :param q_pe_hbm: [B, S, H, R] bf16, per-head pre-rotated RoPE queries.
   :type q_pe_hbm: ``nl.NkiTensor``
   :param c_kv_hbm: [B, S_kv, L] bf16, latent KV cache.
   :type c_kv_hbm: ``nl.NkiTensor``
   :param k_pe_hbm: [B, S_kv, R] bf16, pre-rotated RoPE key cache.
   :type k_pe_hbm: ``nl.NkiTensor``
   :param softmax_scale: Scaling factor applied to the attention scores. Must be positive. DeepSeek's scale is head_dim**-0.5 times a squared mscale correction, so it is always positive in practice.
   :type softmax_scale: ``float``
   :param topk_indices_hbm: int32 topk cache-row indices. Flat [B, S, K] when topk_tiled is False; partition-tiled [num_s_tiles, NUM_TOPK_BATCHES, P_MAX, K // 16] when topk_tiled is True. Required for the sparse path; unused (and may be None) when dense=True.
   :type topk_indices_hbm: ``nl.NkiTensor``
   :param topk_tiled: Select the topk_indices_hbm layout (default False = flat).
   :type topk_tiled: ``bool``
   :param dense: When True, attend over all keys with a causal mask instead of the indexer-selected top-k keys (no indexer inputs required). Default: False.
   :type dense: ``bool``
   :param q_pos_offset_hbm: [1, 1] int32/float32 GLOBAL sequence offset of this rank's query shard (``cp_rank * S_local`` under Context Parallelism), used by the causal mask. A TENSOR rather than a Python int so the framework traces ONE graph for every CP rank instead of specializing (and recompiling) per rank. None (the default) means offset 0 -- the non-CP case.
   :type q_pos_offset_hbm: ``nl.NkiTensor``
   :param precision: precision of the DOWNSTREAM V-up + o_proj kernel (kernel B). This kernel's arithmetic is bf16 regardless; ``precision`` selects only the ``out_attn`` latent COLUMN LAYOUT so it matches what kernel B reads. ``MX`` (default) pre-permutes each head's latent into 4-pack order; ``BF16`` emits NATURAL order (the bf16 o_proj transposes on the Tensor Engine and has no MX layout to feed). See :func:`_mm2_out_aps`. Every GLM-MoE-DSA dim this kernel constrains (L == 512, R == 64, H == 64, topk K == 2048) is already legal, so this layout switch is the only change GLM needs here.
   :type precision: ``MlaPrecision``
   :param use_sbuf_indirect: select how the topk-selected KV rows are gathered.
   :type use_sbuf_indirect: ``bool``
   :return: [B, S, H * L] bf16 latent attention output, row-major h * L + l. Each head's latent columns are pre-permuted into MX 4-pack order when ``precision`` is MX, or left in natural order when BF16.
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * S-sharded across cores; under Context Parallelism the framework gathers the latent KV to the full S_kv before this kernel is called.
   * The MM2 output is written with each head's latent columns pre-permuted into MX 4-pack order (natural latent l = 4*group + sub at physical column sub*(L//4) + group), the cross-kernel layout contract the o_proj kernel's DMA-transpose load depends on. Keep in lockstep with the o_proj load, W_uv load, and both refs.

   **Dimensions**:

   * B: Batch size (must be 1)
   * S: Query sequence length (this rank's S-shard)
   * S_kv: Cache (key/value) sequence length
   * H: Number of attention heads
   * L: Latent (kv_lora_rank) dimension (must be 512 = P_MAX * 4)
   * R: RoPE head dimension

