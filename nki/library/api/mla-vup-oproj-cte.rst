.. meta::
    :description: Standalone MX V-up + MX output projection (S-sharded across cores).
    :date-modified: 08/17/2026

.. currentmodule:: nkilib.experimental.mla.deepseek

MLA V-Up O-Proj CTE Kernel API Reference
========================================

Standalone MX V-up + MX output projection (S-sharded across cores).

KERNEL B of the split DeepSeek-V3.2 sparse-MLA forward. Reads the latent attention output from kernel A (mla_sparse_attention_cte_kernel), V-ups each head's latent with W_uv (MX fp8x4), then projects the H*d_v activation with W_o (MX) into out_hbm[B=1, S, HID]. Each core projects its own queries over the full HID. Intended for Context Encoding with DeepSeek-V3.2 dims (L == 512 kv_lora_rank, d_v == 128, H a multiple of 4 up to 128); requires B == 1 and S divisible by the number of cores. Budget-aware o_proj weight residency: full W_o loaded once when it fits SBUF, else K-slab-streamed double-buffered.

Background
-----------

The ``mla_vupmx_oproj_cte_kernel`` kernel performs standalone MX V-up + MX output projection (S-sharded across cores).

API Reference
--------------

**Source code for this kernel API can be found at**: `mla_vup_oproj_cte.py <https://github.com/aws-neuron/nki-library/blob/2.32/src/nkilib_src/nkilib/experimental/mla/deepseek/mla_vup_oproj_cte.py>`_

mla_vupmx_oproj_cte_kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: mla_vupmx_oproj_cte_kernel(out_attn_hbm: nl.NkiTensor, wuv_qtz_hbm: nl.NkiTensor, wuv_scale_hbm: nl.NkiTensor, wo_qtz_hbm: nl.NkiTensor, wo_scale_hbm: nl.NkiTensor) -> nl.NkiTensor

   Standalone MX V-up + MX output projection (S-sharded across cores).

   :param out_attn_hbm: [B, S, H*L] bf16 latent attention output from kernel A, carrying the cross-kernel MX 4-pack column layout.
   :type out_attn_hbm: ``nl.NkiTensor``
   :param wuv_qtz_hbm: [H*L // 4, d_v] fp8x4 packed MX V-up weight.
   :type wuv_qtz_hbm: ``nl.NkiTensor``
   :param wuv_scale_hbm: [H*L // 128, ceil(d_v / 128)] uint8 compact block-128 scales.
   :type wuv_scale_hbm: ``nl.NkiTensor``
   :param wo_qtz_hbm: [H*d_v // 4, HID] fp8x4 packed MX o_proj weight.
   :type wo_qtz_hbm: ``nl.NkiTensor``
   :param wo_scale_hbm: [H*d_v // 128, ceil(HID / 128)] uint8 compact block-128 scales.
   :type wo_scale_hbm: ``nl.NkiTensor``
   :return: [B, S, HID] bf16 output projection result.
   :rtype: ``nl.ndarray``

   **Notes**:

   * H and L are recovered from tensor shapes (L fixed at 512): wuv_qtz_hbm is [H*L // 4, d_v]. Do NOT pass H as a runtime scalar — the framework materializes it as an HBM tensor, not a trace-time int, so shape math derived from it is garbage.
   * Consumes the cross-kernel MX 4-pack column layout kernel A writes into out_attn (each (k512, sub)'s latent groups contiguous in HBM), so the latent transpose is a contiguous swdge dma_transpose off the Tensor Engine.

   **Dimensions**:

   * B: Batch size (must be 1)
   * S: Sequence length (this rank's S-shard)
   * H: Number of attention heads (a multiple of 4)
   * L: Latent (kv_lora_rank) dimension (fixed at 512 = P_MAX * 4)
   * d_v: Per-head V dimension (must be 128 = P_MAX)
   * HID: Output projection (hidden) dimension

