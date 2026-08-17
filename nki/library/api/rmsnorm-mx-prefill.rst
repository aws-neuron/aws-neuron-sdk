.. meta::
    :description: Fused RMSNorm [T,H] + MX quantization (+ optional router top-K) for prefill.
    :date-modified: 08/17/2026

.. currentmodule:: nkilib.core.rmsnorm

RMSNorm MX Prefill Kernel API Reference
=======================================

Fused RMSNorm [T,H] + MX quantization (+ optional router top-K) for prefill.

Background
-----------

The ``rmsnorm_mx_prefill`` kernel performs fused RMSNorm [T,H] + MX quantization (+ optional router top-K) for prefill.

API Reference
--------------

**Source code for this kernel API can be found at**: `rmsnorm_mx_prefill.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/rmsnorm/rmsnorm_mx_prefill.py>`_

rmsnorm_mx_prefill
^^^^^^^^^^^^^^^^^^

.. py:function:: rmsnorm_mx_prefill(hidden_states: nl.NkiTensor, gamma: nl.NkiTensor, router_weights: nl.NkiTensor = None, router_bias: nl.NkiTensor = None, eps: float = 1e-06, top_k: int = 1, router_act_fn: RouterActFnType = RouterActFnType.SIGMOID, n_group: int = 1, topk_group: int = 1, routed_scaling_factor: float = 1.0, qmx_output_dtype = nl.float8_e4m3fn_x4, pack_scales: bool = True, pack_affinities: bool = False, unpadded_hidden_size: int = None, residual: nl.NkiTensor = None, emit_norm_bf16: bool = False)

   Fused RMSNorm [T,H] + MX quantization (+ optional router top-K) for prefill.

   :param hidden_states: [B, S, H] bf16 input on HBM. Loaded in NATURAL H order -- the kernel applies the swizzle internally during the FP32-packed transpose (see below), so the caller must NOT pre-permute the hidden states.
   :type hidden_states: ``nl.NkiTensor``
   :param gamma: [1, H] or [H] RMSNorm weights on HBM. Natural H order (indexes hidden_states directly); not swizzled.
   :type gamma: ``nl.NkiTensor``
   :param router_weights: [H, E] router weights PRE-PERMUTED into the kernel's internal swizzle H order on HBM. If None, the router is skipped and only the packed quant tensor is returned. Why the permute: the router matmul runs against the swizzle-TRANSPOSED activations (the same transpose quantize_mx needs), whose partition axis is the swizzled H index h_swz = h512*512 + 4*p + q (h512 = H512-block, p = 0..127 partition, q = 0..3 lane) -- NOT natural h. To make logits = sum_h norm[t,h] * W[h,e] come out right, row h of the natural weight must be placed at swizzle slot h_swz. Equivalently, build the permuted weight as W_perm[swizzle_slot] = W[swizzle_h_index[swizzle_slot]]. Reference swizzle (offline host weight-prep): def swizzle_h_index(H):                  # swizzle slot -> original H index num_h512 = H // 512 h512 = torch.arange(num_h512).reshape(num_h512, 1, 1) q    = torch.arange(4).reshape(1, 4, 1) p    = torch.arange(128).reshape(1, 1, 128) return (h512 * 512 + 4 * p + q).reshape(-1)   # shape [H] router_weights_permuted = natural_router_weights[swizzle_h_index(H)]   # [H, E] For router_act_fn == NOAUX_TC, the SAME swizzle-permute applies -- the noaux_tc router still scores against the swizzle-transposed activations, so its weight must be permuted the same way.
   :type router_weights: ``nl.NkiTensor``
   :param router_bias: [1, E] or [E] optional router bias on HBM. For NOAUX_TC this is the e_score_correction_bias (added to the sigmoid scores for group/expert SELECTION only; the returned affinity is normalized from the PRE-bias sigmoid scores). Required for NOAUX_TC.
   :type router_bias: ``nl.NkiTensor``
   :param eps: epsilon for numerical stability.
   :type eps: ``float``
   :param top_k: number of experts to select per token (<= 8). Only used when router_weights set.
   :type top_k: ``int``
   :param router_act_fn: SIGMOID, SOFTMAX, or NOAUX_TC. Only used when router_weights set. NOAUX_TC - group-limited router (see n_group/topk_group/routed_scaling_factor) and emits ONLY the dense expert_affinities [T, E] (no expert_index tensor).
   :type router_act_fn: ``RouterActFnType``
   :param n_group: NOAUX_TC only. Number of expert groups (E must be divisible by n_group; <= 8 for the max8-based per-group selection). Ignored for SIGMOID/SOFTMAX.
   :type n_group: ``int``
   :param topk_group: NOAUX_TC only. Number of groups kept after group-level gating (<= 8). Ignored for SIGMOID/SOFTMAX.
   :type topk_group: ``int``
   :param routed_scaling_factor: NOAUX_TC only. Final multiplier on the L1-normalized top-k affinities. Ignored for SIGMOID/SOFTMAX.
   :type routed_scaling_factor: ``float``
   :param qmx_output_dtype: packed MX output dtype (float8_e4m3fn_x4 default).
   :param pack_scales: controls how the per-block MX scales are laid out in the scale region of each packed output row. quantize_mx emits one uint8 scale per (32-partition quadrant x 4-lane) group, so each H512 tile produces scales that only occupy 4 of every 32 columns after transpose -- 7/8 of an unfolded scale tile is wasted padding. - pack_scales=True (default, "folded"): pack 4 consecutive H512 tiles into one 128-wide block by shifting tile k into within-quadrant offset (k%4)*4. This reclaims the wasted columns, so scale_region = ceil(num_H512/4)*128 (roughly H/16 columns). - pack_scales=False ("unfolded"): one 128-wide block per H512 tile, no folding, so scale_region = num_H512*128 (= H/4 columns) -- ~4x larger
   :type pack_scales: ``bool``
   :param pack_affinities: router only. If True, the dense [T, E] expert affinities are concatenated into each packed row after the scale region (as bf16 reinterpreted into the fp8 row) instead of returned as a separate HBM tensor, so a downstream block gather pulls [hidden | scale | affinities] in one indirect DMA. The total row is padded to a multiple of 4 fp8 columns (the hidden region's fp32-reinterpret transpose requires it). When False (default), expert_affinities is returned as its own [T, E] tensor (legacy layout).
   :type pack_affinities: ``bool``
   :param unpadded_hidden_size: actual (unpadded) hidden size for the RMS mean denominator. When the input H is zero-padded offline (e.g. up to a multiple of 512), the sum-of-squares is taken over the full padded H (the zero pad contributes 0), but the mean divides by unpadded_hidden_size so padding does not skew the norm. Defaults to H (no padding).
   :type unpadded_hidden_size: ``int``
   :param residual: [B, S, H] optional residual on HBM. When set, the kernel adds it to hidden_states before RMSNorm (hidden = hidden_states + residual); the norm/quant/router all consume the sum. The pre-norm sum is also written out (output_residual) for the next layer's residual stream. If None, no residual add is performed.
   :type residual: ``nl.NkiTensor``
   :param emit_norm_bf16: when True, additionally return the token-major bf16 RMSNorm output norm_bf16 [T, H] = (hidden [+ residual]) * inv_rms * gamma, in NATURAL H order -- the same value a standalone RMSNorm produces. This lets one fused launch feed both the MX-quant consumers AND bf16 consumers (attention wq_a/wkv_a, the sparse indexer wk/weights_proj) that need the normed activation un-quantized. Computed fp32 and cast to bf16 on store (matches a torch RMSNorm reference); orthogonal to the router and to residual.
   :type emit_norm_bf16: ``bool``

   **Dimensions**:

   * B: batch, S: sequence, T = B*S tokens

