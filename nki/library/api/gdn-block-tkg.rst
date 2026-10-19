.. meta::
    :description: Fused GDN decode block: in_proj + conv1d + GQA + gated delta-rule in ONE kernel.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.gdn

GDN Block TKG Kernel API Reference
==================================

Fused GDN decode block: in_proj + conv1d + GQA + gated delta-rule in ONE kernel.

Runs the full Qwen3.5 GDN decode layer except out_proj / all_reduce: in_proj matmuls -> gdn_conv1d_decode (conv + silu + slide-window state) -> split conv_out + GQA-repeat q,k -> gdn_tkg (l2norm/beta/g/RMSNormGated + single-token recurrence) -> scatter to core_out.

Background
-----------

The ``gdn_block_tkg`` kernel fuses a full Qwen3.5 gated delta-net (GDN) decode layer, except the output projection and all-reduce, into one kernel: input projections, causal conv1d, GQA expansion, and the gated delta-rule recurrence.

API Reference
--------------

**Source code for this kernel API can be found at**: `gdn_block_tkg.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/gdn/gdn_block_tkg.py>`_

gdn_block_tkg
^^^^^^^^^^^^^

.. py:function:: gdn_block_tkg(hidden: nl.NkiTensor, W_in_qkv: nl.NkiTensor, W_in_z: nl.NkiTensor, W_in_a: nl.NkiTensor, W_in_b: nl.NkiTensor, conv_weight: nl.NkiTensor, conv_state: nl.NkiTensor, A_log: nl.NkiTensor, dt_bias: nl.NkiTensor, norm_weight: nl.NkiTensor, recurrent_state: nl.NkiTensor) -> Tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]

   Fused GDN decode block: in_proj + conv1d + GQA + gated delta-rule in ONE kernel.

   :param hidden: [B, 1, H] @ HBM, bf16 decode-step hidden states.
   :type hidden: ``nl.NkiTensor``
   :param W_in_qkv: [H, conv_dim] @ HBM, bf16 q|k|v in_proj weight (== nn.Linear.weight.T).
   :type W_in_qkv: ``nl.NkiTensor``
   :param W_in_z: [H, value_dim] @ HBM, bf16 gate (z) in_proj weight.
   :type W_in_z: ``nl.NkiTensor``
   :param W_in_a: [H, num_v_heads] @ HBM, bf16 gate-input (a) in_proj weight.
   :type W_in_a: ``nl.NkiTensor``
   :param W_in_b: [H, num_v_heads] @ HBM, bf16 beta-input (b) in_proj weight.
   :type W_in_b: ``nl.NkiTensor``
   :param conv_weight: [conv_dim, K_win] @ HBM, fp32 depthwise conv weight (self.conv1d.weight squeezed).
   :type conv_weight: ``nl.NkiTensor``
   :param conv_state: [B, conv_dim, K_win] @ HBM, bf16 conv window (read at entry).
   :type conv_state: ``nl.NkiTensor``
   :param A_log: [num_v_heads] @ HBM, fp32 per-v-head log decay.
   :type A_log: ``nl.NkiTensor``
   :param dt_bias: [num_v_heads] @ HBM, fp32 per-v-head softplus bias.
   :type dt_bias: ``nl.NkiTensor``
   :param norm_weight: [head_v_dim] @ HBM, fp32 RMSNormGated weight (gamma).
   :type norm_weight: ``nl.NkiTensor``
   :param recurrent_state: [B, num_v_heads, head_k_dim, head_v_dim] @ HBM, fp32 recurrent state (read at entry).
   :type recurrent_state: ``nl.NkiTensor``
   :return: [B, 1, value_dim] @ HBM, RMSNormGated decode output.
   :rtype: ``nl.NkiTensor``
   :return: [B, conv_dim, K_win] @ HBM, updated conv window.
   :rtype: ``nl.NkiTensor``
   :return: [B, num_v_heads, head_k_dim, head_v_dim] @ HBM, updated recurrent state.
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * Functional: inputs are not mutated in place; the caller aliases the returned conv_state / recurrent_state back into the persistent caches.
   * gdn_conv1d_decode and gdn_tkg are reused verbatim as nested @nki.jit calls; internal nl.hbm scratch carries the inter-stage activations.
   * All dims are derived from tensor shapes (no python-int args).

   **Dimensions**:

   * B: Batch size
   * H: Hidden dimension (must be a multiple of 128)
   * conv_dim: Conv channel count = 2*key_dim + value_dim (q|k|v packed)
   * key_dim: Query/key projection dim = num_k_heads * head_k_dim
   * value_dim: Value projection dim = num_v_heads * head_v_dim
   * num_k_heads: Number of key heads
   * num_v_heads: Number of value heads (GQA target)
   * head_k_dim: Key/state head dim (<= 128)
   * head_v_dim: Value head dim (<= 128)
   * gqa: Query heads per key head = num_v_heads / num_k_heads
   * BH: Flattened B * num_v_heads

