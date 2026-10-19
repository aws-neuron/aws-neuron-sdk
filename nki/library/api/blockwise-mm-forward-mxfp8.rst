.. meta::
    :description: MXFP8 forward pass for blockwise (dropless) Mixture of Experts.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.moe_mxfp8.fwd

Blockwise MM Forward MXFP8 Kernel API Reference
===============================================

MXFP8 forward pass for blockwise (dropless) Mixture of Experts.

Computes the MoE FFN output and emits the activation checkpoints the MXFP8 MoE backward (``blockwise_mm_bwd_mxfp8``) consumes, so fwd + bwd form a validated training pair. Tokens are processed in fixed-size blocks already assigned to a single expert each by an upstream router; this kernel never computes routing. Only weights support pre-quantized MXFP8 inputs. Activations (hidden_states) must be BF16 because they are gathered per-block via indirect DMA, which would break MXFP8 32-element quantization groups. When no_indirect_load is True, hidden_states must already contain block-aligned tokens for one expert and both weight tensors must have E=1.

Background
-----------

The ``blockwise_mm_fwd_mxfp8`` kernel computes the MXFP8 forward pass for blockwise (dropless) Mixture of Experts.

API Reference
--------------

**Source code for this kernel API can be found at**: `blockwise_mm_forward_mxfp8.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/moe_mxfp8/fwd/blockwise_mm_forward_mxfp8.py>`_

blockwise_mm_fwd_mxfp8
^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: blockwise_mm_fwd_mxfp8(hidden_states: nl.NkiTensor, expert_affinities_masked: nl.NkiTensor, gate_up_proj_weight: nl.NkiTensor, down_proj_weight: nl.NkiTensor, token_position_to_id: nl.NkiTensor, block_to_expert: nl.NkiTensor, block_size: int = 4096, config: MXFP8MOEFwdConfig = None, gate_up_weight_scales: nl.NkiTensor = None, down_weight_scales: nl.NkiTensor = None, num_valid_tokens: nl.NkiTensor = None) -> tuple

   MXFP8 forward pass for blockwise (dropless) Mixture of Experts.

   :param hidden_states: [T, H], input hidden states (BF16) on HBM.
   :type hidden_states: ``nl.NkiTensor``
   :param expert_affinities_masked: [T*E, 1], expert affinities (fp32) on HBM.
   :type expert_affinities_masked: ``nl.NkiTensor``
   :param gate_up_proj_weight: Gate/up weights on HBM. BF16 defaults to forward-natural [E, I_TP, 2, H], reinterpreted [E, I_TP, 2*H] (F-by-K, F=I_TP, K=2*H): per intermediate channel the 2*H columns are [gate(H), up(H)] contiguous (the transpose of the backward's [E, H, 2, I_TP]). The forward GEMMs contract over H, so gate/up is a K-slice of the 2*H weight (gate K in [0,H), up K in [H,2H)) via rhs_k_offset while the hidden LHS stays H-long. Setting config.gate_up_weight_td.orientation=K_BY_F accepts the backward-natural BF16 layout [E, H, 2, I_TP] instead; gate/up are then selected as free-axis slices. When gate_up_weight_scales is provided the weight is pre-quantized MXFP8 x4 in the K-by-F layout [E, 2*H/4, I_TP] (K=2*H packed by 4 on dim-1).
   :type gate_up_proj_weight: ``nl.NkiTensor``
   :param down_proj_weight: Down weights on HBM. BF16 defaults to forward-natural [E, H, I_TP] (F-by-K, transpose of the backward's [E, I_TP, H]); setting config.down_weight_td.orientation=K_BY_F accepts backward-natural [E, I_TP, H]. The two weight orientations are independent. The down GEMM contracts over I_TP. When down_weight_scales is provided the weight is pre-quantized MXFP8 x4 in the K-by-F layout [E, I_TP/4, H] (K=I_TP packed by 4 on dim-1).
   :type down_proj_weight: ``nl.NkiTensor``
   :param token_position_to_id: [N*B] int32, token -> block-position map (pad id = -1 under skip_dma). Use a dummy [1] tensor when single_expert_dense is True.
   :type token_position_to_id: ``nl.NkiTensor``
   :param block_to_expert: [N, 1] int32, expert index per block. Use a dummy [1, 1] tensor when single_expert_dense is True.
   :type block_to_expert: ``nl.NkiTensor``
   :param block_size: tokens per block (128/256/512/1024/2048/4096).
   :type block_size: ``int``
   :param config: fused forward configuration carrying every non-tensor knob — compute/quant dtypes, activation, sharding + affinity placement, the two per-phase matmul configs (gate_up/down), the checkpoint emission config, clamp limits, skip-DMA mode, and the load-mode / spill / scale-packing / LNC2 flags. When None a default ``MXFP8MOEFwdConfig()`` is used, so the framework can call the kernel with only the tensors + ``block_size``. See ``MXFP8MOEFwdConfig`` for per-field docs and defaults.
   :type config: ``MXFP8MOEFwdConfig``
   :param gate_up_weight_scales: MXFP8 scales for pre-quantized gate/up weights. Passing scales auto-selects the pre-quantized path: gate_up_proj_weight is consumed as MXFP8 x4 in the K-by-F layout [E, 2*H/4, I_TP] and the on-chip quantize is skipped; when None, gate_up_proj_weight is unswizzled BF16 in the orientation selected by config.gate_up_weight_td. The x4 swizzle scheme is taken from config.gate_up_weight_td.quant_scheme (the source of truth); left unset it defaults to 1x32 PE-swizzle on the single_expert_dense (E=1) path and WRAPX/DGT on the routed path, with scale packing on. Set config.gate_up_weight_td=TensorDescriptor(quant_scheme=QuantScheme.WRAPX) to force WRAPX. The offline weight swizzle must match this scheme.
   :type gate_up_weight_scales: ``nl.NkiTensor``
   :param down_weight_scales: MXFP8 scales for pre-quantized down weights. When provided, down_proj_weight is consumed as MXFP8 x4 in the K-by-F layout [E, I_TP/4, H] (same scheme resolution as gate/up); when None, it is unswizzled BF16 in the orientation selected by config.down_weight_td.
   :type down_weight_scales: ``nl.NkiTensor``
   :param num_valid_tokens: [1] int32, how many leading rows of hidden_states hold real tokens. Enables the hardware dynamic block loop. Real tokens MUST be the leading rows. Values above T are clamped to T. Requires single_expert_dense. Output rows and checkpoint slots past the effective count are UNDEFINED; consumers must ignore that region, and the backward must be given the same count.
   :type num_valid_tokens: ``nl.NkiTensor``
   :return: - output_hidden_states (nl.NkiTensor): [T, H] MoE FFN output. followed by the gate/up activation checkpoint, present only when config.checkpoint_config.save_gate_up_proj_act is set: - gate_up_proj_act_checkpoint_T (nl.NkiTensor): clamped gate pre-activation at [block, 0] and up at [block, 1]. Shape follows the store layout selected by config.gate_up_proj_act_td.orientation: K_BY_F -> TRANSPOSED [N, 2, I_TP, B] (B contiguous, the backward's layout); F_BY_K -> DIRECT [N, 2, B, I_TP] (I_TP contiguous).
   :rtype: ``tuple``

   **Dimensions**:

   * T: total tokens (linearized across batch)
   * H: hidden dimension
   * I_TP: intermediate size / tensor-parallel degree
   * E: number of experts
   * B: tokens per block (block_size)

