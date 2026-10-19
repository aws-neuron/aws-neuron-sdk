.. meta::
    :description: MXFP8 backward pass for blockwise Mixture of Experts.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.moe_mxfp8.bwd

Blockwise MM Backward MXFP8 Kernel API Reference
================================================

MXFP8 backward pass for blockwise Mixture of Experts.

Computes gradients for all parameters in a Mixture of Experts layer using MXFP8 quantized matrix multiplication. Processes tokens in blocks assigned to specific experts. Only weights (gate_up_proj_weight, down_proj_weight) support pre-quantized MXFP8 inputs. Activations (hidden_states, output_hidden_states_grad) must be BF16 because they are gathered per-block via indirect DMA using token indices, which breaks MXFP8 32-element quantization group alignment.

Background
-----------

The ``blockwise_mm_bwd_mxfp8`` kernel computes gradients for all parameters of a blockwise Mixture of Experts layer using MXFP8 quantized matrix multiplication, processing tokens in blocks assigned to specific experts. Only the weights support pre-quantized MXFP8 inputs; activations remain BF16 because per-block indirect-DMA gathers break MXFP8 32-element group alignment.

API Reference
--------------

**Source code for this kernel API can be found at**: `blockwise_mm_backward_mxfp8.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/moe_mxfp8/bwd/blockwise_mm_backward_mxfp8.py>`_

blockwise_mm_bwd_mxfp8
^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: blockwise_mm_bwd_mxfp8(hidden_states: nl.NkiTensor, expert_affinities_masked: nl.NkiTensor, gate_up_proj_weight: nl.NkiTensor, down_proj_weight: nl.NkiTensor, token_position_to_id: nl.NkiTensor, block_to_expert: nl.NkiTensor, output_hidden_states_grad: nl.NkiTensor, block_size: int = 4096, gate_up_proj_act_checkpoint_T: Optional[nl.NkiTensor] = None, gate_act_checkpoint_T: nl.NkiTensor = None, intermediate_checkpoint_T: nl.NkiTensor = None, scaled_intermediate_checkpoint_T: nl.NkiTensor = None, down_proj_act_checkpoint: Optional[nl.NkiTensor] = None, config: MXFP8MOEBwdConfig = None, gate_up_weight_scales: nl.NkiTensor = None, down_weight_scales: nl.NkiTensor = None, num_valid_tokens: nl.NkiTensor = None) -> tuple

   MXFP8 backward pass for blockwise Mixture of Experts.

   :param hidden_states: [T, H], Input hidden states (BF16) on HBM.
   :type hidden_states: ``nl.NkiTensor``
   :param expert_affinities_masked: [T * E, 1], Expert affinities on HBM.
   :type expert_affinities_masked: ``nl.NkiTensor``
   :param gate_up_proj_weight: [E, H, 2, I_TP], Gate/up projection weights on HBM.
   :type gate_up_proj_weight: ``nl.NkiTensor``
   :param down_proj_weight: [E, I_TP, H], Down projection weights on HBM.
   :type down_proj_weight: ``nl.NkiTensor``
   :param token_position_to_id: [N * B], Token position to block mapping.
   :type token_position_to_id: ``nl.NkiTensor``
   :param block_to_expert: [N, 1], Expert index per block.
   :type block_to_expert: ``nl.NkiTensor``
   :param output_hidden_states_grad: [T, H], Upstream gradient (BF16) from output.
   :type output_hidden_states_grad: ``nl.NkiTensor``
   :param block_size: Number of tokens per block (128, 256, 512, or 1024).
   :type block_size: ``int``
   :param gate_up_proj_act_checkpoint_T: [N, 2, I_TP, B], Checkpointed gate/up activations (gate_pre = checkpoint[block, 0], up = checkpoint[block, 1]). If None, gate_act_checkpoint_T and intermediate_checkpoint_T must be provided so the kernel can avoid recomputing from this checkpoint.
   :type gate_up_proj_act_checkpoint_T: ``Optional[nl.NkiTensor]``
   :param gate_act_checkpoint_T: [N, I_TP, B], Pre-computed SiLU(gate_pre). If None, recomputed per-block as SiLU(gate_up_proj_act_checkpoint_T[block, 0]).
   :type gate_act_checkpoint_T: ``nl.NkiTensor``
   :param intermediate_checkpoint_T: [N, I_TP, B], Pre-computed gate_act * up. If None, recomputed per-block as gate_act * up. Used for Phase 4 (dW_down).
   :type intermediate_checkpoint_T: ``nl.NkiTensor``
   :param scaled_intermediate_checkpoint_T: [N, I_TP, B], Pre-computed intermediate * expert_affinity (Affinity I mode), saved from the forward pass. If provided, Phase 4 reads its per-block slice directly as the dW_down RHS. If None, Phase 4 reuses Phase 1's scaled_intermediate (already EA-scaled under AFFINITY_ON_I) and transposes it inline — no separate recompute.
   :type scaled_intermediate_checkpoint_T: ``nl.NkiTensor``
   :param down_proj_act_checkpoint: [N, B, H], Pre-computed output_grad * expert_affinity (Affinity H mode). If None, recomputed per-block as output_grad[block] * ea_scale. Used for Phase 1 when affinity_option=AFFINITY_ON_H.
   :type down_proj_act_checkpoint: ``Optional[nl.NkiTensor]``
   :param config: fused backward configuration carrying every non-tensor knob — compute/quant dtypes, activation, sharding + affinity placement, the four per-phase matmul configs (which own their own blocking, spill/reload and scale packing), the P3/P4 transpose modes, the per-operand swizzle modes, clamp limits, skip-DMA mode, and the accumulation / grad-init / single-expert-dense / fast-DMA flags. When None a default ``MXFP8MOEBwdConfig()`` is used, so the framework can call the kernel with only the tensors + ``block_size``. See ``MXFP8MOEBwdConfig`` for per-field docs and defaults.
   :type config: ``MXFP8MOEBwdConfig``
   :param gate_up_weight_scales: MXFP8 scales for pre-quantized gate/up weights.
   :type gate_up_weight_scales: ``nl.NkiTensor``
   :param down_weight_scales: MXFP8 scales for pre-quantized down weights.
   :type down_weight_scales: ``nl.NkiTensor``
   :param num_valid_tokens: [1] int32, how many leading rows of hidden_states hold real tokens. Enables the hardware dynamic block loop. Real tokens MUST be the leading rows. Values above T are clamped to T. Requires single_expert_dense, and is unsupported alongside a scaled-intermediate checkpoint or bias gradients. hidden_states_grad rows and expert_affinities_masked_grad entries past the effective count are UNDEFINED; give the forward the same count.
   :type num_valid_tokens: ``nl.NkiTensor``
   :return: Gradient tensors: - hidden_states_grad (nl.NkiTensor): [T, H], Gradient for hidden states. - expert_affinities_masked_grad (nl.NkiTensor): [T * E, 1], Gradient for affinities. - gate_up_proj_weight_grad (nl.NkiTensor): [E, H, 2, I_TP], Gradient for gate/up weights. - down_proj_weight_grad (nl.NkiTensor): [E, I_TP, H], Gradient for down weights. - gate_and_up_proj_bias_grad (nl.NkiTensor, optional): [E, 2, I_TP], if bias=True. - down_proj_bias_grad (nl.NkiTensor, optional): [E, H], if bias=True.
   :rtype: ``tuple``

   **Dimensions**:

   * T: Total number of input tokens (after linearizing across batch dimension)
   * H: Hidden dimension size
   * I_TP: Intermediate size / tensor parallel degree
   * E: Number of experts
   * B: Number of tokens per block (block_size)
   * N: Total number of blocks (``(T*TopK - (E-1)) / B + E - 1``)

