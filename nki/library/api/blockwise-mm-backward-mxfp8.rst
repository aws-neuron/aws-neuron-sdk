.. meta::
    :description: MXFP8 backward pass for blockwise Mixture of Experts.
    :date-modified: 06/11/2026

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

**Source code for this kernel API can be found at**: `blockwise_mm_backward_mxfp8.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe_mxfp8/bwd/blockwise_mm_backward_mxfp8.py>`_

blockwise_mm_bwd_mxfp8
^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: blockwise_mm_bwd_mxfp8(hidden_states: nl.ndarray, expert_affinities_masked: nl.ndarray, gate_up_proj_weight: nl.ndarray, down_proj_weight: nl.ndarray, token_position_to_id: nl.ndarray, block_to_expert: nl.ndarray, output_hidden_states_grad: nl.ndarray, block_size: int, gate_up_proj_act_checkpoint_T: Optional[nl.ndarray] = None, gate_act_checkpoint_T: nl.ndarray = None, intermediate_checkpoint_T: nl.ndarray = None, scaled_intermediate_checkpoint_T: nl.ndarray = None, down_proj_act_checkpoint: Optional[nl.ndarray] = None, gate_up_weight_scales: nl.ndarray = None, gate_up_weight_is_swizzled: bool = False, down_weight_scales: nl.ndarray = None, down_weight_is_swizzled: bool = False, phase1_config: Optional[MatmulMxfp8KernelConfig] = None, phase2_config: Optional[MatmulMxfp8KernelConfig] = None, phase3_config: Optional[MatmulMxfp8KernelConfig] = None, phase4_config: Optional[MatmulMxfp8KernelConfig] = None, fp8_x4_dtype: type = float8_e4m3fn_x4, spill_reload: bool = False, use_scale_packing: bool = True, run_with_lnc2: bool = True, shard_option: ShardOption = ShardOption.SHARD_ON_FREE, affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_I, compute_dtype: nki.dtype = nl.bfloat16, skip_dma: SkipMode = None, skip_grad_initialization: bool = False, is_tensor_update_accumulating: bool = True, clamp_limits: ClampLimits = None, activation_type: ActFnType = ActFnType.SiLU, bias: bool = False) -> tuple

   MXFP8 backward pass for blockwise Mixture of Experts.

   :param hidden_states: [T, H], Input hidden states (BF16) on HBM.
   :type hidden_states: ``nl.ndarray``
   :param expert_affinities_masked: [T * E, 1], Expert affinities on HBM.
   :type expert_affinities_masked: ``nl.ndarray``
   :param gate_up_proj_weight: [E, H, 2, I_TP], Gate/up projection weights on HBM.
   :type gate_up_proj_weight: ``nl.ndarray``
   :param down_proj_weight: [E, I_TP, H], Down projection weights on HBM.
   :type down_proj_weight: ``nl.ndarray``
   :param token_position_to_id: [N * B], Token position to block mapping.
   :type token_position_to_id: ``nl.ndarray``
   :param block_to_expert: [N, 1], Expert index per block.
   :type block_to_expert: ``nl.ndarray``
   :param output_hidden_states_grad: [T, H], Upstream gradient (BF16) from output.
   :type output_hidden_states_grad: ``nl.ndarray``
   :param block_size: Number of tokens per block (128, 256, 512, or 1024).
   :type block_size: ``int``
   :param gate_up_proj_act_checkpoint_T: [N, 2, I_TP, B], Checkpointed gate/up activations (gate_pre = checkpoint[block, 0], up = checkpoint[block, 1]). **Currently required** — it must be provided (recompute of gate/up activations from a different checkpoint is not yet supported), and ``I_TP`` is derived from its shape. Passing ``None`` will raise.
   :type gate_up_proj_act_checkpoint_T: ``nl.ndarray``
   :param gate_act_checkpoint_T: Reserved; **not currently supported** and must be ``None`` (passing a tensor will raise).
   :type gate_act_checkpoint_T: ``nl.ndarray``
   :param intermediate_checkpoint_T: Reserved; **not currently supported** and must be ``None`` (passing a tensor will raise).
   :type intermediate_checkpoint_T: ``nl.ndarray``
   :param scaled_intermediate_checkpoint_T: Reserved; **not currently supported** and must be ``None`` (passing a tensor will raise). Phase 4 reuses Phase 1's scaled_intermediate (EA-scaled under AFFINITY_ON_I) and transposes it inline.
   :type scaled_intermediate_checkpoint_T: ``nl.ndarray``
   :param down_proj_act_checkpoint: [N, B, H], Pre-computed output_grad * expert_affinity (used only with the unsupported AFFINITY_ON_H mode — see ``affinity_option``). Leave as ``None`` for the supported AFFINITY_ON_I mode.
   :type down_proj_act_checkpoint: ``Optional[nl.ndarray]``
   :param gate_up_weight_scales: MXFP8 scales for pre-quantized gate/up weights.
   :type gate_up_weight_scales: ``nl.ndarray``
   :param gate_up_weight_is_swizzled: Reserved; **not currently supported** and must be ``False`` (passing ``True`` will raise).
   :type gate_up_weight_is_swizzled: ``bool``
   :param down_weight_scales: MXFP8 scales for pre-quantized down weights.
   :type down_weight_scales: ``nl.ndarray``
   :param down_weight_is_swizzled: Reserved; **not currently supported** and must be ``False`` (passing ``True`` will raise).
   :type down_weight_is_swizzled: ``bool``
   :param phase1_config: Per-phase matmul tiling configuration for Phase 1 (dW down-proj gradient). If ``None``, defaults are used. Tune to maximize kernel performance.
   :type phase1_config: ``Optional[MatmulMxfp8KernelConfig]``
   :param phase2_config: Per-phase matmul tiling configuration for Phase 2 (hidden-states gradient). If ``None``, defaults are used.
   :type phase2_config: ``Optional[MatmulMxfp8KernelConfig]``
   :param phase3_config: Per-phase matmul tiling configuration for Phase 3 (gate/up weight gradient). If ``None``, defaults are used.
   :type phase3_config: ``Optional[MatmulMxfp8KernelConfig]``
   :param phase4_config: Per-phase matmul tiling configuration for Phase 4 (down weight gradient). If ``None``, defaults are used.
   :type phase4_config: ``Optional[MatmulMxfp8KernelConfig]``
   :param fp8_x4_dtype: MXFP8 packed data type (default: float8_e4m3fn_x4).
   :type fp8_x4_dtype: ``type``
   :param spill_reload: Whether to spill quantized tiles to HBM for K-block reuse.
   :type spill_reload: ``bool``
   :param use_scale_packing: Whether to use packed scale layout for MXFP8 quantization.
   :type use_scale_packing: ``bool``
   :param run_with_lnc2: Whether to shard across 2 LNC cores.
   :type run_with_lnc2: ``bool``
   :param shard_option: LNC2 sharding strategy. **Currently only ``SHARD_ON_FREE`` is supported** (the default); other values will raise.
   :type shard_option: ``ShardOption``
   :param affinity_option: Where the expert affinity scalar is folded into the FFN chain. **Currently only ``AFFINITY_ON_I`` is supported**; passing ``AFFINITY_ON_H`` will raise.
   :type affinity_option: ``AffinityOption``
   :param compute_dtype: Dtype for SBUF/HBM intermediates (default: bf16).
   :type compute_dtype: ``nki.dtype``
   :param skip_dma: OOB handling mode for indirect DMA token gathers.
   :type skip_dma: ``SkipMode``
   :param skip_grad_initialization: If True, skip the zero-init of grad outputs.
   :type skip_grad_initialization: ``bool``
   :param is_tensor_update_accumulating: If True (default), the Phase 2 hidden_states_grad scatter does a read-modify-write so multiple experts contributing to the same token (top-K > 1 routing) accumulate correctly. If False, the scatter overwrites — correct only when each token is touched by exactly one block (top-K = 1).
   :type is_tensor_update_accumulating: ``bool``
   :param clamp_limits: Optional gradient clamping limits. When set, masks out gradients that exceed the specified bounds.
   :type clamp_limits: ``ClampLimits``
   :param activation_type: NOT YET IMPLEMENTED. SiLU is hardcoded in the MXFP8 dropless impl; passing a different activation will raise.
   :type activation_type: ``ActFnType``
   :param bias: Whether to compute bias gradients (default: False).
   :type bias: ``bool``
   :return: Tuple of gradient tensors: - hidden_states_grad (nl.ndarray): [T, H], Gradient for hidden states. - expert_affinities_masked_grad (nl.ndarray): [T * E, 1], Gradient for affinities. - gate_up_proj_weight_grad (nl.ndarray): [E, H, 2, I_TP], Gradient for gate/up weights. - down_proj_weight_grad (nl.ndarray): [E, I_TP, H], Gradient for down weights. - gate_and_up_proj_bias_grad (nl.ndarray, optional): [E, 2, I_TP], if bias=True. - down_proj_bias_grad (nl.ndarray, optional): [E, H], if bias=True. Returns a 4-element tuple, or a 6-element tuple when ``bias=True``.
   :rtype: ``tuple``

   **Dimensions**:

   * T: Total number of input tokens (after linearizing across batch dimension)
   * H: Hidden dimension size
   * I_TP: Intermediate size / tensor parallel degree
   * E: Number of experts
   * B: Number of tokens per block (block_size)
   * N: Total number of blocks (``(T*TopK - (E-1)) / B + E - 1``)

