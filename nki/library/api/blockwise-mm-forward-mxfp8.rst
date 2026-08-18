.. meta::
    :description: MXFP8 forward pass for blockwise (dropless) Mixture of Experts.
    :date-modified: 08/17/2026

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

**Source code for this kernel API can be found at**: `blockwise_mm_forward_mxfp8.py <https://github.com/aws-neuron/nki-library/blob/2.32/src/nkilib_src/nkilib/experimental/moe_mxfp8/fwd/blockwise_mm_forward_mxfp8.py>`_

blockwise_mm_fwd_mxfp8
^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: blockwise_mm_fwd_mxfp8(hidden_states: nl.ndarray, expert_affinities_masked: nl.ndarray, gate_up_proj_weight: nl.ndarray, down_proj_weight: nl.ndarray, token_position_to_id: nl.ndarray, block_to_expert: nl.ndarray, block_size: int, gate_up_weight_scales: nl.ndarray = None, gate_up_weight_is_swizzled: bool = False, down_weight_scales: nl.ndarray = None, down_weight_is_swizzled: bool = False, gate_up_config: Optional[MatmulMxfp8KernelConfig] = None, down_config: Optional[MatmulMxfp8KernelConfig] = None, fp8_x4_dtype: type = float8_e4m3fn_x4, spill_reload: bool = False, use_scale_packing: bool = True, run_with_lnc2: bool = True, shard_option: ShardOption = ShardOption.SHARD_ON_BLOCK, affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_I, compute_dtype: nki.dtype = nl.bfloat16, skip_dma: SkipMode = None, is_tensor_update_accumulating: bool = True, no_indirect_load: bool = False, clamp_limits: ClampLimits = None, activation_type: ActFnType = ActFnType.SiLU, bias: bool = False, checkpoint_config: Optional[MXFP8MOECheckpointConfig] = None) -> tuple

   MXFP8 forward pass for blockwise (dropless) Mixture of Experts.

   :param hidden_states: [T, H], input hidden states (BF16) on HBM.
   :type hidden_states: ``nl.ndarray``
   :param expert_affinities_masked: [T*E, 1], expert affinities (fp32) on HBM.
   :type expert_affinities_masked: ``nl.ndarray``
   :param gate_up_proj_weight: [E, 2, I_TP, H], gate/up weights on HBM in forward-natural [out, in] orientation (the transpose of the backward's [E, H, 2, I_TP]). The forward GEMMs contract over the input dim H, so the per-expert DGT load needs H as the contraction axis.
   :type gate_up_proj_weight: ``nl.ndarray``
   :param down_proj_weight: [E, H, I_TP], down weights on HBM in forward-natural [out, in] orientation (transpose of the backward's [E, I_TP, H]); the down GEMM contracts over I_TP.
   :type down_proj_weight: ``nl.ndarray``
   :param token_position_to_id: [N*B] int32, token -> block-position map (pad id = -1 under skip_dma). Use a dummy [1] tensor when no_indirect_load is True.
   :type token_position_to_id: ``nl.ndarray``
   :param block_to_expert: [N, 1] int32, expert index per block. Use a dummy [1, 1] tensor when no_indirect_load is True.
   :type block_to_expert: ``nl.ndarray``
   :param block_size: tokens per block (128/256/512/1024/2048/4096).
   :type block_size: ``int``
   :param gate_up_weight_scales: MXFP8 scales for pre-quantized gate/up.
   :type gate_up_weight_scales: ``nl.ndarray``
   :param gate_up_weight_is_swizzled: whether gate/up weights are pre-swizzled.
   :type gate_up_weight_is_swizzled: ``bool``
   :param down_weight_scales: MXFP8 scales for pre-quantized down.
   :type down_weight_scales: ``nl.ndarray``
   :param down_weight_is_swizzled: whether down weights are pre-swizzled. gate_up_config / down_config (MatmulMxfp8KernelConfig, optional): per-phase matmul blocking. When None, defaults are used.
   :type down_weight_is_swizzled: ``bool``
   :param fp8_x4_dtype: MXFP8 packed weight dtype (default float8_e4m3fn_x4).
   :type fp8_x4_dtype: ``type``
   :param spill_reload: spill quantized tiles to HBM for K-block reuse.
   :type spill_reload: ``bool``
   :param use_scale_packing: packed MXFP8 scale layout.
   :type use_scale_packing: ``bool``
   :param run_with_lnc2: shard across 2 LNC cores.
   :type run_with_lnc2: ``bool``
   :param shard_option: sharding strategy (default SHARD_ON_BLOCK).
   :type shard_option: ``ShardOption``
   :param affinity_option: affinity placement; must match the backward (AFFINITY_ON_I — the forward folds affinity on the intermediate).
   :type affinity_option: ``AffinityOption``
   :param compute_dtype: dtype for SBUF/HBM intermediates + checkpoints (bf16).
   :type compute_dtype: ``nki.dtype``
   :param skip_dma: OOB handling for indirect-DMA token gather/scatter.
   :type skip_dma: ``SkipMode``
   :param is_tensor_update_accumulating: when True (top_k>1) the output scatter does read-modify-write so experts touching the same token accumulate. Ignored when no_indirect_load is True because direct outputs do not scatter.
   :type is_tensor_update_accumulating: ``bool``
   :param no_indirect_load: use contiguous single-expert inputs/weights and skip token-index gather, expert-indexed weight loads, affinity gather, and scatter.
   :type no_indirect_load: ``bool``
   :param clamp_limits: optional gate/up clamp, applied BEFORE the gate_up checkpoint + SiLU so the checkpoint matches the backward.
   :type clamp_limits: ``ClampLimits``
   :param activation_type: SiLU only (hardcoded in the dropless impl).
   :type activation_type: ``ActFnType``
   :param bias: whether gate/up + down biases are added (reserved surface).
   :type bias: ``bool``
   :param checkpoint_config: per-checkpoint save flags selecting which activation checkpoints the forward emits for the backward. When a checkpoint is disabled the kernel skips computing + storing it and does not allocate/return it. Defaults to saving both.
   :type checkpoint_config: ``Optional[MXFP8MOECheckpointConfig]``
   :return: - output_hidden_states (nl.ndarray): [T, H] MoE FFN output. followed by each saved checkpoint, in this fixed order (an entry is present only when its checkpoint_config flag is set): - gate_up_proj_act_checkpoint_T (nl.ndarray): [N, 2, I_TP, B], clamped gate pre-activation at [block, 0] and up at [block, 1] (B contiguous); present when checkpoint_config.save_gate_up_proj_act. - scaled_intermediate_checkpoint_T (nl.ndarray): [N, I_TP, B], SiLU(gate)*up*EA transposed; present when checkpoint_config.save_scaled_intermediate.
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * T: total tokens (linearized across batch)
   * H: hidden dimension
   * I_TP: intermediate size / tensor-parallel degree
   * E: number of experts
   * B: tokens per block (block_size)

