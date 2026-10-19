.. meta::
    :description: Blockwise MXFP MoE kernel, SINGLE LOGICAL CORE (LNC1) only. Use as standalone kernel.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.deepseek_v32.moe_cte

Blockwise MM Block MX Kernel API Reference
==========================================

Blockwise MXFP MoE kernel, SINGLE LOGICAL CORE (LNC1) only. Use as standalone kernel.

The blockwise matrix multiplication (matmul) kernel implements a Mixture of Experts (MoE) layer at a block granularity, offering an alternative to token dropping approaches. This method assumes that tokens have already been assigned to blocks, as specified by the user through the token_position_to_id parameter. This kernel loops over all blocks, without considering they are padded or non-padded blocks. Supports both MXFP4 and MXFP8 weight quantization. LNC1 ONLY: the kernel asserts a single logical Neuron core at entry. There is no block distribution across shards, no shard offset on the output (which is a plain [T, H] buffer), and no closing core_barrier / cross-shard reduce -- the one core's accumulation is the final result. Launching this at LNC2 fails the entry assert; use core/moe/moe_cte/bwmm_shard_on_block_mx.py for two cores. Intended Usage: - Block size B: 128-1024 tokens - Total tokens T: 32k - Hidden dimension H: 512-8192 - Intermediate dimension I_TP: 384-3072 - Number of experts E: 8-128

Background
-----------

The ``bwmm_block_mx`` kernel implements a blockwise MXFP4/MXFP8 Mixture of Experts layer for DeepSeek-V3.2 prefill on a single logical NeuronCore (LNC1). Tokens are pre-assigned to blocks by the caller through ``token_position_to_id``.

API Reference
--------------

**Source code for this kernel API can be found at**: `bwmm_block_mx.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/deepseek_v32/moe_cte/bwmm_block_mx.py>`_

bwmm_block_mx
^^^^^^^^^^^^^

.. py:function:: bwmm_block_mx(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight, token_position_to_id, block_to_expert, conditions: nl.NkiTensor | None = None, gate_and_up_proj_bias: nl.NkiTensor | None = None, down_proj_bias: nl.NkiTensor | None = None, gate_up_proj_scale: nl.NkiTensor | None = None, down_proj_scale: nl.NkiTensor | None = None, block_size = None, n_static_blocks: int = -1, n_dynamic_blocks: int = -1, top_k: int = 1, ep_degree: int = 1, gate_up_activations_T = None, down_activations = None, activation_function: ActFnType = ActFnType.SiLU, skip_dma: SkipMode = SkipMode(False, False), compute_dtype = nl.bfloat16, weight_dtype: Any = None, is_tensor_update_accumulating = True, expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE, gate_clamp_upper_limit: float | None = None, gate_clamp_lower_limit: float | None = None, up_clamp_lower_limit: float | None = None, up_clamp_upper_limit: float | None = None, use_packed_scales: bool = False, quantization_type: QuantizationType = QuantizationType.NONE, gate_up_in_scale: nl.NkiTensor | None = None, down_in_scale: nl.NkiTensor | None = None)

   Blockwise MXFP MoE kernel, SINGLE LOGICAL CORE (LNC1) only. Use as standalone kernel.

   :param hidden_states: Tensor of input hidden states on HBM of size (T+1, H). The reason it is T+1 is because padding token position is set to T. with skip_dma, id will be set to -1, so this shape can be (T, H). Similarly for expert_affinities_masked, output
   :param expert_affinities_masked: Tensor of expert affinities corresponding to each token of size ((T+1) * E, 1).
   :param gate_up_proj_weight: Tensor of concatenated gate and up projection weights on HBM (E, H, 2, I). Supports MXFP4 (nl.float4_e2m1fn_x4) and MXFP8 (nl.float8_e4m3fn_x4, nl.float8_e5m2_x4).
   :param down_proj_weight: Tensor of down projection weights on HBM (E, I, H). Supports MXFP4 (nl.float4_e2m1fn_x4) and MXFP8 (nl.float8_e4m3fn_x4, nl.float8_e5m2_x4).
   :param block_size: Number of tokens per block
   :param token_position_to_id: Tensor of block index of the corresponding tokens on HBM (N * B,) Note that we include tokens included for padding purposes and N * B >= T. For padding token, id is set to T. with skip_dma, id will be set to -1.
   :param block_to_expert: Tensor of expert indices of corresponding blocks on HBM (N, 1)
   :param num_static_block: Optional. Number of non-padded blocks if known (default: -1).
   :param n_dynamic_blocks: Number of blocks to process with dynamic loop when n_static_blocks is not specified (default: 55, empirically tuned for GPT-OSS).
   :type n_dynamic_blocks: ``int``
   :param gate_and_up_proj_bias: nl.NkiTensor | None = None, Optional. A tensor of shape [E, 2, I]. Note that if activation function is Swiglu, we expect up_bias = up_bias + 1
   :type gate_and_up_proj_bias: ``nl.NkiTensor | None``
   :param down_proj_bias: nl.NkiTensor | None = None. Optional argument. A tensor of shape [E, H] # Arguments for quantization scales
   :type down_proj_bias: ``nl.NkiTensor | None``
   :param gate_up_proj_scale: nl.NkiTensor | None = None. uint8 MX scales of shape [E, _pmax // _q_height, 2, n_H512_tile, I] (standard layout) or [E, _pmax, n_packed_gup, 2, I] (use_packed_scales=True). Under STATIC_MX it instead carries the per-expert gate/up weight scales as fp32 [E, 2, 1] (idx 0 = gate, idx 1 = up).
   :type gate_up_proj_scale: ``nl.NkiTensor | None``
   :param down_proj_scale: nl.NkiTensor | None = None. uint8 MX scales of shape [E, p_I // _q_height, n_total_I512_tile, H] (standard layout) or [E, _pmax, n_packed_down, H] (use_packed_scales=True). Under STATIC_MX it instead carries the per-expert down weight scale as fp32 [E, 1]. # Unsupported output tensors. Please set to None.
   :type down_proj_scale: ``nl.NkiTensor | None``
   :param gate_up_activations_T: nl.NkiTensor | None = None. Currently not supported.
   :param down_activations: nl.NkiTensor | None = None. Currently not supported # meta parameters
   :param activation_function: one of the Enum in nkilib.core.utils.common_types.ActFnType. Indicate what activation function to use in the MLP block
   :type activation_function: ``ActFnType``
   :param skip_dma: SkipMode = SkipMode(False, False), compute_dtype=nl.bfloat16,
   :type skip_dma: ``SkipMode``
   :param weight_dtype: Target dtype for weight conversion when weights are passed as uint/int/float types. For MXFP4: nl.float4_e2m1fn_x4 For MXFP8: nl.float8_e4m3fn_x4 or nl.float8_e5m2_x4 If None, auto-detects (defaults to e4m3fn for MXFP8)
   :type weight_dtype: ``Any``
   :param is_tensor_update_accumulating: bool. Indicate whether we need to accumulate the results over multiple blocks
   :param expert_affinities_scaling_mode: one of the Enum in nkilib.core.utils.common_types.ExpertAffinityScaleMode. Indicate if the kernel is doing post or pre scaling.
   :param n_block_per_iter: int. Currently unsupported #parameters for clipping the MLP projections
   :param gate_clamp_upper_limit: float | None = None,
   :type gate_clamp_upper_limit: ``float | None``
   :param gate_clamp_lower_limit: float | None = None,
   :type gate_clamp_lower_limit: ``float | None``
   :param up_clamp_lower_limit: float | None = None,
   :type up_clamp_lower_limit: ``float | None``
   :param up_clamp_upper_limit: float | None = None
   :type up_clamp_upper_limit: ``float | None``
   :param skip_dma: Whether to skip DMA operations (default: False)
   :type skip_dma: ``SkipMode``
   :return: Tensor of output hidden states on HBM of size (T+1, H).
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * All input/output tensors must have the same floating point dtype
   * token_position_to_id and block_to_expert must be np.int32 tensors

   **Dimensions**:

   * H: Hidden dimension size
   * T: Total number of input tokens (after linearizing across the batch dimension)
   * B: Number of tokens per block
   * N: Total number of blocks
   * E: Number of experts

