.. meta::
    :description: Return (num_cores, shard_id) for LNC2 sharding.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.mlp_mxfp8.mlp_bwd_mxfp8

MLP Backward MXFP8 Kernel API Reference
========================================

Return (num_cores, shard_id) for LNC2 sharding.

Background
-----------

The ``get_program_sharding_info`` kernel returns the LNC2 sharding configuration (num_cores, shard_id), used by the MXFP8 MLP backward pass to distribute computation across logical cores.

API Reference
--------------

**Source code for this kernel API can be found at**: `mlp_bwd_mxfp8_kernel.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/mlp_mxfp8/mlp_bwd_mxfp8/mlp_bwd_mxfp8_kernel.py>`_

get_program_sharding_info
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: get_program_sharding_info(run_with_lnc2: bool) -> tuple

   Return (num_cores, shard_id) for LNC2 sharding.


compute_phase1_down_proj_mm_grad_mxfp8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: compute_phase1_down_proj_mm_grad_mxfp8(output_grad_td: TensorDescriptor, gate_pre_td: TensorDescriptor, gate_act_td: TensorDescriptor, up_td: TensorDescriptor, d_gate_up_td: TensorDescriptor, scratch_td: TensorDescriptor, down_weight_td: TensorDescriptor, s_base: int, dtype: type, fp8_x4_dtype: type, config: MatmulMxfp8KernelConfig = None, spill_reload: bool = True, use_scale_packing: bool = True, run_with_lnc2: bool = True, clamp_limits: ClampLimits = None) -> None

   Phase 1: Compute gradient through the down projection and SwiGLU gate.

   :param output_grad_td: [S, H], incoming gradient (is_f_by_k=True).
   :type output_grad_td: ``TensorDescriptor``
   :param gate_pre_td: [S, I], checkpointed gate pre-activation.
   :type gate_pre_td: ``TensorDescriptor``
   :param gate_act_td: [S, I], checkpointed gate post-activation.
   :type gate_act_td: ``TensorDescriptor``
   :param up_td: [S, I], checkpointed up projection.
   :type up_td: ``TensorDescriptor``
   :param d_gate_up_td: [S, 2I], output: fused gate || up gradient.
   :type d_gate_up_td: ``TensorDescriptor``
   :param scratch_td: [2I, S], output: transposed d_gate || d_up.
   :type scratch_td: ``TensorDescriptor``
   :param down_weight_td: [I, H], transposed down projection weights (is_f_by_k=True).
   :type down_weight_td: ``TensorDescriptor``
   :param s_base: Row offset into the full [S, ...] tensors for this LNC core.
   :type s_base: ``int``
   :param dtype: Data type for computation (nl.bfloat16).
   :type dtype: ``type``
   :param fp8_x4_dtype: MXFP8 quantized data type (e.g. float8_e4m3fn_x4).
   :type fp8_x4_dtype: ``type``
   :param config: Per-phase matmul tiling configuration. Replaces the previous ``TILES_IN_BLOCK_M/N/K`` arguments.
   :type config: ``MatmulMxfp8KernelConfig``
   :param clamp_limits: Optional activation clamp limits applied during the gradient computation.
   :type clamp_limits: ``ClampLimits``

compute_phase2_hidden_states_grad_mxfp8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: compute_phase2_hidden_states_grad_mxfp8(hidden_states_grad_td: TensorDescriptor, gate_up_weight_td: TensorDescriptor, d_gate_up_td: TensorDescriptor, s_base: int, dtype: type, fp8_x4_dtype: type, config: MatmulMxfp8KernelConfig = None, spill_reload: bool = True, use_scale_packing: bool = True, run_with_lnc2: bool = True) -> None

   Phase 2: Compute gradient w.r.t. input hidden states.

   :param hidden_states_grad_td: [S, H], output: dL/d_hidden.
   :type hidden_states_grad_td: ``TensorDescriptor``
   :param gate_up_weight_td: [H, 2I], transposed fused gate+up projection weights (is_f_by_k=True).
   :type gate_up_weight_td: ``TensorDescriptor``
   :param d_gate_up_td: [S, 2I], fused gate || up gradient (is_f_by_k=True).
   :type d_gate_up_td: ``TensorDescriptor``
   :param s_base: Row offset for this LNC core's shard.
   :type s_base: ``int``
   :param dtype: Data type for computation (nl.bfloat16).
   :type dtype: ``type``
   :param fp8_x4_dtype: MXFP8 quantized data type (e.g. float8_e4m3fn_x4).
   :type fp8_x4_dtype: ``type``
   :param config: Per-phase matmul tiling configuration. Replaces the previous ``TILES_IN_BLOCK_M/N/K`` arguments.
   :type config: ``MatmulMxfp8KernelConfig``

compute_phase3_gate_up_weight_grad_mxfp8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: compute_phase3_gate_up_weight_grad_mxfp8(weight_grad_td: TensorDescriptor, hidden_states_T_td: TensorDescriptor, grad_T_td: TensorDescriptor, dtype: type, fp8_x4_dtype: type, config: MatmulMxfp8KernelConfig = None, spill_reload: bool = True, use_scale_packing: bool = True, run_with_lnc2: bool = True) -> None

   Phase 3: Compute gradient w.r.t. gate and up weight matrices as a single matmul.

   :param weight_grad_td: [2I, H], output: [dW_gate; dW_up].
   :type weight_grad_td: ``TensorDescriptor``
   :param hidden_states_T_td: [H, S], transposed input hidden states (is_f_by_k=True).
   :type hidden_states_T_td: ``TensorDescriptor``
   :param grad_T_td: [2I, S], transposed gate+up gradients (is_f_by_k=True, is_col_parallel_sharded=True for LNC2).
   :type grad_T_td: ``TensorDescriptor``
   :param dtype: Data type for computation (nl.bfloat16).
   :type dtype: ``type``
   :param fp8_x4_dtype: MXFP8 quantized data type.
   :type fp8_x4_dtype: ``type``
   :param config: Per-phase matmul tiling configuration. Replaces the previous ``TILES_IN_BLOCK_M/N/K`` arguments.
   :type config: ``MatmulMxfp8KernelConfig``

   **Dimensions**:

   * S: Sequence length.
   * H: Hidden dimension size.

compute_phase4_down_weight_grad_mxfp8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: compute_phase4_down_weight_grad_mxfp8(down_weight_grad_td: TensorDescriptor, output_grad_T_td: TensorDescriptor, intermediate_T_td: TensorDescriptor, h_base: int, dtype: type, fp8_x4_dtype: type, config: MatmulMxfp8KernelConfig = None, spill_reload: bool = True, use_scale_packing: bool = True, run_with_lnc2: bool = True) -> None

   Phase 4: Compute gradient w.r.t. down projection weight matrix.

   :param down_weight_grad_td: [H, I], output: dW_down.
   :type down_weight_grad_td: ``TensorDescriptor``
   :param output_grad_T_td: [H, S], transposed output gradient (is_f_by_k=True).
   :type output_grad_T_td: ``TensorDescriptor``
   :param intermediate_T_td: [I, S], transposed intermediate activations (is_f_by_k=True).
   :type intermediate_T_td: ``TensorDescriptor``
   :param h_base: Row offset into the H dimension for this LNC core.
   :type h_base: ``int``
   :param dtype: Data type for computation (nl.bfloat16).
   :type dtype: ``type``
   :param fp8_x4_dtype: MXFP8 quantized data type.
   :type fp8_x4_dtype: ``type``
   :param config: Per-phase matmul tiling configuration. Replaces the previous ``TILES_IN_BLOCK_M/N/K`` arguments.
   :type config: ``MatmulMxfp8KernelConfig``

mlp_backward_mxfp8_base_nki
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: mlp_backward_mxfp8_base_nki(output_grad_td: TensorDescriptor, gate_pre_td: TensorDescriptor, gate_act_td: TensorDescriptor, up_td: TensorDescriptor, gate_up_weight_T_td: TensorDescriptor, down_weight_T_td: TensorDescriptor, d_gate_up_td: TensorDescriptor, hidden_states_T_td: TensorDescriptor, output_grad_T_td: TensorDescriptor, intermediate_T_td: TensorDescriptor, scratch_td: TensorDescriptor, hidden_states_grad_td: TensorDescriptor, weight_grad_td: TensorDescriptor, down_weight_grad_td: TensorDescriptor, run_with_lnc2: bool = True, matmul_config: MlpBwdMatmulConfig = None, fp8_x4_dtype: type = float8_e4m3fn_x4, spill_reload: bool = True, use_scale_packing: bool = True, clamp_limits: ClampLimits = None) -> tuple

   MXFP8 SwiGLU MLP backward pass (base kernel).

   :param output_grad_td: [S, H], incoming gradient dL/d_output (is_f_by_k=True).
   :type output_grad_td: ``TensorDescriptor``
   :param gate_pre_td: [S, I], gate pre-activation (before SiLU).
   :type gate_pre_td: ``TensorDescriptor``
   :param gate_act_td: [S, I], gate post-activation (SiLU(gate_pre)).
   :type gate_act_td: ``TensorDescriptor``
   :param up_td: [S, I], up projection (hidden @ W_up.T).
   :type up_td: ``TensorDescriptor``
   :param gate_up_weight_T_td: [H, 2I], transposed fused gate+up projection weights.
   :type gate_up_weight_T_td: ``TensorDescriptor``
   :param down_weight_T_td: [I, H], transposed down projection weights.
   :type down_weight_T_td: ``TensorDescriptor``
   :param d_gate_up_td: [S, 2I], scratch: fused gate || up gradient.
   :type d_gate_up_td: ``TensorDescriptor``
   :param hidden_states_T_td: [H, S], pre-transposed input hidden states.
   :type hidden_states_T_td: ``TensorDescriptor``
   :param output_grad_T_td: [H, S], pre-transposed output gradient.
   :type output_grad_T_td: ``TensorDescriptor``
   :param intermediate_T_td: [I, S], pre-transposed intermediate activations.
   :type intermediate_T_td: ``TensorDescriptor``
   :param scratch_td: [2I, S], scratch: transposed d_gate || d_up.
   :type scratch_td: ``TensorDescriptor``
   :param hidden_states_grad_td: [S, H], output: dL/d_hidden.
   :type hidden_states_grad_td: ``TensorDescriptor``
   :param weight_grad_td: [2I, H], output: fused [dW_gate; dW_up].
   :type weight_grad_td: ``TensorDescriptor``
   :param down_weight_grad_td: [H, I], output: dL/dW_down.
   :type down_weight_grad_td: ``TensorDescriptor``
   :param run_with_lnc2: Whether to shard across 2 LNC cores.
   :type run_with_lnc2: ``bool``
   :param matmul_config: Per-phase matmul tiling configuration. Replaces the previous per-phase ``phase*_tiles_*`` arguments.
   :type matmul_config: ``MlpBwdMatmulConfig``
   :param fp8_x4_dtype: MXFP8 quantized data type.
   :type fp8_x4_dtype: ``type``
   :param clamp_limits: Optional activation clamp limits.
   :type clamp_limits: ``ClampLimits``
   :return: (hidden_states_grad [S, H], gate_up_weight_grad [2I, H], down_weight_grad [H, I]).
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * S: Sequence length.
   * H: Hidden dimension size.

mlp_backward_mxfp8_nki
^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: mlp_backward_mxfp8_nki(output_grad: nl.ndarray, hidden_states: nl.ndarray, down_proj_weight: nl.ndarray = None, gate_up_weights: nl.ndarray = None, gate_up_weight_T: nl.ndarray = None, gate_up_weight_T_scales: nl.ndarray = None, gate_up_weights_scales: nl.ndarray = None, down_weight_T: nl.ndarray = None, down_weight_T_scales: nl.ndarray = None, output_grad_T: nl.ndarray = None, output_grad_T_scales: nl.ndarray = None, hidden_states_T: nl.ndarray = None, hidden_states_T_scales: nl.ndarray = None, gate_pre: nl.ndarray = None, gate_act: nl.ndarray = None, up: nl.ndarray = None, intermediate: nl.ndarray = None, run_with_lnc2: bool = True, matmul_config: MlpBwdMatmulConfig = None, fp8_x4_dtype: type = float8_e4m3fn_x4, spill_reload: bool = True, use_scale_packing: bool = True, clamp_limits: ClampLimits = None) -> tuple

   MXFP8 SwiGLU MLP backward pass with activation checkpointing support.

   :param output_grad: [S, H], incoming gradient dL/d_output.
   :type output_grad: ``nl.ndarray``
   :param hidden_states: [S, H], original input (for recompute + weight grad).
   :type hidden_states: ``nl.ndarray``
   :param down_proj_weight: [I, H], down projection weights (phase 1).
   :type down_proj_weight: ``nl.ndarray``
   :param gate_up_weights: [2I, H], fused gate+up weights (for recompute).
   :type gate_up_weights: ``nl.ndarray``
   :param gate_up_weight_T: [H, 2I], transposed fused gate+up projection weights (phase 2). Optionally pre-quantized MXFP8 via ``gate_up_weight_T_scales``.
   :type gate_up_weight_T: ``nl.ndarray``
   :param gate_up_weight_T_scales: MXFP8 scales for pre-quantized ``gate_up_weight_T``.
   :type gate_up_weight_T_scales: ``nl.ndarray``
   :param gate_up_weights_scales: MXFP8 scales for pre-quantized ``gate_up_weights`` (recompute RHS).
   :type gate_up_weights_scales: ``nl.ndarray``
   :param down_weight_T: [I, H], transposed down projection weights. Optionally pre-quantized via ``down_weight_T_scales``.
   :type down_weight_T: ``nl.ndarray``
   :param down_weight_T_scales: MXFP8 scales for pre-quantized ``down_weight_T``.
   :type down_weight_T_scales: ``nl.ndarray``
   :param output_grad_T: [H, S], pre-transposed output gradient. Optionally pre-quantized via ``output_grad_T_scales``.
   :type output_grad_T: ``nl.ndarray``
   :param output_grad_T_scales: MXFP8 scales for pre-quantized ``output_grad_T``.
   :type output_grad_T_scales: ``nl.ndarray``
   :param hidden_states_T: [H, S], pre-transposed input hidden states. Optionally pre-quantized via ``hidden_states_T_scales``.
   :type hidden_states_T: ``nl.ndarray``
   :param hidden_states_T_scales: MXFP8 scales for pre-quantized ``hidden_states_T``.
   :type hidden_states_T_scales: ``nl.ndarray``
   :param gate_pre: [S, I], checkpointed gate pre-activation, or None.
   :type gate_pre: ``nl.ndarray``
   :param gate_act: [S, I], checkpointed SiLU(gate_pre), or None.
   :type gate_act: ``nl.ndarray``
   :param up: [S, I], checkpointed up projection, or None.
   :type up: ``nl.ndarray``
   :param intermediate: [S, I], checkpointed gate_act * up, or None.
   :type intermediate: ``nl.ndarray``
   :param run_with_lnc2: Whether to shard across 2 LNC cores.
   :type run_with_lnc2: ``bool``
   :param matmul_config: Per-phase matmul tiling configuration. Replaces the previous per-phase ``phase*_tiles_*`` / ``recompute_tiles_*`` arguments.
   :type matmul_config: ``MlpBwdMatmulConfig``
   :param fp8_x4_dtype: MXFP8 quantized data type.
   :type fp8_x4_dtype: ``type``
   :param clamp_limits: Optional activation clamp limits.
   :type clamp_limits: ``ClampLimits``
   :return: (hidden_states_grad [S, H], gate_up_weight_grad [2I, H], down_proj_weight_grad [H, I]).
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * S: Sequence length.
   * H: Hidden dimension size.

