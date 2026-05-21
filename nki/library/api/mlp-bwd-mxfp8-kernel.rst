.. meta::
    :description: Return (num_cores, shard_id) for LNC2 sharding.
    :date-modified: 05/21/2026

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

.. py:function:: compute_phase1_down_proj_mm_grad_mxfp8(output_grad_td: TensorDescriptor, gate_pre_td: TensorDescriptor, gate_act_td: TensorDescriptor, up_td: TensorDescriptor, d_gate_td: TensorDescriptor, d_up_td: TensorDescriptor, scratch_td: TensorDescriptor, down_weight_td: TensorDescriptor, s_base: int, dtype: type, fp8_x4_dtype: type, TILES_IN_BLOCK_M: int = 8, TILES_IN_BLOCK_N: int = 1, TILES_IN_BLOCK_K: int = 8, spill_reload: bool = True, use_scale_packing: bool = True, run_with_lnc2: bool = True) -> None

   Phase 1: Compute gradient through the down projection and SwiGLU gate.

   :param output_grad_td: [S, H], incoming gradient (is_f_by_k=True).
   :type output_grad_td: ``TensorDescriptor``
   :param gate_pre_td: [S, I], checkpointed gate pre-activation.
   :type gate_pre_td: ``TensorDescriptor``
   :param gate_act_td: [S, I], checkpointed gate post-activation.
   :type gate_act_td: ``TensorDescriptor``
   :param up_td: [S, I], checkpointed up projection.
   :type up_td: ``TensorDescriptor``
   :param d_gate_td: [S, I], output: gate gradient.
   :type d_gate_td: ``TensorDescriptor``
   :param d_up_td: [S, I], output: up gradient.
   :type d_up_td: ``TensorDescriptor``
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
   :param TILES_IN_BLOCK_M: Number of M tiles per block.
   :type TILES_IN_BLOCK_M: ``int``
   :param TILES_IN_BLOCK_N: Number of N tiles per block.
   :type TILES_IN_BLOCK_N: ``int``
   :param TILES_IN_BLOCK_K: Number of K tiles to accumulate in PSUM.
   :type TILES_IN_BLOCK_K: ``int``

compute_phase2_hidden_states_grad_mxfp8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: compute_phase2_hidden_states_grad_mxfp8(hidden_states_grad_td: TensorDescriptor, gate_weight_td: TensorDescriptor, up_weight_td: TensorDescriptor, d_gate_td: TensorDescriptor, d_up_td: TensorDescriptor, s_base: int, dtype: type, fp8_x4_dtype: type, TILES_IN_BLOCK_M: int = 8, TILES_IN_BLOCK_N: int = 1, TILES_IN_BLOCK_K: int = 8, spill_reload: bool = True, use_scale_packing: bool = True, run_with_lnc2: bool = True) -> None

   Phase 2: Compute gradient w.r.t. input hidden states.

   :param hidden_states_grad_td: [S, H], output: dL/d_hidden.
   :type hidden_states_grad_td: ``TensorDescriptor``
   :param gate_weight_td: [H, I], transposed gate projection weights (is_f_by_k=True).
   :type gate_weight_td: ``TensorDescriptor``
   :param up_weight_td: [H, I], transposed up projection weights (is_f_by_k=True).
   :type up_weight_td: ``TensorDescriptor``
   :param d_gate_td: [S, I], gate gradient (is_f_by_k=True).
   :type d_gate_td: ``TensorDescriptor``
   :param d_up_td: [S, I], up gradient (is_f_by_k=True).
   :type d_up_td: ``TensorDescriptor``
   :param s_base: Row offset for this LNC core's shard.
   :type s_base: ``int``
   :param dtype: Data type for computation (nl.bfloat16).
   :type dtype: ``type``
   :param fp8_x4_dtype: MXFP8 quantized data type (e.g. float8_e4m3fn_x4).
   :type fp8_x4_dtype: ``type``
   :param TILES_IN_BLOCK_M: Number of M tiles per block.
   :type TILES_IN_BLOCK_M: ``int``
   :param TILES_IN_BLOCK_N: Number of N tiles per block.
   :type TILES_IN_BLOCK_N: ``int``
   :param TILES_IN_BLOCK_K: Number of K tiles to accumulate in PSUM.
   :type TILES_IN_BLOCK_K: ``int``

compute_phase3_gate_up_weight_grad_mxfp8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: compute_phase3_gate_up_weight_grad_mxfp8(weight_grad_td: TensorDescriptor, hidden_states_T_td: TensorDescriptor, grad_T_td: TensorDescriptor, dtype: type, fp8_x4_dtype: type, TILES_IN_BLOCK_M: int = 4, TILES_IN_BLOCK_N: int = 1, TILES_IN_BLOCK_K: int = 8, spill_reload: bool = True, use_scale_packing: bool = True, run_with_lnc2: bool = True) -> None

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
   :param TILES_IN_BLOCK_M: Number of M tiles per block.
   :type TILES_IN_BLOCK_M: ``int``
   :param TILES_IN_BLOCK_N: Number of N tiles per block.
   :type TILES_IN_BLOCK_N: ``int``
   :param TILES_IN_BLOCK_K: Number of K tiles to accumulate in PSUM.
   :type TILES_IN_BLOCK_K: ``int``

   **Dimensions**:

   * S: Sequence length.
   * H: Hidden dimension size.

compute_phase4_down_weight_grad_mxfp8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: compute_phase4_down_weight_grad_mxfp8(down_weight_grad_td: TensorDescriptor, output_grad_T_td: TensorDescriptor, hidden_T_td: TensorDescriptor, h_base: int, dtype: type, fp8_x4_dtype: type, TILES_IN_BLOCK_M: int = 4, TILES_IN_BLOCK_N: int = 1, TILES_IN_BLOCK_K: int = 8, spill_reload: bool = True, use_scale_packing: bool = True, run_with_lnc2: bool = True) -> None

   Phase 4: Compute gradient w.r.t. down projection weight matrix.

   :param down_weight_grad_td: [H, I], output: dW_down.
   :type down_weight_grad_td: ``TensorDescriptor``
   :param output_grad_T_td: [H, S], transposed output gradient (is_f_by_k=True).
   :type output_grad_T_td: ``TensorDescriptor``
   :param hidden_T_td: [I, S], transposed intermediate activations (is_f_by_k=True).
   :type hidden_T_td: ``TensorDescriptor``
   :param h_base: Row offset into the H dimension for this LNC core.
   :type h_base: ``int``
   :param dtype: Data type for computation (nl.bfloat16).
   :type dtype: ``type``
   :param fp8_x4_dtype: MXFP8 quantized data type.
   :type fp8_x4_dtype: ``type``
   :param TILES_IN_BLOCK_M: Number of M tiles per block.
   :type TILES_IN_BLOCK_M: ``int``
   :param TILES_IN_BLOCK_N: Number of N tiles per block.
   :type TILES_IN_BLOCK_N: ``int``
   :param TILES_IN_BLOCK_K: Number of K tiles to accumulate in PSUM.
   :type TILES_IN_BLOCK_K: ``int``

mlp_backward_mxfp8_base_nki
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: mlp_backward_mxfp8_base_nki(output_grad_td: TensorDescriptor, hidden_states_td: TensorDescriptor, gate_pre_td: TensorDescriptor, gate_act_td: TensorDescriptor, up_td: TensorDescriptor, hidden_td: TensorDescriptor, gate_weight_T_td: TensorDescriptor, up_weight_T_td: TensorDescriptor, down_weight_T_td: TensorDescriptor, d_gate_td: TensorDescriptor, d_up_td: TensorDescriptor, hidden_states_T_td: TensorDescriptor, output_grad_T_td: TensorDescriptor, hidden_T_td: TensorDescriptor, scratch_td: TensorDescriptor, hidden_states_grad_td: TensorDescriptor, weight_grad_td: TensorDescriptor, down_weight_grad_td: TensorDescriptor, run_with_lnc2: bool = True, phase1_tiles_m: int = 8, phase1_tiles_n: int = 1, phase1_tiles_k: int = 8, phase2_tiles_m: int = 8, phase2_tiles_n: int = 1, phase2_tiles_k: int = 8, phase3_tiles_m: int = 4, phase3_tiles_n: int = 1, phase3_tiles_k: int = 8, phase4_tiles_m: int = 4, phase4_tiles_n: int = 1, phase4_tiles_k: int = 8, fp8_x4_dtype: type = float8_e4m3fn_x4, spill_reload: bool = True, use_scale_packing: bool = True) -> tuple

   MXFP8 SwiGLU MLP backward pass (base kernel).

   :param output_grad_td: [S, H], incoming gradient dL/d_output (is_f_by_k=True).
   :type output_grad_td: ``TensorDescriptor``
   :param hidden_states_td: [S, H], original input (for phase 3 weight grad).
   :type hidden_states_td: ``TensorDescriptor``
   :param gate_pre_td: [S, I], gate pre-activation (before SiLU).
   :type gate_pre_td: ``TensorDescriptor``
   :param gate_act_td: [S, I], gate post-activation (SiLU(gate_pre)).
   :type gate_act_td: ``TensorDescriptor``
   :param up_td: [S, I], up projection (hidden @ W_up.T).
   :type up_td: ``TensorDescriptor``
   :param hidden_td: [S, I], gated intermediate (gate_act * up, for phase 4).
   :type hidden_td: ``TensorDescriptor``
   :param gate_weight_T_td: [H, I], transposed gate projection weights.
   :type gate_weight_T_td: ``TensorDescriptor``
   :param up_weight_T_td: [H, I], transposed up projection weights.
   :type up_weight_T_td: ``TensorDescriptor``
   :param down_weight_T_td: [I, H], transposed down projection weights.
   :type down_weight_T_td: ``TensorDescriptor``
   :param d_gate_td: [S, I], scratch: gate gradient.
   :type d_gate_td: ``TensorDescriptor``
   :param d_up_td: [S, I], scratch: up gradient.
   :type d_up_td: ``TensorDescriptor``
   :param hidden_states_T_td: [H, S], pre-transposed input hidden states.
   :type hidden_states_T_td: ``TensorDescriptor``
   :param output_grad_T_td: [H, S], pre-transposed output gradient.
   :type output_grad_T_td: ``TensorDescriptor``
   :param hidden_T_td: [I, S], pre-transposed intermediate activations.
   :type hidden_T_td: ``TensorDescriptor``
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
   :param phase1_tiles_m: M blocking for phase 1.
   :type phase1_tiles_m: ``int``
   :param phase1_tiles_n: N blocking for phase 1.
   :type phase1_tiles_n: ``int``
   :param phase1_tiles_k: K blocking for phase 1.
   :type phase1_tiles_k: ``int``
   :param phase2_tiles_m: M blocking for phase 2.
   :type phase2_tiles_m: ``int``
   :param phase2_tiles_n: N blocking for phase 2.
   :type phase2_tiles_n: ``int``
   :param phase2_tiles_k: K blocking for phase 2.
   :type phase2_tiles_k: ``int``
   :param phase3_tiles_m: M blocking for phase 3.
   :type phase3_tiles_m: ``int``
   :param phase3_tiles_n: N blocking for phase 3.
   :type phase3_tiles_n: ``int``
   :param phase3_tiles_k: K blocking for phase 3.
   :type phase3_tiles_k: ``int``
   :param phase4_tiles_m: M blocking for phase 4.
   :type phase4_tiles_m: ``int``
   :param phase4_tiles_n: N blocking for phase 4.
   :type phase4_tiles_n: ``int``
   :param phase4_tiles_k: K blocking for phase 4.
   :type phase4_tiles_k: ``int``
   :param fp8_x4_dtype: MXFP8 quantized data type.
   :type fp8_x4_dtype: ``type``
   :return: (hidden_states_grad [S, H], gate_up_weight_grad [2I, H], down_weight_grad [H, I]).
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * S: Sequence length.
   * H: Hidden dimension size.

mlp_backward_mxfp8_nki
^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: mlp_backward_mxfp8_nki(output_hidden_states_grad: nl.ndarray, hidden_states: nl.ndarray, gate_proj_weight_T: nl.ndarray, up_proj_weight_T: nl.ndarray, down_proj_weight_T: nl.ndarray, gate_up_weights: nl.ndarray, d_gate_scratch: nl.ndarray, d_up_scratch: nl.ndarray, hidden_states_T: nl.ndarray, output_grad_T: nl.ndarray, hidden_T: nl.ndarray, silu_up_mul_gate_grad_T_scratch: nl.ndarray, gate_pre_scratch: nl.ndarray, gate_act_scratch: nl.ndarray, up_scratch: nl.ndarray, hidden_scratch: nl.ndarray, gate_pre: nl.ndarray = None, gate_act: nl.ndarray = None, up: nl.ndarray = None, hidden: nl.ndarray = None, run_with_lnc2: bool = True, phase1_tiles_m: int = 8, phase1_tiles_n: int = 1, phase1_tiles_k: int = 8, phase2_tiles_m: int = 8, phase2_tiles_n: int = 1, phase2_tiles_k: int = 8, phase3_tiles_m: int = 4, phase3_tiles_n: int = 1, phase3_tiles_k: int = 8, phase4_tiles_m: int = 4, phase4_tiles_n: int = 1, phase4_tiles_k: int = 8, recompute_tiles_m: int = 8, recompute_tiles_n: int = 1, recompute_tiles_k: int = 8, fp8_x4_dtype: type = float8_e4m3fn_x4, spill_reload: bool = True, use_scale_packing: bool = True, output_grad_scales: nl.ndarray = None, output_grad_is_swizzled: bool = False, down_weight_scales: nl.ndarray = None, down_weight_is_swizzled: bool = False, gate_weight_scales: nl.ndarray = None, gate_weight_is_swizzled: bool = False, up_weight_scales: nl.ndarray = None, up_weight_is_swizzled: bool = False, hidden_states_T_scales: nl.ndarray = None, hidden_states_T_is_swizzled: bool = False, hidden_states_scales: nl.ndarray = None, hidden_states_is_swizzled: bool = False, gate_up_weights_scales: nl.ndarray = None, gate_up_weights_is_swizzled: bool = False, output_grad_T_scales: nl.ndarray = None, output_grad_T_is_swizzled: bool = False, hidden_T_scales: nl.ndarray = None, hidden_T_is_swizzled: bool = False) -> tuple

   MXFP8 SwiGLU MLP backward pass with activation checkpointing support.

   :param output_hidden_states_grad: [S, H], incoming gradient dL/d_output.
   :type output_hidden_states_grad: ``nl.ndarray``
   :param hidden_states: [S, H], original input (for recompute + phase 3).
   :type hidden_states: ``nl.ndarray``
   :param gate_proj_weight_T: [H, I], transposed gate projection weights (phase 2).
   :type gate_proj_weight_T: ``nl.ndarray``
   :param up_proj_weight_T: [H, I], transposed up projection weights (phase 2).
   :type up_proj_weight_T: ``nl.ndarray``
   :param down_proj_weight_T: [I, H], transposed down projection weights (phase 1).
   :type down_proj_weight_T: ``nl.ndarray``
   :param gate_up_weights: [2I, H], fused gate+up weights (for recompute).
   :type gate_up_weights: ``nl.ndarray``
   :param d_gate_scratch: [S, I], scratch: gate gradient.
   :type d_gate_scratch: ``nl.ndarray``
   :param d_up_scratch: [S, I], scratch: up gradient.
   :type d_up_scratch: ``nl.ndarray``
   :param hidden_states_T: [H, S], pre-transposed input hidden states.
   :type hidden_states_T: ``nl.ndarray``
   :param output_grad_T: [H, S], pre-transposed output gradient.
   :type output_grad_T: ``nl.ndarray``
   :param hidden_T: [I, S], pre-transposed intermediate activations.
   :type hidden_T: ``nl.ndarray``
   :param silu_up_mul_gate_grad_T_scratch: [2I, S], scratch: transposed d_gate || d_up.
   :type silu_up_mul_gate_grad_T_scratch: ``nl.ndarray``
   :param gate_pre_scratch: [S, I], scratch buffer for gate_pre.
   :type gate_pre_scratch: ``nl.ndarray``
   :param gate_act_scratch: [S, I], scratch buffer for gate_act.
   :type gate_act_scratch: ``nl.ndarray``
   :param up_scratch: [S, I], scratch buffer for up.
   :type up_scratch: ``nl.ndarray``
   :param hidden_scratch: [S, I], scratch buffer for hidden.
   :type hidden_scratch: ``nl.ndarray``
   :param gate_pre: [S, I], checkpointed gate pre-activation, or None.
   :type gate_pre: ``nl.ndarray``
   :param gate_act: [S, I], checkpointed SiLU(gate_pre), or None.
   :type gate_act: ``nl.ndarray``
   :param up: [S, I], checkpointed up projection, or None.
   :type up: ``nl.ndarray``
   :param hidden: [S, I], checkpointed gate_act * up, or None. Pre-swizzled/pre-quantized input support: Each matmul operand tensor accepts an optional ``*_scales`` (nl.ndarray) and ``*_is_swizzled`` (bool) pair. When both are default (None/False), the tensor is treated as unswizzled BF16 — identical to prior behavior. output_grad_scales, output_grad_is_swizzled: Phase 1 LHS (output_hidden_states_grad). down_weight_scales, down_weight_is_swizzled: Phase 1 RHS (down_proj_weight_T). gate_weight_scales, gate_weight_is_swizzled: Phase 2 RHS (gate_proj_weight_T). up_weight_scales, up_weight_is_swizzled: Phase 2 RHS (up_proj_weight_T). hidden_states_T_scales, hidden_states_T_is_swizzled: Phase 3 RHS (hidden_states_T). hidden_states_scales, hidden_states_is_swizzled: Recompute LHS (hidden_states). gate_up_weights_scales, gate_up_weights_is_swizzled: Recompute RHS (gate_up_weights). output_grad_T_scales, output_grad_T_is_swizzled: Phase 4 LHS (output_grad_T). hidden_T_scales, hidden_T_is_swizzled: Phase 4 RHS (hidden_T).
   :type hidden: ``nl.ndarray``
   :return: (hidden_states_grad [S, H], gate_up_weight_grad [2I, H], down_proj_weight_grad [H, I]).
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * S: Sequence length.
   * H: Hidden dimension size.

