.. meta::
    :description: MoE TKG kernel implements Mixture of Experts MLP optimized for Token Generation.
    :date-modified: 02/13/2026

.. currentmodule:: nkilib.core.moe_tkg

MoE TKG Kernel API Reference
=============================

Implements Mixture of Experts (MoE) MLP computation optimized for Token Generation with support for both all-expert and selective-expert modes.

The kernel supports:

* All-expert mode (process all experts for all tokens)
* Selective-expert mode (process only top-K selected experts)
* Multiple quantization types (FP8 row/static, MxFP4)
* Expert affinity scaling (post-scale mode)
* Expert affinity masking for distributed inference
* Various activation functions (SiLU, GELU, ReLU)
* Optional bias terms for projections
* Clamping for gate and up projections
* SBUF or HBM output allocation

Background
--------------

The ``MoE TKG`` kernel is designed for Mixture of Experts models during token generation (decoding) phase where the batch size and sequence length are typically small (T ≤ 128). The kernel performs the core MoE MLP computation:

1. **Gate Projection**: ``gate_out = hidden @ gate_weights``
2. **Up Projection**: ``up_out = hidden @ up_weights``
3. **Activation**: ``act_gate = activation_fn(gate_out)``
4. **Element-wise Multiply**: ``intermediate = act_gate * up_out``
5. **Down Projection**: ``expert_out = intermediate @ down_weights``
6. **Affinity Scaling**: ``output = sum(expert_out * affinity)`` (if enabled)

The kernel supports two operational modes:

* **All-Expert Mode**: Processes all experts for all tokens, useful for distributed inference scenarios
* **Selective-Expert Mode**: Processes only the top-K selected experts per token, reducing computation

API Reference
----------------

**Source code for this kernel API can be found at**: `moe_tkg.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/moe/moe_tkg/moe_tkg.py>`_

moe_tkg
^^^^^^^^^^^^^^^

.. py:function:: moe_tkg(hidden_input: nl.ndarray, expert_gate_up_weights: nl.ndarray, expert_down_weights: nl.ndarray, expert_affinities: nl.ndarray, expert_index: nl.ndarray, is_all_expert: bool, rank_id: Optional[nl.ndarray] = None, expert_gate_up_bias: Optional[nl.ndarray] = None, expert_down_bias: Optional[nl.ndarray] = None, expert_gate_up_weights_scale: Optional[nl.ndarray] = None, expert_down_weights_scale: Optional[nl.ndarray] = None, hidden_input_scale: Optional[nl.ndarray] = None, gate_up_input_scale: Optional[nl.ndarray] = None, down_input_scale: Optional[nl.ndarray] = None, mask_unselected_experts: bool = False, expert_affinities_eager: Optional[nl.ndarray] = None, expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE, activation_fn: ActFnType = ActFnType.SiLU, output_dtype=None, gate_clamp_upper_limit: Optional[float] = None, gate_clamp_lower_limit: Optional[float] = None, up_clamp_upper_limit: Optional[float] = None, up_clamp_lower_limit: Optional[float] = None, output_in_sbuf: bool = False, is_all_expert_dynamic: bool = False) -> nl.ndarray

   Mixture of Experts (MoE) MLP token generation kernel.

   Performs MoE computation with support for both all-expert and selective-expert modes.
   Supports various quantization types including FP8 row/static quantization and MxFP4.
   Optimized for token generation scenarios with T ≤ 128 (except MX all-expert mode).

   :param hidden_input: Input hidden states tensor with shape ``[T, H]`` in HBM or ``[H0, T, H1]`` in SBUF
   :type hidden_input: ``nl.ndarray``
   :param expert_gate_up_weights: Fused gate and up projection weights. Shape ``[E_L, H, 2, I]`` for bf16/fp16 or ``[E_L, 128, 2, ceil(H/512), I]`` for MxFP4
   :type expert_gate_up_weights: ``nl.ndarray``
   :param expert_down_weights: Down projection weights. Shape ``[E_L, I, H]`` for bf16/fp16 or ``[E_L, I_p, ceil(I/512), H]`` for MxFP4
   :type expert_down_weights: ``nl.ndarray``
   :param expert_affinities: Expert routing weights/affinities with shape ``[T, E]``. For all-expert mode with affinity scaling, this will be sliced to ``[T, E_L]`` internally.
   :type expert_affinities: ``nl.ndarray``
   :param expert_index: Top-K expert indices per token with shape ``[T, K]``
   :type expert_index: ``nl.ndarray``
   :param is_all_expert: If ``True``, process all experts for all tokens; otherwise, process only selected top-K experts
   :type is_all_expert: ``bool``
   :param rank_id: Rank ID tensor specifying which worker processes experts ``[E_L * rank_id, E_L * (rank_id + 1))``. Shape ``[1, 1]``. Required for all-expert mode with affinity scaling enabled.
   :type rank_id: ``nl.ndarray``, optional
   :param expert_gate_up_bias: Bias for gate/up projections. Shape ``[E_L, 2, I]`` for non-MX or ``[E_L, I_p, 2, ceil(I/512), 4]`` for MX.
   :type expert_gate_up_bias: ``nl.ndarray``, optional
   :param expert_down_bias: Bias for down projection with shape ``[E_L, H]``
   :type expert_down_bias: ``nl.ndarray``, optional
   :param expert_gate_up_weights_scale: Quantization scales for gate/up weights. Shape ``[E_L, 2, I]`` for FP8 row quantization, ``[E_L, 2, 1]`` for FP8 static quantization, or ``[E_L, 128/8, 2, ceil(H/512), I]`` for MxFP4.
   :type expert_gate_up_weights_scale: ``nl.ndarray``, optional
   :param expert_down_weights_scale: Quantization scales for down weights. Shape ``[E_L, H]`` for FP8 row quantization, ``[E_L, 1]`` for FP8 static quantization, or ``[E_L, I_p/8, ceil(I/512), H]`` for MxFP4.
   :type expert_down_weights_scale: ``nl.ndarray``, optional
   :param hidden_input_scale: FP8 dequantization scale for the hidden input tensor. Used for static quantization of the input.
   :type hidden_input_scale: ``nl.ndarray``, optional
   :param gate_up_input_scale: FP8 dequantization scales for gate/up input. Shape ``[E_L, 1]``. Used for static quantization.
   :type gate_up_input_scale: ``nl.ndarray``, optional
   :param down_input_scale: FP8 dequantization scales for down input. Shape ``[E_L, 1]``. Used for static quantization.
   :type down_input_scale: ``nl.ndarray``, optional
   :param mask_unselected_experts: Whether to apply expert affinity masking based on expert_index. When ``True``, affinities are masked to zero for experts not selected by each token. Only used in all-expert mode with affinity scaling.
   :type mask_unselected_experts: ``bool``
   :param expert_affinities_eager: Eager expert affinities with shape ``[T, K]``. Not used in all-expert mode.
   :type expert_affinities_eager: ``nl.ndarray``, optional
   :param expert_affinities_scaling_mode: When to apply affinity scaling. Supported values: ``NO_SCALE``, ``POST_SCALE``. Default is ``NO_SCALE``.
   :type expert_affinities_scaling_mode: ``ExpertAffinityScaleMode``
   :param activation_fn: Activation function type. Default is ``SiLU``.
   :type activation_fn: ``ActFnType``
   :param output_dtype: Output tensor data type. Defaults to ``None``; if ``None``, uses ``hidden_input`` dtype.
   :type output_dtype: ``nl.dtype``, optional
   :param gate_clamp_upper_limit: Upper bound value to clamp gate projection results
   :type gate_clamp_upper_limit: ``float``, optional
   :param gate_clamp_lower_limit: Lower bound value to clamp gate projection results
   :type gate_clamp_lower_limit: ``float``, optional
   :param up_clamp_upper_limit: Upper bound value to clamp up projection results
   :type up_clamp_upper_limit: ``float``, optional
   :param up_clamp_lower_limit: Lower bound value to clamp up projection results
   :type up_clamp_lower_limit: ``float``, optional
   :param output_in_sbuf: If ``True``, allocate output in SBUF with same shape as hidden_input. If ``False`` (default), allocate output in HBM with shape ``[T, H]``.
   :type output_in_sbuf: ``bool``
   :param is_all_expert_dynamic: If ``True``, enables dynamic expert selection in all-expert mode, where the set of active experts can vary per token. Default: ``False``.
   :type is_all_expert_dynamic: ``bool``
   :return: Output tensor with MoE computation results. Shape ``[T, H]`` or same shape as hidden_input if output_in_sbuf=True.
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * T: Number of tokens (batch_size × seq_len)
   * H: Hidden dimension
   * I: Intermediate dimension
   * E: Number of global experts
   * E_L: Number of local experts processed by this kernel
   * K: Top-K experts per token
   * I_p: I//4 if I ≤ 512 else 128

   **Supported Data Types**:

   * Input: bfloat16, float16, float4_e2m1fn_x4 (MxFP4)

   **Constraints**:

   * T ≤ 128 (batch_size × seq_len must be ≤ 128, except for MX all-expert mode)
   * ``PRE_SCALE`` and ``PRE_SCALE_DELAYED`` modes are not supported
   * Static quantization (``gate_up_input_scale`` and ``down_input_scale``) is not currently supported
   * MX kernels require ``expert_gate_up_weights_scale`` and ``expert_down_weights_scale`` to be set
   * All-expert mode with affinity scaling requires ``rank_id`` parameter
   * All-expert mode does not support ``expert_affinities_eager``

Implementation Details
-------------------------

The kernel implementation includes several key optimizations:

1. **Dual Mode Operation**: Supports both all-expert and selective-expert modes with separate optimized implementations for each.

2. **Quantization Support**: Handles multiple quantization schemes:
   
   * **FP8 Row Quantization**: Per-row scaling for weights
   * **FP8 Static Quantization**: Single scale per weight matrix
   * **MxFP4**: Microscaling FP4 format with block-wise scaling

3. **Expert Affinity Masking**: For distributed inference in all-expert mode, masks expert affinities based on rank ID to ensure each worker processes only its assigned experts.

4. **Fused Gate-Up Projection**: Gate and up projection weights are fused into a single tensor for efficient memory access and computation.

5. **Affinity Scaling Modes**:
   
   * **NO_SCALE**: No affinity scaling applied
   * **POST_SCALE**: Apply affinity scaling after expert computation (recommended)

6. **Activation Function Support**: Supports various activation functions including SiLU (default), GELU, and ReLU.

7. **Optional Clamping**: Supports clamping of gate and up projection outputs for numerical stability.

8. **Flexible Output Allocation**: Supports output allocation in either HBM or SBUF for integration with larger kernels.

9. **MX-Specific Optimizations**: MX all-expert mode supports larger batch sizes and includes K-dimension sharding for selective-expert mode.



See Also
-----------

* :doc:`MoE CTE Kernel API Reference </nki/library/api/moe-cte>`
* :doc:`Router Top-K Kernel API Reference </nki/library/api/router-topk>`
* :doc:`MLP Kernel API Reference </nki/library/api/mlp>`
