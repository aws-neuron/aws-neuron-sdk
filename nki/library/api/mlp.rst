.. meta::
    :description: MLP kernel implements Multi-Layer Perceptron with optional normalization fusion and quantization.
    :date-modified: 11/12/2025

.. currentmodule:: nkilib.core.mlp

MLP Kernel API Reference
=========================

Implements Multi-Layer Perceptron with optional normalization fusion and quantization support.

The kernel supports:

* Both context encoding (CTE) and token generation (TKG) modes
* Optional normalization fusion (RMSNorm, LayerNorm)
* Various activation functions
* Residual connections via fused addition
* Flexible tensor layouts and column tiling optimizations
* Bias addition for all projections and normalization
* FP8 quantization (static and row-wise, TKG mode only)
* Gate and up projection result clamping
* Optional gate projection skipping
* SBUF output for kernel fusion

Background
-----------

The ``MLP`` kernel is a critical component in transformer architectures, responsible for processing token representations after the attention mechanism. This kernel optimizes the MLP computation by fusing it with optional normalization and supporting various optimizations for both context encoding and token generation scenarios.

.. note::
    This kernel automatically selects between TKG (Token Generation) and CTE (Context Encoding) implementations based on the batch size × sequence length, ensuring optimal performance across different use cases.

API Reference
--------------

**Source code for this kernel API can be found at**: `mlp.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/mlp/mlp.py>`_

mlp
^^^

.. py:function:: mlp(hidden_tensor: nl.ndarray, gate_proj_weights_tensor: nl.ndarray, up_proj_weights_tensor: nl.ndarray, down_proj_weights_tensor: nl.ndarray, normalization_weights_tensor: Optional[nl.ndarray] = None, gate_proj_bias_tensor: Optional[nl.ndarray] = None, up_proj_bias_tensor: Optional[nl.ndarray] = None, down_proj_bias_tensor: Optional[nl.ndarray] = None, normalization_bias_tensor: Optional[nl.ndarray] = None, fused_add_tensor: Optional[nl.ndarray] = None, store_fused_add_result: bool = False, activation_fn: ActFnType = ActFnType.SiLU, normalization_type: NormType = NormType.NO_NORM, quantization_type: QuantizationType = QuantizationType.NONE, gate_w_scale: Optional[nl.ndarray] = None, up_w_scale: Optional[nl.ndarray] = None, down_w_scale: Optional[nl.ndarray] = None, gate_up_in_scale: Optional[nl.ndarray] = None, down_in_scale: Optional[nl.ndarray] = None, quant_clipping_bound: float = 0.0, output_dtype = None, store_output_in_sbuf: bool = False, eps: float = 1e-6, skip_gate_proj: bool = False, use_tkg_gate_up_proj_column_tiling: bool = True, use_tkg_down_proj_column_tiling: bool = True, use_tkg_down_proj_optimized_layout: bool = False, gate_clamp_upper_limit: Optional[float] = None, gate_clamp_lower_limit: Optional[float] = None, up_clamp_upper_limit: Optional[float] = None, up_clamp_lower_limit: Optional[float] = None, force_cte_mode: bool = False) -> list[nl.ndarray]

   MLP (Multi-Layer Perceptron) Kernel implementation.

   Performs the standard MLP computation with support for both context encoding (CTE) and
   token generation (TKG) modes. Automatically selects the appropriate implementation based
   on input dimensions and supports various optimizations.

   :param hidden_tensor: Input hidden states tensor with shape [B, S, H] or SBUF layout.
   :type hidden_tensor: ``nl.ndarray``
   :param gate_proj_weights_tensor: Gate projection weight matrix with shape [H, I].
   :type gate_proj_weights_tensor: ``nl.ndarray``
   :param up_proj_weights_tensor: Up projection weight matrix with shape [H, I].
   :type up_proj_weights_tensor: ``nl.ndarray``
   :param down_proj_weights_tensor: Down projection weight matrix with shape [I, H].
   :type down_proj_weights_tensor: ``nl.ndarray``
   :param normalization_weights_tensor: Normalization weights with shape [1, H].
   :type normalization_weights_tensor: ``nl.ndarray``, optional
   :param gate_proj_bias_tensor: Bias tensor for gate projection with shape [1, I].
   :type gate_proj_bias_tensor: ``nl.ndarray``, optional
   :param up_proj_bias_tensor: Bias tensor for up projection with shape [1, I].
   :type up_proj_bias_tensor: ``nl.ndarray``, optional
   :param down_proj_bias_tensor: Bias tensor for down projection with shape [1, H].
   :type down_proj_bias_tensor: ``nl.ndarray``, optional
   :param normalization_bias_tensor: Bias tensor for normalization with shape [1, H]. Only applicable for layer normalization.
   :type normalization_bias_tensor: ``nl.ndarray``, optional
   :param fused_add_tensor: Tensor to fuse for the residual connection.
   :type fused_add_tensor: ``nl.ndarray``, optional
   :param store_fused_add_result: If True, stores the fused_add output to HBM, and the kernel returns both the fused_add output and the MLP output. Default: False.
   :type store_fused_add_result: ``bool``
   :param activation_fn: Activation function type.
   :type activation_fn: ``ActFnType``
   :param normalization_type: Type of normalization.
   :type normalization_type: ``NormType``
   :param quantization_type: Quantization type to use (default: QuantizationType.NONE). Supported values are QuantizationType.STATIC and QuantizationType.ROW. Quantization is only supported in TKG mode.
   :type quantization_type: ``QuantizationType``
   :param gate_w_scale: FP8 dequantization scales for gate weights. Shape is [128, I] for row-wise quantization, [128, 1] for static quantization. Defaults to None.
   :type gate_w_scale: ``nl.ndarray``, optional
   :param up_w_scale: FP8 dequantization scales for up weights. Shape is [128, I] for row-wise quantization, [128, 1] for static quantization. Defaults to None.
   :type up_w_scale: ``nl.ndarray``, optional
   :param down_w_scale: FP8 dequantization scales for down weights. Shape is [128, I] for row-wise quantization, [128, 1] for static quantization. Defaults to None.
   :type down_w_scale: ``nl.ndarray``, optional
   :param gate_up_in_scale: FP8 dequantization scales for gate and up input. Used for static quantization with shape [128, 1]. Defaults to None.
   :type gate_up_in_scale: ``nl.ndarray``, optional
   :param down_in_scale: FP8 dequantization scales for down input. Used for static quantization with shape [128, 1]. Defaults to None.
   :type down_in_scale: ``nl.ndarray``, optional
   :param quant_clipping_bound: Clipping bound for quantization. Default: 0.0.
   :type quant_clipping_bound: ``float``
   :param output_dtype: Output tensor data type. Defaults to None; if None, the hidden tensor's ``dtype`` is used.
   :type output_dtype: ``nki.dtype``
   :param store_output_in_sbuf: If True, stores the output in SBUF instead of HBM, allowing the next layer to read it directly without an additional load operation. This option is only available in TKG mode where output tensor is small enough to fit in SBUF. Default: False.
   :type store_output_in_sbuf: ``bool``
   :param eps: Epsilon value for numerical stability.
   :type eps: ``float``
   :param skip_gate_proj: Skip gate projection.
   :type skip_gate_proj: ``bool``
   :param use_tkg_gate_up_proj_column_tiling: If True, uses column tiling for the gate and up projection in TKG mode. Default: True.
   :type use_tkg_gate_up_proj_column_tiling: ``bool``
   :param use_tkg_down_proj_column_tiling: If True, uses column tiling for the down projection in TKG mode. Default: True.
   :type use_tkg_down_proj_column_tiling: ``bool``
   :param use_tkg_down_proj_optimized_layout: If True, the standard down_weight tensor (``shape [I, H]``) is reinterpreted as ``[I, lnc, 128, H // (128 * lnc)]``, then transposed to ``[I, lnc, H // (128 * lnc), 128]``. This layout provides unit-stride weight loading, reducing the matrix multiplication initiation interval. Only applied when ``use_tkg_down_proj_column_tiling`` is False. Default: False.
   :type use_tkg_down_proj_optimized_layout: ``bool``
   :param gate_clamp_upper_limit: Upper bound value to clamp on gate projection results, does not perform clamping if the value is set to None.
   :type gate_clamp_upper_limit: ``float``, optional
   :param gate_clamp_lower_limit: Lower bound value to clamp on gate projection results, does not perform clamping if the value is set to None.
   :type gate_clamp_lower_limit: ``float``, optional
   :param up_clamp_upper_limit: Upper bound value to clamp on up projection results, does not perform clamping if the value is set to None.
   :type up_clamp_upper_limit: ``float``, optional
   :param up_clamp_lower_limit: Lower bound value to clamp on up projection results, does not perform clamping if the value is set to None.
   :type up_clamp_lower_limit: ``float``, optional
   :param force_cte_mode: If True, forces the use of CTE mode. Default: False.
   :type force_cte_mode: ``bool``
   :return: The MLP output tensor(s). HBM output: Tensor with shape [B, S, H]. SBUF output: Shape depends on the mode setting. CTE: Not applicable. TKG when ``use_tkg_down_proj_column_tiling`` is ``True = [BxS, H]``. TKG when ``use_tkg_down_proj_column_tiling`` is ``False = [128(p_max), H/128, BxS``]``. If ``store_fused_add_result`` is ``True``, returns a list containing both the output and the stored fused output.
   :rtype: ``list[nl.ndarray]``

   **Notes**:

   * Automatically dispatches to either CTE or TKG implementation based on batch size and sequence length.
   * Token generation mode (TKG) is used for small batch/sequence dimensions (``batch_size × sequence_length ≤ 96``), while context encoding (CTE) handles larger inputs.
   * Column tiling and tensor layout optimization (``use_tkg_down_proj_optimized_layout``) are valid only in TKG mode.
   * FP8 quantization support is available only in TKG mode.
   * Supported input data types: ``nl.bfloat16``, ``nl.float16``, ``nl.float32``

Implementation Details
-----------------------

The kernel implementation includes several key optimizations:

1. **Dual Implementation Strategy**: Automatically selects between CTE (Context Encoding) and TKG (Token Generation) implementations based on ``batch_size × sequence_length``.

2. **Normalization Fusion**: Optionally fuses RMSNorm or LayerNorm operations with the MLP computation for improved performance.

3. **FP8 Quantization**: Supports FP8 quantization with both static and row-wise dequantization scales. Available only in TKG mode for weights and activations.

4. **Flexible Tensor Layouts**: Supports column tiling optimizations and tensor layout optimizations in TKG mode to improve memory access patterns.

5. **Activation Function Options**: Supports multiple activation functions, including SiLU (Swish), GELU, and ReLU.

6. **Result Clamping**: Provides optional clamping of gate and up projection results with configurable upper and lower bounds.

7. **Gate Projection Skipping**: Allows skipping the gate projection computation when ``skip_gate_proj`` is enabled.

8. **Residual Connection Fusion**: Can incorporate residual connections through fused_add_tensor for improved performance.

9. **SBUF Output Option**: Provides the option to keep output in SBUF for fusion with subsequent operations (TKG mode only).

10. **Bias Addition**: Supports optional bias addition for gate, up, and down projections, as well as for normalization.

11. **Optimized Weight Loading**: In TKG mode, ``use_tkg_down_proj_optimized_layout`` enables unit-stride weight loading to reduce matrix multiplication initiation interval.

12. **Multi-Precision Support**: Supports ``bfloat16``, ``float16``, and ``float32`` input data types for flexible precision requirements.

See Also
--------

* :doc:`QKV Kernel API Reference </nki/library/api/qkv>`
* :doc:`RMSNorm-Quant Kernel API Reference </nki/library/api/rmsnorm-quant>`
