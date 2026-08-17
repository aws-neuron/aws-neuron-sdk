.. meta::
    :description: Backward pass for matrix multiplication with MXFP8 quantization.
    :date-modified: 08/17/2026

.. currentmodule:: nkilib.experimental.matmul_mxfp8

Matmul MXFP8 Backward Kernel API Reference
==========================================

Backward pass for matrix multiplication with MXFP8 quantization.

Computes both input gradients (dX) and weight gradients (dW) for a linear layer. Forward pass convention: Y = X @ W^T, where X is [M, K], W is [N, K], Y is [M, N] Backward pass (two separate matmuls with different dimensions): dX = dY @ W     (shape [M, K]):  M_logical=M, K_contraction=N, N_logical=K dW = dY^T @ X   (shape [N, K]):  M_logical=N, K_contraction=M, N_logical=K

Background
-----------

The ``matmul_mxfp8_backward`` kernel computes the backward pass for matrix multiplication with MXFP8 quantization.

API Reference
--------------

**Source code for this kernel API can be found at**: `matmul_mxfp8_generic_backward_kernel.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/matmul_mxfp8/matmul_mxfp8_generic_backward_kernel.py>`_

matmul_mxfp8_backward
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: matmul_mxfp8_backward(output_grad, weights, input_activation, input_grad_config: MatmulMxfp8KernelConfig = None, weight_grad_config: MatmulMxfp8KernelConfig = None, tile_loop_order: str = 'mnk', float8_dtype: str = 'float8_e5m2', output_dtype = nl.bfloat16, run_with_lnc2: bool = True, lnc_2_shard_rhs: bool = True, output_grad_scales = None, weight_scales = None, input_scales = None, use_scale_packing: bool = False, spill_reload: bool = False, output_grad_is_swizzled: bool = False, weights_is_swizzled: bool = False, input_is_swizzled: bool = False) -> tuple

   Backward pass for matrix multiplication with MXFP8 quantization.

   :param output_grad: Output gradient (dY), shape [M, N] in BF16.
   :param weights: Weight matrix (W), shape [N, K] in BF16.
   :param input_activation: Input activation (X), shape [M, K] in BF16.
   :param input_grad_config: MatmulMxfp8KernelConfig for the dX phase (auto-resolved if None).
   :type input_grad_config: ``MatmulMxfp8KernelConfig``
   :param weight_grad_config: MatmulMxfp8KernelConfig for the dW phase (auto-resolved if None).
   :type weight_grad_config: ``MatmulMxfp8KernelConfig``
   :param tile_loop_order: Tile processing order within blocks, default 'mnk'.
   :type tile_loop_order: ``str``
   :param float8_dtype: FP8 dtype for quantization, default "float8_e5m2".
   :type float8_dtype: ``str``
   :param output_dtype: Output data type, default nl.bfloat16.
   :param run_with_lnc2: Enable LNC2 parallelization, default True.
   :type run_with_lnc2: ``bool``
   :param lnc_2_shard_rhs: Shard on N dimension (RHS), default True.
   :type lnc_2_shard_rhs: ``bool``
   :param output_grad_scales: Optional pre-computed scales for output gradient.
   :param weight_scales: Optional pre-computed scales for weights.
   :param input_scales: Optional pre-computed scales for input activation.
   :param use_scale_packing: Assert packed scales for pre-quantized inputs.
   :type use_scale_packing: ``bool``
   :param spill_reload: Spill quantized blocks to HBM for reuse.
   :type spill_reload: ``bool``
   :param output_grad_is_swizzled: Whether output gradient is pre-swizzled.
   :type output_grad_is_swizzled: ``bool``
   :param weights_is_swizzled: Whether weights are pre-swizzled.
   :type weights_is_swizzled: ``bool``
   :param input_is_swizzled: Whether input activation is pre-swizzled.
   :type input_is_swizzled: ``bool``
   :return: (input_grad, weight_grad) where: - input_grad: Shape [M, K], gradient with respect to input - weight_grad: Shape [N, K], gradient with respect to weights
   :rtype: ``nl.ndarray``

