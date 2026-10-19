.. meta::
    :description: 3D Convolution using tensor engine with K-replication strategy and W-contiguous tiling.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.conv

Conv3D Kernel API Reference
===========================

3D Convolution using tensor engine with K-replication strategy and W-contiguous tiling.

Implements a 3D convolution operation (x_in * filters + bias) optimized for NeuronCore using a K-replication strategy for filter loading and W-contiguous tiling for output computation. Supports configurable stride, padding, dilation, optional bias, optional activation function, and LNC sharding. Intended Usage Range: B: 1-128 C_in: 3-1280, C_out: 3-2048 D: 1-1024, H: 1-1024, W: 1-1024 K_d: 1-64, K_h: 1-64, K_w: 1-64 Stride: 1-64 per dimension Dilation: 1-64 per dimension

Background
-----------

The ``conv3d`` kernel implements 3D convolution using the tensor engine with a K-replication strategy for filter loading and W-contiguous tiling for output computation. It supports configurable stride, padding, dilation, optional bias, activation fusion, and LNC sharding.

API Reference
--------------

**Source code for this kernel API can be found at**: `conv3d.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/conv/conv3d.py>`_

conv3d
^^^^^^

.. py:function:: conv3d(x_in: nl.NkiTensor, filters: nl.NkiTensor, bias: Optional[nl.NkiTensor] = None, stride: tuple[int, int, int] = (1, 1, 1), padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0), dilation: tuple[int, int, int] = (1, 1, 1), activation_fn: Optional[ActFnType] = None, lnc_shard: bool = False, batch_norm_mode: BatchNormMode = BatchNormMode.NONE, output_pre_norm: Optional[bool] = False, batch_norm_eps: Optional[float] = 1e-05, gamma: Optional[nl.NkiTensor] = None, beta: Optional[nl.NkiTensor] = None, momentum: Optional[float] = 0.1, running_means: Optional[nl.NkiTensor] = None, running_variances: Optional[nl.NkiTensor] = None, residual_add_loc: ResidualAddLoc = ResidualAddLoc.NONE, residuals_in: Optional[nl.NkiTensor] = None, sbm: Optional[SbufManager] = None, use_auto_allocation: bool = False) -> nl.NkiTensor

   3D Convolution using tensor engine with K-replication strategy and W-contiguous tiling.

   :param x_in: [B, C_in, D, H, W], Input tensor on HBM.
   :type x_in: ``nl.NkiTensor``
   :param filters: [K_d, K_h, K_w, C_in, C_out], Filter weights on HBM.
   :type filters: ``nl.NkiTensor``
   :param bias: [C_out], Optional bias tensor on HBM.
   :type bias: ``Optional[nl.NkiTensor]``
   :param stride: (stride_d, stride_h, stride_w), Convolution strides.
   :type stride: ``tuple[int, int, int]``
   :param padding: (pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right), Padding for each spatial dimension.
   :type padding: ``tuple[int, int, int, int, int, int]``
   :param dilation: (dilation_d, dilation_h, dilation_w), Dilation factors.
   :type dilation: ``tuple[int, int, int]``
   :param activation_fn: Optional activation function to apply after conv. When a batchnorm is fused, the activation is instead applied after the fused batchnorm (in TRAINING mode the statistics are still computed on the raw convolution output).
   :type activation_fn: ``Optional[ActFnType]``
   :param lnc_shard: Does nothing. Will be deprecated in next release.
   :type lnc_shard: ``bool``
   :param batch_norm_mode: Fused-batchnorm mode. Default BatchNormMode.NONE.
   :type batch_norm_mode: ``BatchNormMode``
   :param output_pre_norm: When True, additionally return the raw pre-batchnorm convolution output as conv_out. Only valid in BatchNormMode.TRAINING, which already stages that tensor in HBM; rejected in EVAL (which never materializes it) and ignored in NONE. Default False.
   :type output_pre_norm: ``Optional[bool]``
   :param batch_norm_eps: Epsilon added to the variance for numerical stability when a batchnorm is fused. Default 1e-5.
   :type batch_norm_eps: ``Optional[float]``
   :param gamma: [C_out, 1], per-channel batchnorm scale. Required when a batchnorm is fused.
   :type gamma: ``Optional[nl.NkiTensor]``
   :param beta: [C_out, 1], per-channel batchnorm shift. Required when a batchnorm is fused.
   :type beta: ``Optional[nl.NkiTensor]``
   :param momentum: Momentum for the running-statistics update in BatchNormMode.TRAINING, matching torch.nn.BatchNorm2d: x_running = (1 - momentum) * x_running + momentum * x_new. Default 0.1.
   :type momentum: ``Optional[float]``
   :param running_means: [C_out, 1], float32 running mean carried across steps. Required when a batchnorm is fused. In BatchNormMode.EVAL this is the mean used for the normalization.
   :type running_means: ``Optional[nl.NkiTensor]``
   :param running_variances: [C_out, 1], float32 running variance carried across steps. Required when a batchnorm is fused. In BatchNormMode.EVAL this is the variance used for the normalization.
   :type running_variances: ``Optional[nl.NkiTensor]``
   :param residual_add_loc: Insertion point of the fused residual add. Default ResidualAddLoc.NONE (no residual add).
   :type residual_add_loc: ``ResidualAddLoc``
   :param residuals_in: [B, C_out, D_out, H_out, W_out], residual tensor added to the output. Required when residual_add_loc is not NONE, ignored otherwise. Same dtype as x_in.
   :type residuals_in: ``Optional[nl.NkiTensor]``
   :param sbm: Optional caller-provided SBUF manager for allocation. When provided, its allocation mode must match use_auto_allocation.
   :type sbm: ``Optional[SbufManager]``
   :param use_auto_allocation: Whether to use auto allocation for SBUF buffers.
   :type use_auto_allocation: ``bool``
   :return: [B, C_out, D_out, H_out, W_out], Output tensor on HBM. NONE and EVAL return only y_out (EVAL computes no statistics and leaves the running ones untouched). TRAINING returns (y_out, means, variances, updated_running_means, updated_running_variances): means and variances are [C_out] float32 raw (pre-normalization) statistics with correction 0; the updated_* pair is [C_out, 1] float32 on shared HBM and uses the correction-1 (unbiased) variance, matching torch.nn.BatchNorm2d. With output_pre_norm=True, conv_out (same shape and dtype as y_out) is inserted right after y_out.
   :rtype: ``nl.NkiTensor``

   **Dimensions**:

   * B: Batch size
   * C_in: Number of input channels
   * C_out: Number of output channels
   * D: Input depth
   * H: Input height
   * W: Input width
   * K_d: Filter depth
   * K_h: Filter height
   * K_w: Filter width
   * D_out: Output depth = (D + pad_d_left + pad_d_right - dilation_d * (K_d - 1) - 1) // stride_d + 1
   * H_out: Output height = (H + pad_h_top + pad_h_bottom - dilation_h * (K_h - 1) - 1) // stride_h + 1

