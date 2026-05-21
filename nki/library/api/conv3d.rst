.. meta::
    :description: 3D Convolution using tensor engine with K-replication strategy and W-contiguous tiling.
    :date-modified: 05/21/2026

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

**Source code for this kernel API can be found at**: `conv3d.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/conv/conv3d.py>`_

conv3d
^^^^^^

.. py:function:: conv3d(x_in: nl.ndarray, filters: nl.ndarray, bias: Optional[nl.ndarray] = None, stride: tuple[int, int, int] = (1, 1, 1), padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0), dilation: tuple[int, int, int] = (1, 1, 1), activation_fn: Optional[ActFnType] = None, lnc_shard: bool = False) -> nl.ndarray

   3D Convolution using tensor engine with K-replication strategy and W-contiguous tiling.

   :param x_in: [B, C_in, D, H, W], Input tensor on HBM.
   :type x_in: ``nl.ndarray``
   :param filters: [K_d, K_h, K_w, C_in, C_out], Filter weights on HBM.
   :type filters: ``nl.ndarray``
   :param bias: [C_out], Optional bias tensor on HBM.
   :type bias: ``Optional[nl.ndarray]``
   :param stride: (stride_d, stride_h, stride_w), Convolution strides.
   :type stride: ``tuple[int, int, int]``
   :param padding: (pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right), Padding for each spatial dimension.
   :type padding: ``tuple[int, int, int, int, int, int]``
   :param dilation: (dilation_d, dilation_h, dilation_w), Dilation factors.
   :type dilation: ``tuple[int, int, int]``
   :param activation_fn: Optional activation function to apply after conv.
   :type activation_fn: ``Optional[ActFnType]``
   :param lnc_shard: Enable LNC sharding across neuron cores.
   :type lnc_shard: ``bool``
   :return: [B, C_out, D_out, H_out, W_out], Output tensor on HBM.
   :rtype: ``nl.ndarray``

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

