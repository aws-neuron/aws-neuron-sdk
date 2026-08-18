.. meta::
    :description: 3D convolution with temporal unrolling and column tiling for small C_out.
    :date-modified: 08/17/2026

.. currentmodule:: nkilib.experimental.conv

Conv3D Temporal Unroll Kernel API Reference
===========================================

3D convolution with temporal unrolling and column tiling for small C_out.

The companion ``should_use_temporal_unroll`` advisory check reports whether a given problem shape benefits from temporal unrolling before you call the kernel. It returns True when: D_out temporal positions fit in a single PSUM bank column-tiled, C_in is large enough to amortize filter caching, and W_out exceeds F_MAX so multiple W tiles are needed (where the baseline is slow).

Background
-----------

The ``conv3d_temporal_unroll`` kernel performs 3D convolution with temporal unrolling and column tiling for small C_out.

API Reference
--------------

**Source code for this kernel API can be found at**: `conv3d_temporal_unroll.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/conv/conv3d_temporal_unroll.py>`_

should_use_temporal_unroll
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: should_use_temporal_unroll(C_out: int, D_out: int, C_in: int, K_d: int, W_out: int) -> bool

   Advisory check: whether conv3d_temporal_unroll is applicable for this shape.

   :param C_out: Number of output channels.
   :type C_out: ``int``
   :param D_out: Number of output depth positions.
   :type D_out: ``int``
   :param C_in: Number of input channels.
   :type C_in: ``int``
   :param K_d: Filter depth dimension.
   :type K_d: ``int``
   :param W_out: Output width.
   :type W_out: ``int``
   :return: True if temporal unroll should be used.
   :rtype: ``nl.ndarray``

conv3d_temporal_unroll
^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: conv3d_temporal_unroll(x_in: nl.ndarray, filters: nl.ndarray, bias: Optional[nl.ndarray] = None, stride: tuple[int, int, int] = (1, 1, 1), padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0), dilation: tuple[int, int, int] = (1, 1, 1), activation_fn: Optional[ActFnType] = None, lnc_shard: bool = False) -> nl.ndarray

   3D convolution with temporal unrolling and column tiling for small C_out.

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
   :param lnc_shard: Enable LNC sharding across neuron cores (shards on H_out).
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

