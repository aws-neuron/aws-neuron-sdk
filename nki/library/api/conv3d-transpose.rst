.. meta::
    :description: 3D Transposed Convolution kernel.
    :date-modified: 08/17/2026

.. currentmodule:: nkilib.experimental.conv

Conv3D Transpose Kernel API Reference
=====================================

3D Transposed Convolution kernel.

Implements ConvTranspose3d operation by delegating to an embedded 3D convolution core with remapped parameters. The transposed convolution is achieved by mapping forward stride to input_dilation, remapping padding to dilation * (K - 1) - padding, and fixing kernel stride at 1.

Background
-----------

The ``conv3d_transpose`` kernel implements a 3D transposed convolution (ConvTranspose3d).

API Reference
--------------

**Source code for this kernel API can be found at**: `conv3d_transpose.py <https://github.com/aws-neuron/nki-library/blob/2.32/src/nkilib_src/nkilib/experimental/conv/conv3d_transpose.py>`_

conv3d_transpose
^^^^^^^^^^^^^^^^

.. py:function:: conv3d_transpose(x_in: nl.NkiTensor, filters: nl.NkiTensor, bias: Optional[nl.NkiTensor] = None, stride: tuple[int, int, int] = (1, 1, 1), padding: tuple[int, int, int] = (0, 0, 0), dilation: tuple[int, int, int] = (1, 1, 1), activation_fn: Optional[ActFnType] = None, lnc_shard: bool = False, filter_shape: str = _FILTER_SHAPE_KDHW_CI_CO, sbm: Optional[SbufManager] = None, use_auto_allocation: bool = False) -> nl.NkiTensor

   3D Transposed Convolution kernel.

   :param x_in: [B, C_in, D, H, W], Input tensor on HBM.
   :type x_in: ``nl.NkiTensor``
   :param filters: Filter weights on HBM with spatial axes flipped. Shape depends on filter_shape: - "KDHW_CI_CO" (default): [K_d, K_h, K_w, C_in, C_out] - "KDHW_CO_CI":           [K_d, K_h, K_w, C_out, C_in]
   :type filters: ``nl.NkiTensor``
   :param bias: [C_out], Optional bias tensor on HBM.
   :type bias: ``Optional[nl.NkiTensor]``
   :param stride: (stride_d, stride_h, stride_w), Convolution strides.
   :type stride: ``tuple[int, int, int]``
   :param padding: (pad_d, pad_h, pad_w), Padding for each spatial dimension.
   :type padding: ``tuple[int, int, int]``
   :param dilation: (dilation_d, dilation_h, dilation_w), Filter dilation factors.
   :type dilation: ``tuple[int, int, int]``
   :param activation_fn: Optional activation function to apply after convolution.
   :type activation_fn: ``Optional[ActFnType]``
   :param lnc_shard: Enable LNC sharding across neuron cores.
   :type lnc_shard: ``bool``
   :param filter_shape: Storage layout of the filters tensor. Default is "KDHW_CI_CO".
   :type filter_shape: ``str``
   :param sbm: Optional caller-provided SBUF manager. When None, the kernel creates its own SbufManager.
   :type sbm: ``Optional[SbufManager]``
   :param use_auto_allocation: Must equal sbm.is_auto_alloc() when sbm is provided. When sbm is None this flag is unused.
   :type use_auto_allocation: ``bool``
   :return: [B, C_out, D_out, H_out, W_out], Output tensor on HBM.
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * B: Batch size
   * C_in: Number of input channels
   * C_out: Number of output channels
   * D: Input depth dimension
   * H: Input height dimension
   * W: Input width dimension
   * K_d: Filter kernel depth
   * K_h: Filter kernel height
   * K_w: Filter kernel width
   * D_out: Output depth = (D - 1) * stride_d + dilation_d * (K_d - 1) - 2 * pad_d + 1
   * H_out: Output height = (H - 1) * stride_h + dilation_h * (K_h - 1) - 2 * pad_h + 1

