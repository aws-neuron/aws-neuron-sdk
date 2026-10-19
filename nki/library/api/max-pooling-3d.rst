.. meta::
    :description: 3D max pool kernel over the (D, H, W) axes. Pooling runs independently per (batch, channel).
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.pooling

Max Pooling 3D Kernel API Reference
===================================

3D max pool kernel over the (D, H, W) axes. Pooling runs independently per (batch, channel).

Background
-----------

The ``max_pooling_3d`` kernel computes 3D max pooling over the (D, H, W) axes, independently for each (batch, channel) pair. It supports channels-first (NCDHW) and channels-last (NDHWC) layouts.

API Reference
--------------

**Source code for this kernel API can be found at**: `max_pooling_3d.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/pooling/max_pooling_3d.py>`_

max_pooling_3d
^^^^^^^^^^^^^^

.. py:function:: max_pooling_3d(src_tensor: nl.NkiTensor, kernel_size: Union[int, tuple], stride: Optional[Union[int, tuple]] = None, padding: Union[int, tuple] = 0, data_format: str = 'NCDHW', output_format: Optional[str] = None) -> nl.NkiTensor

   3D max pool kernel over the (D, H, W) axes. Pooling runs independently per (batch, channel).

   :param src_tensor: Input in HBM.  Shape is (B, C, D, H, W) for data_format="NCDHW" (channels-first) or (B, D, H, W, C) for data_format="NDHWC" (channels-last).
   :type src_tensor: ``nl.NkiTensor``
   :param kernel_size: Pooling window (kD, kH, kW) (int broadcasts).
   :type kernel_size: ``Union[int, tuple]``
   :param stride: Stride (sD, sH, sW); defaults to kernel_size.
   :type stride: ``Optional[Union[int, tuple]]``
   :param padding: Implicit -inf padding (pD, pH, pW).
   :type padding: ``Union[int, tuple]``
   :param data_format: Input layout, "NCDHW" (default) or "NDHWC".
   :type data_format: ``str``
   :param output_format: Output layout, "NCDHW" or "NDHWC"; defaults to data_format (output matches input).
   :type output_format: ``Optional[str]``
   :return: Pooled result in HBM, in output_format layout: (B, C, D_out, H_out, W_out) or (B, D_out, H_out, W_out, C) with out[a] = (in[a] + 2*pad[a] - kernel[a]) // stride[a] + 1.
   :rtype: ``nl.NkiTensor``

   **Dimensions**:

   * B: batch.

