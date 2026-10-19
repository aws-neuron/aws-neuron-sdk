.. meta::
    :description: 3D Convolution backward pass.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.conv

Conv3D Backward Kernel API Reference
====================================

3D Convolution backward pass.

Computes any subset of {dx, dw, db} for a 3D convolution forward pass. The planner selects a strategy and per-gradient compute method from the problem geometry.

Background
-----------

The ``conv3d_bwd`` kernel computes any subset of the input, weight, and bias gradients (``dx``, ``dw``, ``db``) of a 3D convolution. A planner selects the strategy and per-gradient compute method from the problem geometry.

API Reference
--------------

**Source code for this kernel API can be found at**: `conv3d_bwd.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/conv/conv3d_bwd.py>`_

conv3d_bwd
^^^^^^^^^^

.. py:function:: conv3d_bwd(dy: nl.NkiTensor, x_in: nl.NkiTensor, filters: nl.NkiTensor, stride: tuple[int, int, int] = (1, 1, 1), padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0), dilation: tuple[int, int, int] = (1, 1, 1), compute_dx: bool = True, compute_dw: bool = True, compute_db: bool = True) -> tuple[nl.NkiTensor, ...]

   3D Convolution backward pass.

   :param dy: [B, C_out, D_out, H_out, W_out], Output gradient on HBM.
   :type dy: ``nl.NkiTensor``
   :param x_in: [B, C_in, D, H, W], Forward input on HBM.
   :type x_in: ``nl.NkiTensor``
   :param filters: [K_d, K_h, K_w, C_in, C_out], Forward filters on HBM.
   :type filters: ``nl.NkiTensor``
   :param stride: (stride_d, stride_h, stride_w), Forward convolution strides.
   :type stride: ``tuple[int, int, int]``
   :param padding: (pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right), Forward padding.
   :type padding: ``tuple[int, int, int, int, int, int]``
   :param dilation: (dilation_d, dilation_h, dilation_w), Forward dilation factors.
   :type dilation: ``tuple[int, int, int]``
   :param compute_dx: Whether to compute the input gradient dx.
   :type compute_dx: ``bool``
   :param compute_dw: Whether to compute the filter gradient dw.
   :type compute_dw: ``bool``
   :param compute_db: Whether to compute the bias gradient db.
   :type compute_db: ``bool``

   **Notes**:

   * Supports the same 6-value (possibly asymmetric) padding as the forward conv3d kernel for all of dx, dw, and db.
   * The planner selects the strategy and per-gradient compute method from the problem geometry.

   **Dimensions**:

   * B: Batch size
   * C_in: Number of input channels

