.. meta::
    :description: Multi-scale deformable attention backward pass kernel.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.deformable_attention

MS Deformable Attention Backward Kernel API Reference
=====================================================

Multi-scale deformable attention backward pass kernel.

Computes gradients with respect to value, sampling_locations, and attention_weights given the downstream gradient.

Background
-----------

The ``ms_deformable_attention_bwd`` kernel computes the backward pass for multi-scale deformable attention, producing gradients with respect to ``value``, ``sampling_locations``, and ``attention_weights`` from the downstream gradient using bilinear-interpolation derivatives. It supports both ``BLNC``/``BNLC`` value layouts and ``BQHLP2``/``B2QHLP`` sampling-location layouts.

API Reference
--------------

**Source code for this kernel API can be found at**: `ms_deformable_attention_bwd.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/deformable_attention/ms_deformable_attention_bwd.py>`_

ms_deformable_attention_bwd
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: ms_deformable_attention_bwd(grad_output: nl.ndarray, value: nl.ndarray, spatial_shapes: tuple, level_start_index: tuple, sampling_locations: nl.ndarray, attention_weights: nl.ndarray, value_layout: str = 'BLNC', sampling_locations_layout: str = 'BQHLP2', align_corners: bool = False, padding_mode: str = 'zeros') -> Tuple[nl.ndarray, nl.ndarray, nl.ndarray]

   Multi-scale deformable attention backward pass kernel.

   :param grad_output: Gradient from downstream in HBM, shape (B, N_q, N_h * C_h)
   :type grad_output: ``nl.ndarray``
   :param value: Value tensor in HBM. Shape depends on value_layout: - If value_layout="BLNC": (B, L, N_h, C_h) - If value_layout="BNLC": (B, N_h, L, C_h)
   :type value: ``nl.ndarray``
   :param spatial_shapes: Tuple of (H_i, W_i) tuples specifying spatial dimensions for each level
   :type spatial_shapes: ``tuple``
   :param level_start_index: Tuple of start indices for each level in the flattened L dimension
   :type level_start_index: ``tuple``
   :param sampling_locations: Normalized sampling coordinates in HBM. Shape depends on layout: - If sampling_locations_layout="BQHLP2": (B, N_q, N_h, N_l, N_p, 2) - If sampling_locations_layout="B2QHLP": (B, 2, N_q, N_h, N_l, N_p)
   :type sampling_locations: ``nl.ndarray``
   :param attention_weights: Attention weights in HBM, shape (B, N_q, N_h, N_l, N_p)
   :type attention_weights: ``nl.ndarray``
   :param value_layout: Layout of value tensor, either "BLNC" or "BNLC". Default: "BLNC"
   :type value_layout: ``str``
   :param sampling_locations_layout: Layout of sampling_locations, either "BQHLP2" or "B2QHLP". Default: "BQHLP2"
   :type sampling_locations_layout: ``str``
   :param align_corners: If True, coordinates map [0,1] to [0, H-1]. If False, map to [-0.5, H-0.5]. Default: False
   :type align_corners: ``bool``
   :param padding_mode: Padding mode for out-of-bounds coordinates, either "zeros" or "border". Default: "zeros"
   :type padding_mode: ``str``
   :return: Gradient w.r.t. value in HBM, same shape and layout as input value
   :rtype: ``nl.ndarray``
   :return: Gradient w.r.t. sampling_locations in HBM, same shape and layout as input
   :rtype: ``nl.ndarray``
   :return: Gradient w.r.t. attention_weights in HBM, shape (B, N_q, N_h, N_l, N_p)
   :rtype: ``nl.ndarray``

   **Notes**:

   * Computes actual gradients using bilinear interpolation derivatives
   * Supports both BLNC and BNLC value layouts
   * Supports both BQHLP2 and B2QHLP sampling_locations layouts
   * Padding modes: "zeros" (OOB returns 0) and "border" (clamps to edge)

   **Dimensions**:

   * B: Batch size
   * N_q: Number of queries
   * N_h: Number of attention heads
   * C_h: Channels per head
   * N_l: Number of feature pyramid levels
   * N_p: Number of sampling points per query per head per level
   * L: Total flattened spatial dimension (sum of H_i * W_i across all levels)
   * H_i: Height of feature map at level i

