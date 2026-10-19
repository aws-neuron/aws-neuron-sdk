.. meta::
    :description: Multi-scale deformable attention backward pass kernel.
    :date-modified: 10/19/2026

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

**Source code for this kernel API can be found at**: `ms_deformable_attention_bwd.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/deformable_attention/ms_deformable_attention_bwd.py>`_

ms_deformable_attention_bwd
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: ms_deformable_attention_bwd(grad_output: nl.NkiTensor, value: nl.NkiTensor, spatial_shapes: tuple, level_start_index: tuple, sampling_locations: nl.NkiTensor, attention_weights: nl.NkiTensor, value_layout: str = 'BLNC', sampling_locations_layout: str = 'BQHLP2', align_corners: bool = False, padding_mode: str = 'zeros', max_gather_indices_per_indirect: Optional[int] = None, max_scatter_indices_per_indirect: Optional[int] = None, gather_method: str = 'transpose', num_scatter_buffers: Optional[int] = None, compute_grad_value: bool = True, compute_grad_sampling_locations: bool = True, compute_grad_attention_weights: bool = True, iota_k_scale_in: Optional[nl.NkiTensor] = None, grad_value_buffers_in: Optional[nl.NkiTensor] = None, prev_reduced_grad_value: Optional[nl.NkiTensor] = None, dump_iota_k_scale: bool = False, dump_grad_value_buffers: bool = False, dump_reduced_grad_value: bool = False) -> Tuple[nl.NkiTensor, ...]

   Multi-scale deformable attention backward pass kernel.

   :param grad_output: Gradient from downstream in HBM, shape (B, N_q, N_h * C_h)
   :type grad_output: ``nl.NkiTensor``
   :param value: Value tensor in HBM. Shape depends on value_layout: - If value_layout="BLNC": (B, L, N_h, C_h) - If value_layout="BNLC": (B, N_h, L, C_h)
   :type value: ``nl.NkiTensor``
   :param spatial_shapes: Tuple of (H_i, W_i) tuples specifying spatial dimensions for each level
   :type spatial_shapes: ``tuple``
   :param level_start_index: Tuple of start indices for each level in the flattened L dimension
   :type level_start_index: ``tuple``
   :param sampling_locations: Normalized sampling coordinates in HBM. Shape depends on layout: - If sampling_locations_layout="BQHLP2": (B, N_q, N_h, N_l, N_p, 2) - If sampling_locations_layout="B2QHLP": (B, 2, N_q, N_h, N_l, N_p)
   :type sampling_locations: ``nl.NkiTensor``
   :param attention_weights: Attention weights in HBM, shape (B, N_q, N_h, N_l, N_p)
   :type attention_weights: ``nl.NkiTensor``
   :param value_layout: Layout of value tensor, either "BLNC" or "BNLC". Default: "BLNC"
   :type value_layout: ``str``
   :param sampling_locations_layout: Layout of sampling_locations, either "BQHLP2" or "B2QHLP". Default: "BQHLP2"
   :type sampling_locations_layout: ``str``
   :param align_corners: If True, coordinates map [0,1] to [0, H-1]. If False, map to [-0.5, H-0.5]. Default: False
   :type align_corners: ``bool``
   :param padding_mode: Padding mode for out-of-bounds coordinates, either "zeros" or "border". Default: "zeros"
   :type padding_mode: ``str``
   :param max_gather_indices_per_indirect: Cap on gather indices (P_MAX * M_batch) per batched gather. Setting this (not None) enables the batched indirect gather path (M_batch = cap // P_MAX); None disables it (each (head, level) gathered separately). Default: None
   :type max_gather_indices_per_indirect: ``Optional[int]``
   :param max_scatter_indices_per_indirect: Controls the scatter-add indirection budget. If None, scatters over all P_MAX query partitions with no column batching. If set and <= P_MAX, enables the "fewer indirections" path: at most cap query partitions are scattered per dma_compute. Default: None
   :type max_scatter_indices_per_indirect: ``Optional[int]``
   :param gather_method: How values are gathered: "transpose" (dma_transpose to [channel, query] then nc_transpose to [query, channel]) or "copy" (indirect dma_copy to [query, channel]). Scatter always uses dma_compute. Default: "transpose"
   :type gather_method: ``str``
   :param num_scatter_buffers: Number of K-replicated scatter buffer sets held in HBM. Defaults to 1 when None.
   :type num_scatter_buffers: ``Optional[int]``
   :param compute_grad_value: Compute ``grad_value``. At least one of the three ``compute_grad_*`` flags must be True. Default: True.
   :type compute_grad_value: ``bool``
   :param compute_grad_sampling_locations: Compute ``grad_sampling_locations``. Default: True.
   :type compute_grad_sampling_locations: ``bool``
   :param compute_grad_attention_weights: Compute ``grad_attention_weights``. Default: True.
   :type compute_grad_attention_weights: ``bool``
   :param iota_k_scale_in: Precomputed K-scaling index tensor in HBM, shape (P_MAX, k_scale_cols), uint32. If given, the kernel copies it in instead of building it. Default: None
   :type iota_k_scale_in: ``Optional[nl.NkiTensor]``
   :param grad_value_buffers_in: Persistent K replicated scatter buffers in HBM, shape (lnc*K, L, N_h, C_h) (BLNC) / (lnc*K, N_h, L, C_h) (BNLC). If given, the kernel accumulates into it and skips zeroing; if None, the buffers are allocated and zeroed. Default: None
   :type grad_value_buffers_in: ``Optional[nl.NkiTensor]``
   :param prev_reduced_grad_value: Reduced cumulative grad_value through the previous call, shape/layout of a single value map (L, N_h, C_h) / (N_h, L, C_h). When given, grad_value = reduce(K-buffers) - prev_reduced_grad_value, recovering this call's own contribution. Default: None (grad_value = reduce(K-buffers)).
   :type prev_reduced_grad_value: ``Optional[nl.NkiTensor]``
   :param dump_iota_k_scale: If True, also return the built/ingested iota_k_scale in HBM so a later call can pass it as iota_k_scale_in. Default: False
   :type dump_iota_k_scale: ``bool``
   :param dump_grad_value_buffers: If True, also return the raw K replicated scatter buffers in HBM so a later call can pass them as grad_value_buffers_in. Default: False
   :type dump_grad_value_buffers: ``bool``
   :param dump_reduced_grad_value: If True, also return reduce(K-buffers) (cumulative through this call) in HBM so a later call can pass it as prev_reduced_grad_value. Default: False
   :type dump_reduced_grad_value: ``bool``
   :return: Gradient w.r.t. value in HBM, same shape and layout as input value
   :return: Gradient w.r.t. sampling_locations in HBM, same shape and layout as input
   :return: Gradient w.r.t. attention_weights in HBM, shape (B, N_q, N_h, N_l, N_p) Followed, in this fixed order, by whichever of these are enabled: iota_k_scale (dump_iota_k_scale), grad_value_buffers (dump_grad_value_buffers), reduced_grad_value (dump_reduced_grad_value).
   :rtype: ``Tuple[nl.NkiTensor, ...]``

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

