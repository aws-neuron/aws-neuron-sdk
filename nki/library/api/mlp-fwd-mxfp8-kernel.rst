.. meta::
    :description: Fused gate/up + SiLU + multiply + down projection using TensorDescriptors.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.mlp_mxfp8.mlp_fwd_mxfp8

MLP Forward MXFP8 Kernel API Reference
=======================================

Fused gate/up + SiLU + multiply + down projection using TensorDescriptors.

Uses HBM for the intermediate tensor. All loads use load_and_quantize_tile with TileLocation/TensorDescriptor. For each M-block of TILES_IN_BLOCK_M tiles: Phase 1: gate/up matmul -> SiLU(gate) * up -> write intermediate to HBM Phase 2: read intermediate from HBM via DGT, matmul with down weights

Background
-----------

The ``compute_fused_gate_up_down_mxfp8`` kernel implements a fused SwiGLU MLP forward pass (gate/up projection, SiLU activation, element-wise multiply, and down projection) using MXFP8 quantized matmuls with TensorDescriptors.

API Reference
--------------

**Source code for this kernel API can be found at**: `mlp_fwd_mxfp8_kernel.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/mlp_mxfp8/mlp_fwd_mxfp8/mlp_fwd_mxfp8_kernel.py>`_

compute_fused_gate_up_down_mxfp8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: compute_fused_gate_up_down_mxfp8(hidden_td: TensorDescriptor, gate_up_td: TensorDescriptor, down_w_td: TensorDescriptor, int_td: TensorDescriptor, output_td: TensorDescriptor, s_base_offset: int, dtype, TILES_IN_BLOCK_M: int = 8, TILES_IN_BLOCK_N_GU: int = 1, TILES_IN_BLOCK_K_GU: int = 8, TILES_IN_BLOCK_M_DOWN: int = 8, TILES_IN_BLOCK_N_DOWN: int = 1, TILES_IN_BLOCK_K_DOWN: int = 8, save_gate_pre_td: TensorDescriptor = None, save_gate_act_td: TensorDescriptor = None, save_up_td: TensorDescriptor = None, spill_reload: bool = True, use_scale_packing: bool = True, run_with_lnc2: bool = True)

   Fused gate/up + SiLU + multiply + down projection using TensorDescriptors.

   :param hidden_td: [S, H], input hidden states (is_f_by_k=True).
   :type hidden_td: ``TensorDescriptor``
   :param gate_up_td: [2I, H], fused gate+up weight matrix (is_f_by_k=True).
   :type gate_up_td: ``TensorDescriptor``
   :param down_w_td: [H, I], down projection weights (is_f_by_k=True).
   :type down_w_td: ``TensorDescriptor``
   :param int_td: [S, I], scratch buffer for gated intermediate activations (is_f_by_k=True).
   :type int_td: ``TensorDescriptor``
   :param output_td: [S_local, H], output buffer (may be a slice for LNC sharding).
   :type output_td: ``TensorDescriptor``
   :param s_base_offset: Row offset into the full [S, ...] tensors for this LNC core.
   :type s_base_offset: ``int``
   :param dtype: Output data type (e.g. nl.bfloat16).
   :param TILES_IN_BLOCK_M: Number of M tiles per block for gate/up phase.
   :type TILES_IN_BLOCK_M: ``int``
   :param TILES_IN_BLOCK_N_GU: Number of N tiles per block for gate/up phase.
   :type TILES_IN_BLOCK_N_GU: ``int``
   :param TILES_IN_BLOCK_K_GU: Number of K tiles per block for gate/up phase.
   :type TILES_IN_BLOCK_K_GU: ``int``
   :param TILES_IN_BLOCK_M_DOWN: Number of M tiles per block for down phase.
   :type TILES_IN_BLOCK_M_DOWN: ``int``
   :param TILES_IN_BLOCK_N_DOWN: Number of N tiles per block for down phase.
   :type TILES_IN_BLOCK_N_DOWN: ``int``
   :param TILES_IN_BLOCK_K_DOWN: Number of K tiles per block for down phase.
   :type TILES_IN_BLOCK_K_DOWN: ``int``
   :param save_gate_pre_td: [S, I], optional TD to checkpoint gate pre-activation, or None.
   :type save_gate_pre_td: ``TensorDescriptor``
   :param save_gate_act_td: [S, I], optional TD to checkpoint SiLU(gate_pre), or None.
   :type save_gate_act_td: ``TensorDescriptor``
   :param save_up_td: [S, I], optional TD to checkpoint up projection, or None.
   :type save_up_td: ``TensorDescriptor``

   **Dimensions**:

   * S: Sequence length (number of tokens).
   * H: Hidden dimension size.

mlp_forward_mxfp8_nki
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: mlp_forward_mxfp8_nki(hidden: nl.ndarray, gate_up_weights: nl.ndarray, down_weights: nl.ndarray, intermediate_hbm: nl.ndarray, run_with_lnc2: bool = True, gate_up_tiles_m: int = 8, gate_up_tiles_n: int = 1, gate_up_tiles_k: int = 8, down_tiles_m: int = 8, down_tiles_n: int = 1, down_tiles_k: int = 8, fp8_x4_dtype = float8_e4m3fn_x4, save_gate_pre: nl.ndarray = None, save_gate_act: nl.ndarray = None, save_up: nl.ndarray = None, save_hidden: nl.ndarray = None, dtype = nl.bfloat16, spill_reload: bool = True, use_scale_packing: bool = True, hidden_scales: nl.ndarray = None, gate_up_scales: nl.ndarray = None, down_scales: nl.ndarray = None, hidden_is_swizzled: bool = False, gate_up_is_swizzled: bool = False, down_is_swizzled: bool = False) -> nl.ndarray

   MXFP8 SwiGLU MLP forward pass with optional activation checkpointing.

   :param hidden: [S, H], input hidden states.
   :type hidden: ``nl.ndarray``
   :param gate_up_weights: [2I, H], fused weight matrix — rows [0:I] = W_gate, rows [I:2I] = W_up.
   :type gate_up_weights: ``nl.ndarray``
   :param down_weights: [H, I], down projection weights (W_down).
   :type down_weights: ``nl.ndarray``
   :param intermediate_hbm: [S, I], scratch buffer for gated intermediate activations.
   :type intermediate_hbm: ``nl.ndarray``
   :param run_with_lnc2: Whether to shard across 2 LNC cores.
   :type run_with_lnc2: ``bool``
   :param gate_up_tiles_m: Number of M tiles per block for gate/up phase.
   :type gate_up_tiles_m: ``int``
   :param gate_up_tiles_n: Number of N tiles per block for gate/up phase.
   :type gate_up_tiles_n: ``int``
   :param gate_up_tiles_k: Number of K tiles per block for gate/up phase.
   :type gate_up_tiles_k: ``int``
   :param down_tiles_m: Number of M tiles per block for down phase.
   :type down_tiles_m: ``int``
   :param down_tiles_n: Number of N tiles per block for down phase.
   :type down_tiles_n: ``int``
   :param down_tiles_k: Number of K tiles per block for down phase.
   :type down_tiles_k: ``int``
   :param fp8_x4_dtype: MXFP8 quantized data type for nc_matmul_mx.
   :param save_gate_pre: [S, I], HBM buffer to checkpoint gate pre-activation, or None.
   :type save_gate_pre: ``nl.ndarray``
   :param save_gate_act: [S, I], HBM buffer to checkpoint SiLU(gate_pre), or None.
   :type save_gate_act: ``nl.ndarray``
   :param save_up: [S, I], HBM buffer to checkpoint up projection, or None.
   :type save_up: ``nl.ndarray``
   :param save_hidden: [S, I], HBM buffer to checkpoint gate_act * up, or None (same data as intermediate_hbm but kept as a separate named output for clarity in the fwd/bwd contract).
   :type save_hidden: ``nl.ndarray``
   :param dtype: Output data type (e.g. nl.bfloat16).
   :param spill_reload: Whether to spill quantized operands to HBM for reload across K-blocks.
   :type spill_reload: ``bool``
   :param use_scale_packing: Whether to pack MXFP8 scales into compact format.
   :type use_scale_packing: ``bool``
   :param hidden_scales: MXFP8 scales for pre-quantized hidden, or None (raw BF16).
   :type hidden_scales: ``nl.ndarray``
   :param gate_up_scales: MXFP8 scales for pre-quantized gate_up_weights, or None.
   :type gate_up_scales: ``nl.ndarray``
   :param down_scales: MXFP8 scales for pre-quantized down_weights, or None.
   :type down_scales: ``nl.ndarray``
   :param hidden_is_swizzled: True if hidden is pre-swizzled [K/4, F*4] BF16.
   :type hidden_is_swizzled: ``bool``
   :param gate_up_is_swizzled: True if gate_up_weights is pre-swizzled.
   :type gate_up_is_swizzled: ``bool``
   :param down_is_swizzled: True if down_weights is pre-swizzled.
   :type down_is_swizzled: ``bool``
   :return: [S, H], MLP output hidden states.
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * S: Sequence length (number of tokens).
   * H: Hidden dimension size.

