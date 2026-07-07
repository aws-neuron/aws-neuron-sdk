.. meta::
    :description: Complete release notes for the NKI Library component across all AWS Neuron SDK versions.
    :keywords: nki library, nki-lib, release notes, aws neuron sdk
    :date-modified: 06/11/2026

.. _nki-lib_rn:

Release Notes for Neuron Component: NKI Library
================================================

The release notes for the NKI Library Neuron component. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _nki-lib-2-31-0-rn:

NKI Library (NKI-Lib) (Neuron 2.31.0 Release)
--------------------------------------------------------------------

Date of Release: 07/07/2026

What's New
~~~~~~~~~~

This release adds 14 new experimental kernels spanning deformable attention (forward and backward), indexed gather/scatter, MoE training collectives (all-to-all-v dispatch/combine, batch sharding, HBM collectives), ring attention unpermute, fused RMSNorm + router top-K, MXFP8 blockwise MoE backward, DeepSeek MLA QKV projection, and a block-sharded MoE context-encoding kernel. The KV-parallel segmented prefill attention kernel has been renamed from ``kv_parallel_segmented_prefill`` to ``attention_kv_parallel_segmented_cte`` (see Breaking Changes). Existing kernels gain a unified ``dtype_mode`` precision selector, FP8-packed KV paths (``fp8_packed``), expanded MX quantization, position-bias support in attention, and additional context-parallel KV options. PyTorch reference implementations are added for 44 kernels. This release also includes a major rewrite of the experimental MXFP8 MLP backward kernel — see Breaking Changes.

New Experimental Kernels
^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`MS Deformable Attention </nki/library/api/ms-deformable-attention>` — Multi-scale deformable attention that samples values from multiple feature-pyramid levels at learned locations using bilinear interpolation and indirect DMA transpose.
* :doc:`MS Deformable Attention Backward </nki/library/api/ms-deformable-attention-bwd>` — Backward pass for multi-scale deformable attention, computing gradients w.r.t. value, sampling locations, and attention weights.
* :doc:`Gather </nki/library/api/gather>` — Gathers rows from a 2D input by a 1D index using indirect DMA load (equivalent to PyTorch ``input[index]`` on dim 0).
* :doc:`Scatter-Add </nki/library/api/scatter-add>` — Scatter-adds rows from ``src`` into a 2D input by a 1D index (equivalent to PyTorch ``input.scatter_add(dim=0, ...)``).
* :doc:`Permute A2AV </nki/library/api/permute-a2av>` — MoE training dispatch: permutes tokens by destination EP rank and exchanges them via ``all_to_all_v``.
* :doc:`Unpermute A2AV </nki/library/api/unpermute-a2av>` — MoE training combine: exchanges expert output via ``all_to_all_v`` and unpermutes tokens back to their original order (supports top-k > 1).
* :doc:`QKV Batch Shard </nki/library/api/batch-shard>` — Q projection transition from a tensor-parallel layout to a tensor-and-data-parallel layout for batch-sharded attention.
* :doc:`Collective Communication Kernels </nki/library/api/collectives>` — HBM-based ``all_reduce``, ``all_gather``, ``reduce_scatter``, and ``all_to_all`` kernels plus per-rank slice-selection helpers.
* :doc:`Ring Attention Unpermute </nki/library/api/ring-attention-unpermute>` — Reorders striped ring-attention output back to contiguous sequence order after context-parallel attention.
* :doc:`RMSNorm Router Top-K A2AV </nki/library/api/rmsnorm-router-topk-a2av>` — Fused RMSNorm + router top-K for MoE token generation with small token counts, feeding the all-to-all-v dispatch path.
* :doc:`RMSNorm Router Top-K TKG </nki/library/api/rmsnorm-router-topk-tkg>` — Fused RMSNorm (+ optional MX quantize) + router top-K for MoE token generation.
* :doc:`Blockwise MM Backward MXFP8 </nki/library/api/blockwise-mm-backward-mxfp8>` — MXFP8-quantized backward pass for blockwise Mixture of Experts, computing gradients for all parameters per expert-assigned block.
* :doc:`QKV CTE MLA </nki/library/api/qkv-cte-mla>` — DeepSeek Multi-head Latent Attention (MLA) QKV projection with MX (fp8) quantization for context encoding, including two-stage low-rank projections, fused RMSNorm, and RoPE.
* :doc:`Blockwise MM Shard-on-Block </nki/library/api/bwmm-shard-on-block-v2>` — Block-sharded blockwise matrix multiplication for context-encoding MoE layers with a hybrid static/dynamic loop structure.

Improvements
~~~~~~~~~~~~~~~

* :doc:`Attention CTE </nki/library/api/attention-cte>`: Added ``position_bias``, ``bias_layout``, and ``bias_band_params`` parameters for additive position-bias attention.
* :doc:`Attention TKG </nki/library/api/attention-tkg>`: Added ``max_context_len`` and ``dtype_mode`` parameters.
* :doc:`Attention Segmented CTE </nki/library/api/attention-segmented-cte>`: Added ``fp8_packed`` plus KV-parallel context-parallel parameters (``kvp_q_offset``, ``kvp_rank_id``, ``kvp_group_size``, and prior-block bookkeeping).
* :doc:`MLP </nki/library/api/mlp>`: Added ``use_contiguous_x4_gate_up``, ``dtype_mode``, and ``gate_up_w_layout`` parameters for expanded layout and precision control.
* :doc:`MoE CTE </nki/library/api/moe-cte>`: Added ``gate_up_in_scale`` and ``down_in_scale`` input-scale parameters.
* :doc:`MoE TKG </nki/library/api/moe-tkg>`: Added ``output_layout``, ``output``, and ``dtype_mode`` parameters.
* :doc:`QKV </nki/library/api/qkv>`: Added ``fp8_packed`` and ``dtype_mode`` parameters (also added to the CTE and TKG variants).
* :doc:`Output Projection CTE </nki/library/api/output-projection-cte>`: Added ``dtype_mode`` and ``compact_weight_scales`` parameters.
* :doc:`Output Projection TKG </nki/library/api/output-projection-tkg>`: Added ``dtype_mode`` parameter.
* :doc:`RMSNorm-Quant </nki/library/api/rmsnorm-quant>`: Added ``dtype_mode`` parameter.
* :doc:`Ring Attention Forward </nki/library/api/ring-attention-fwd>` / :doc:`Backward </nki/library/api/ring-attention-bwd>`: Added ``bound_min``/``bound_max`` parameters for sequence-packing KV range bounds.
* :doc:`Attention Block TKG </nki/library/api/attention-block-tkg>`: Added ``fp8_packed`` (adds support for a packed FP8 layout for the K cache with an efficient DMA-transpose load), ``KVDP_rank`` (enables use of non-contiguous replica groups with KVDP), ``max_context_len`` (enables dynamic loops to stop execution once all sequences are done), and ``dtype_mode`` parameters.
* :doc:`Transformer TKG </nki/library/api/transformer-tkg>`: Added ``dtype_mode`` parameter.
* :doc:`Matmul MXFP8 </nki/library/api/matmul-mxfp8-generic-kernel>`: Added ``load_with_PE_swizzle``, ``lhs_is_f_by_k``, and ``rhs_is_f_by_k`` parameters.
* :doc:`Blockwise MM Backward </nki/library/api/blockwise-mm-backward>`: Added gradient output-tensor parameters (``hidden_states_grad_out``, ``expert_affinities_masked_grad_out``, ``gate_up_proj_weight_grad_out``, ``down_proj_weight_grad_out``) and ``accumulation_dtype``.
* :doc:`Top-K Reduce </nki/library/api/topk-reduce>`: Added ``token_base_index`` parameter.
* Added a unified ``dtype_mode`` precision selector across attention, QKV, MLP, MoE TKG, output projection, RMSNorm-Quant, and transformer kernels.
* Added PyTorch reference implementations for 44 kernels for testing and validation.

Breaking Changes
~~~~~~~~~~~~~~~~

* :doc:`Attention KV-Parallel Segmented CTE </nki/library/api/attention-kv-parallel-segmented-cte>`: The ``kv_parallel_segmented_prefill`` kernel has been renamed to ``attention_kv_parallel_segmented_cte`` (file ``core/attention/kv_parallel_segmented_prefill.py`` → ``core/attention/attention_kv_parallel_segmented_cte.py``). Callers must update both their import path and the function name.
* :doc:`Attention BWD </nki/library/api/attention-cte>`: In ``load_kv``, ``load_q_dy``, and ``compute_rowsum_single_tile``, the single-head-dim parameters ``d_head_n_tiles``, ``d_head_tile_size``, and ``offset`` have been removed and replaced by separate QK/V variants (``d_head_qk_n_tiles``, ``d_head_v_n_tiles``, ``d_head_qk_tile_size``, ``d_head_v_tile_size``, ``offset_k``/``offset_v``/``offset_q``/``offset_dy``). New parameters are inserted in the middle of these signatures, so positional callers beyond the head-dim arguments must switch to keyword arguments.
* :doc:`Attention Segmented CTE </nki/library/api/attention-segmented-cte>`: The ``kvp_offset`` parameter has been removed from ``attention_segmented_cte`` and ``fused_segmented_attention_impl``, replaced by the explicit context-parallel offsets ``kvp_cp_offset_int`` and ``kvp_seg_block_offset_int``. Callers using ``kvp_offset`` must migrate.
* :doc:`MoE TKG </nki/library/api/moe-tkg>`: The ``outp_layout`` parameter of ``moe_tkg`` has been renamed to ``output_layout``. The ``global_token_indices`` parameter has been removed from ``down_projection_mx``. Callers using the old names must update.
* :doc:`MoE CTE </nki/library/api/moe-cte>`: The ``use_dma_tp`` parameter has been removed from ``sbuf_layout_adapter``. In ``bwmm_shard_on_block_mx``, the ``dbg_tensors`` debug parameter has been removed from ``compute_one_block``, ``process_static_blocks``, and ``process_dynamic_blocks``, the ``n_dynamic_blocks`` default changed from ``55`` to ``-1``, and new parameters (``top_k``, ``ep_degree``) are inserted in the middle of ``bwmm_shard_on_block_mx``, shifting positional arguments.
* :doc:`RMSNorm-Quant </nki/library/api/rmsnorm-quant>`: The ``auto_resolve_fp8_dtype`` parameter has been removed from ``rmsnorm_quant_kernel`` and ``build_rms_norm_quant_constants``; precision is now selected via ``dtype_mode``.
* :doc:`Router Top-K </nki/library/api/router-topk>`: The ``name`` parameter has been removed from ``router_topk_input_w_load``.
* :doc:`MLP </nki/library/api/mlp>`: The new ``use_contiguous_x4_gate_up`` parameter is inserted before ``gate_clamp_upper_limit``, shifting positional arguments. Callers using positional arguments beyond that point must switch to keyword arguments. The ``output`` parameter was also removed from the internal ``input_norm_load`` helper, and ``src_proj_int_dim_size`` was replaced by ``src_proj_int_dim_tile_count`` in ``calc_batch_seqlen_dim_tile_size``.
* :doc:`QKV </nki/library/api/qkv>`: The new ``fp8_packed`` parameter is inserted before ``block_size`` in both ``qkv`` and ``qkv_cte``, shifting positional arguments for callers not using keyword arguments.
* :doc:`Attention Block TKG </nki/library/api/attention-block-tkg>`: The new ``fp8_packed`` parameter is inserted before ``k_scale``, shifting positional arguments for callers not using keyword arguments.
* :doc:`Matmul MXFP8 </nki/library/api/matmul-mxfp8-generic-kernel>`: The ``output_dtype`` default changed from ``nl.float32`` to ``nl.bfloat16``, and the ``lnc_2_shard_rhs`` default changed from ``True`` to ``None`` (auto-select to shard the larger output dimension). Callers relying on the previous defaults must set these explicitly.
* :doc:`MLP Backward MXFP8 </nki/library/api/mlp-bwd-mxfp8-kernel>`: The kernel's configuration API was redesigned. The ``BlockConfig`` and ``MatmulConfig`` classes and the ``get_autotuned_config``, ``get_config_for_shape``, and ``recompute_hidden`` functions were removed; per-phase tuning is now passed via ``MatmulMxfp8KernelConfig`` objects (``phase1_config`` … ``phase4_config``). The many explicit per-phase tile-count, scratch-buffer, weight-scale, and ``*_is_swizzled`` arguments to ``mlp_backward_mxfp8_nki``/``mlp_backward_mxfp8_base_nki`` were removed in favor of the config objects and consolidated tensor arguments. Callers of this experimental kernel must migrate to the new config-object interface.

Bug Fixes
~~~~~~~~~

* **Attention Segmented CTE**: Added a fallback path for packed FP8 in the segmented prefill kernel; fixed NaNs from reading uninitialized memory; fixed prescale accuracy when ``q_active <= 512``; added batching support for sinks.
* **Attention CTE/TKG**: Run ``cascaded_max`` index path in fp32 to avoid bf16 rounding to out-of-range tokens; reserve PSUM bank 7 for ``nc_transpose`` when the env var is set; pass the full tensor to ``tp_broadcast`` instead of pre-indexing the free dim in attention TKG.
* **QKV**: Support BxS not divisible by 4 in QKV TKG MX/STATIC_MX/ROW_MX and output-projection TKG MX paths; accept weight HBM tensors with torch element type in ``qkv_cte``/``qkv_tkg`` STATIC_MX paths; skip Q output allocation and store when ``num_q_heads=0``.
* **MoE CTE**: Fixed 3D destination shape in the fused gate+up FP8 path of shard-on-I CTE; fixed out-of-bounds when processing only one block in ``bwmm_shard_on_block_mx``; zero-init partial-write SBUF allocations in shard-on-block.
* **MoE TKG**: Replaced software DGE with hardware DGE for DMA in dynamic loops; corrected down-matmul activation indexing in the Llama3-70B high-batch double-row path.
* **MoE Backward**: Affinity-weight the MLP2 (down-projection) bias gradient, including the MXFP8 path.
* **Router Top-K**: Bound the ``nc_n_gather`` read range in rotational top-K to prevent uninitialized SBUF reads; sort rotational top-K output when ``sorted=True`` regardless of padding; use ``nl.maximum`` instead of ``nl.max`` in row quantization.
* **MLP CTE**: Fixed access-pattern indexing.
* **MLP Backward MXFP8**: Use BFLOAT16 SBUF sizing and per-matmul config; cap ``TILES_IN_LOAD_M/N`` by the DGT transpose limit.
* **Ring Attention**: Use indirect DMA instead of ``register_store`` for rank-ID load in ring attention forward; chunk the forward/backward torch references to avoid OOM.
* **Conv3D**: Fixed memory configuration in cases where total memory exceeded SBUF.
* **RMSNorm MX Quantize TKG**: Refactored ``_spill_tiled_sb_to_hbm`` with ``TensorView`` to fix partition-stride validation.
* **KVDP Attention**: Support strided replica groups.
* **Top-K Reduce**: Updated to use 1-indexed token indices and handle sparse K.

Known Issues
~~~~~~~~~~~~


.. _nki-lib-2-30-0-rn:

NKI Library (NKI-Lib) (Neuron 2.30.0 Release)
--------------------------------------------------------------------

Date of Release: 05/21/2026

What's New
~~~~~~~~~~

This release adds 19 new experimental kernels spanning attention, convolution, MLP training, MoE, optimizer, padding, quantization, RNG, state-space models, and collective communication subkernels. It also introduces 3 new core kernels including segmented attention with block-based KV cache, KV-parallel prefill, and FP8 quantization. Existing kernels receive context parallelism support, QK-norm fusion, transposed input layouts, and expanded MX quantization paths. PyTorch reference implementations are added for 29 kernels.

New Core Kernels
^^^^^^^^^^^^^^^^

* :doc:`Attention Segmented CTE </nki/library/api/attention-segmented-cte>` — Segmented attention computation with block-based KV cache and prefix caching support, processing the KV cache in configurable segments.
* :doc:`KV-Parallel Segmented Prefill </nki/library/api/attention-kv-parallel-segmented-cte>` — KV-parallel segmented prefill attention kernel. (Renamed to ``attention_kv_parallel_segmented_cte`` in 2.31.0.)
* :doc:`FP8 Quantize </nki/library/api/fp8-quantize>` — Static and row-wise dynamic FP8 quantization kernels with pre-combined dequantization scale support.

New Experimental Kernels
^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`Ring Attention Forward </nki/library/api/ring-attention-fwd>` — Ring attention forward pass for context parallelism across multiple workers using collective permute with latency hiding.
* :doc:`Ring Attention Backward </nki/library/api/ring-attention-bwd>` — Ring attention backward pass SPMD kernel for context parallelism.
* :doc:`Conv3D </nki/library/api/conv3d>` — 3D convolution using tensor engine with K-replication strategy and W-contiguous tiling. Supports stride, padding, dilation, bias, and activation fusion.
* :doc:`Foreach Elementwise </nki/library/api/foreach-elementwise>` — Suite of fused elementwise kernels (add, sub, mul, div, addcdiv, addcmul, lerp, sqrt) with SPMD tiling.
* :doc:`Foreach Norm </nki/library/api/foreach-norm>` — L1, L2, and Linf norm computation kernels with SPMD parallelization.
* :doc:`Matmul MXFP8 </nki/library/api/matmul-mxfp8-generic-kernel>` — Generic matrix multiplication with MXFP8 quantization, supporting pre-quantized and BF16 inputs with LNC2 parallelization.
* :doc:`MLP Forward MXFP8 </nki/library/api/mlp-fwd-mxfp8-kernel>` — MXFP8 SwiGLU MLP forward pass with optional activation checkpointing.
* :doc:`MLP Backward MXFP8 </nki/library/api/mlp-bwd-mxfp8-kernel>` — MXFP8 SwiGLU MLP backward pass with 4-phase gradient computation and activation checkpointing.
* :doc:`MX MoE Block TKG Wrapper </nki/library/api/mx-moe-block-tkg-wrapper>` — Wrapper that bitcasts unsigned integer weights to MX x4 dtype before calling the MoE block kernel.
* :doc:`Fused Adam/AdamW </nki/library/api/fused-adam>` — Fused Adam (L2 regularization) and AdamW (decoupled weight decay) optimizer kernels with AMSGrad support.
* :doc:`Pad </nki/library/api/pad>` — Multi-mode tensor padding (constant, replicate, reflect, circular) following PyTorch semantics.
* :doc:`Quantize MXFP8 </nki/library/api/quantize-mxfp8>` — Block-wise BF16-to-MXFP8 quantization kernel with packed scale support.
* :doc:`RNG </nki/library/api/rng>` — Random number generation kernels using GPSIMD engine with state management.
* :doc:`Linear Scan </nki/library/api/linear-scan>` — First-order linear recurrence computation along the last dimension.
* :doc:`Selective Scan </nki/library/api/selective-scan>` — Selective scan (SSM) as in Mamba models.
* :doc:`SSD </nki/library/api/ssd>` — State Space Duality scan for Mamba-2 models.

New Experimental Subkernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`Argsort Unstable </nki/library/api/argsort-unstable>` — Unstable argsort on 1D input buffer.
* :doc:`Build All-to-All-V Metadata </nki/library/api/build-all-to-all-v-metadata>` — Builds metadata buffer for all_to_all_v collective from MoE routing decisions.
* :doc:`Permute Routed Tokens </nki/library/api/permute-routed-tokens>` — Sorts tokens by expert and packs hidden states, affinities, and token indices for MoE dispatch.

Improvements
~~~~~~~~~~~~~~~

* :doc:`Attention CTE </nki/library/api/attention-cte>`: Added ``cp_striped_input`` for striped context parallelism and ``skip_output_normalization`` parameter.
* :doc:`Attention BWD </nki/library/api/attention-cte>`: Added ``transpose_dv`` and ``cp_offset`` parameters for context parallelism support.
* :doc:`MLP </nki/library/api/mlp>`: Added ``mode``, ``mx_dummy_scale_hbm``, ``transposed_in``, and ``transposed_out`` parameters for expanded layout and quantization support.
* :doc:`MoE CTE </nki/library/api/moe-cte>`: Added ``gate_up_proj_scale`` and ``down_proj_scale`` parameters for per-projection scaling.
* :doc:`MoE TKG </nki/library/api/moe-tkg>`: Added ``expert_gate_up_input_scale``, ``expert_down_input_scale``, ``input_dequant_scale``, ``all_to_all_v_strategy``, and ``outp_layout`` parameters. Replaced boolean sharding flags with ``LNCShardingStrategy`` enum.
* :doc:`MoE Block TKG </nki/library/api/moe-cte>`: Added ``expert_gate_up_input_scale``, ``expert_down_input_scale``, ``is_all_expert_dynamic``, ``block_size``, ``inp_layout``, and ``outp_layout`` parameters.
* :doc:`QKV </nki/library/api/qkv>`: Added ``k_cos_cache``, ``k_sin_cache``, ``transpose_k_cache`` for transposed K cache write, ``transposed_in`` for transposed input layout, ``qk_norm_pre_rope``/``qk_norm_post_rope`` for fused QK-norm, ``strided_input_config``, and ``output_hbm`` parameters.
* :doc:`RMSNorm-Quant </nki/library/api/rmsnorm-quant>`: Added ``pre_norm_gamma``, ``residual``, and ``auto_resolve_fp8_dtype`` parameters.
* :doc:`Attention Block TKG </nki/library/api/attention-block-tkg>`: Added ``transposed_in``, ``is_h_transposed_by_4``, ``KVDP_collective_mode``, ``pos_ids``, ``swa_start_pos_ids``, and ``S_ctx`` parameters for expanded KVDP and sliding window support.
* :doc:`Transformer TKG </nki/library/api/transformer-tkg>`: Added ``attention_mask`` parameter (replaces removed ``mask_cache``/``mask_active``).
* :doc:`Blockwise MM Backward </nki/library/api/blockwise-mm-backward>`: Added ``skip_grad_initialization`` and ``blocking_params`` parameters.
* Added Neurotile (``neurotile``), an **experimental** tile iterator library for NKI that replaces manual tiling loops, and DMA orchestration with composable, high-level primitives. Production NKI kernels are dominated by boilerplate (~70% of code), which includes: manual index arithmetic, SBUF management, DMA patterns, and sharding logic. Neurotile eliminates this boilerplate while preserving explicit control over NKI ISA instructions for the algorithmic core (~30% of code). Get tHe experimental version ``neurotile`` from `the NKI Library GitHub repo <https://github.com/aws-neuron/nki-library/tree/main/src/nkilib_src/nkilib/experimental>`__.
* Added PyTorch reference implementations for 29 kernels for testing and validation.

Breaking Changes
~~~~~~~~~~~~~~~~

* :doc:`MoE TKG </nki/library/api/moe-tkg>`: Parameters ``gate_up_input_scale`` and ``down_input_scale`` have been renamed to ``expert_gate_up_input_scale`` and ``expert_down_input_scale`` respectively. Callers using the old names must update.
* :doc:`MoE TKG </nki/library/api/moe-tkg>`: Replaced boolean sharding flags (``shard_on_I``, ``shard_on_T``) with ``sharding_strategy`` enum in ``down_projection_mx`` interfaces. Callers using ``shard_on_I`` or ``shard_on_T`` keyword arguments must migrate to the new ``sharding_strategy`` parameter.
* :doc:`MoE TKG </nki/library/api/moe-tkg>`: The ``gate_up_projection_mx_shard_I`` function has been removed. Use the unified ``gate_up_projection_mx`` function with the appropriate sharding strategy.
* :doc:`MoE TKG </nki/library/api/moe-tkg>`: Parameters ``is_all_expert_dynamic`` and ``block_size`` removed from ``init_all_expert_mx_configs``. These are now determined automatically from the kernel configuration.
* :doc:`MoE CTE </nki/library/api/moe-cte>`: The ``zeros`` parameter removed from ``reduce_outputs``. The ``skip_dma`` parameter default changed from ``SkipMode(False, False)`` to ``None``. New parameters ``gate_up_proj_scale`` and ``down_proj_scale`` inserted before ``gate_up_activations_T``, shifting positional arguments.
* :doc:`MLP </nki/library/api/mlp>`: The ``bias`` parameter removed from ``down_projection``. The ``unsharded_weight``, ``shard_dim_hidden``, and ``shard_dim_intr`` parameters removed from ``gate_up_projection``. Functions ``down_projection_lhs_rhs_swap`` and ``gate_up_projection_lhs_rhs_swap`` moved to separate files. The ``convert_weight_scale_params_to_views`` utility function removed. The ``store_fused_add_result`` parameter removed from ``input_fused_add``. New parameter ``mode`` inserted before ``sbm``, shifting positional arguments.
* :doc:`QKV </nki/library/api/qkv>`: New parameters ``k_cos_cache`` and ``k_sin_cache`` inserted before ``d_head``, shifting positional arguments for callers not using keyword arguments.
* :doc:`Attention BWD </nki/library/api/attention-cte>`: The ``softmax_scale`` parameter removed from ``load_q_dy``. New parameters inserted in ``setup_config`` and ``recompute_qk_softmax``, shifting positional arguments.
* :doc:`Transformer TKG </nki/library/api/transformer-tkg>`: Parameters ``mask_cache`` and ``mask_active`` removed and replaced by ``attention_mask``. Callers must update to use the new parameter.
* :doc:`Blockwise MM Backward </nki/library/api/blockwise-mm-backward>`: New parameter ``skip_grad_initialization`` inserted before ``shard_option``, shifting positional arguments.
* :doc:`Attention Block TKG </nki/library/api/attention-block-tkg>`: New parameter ``transposed_in`` inserted before ``softmax_scale``, and ``is_h_transposed_by_4`` inserted before ``KVDP``, shifting positional arguments.
* :doc:`QKV </nki/library/api/qkv>` (CTE variant): New parameter ``transpose_k_cache`` inserted between ``use_block_kv`` and ``block_size``, shifting positional arguments for callers not using keyword arguments.
* Removed ``apply_clamp`` from ``bwmm_shard_on_I_mx`` and ``validate_shapes_quantize_mx`` from ``norm_tkg_utils``.
* Removed ``core/mlp/mlp_tkg/mlp_proj_mxfp4_torch.py`` (MXFP4 PyTorch reference replaced by updated implementation).

Bug Fixes
~~~~~~~~~

Core Kernel Fixes
^^^^^^^^^^^^^^^^^

* **QKV**: Fixed invalid ``k_cache`` reshape when ``k_transpose`` is enabled.
* **MoE TKG**: Fixed TKG MX down projection weight layout alignment with CTE (I-contiguous x4).
* **Attention Segmented CTE**: Fixed sink token issue in segmented prefill kernel.
* **Router Top-K**: Fixed large vocab handling in ``rotational_topk`` when BxS fits in pmax.
* **Attention TKG**: Fixed tensor 4-byte alignment to resolve non-determinism error.
* **MoE TKG**: Fixed DLoC RMSNorm threshold (raised from T >= 512 to T > 512).
* **Attention BWD**: Fixed post-scale for softmax.
* **Attention CTE**: Fixed Scalar Engine bottleneck in softmax normalization.
* **Attention TKG**: Fixed SWA prior mask end clamping to ``pos_ids[b, 0]`` for speculation.
* **MoE TKG**: Fixed router weight range widening scoped to fp16 configs only.
* **MLP TKG**: Fixed weight tile layout to correctly fold contiguous groups of 4 elements onto the free dimension for gate/up and down projection contraction axes.
* **Attention BWD**: Fixed D statistic computation hardwired to fp32.
* **QKV**: Fixed K Transpose write to FP8 KV Cache.
* Fixed ``output_specs`` no longer being overwritten with given ``output_names``.

Experimental Kernel Fixes
^^^^^^^^^^^^^^^^^^^^^^^^^

* **MXFP8 Utils**: Fixed preserve 4D shape in DGT load dst slice to fix ``dma_transpose`` assertion.
* **Attention Block TKG**: Fixed ``pos_ids`` slicing per KVDP rank.
* **MLP Backward MXFP8**: Fixed phase2 ``TILES_IN_BLOCK_K`` for shape (4096, 4096, 3072).
* **MLP Forward MXFP8**: Fixed kernel now supports 128-divisible shapes.
* Fixed invalid ``k_cache`` reshape when ``k_transpose`` is enabled in QKV.
* Fixed preserve 4D shape in DGT load dst slice to fix ``dma_transpose`` assertion.
* Fixed TKG MX down projection weight layout alignment with CTE (I-contiguous x4).
* Fixed sink token issue in segmented prefill kernel.
* Fixed large vocab handling in ``rotational_topk`` when BxS fits in pmax.
* Fixed tensor 4-byte alignment to resolve non-determinism error.
* Fixed DLoC RMSNorm threshold (raised from T >= 512 to T > 512).
* Fixed ``pos_ids`` slicing per KVDP rank in attention block TKG.
* Fixed post-scale for softmax in ``attention_bwd``.
* Fixed Scalar Engine bottleneck in ``attention_cte`` softmax normalization.
* Fixed MLP backward phase2 ``TILES_IN_BLOCK_K`` for shape (4096, 4096, 3072).
* Fixed SWA prior mask end clamping to ``pos_ids[b, 0]`` for speculation.
* Fixed router weight range widening scoped to fp16 configs only.
* Fixed MLP TKG weight tile allocation size and fallback logic.
* Fixed ``attention_bwd`` D statistic computation hardwired to fp32.
* Fixed K Transpose write to FP8 KV Cache in QKV.
* Fixed ``output_specs`` no longer being overwritten with given ``output_names``.
* Fixed MLP forward MXFP8 kernel now supports 128-divisible shapes.

.. _nki-lib-2-29-0-rn:

NKI Library (NKI-Lib) (Neuron 2.29.0 Release)
--------------------------------------------------------------------

Date of Release: 04/09/2026

What's New
~~~~~~~~~~

This release promotes ``find_nonzero_indices`` from experimental to a core subkernel and adds 7 new experimental kernels (Conv1D, Transformer TKG, 3 collective communication kernels, Top-K Reduce, and Dynamic Elementwise Add). Existing kernels receive sequence packing support, MXFP quantization paths, and expanded dimension limits. PyTorch reference implementations are added for 22 kernels.

New Core Additions
^^^^^^^^^^^^^^^^^^

* :doc:`find_nonzero_indices </nki/library/api/find-nonzero-indices>` (promoted from experimental) — Finds indices of nonzero elements along the T dimension using GpSimd ``nonzero_with_count`` ISA. Optimized for LNC2 sharding. Supports token counts up to 65536 and column counts up to 128.

New Experimental Kernels
^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`Conv1D </nki/library/api/conv1d>` — 1D convolution using tensor engine with replication strategy. Supports stride, padding, dilation, optional bias, activation fusion, and LNC sharding.
* :doc:`Transformer TKG </nki/library/api/transformer-tkg>` — Multi-layer transformer forward pass megakernel for token generation. Executes attention block, all-reduce, MLP, and residual connections across a configurable number of layers.
* :doc:`Fine-Grained All-Gather </nki/library/api/fg-allgather>` — Ring-based all-gather for TRN2 using collective permute with double buffering to overlap communication and data movement.
* :doc:`FGCC (All-Gather + Matmul) </nki/library/api/fgcc>` — Fused all-gather and matrix multiplication for TRN2, overlapping communication with compute.
* :doc:`SBUF-to-SBUF All-Gather </nki/library/api/sb2sb-allgather>` — Two variants: ``allgather_sb2sb`` for small tensors fitting in SBUF and ``allgather_sb2sb_tiled`` with tiling and LNC support for larger tensors.
* :doc:`Top-K Reduce </nki/library/api/topk-reduce>` — Gathers scattered rows by packed global token index and reduces along the K dimension for MoE output. Supports LNC sharding on the hidden dimension.
* :doc:`Dynamic Elementwise Add </nki/library/api/dynamic-elementwise-add>` — Elementwise addition with runtime-variable M-dimension tiling using dynamic loop bounds.

Improvements
~~~~~~~~~~~~~~~

* :doc:`Attention CTE Kernel </nki/library/api/attention-cte>`: Added ``mm_out_dtype`` parameter for controlling matmul output dtype. Added ``bound_min``/``bound_max`` parameters for sequence packing support (per-query KV range bounds). Increased max batch size from 32 to 512. Increased max sequence length from 36864 to 131072.
* :doc:`Attention BWD Kernel </nki/library/api/attention-cte>`: Added ``bound_min``/``bound_max`` parameters for sequence packing support. Added support for large batch size.
* :doc:`Attention TKG Kernel </nki/library/api/attention-tkg>`: Added ``start_pos_ids`` parameter for explicit KV cache position control to support sliding window masking.
* :doc:`Attention Block TKG Kernel </nki/library/api/attention-block-tkg>`: Added ``rmsnorm_QK_pre_rope_W_Q``/``rmsnorm_QK_pre_rope_W_K`` parameters for fused QK-norm before RoPE. Added KVDP attention sharding support (``KVDP``, ``KVDP_replica_group``). Added ``enable_fa_s_prior_tiling`` for overriding flash attention s_prior tiling.
* :doc:`MLP Kernel </nki/library/api/mlp>`: Added ``sbm`` (BufferManager) parameter for custom SBUF memory management. Added MXFP4/MXFP8 quantization path.
* :doc:`MoE TKG Kernel </nki/library/api/moe-tkg>`: Added new dynamic all-expert algorithm that uses ``block_size`` and ``is_all_expert_dynamic`` args. Expanded support for small I and added support for sharding on T in all-expert MX kernel.
* :doc:`Output Projection CTE Kernel </nki/library/api/output-projection-cte>`: Added ``output_dtype`` parameter for controlling output data type.
* :doc:`Output Projection TKG Kernel </nki/library/api/output-projection-tkg>`: Added ``sbm`` (BufferManager) parameter for custom SBUF memory management.
* :doc:`QKV Kernel </nki/library/api/qkv>`: Added ``is_h_dim_4h_transposed`` and ``weight_layout`` parameters for flexible weight layout support.
* **rmsnorm_tkg** / **layernorm_tkg**: Added ``shard_on_h`` parameter for sharding on the hidden dimension.
* Added PyTorch reference implementations for 22 kernels for testing and validation.

Breaking Changes
~~~~~~~~~~~~~~~~

* :doc:`Router Top-K Kernel </nki/library/api/router-topk>`: The ``output_in_sbuf``, ``x_input_in_sbuf``, and ``expert_affin_in_sb`` parameters have been removed. The kernel now auto-detects SBUF inputs from the tensor buffer type. Callers passing these keyword arguments must remove them.
* :doc:`QKV Kernel </nki/library/api/qkv>`: The ``is_input_swizzled`` parameter has been removed and replaced by ``is_h_dim_4h_transposed`` (same position, same default ``False``) and a new ``weight_layout`` parameter. Callers using ``is_input_swizzled`` by name must rename to ``is_h_dim_4h_transposed``.
* :doc:`QKV Kernel </nki/library/api/qkv>` (TKG variant): New parameter ``is_h_dim_4h_transposed`` has been inserted after ``quantization_type``. Callers using positional arguments for ``qkv_w_scale`` or later parameters must update to use keyword arguments.
* :doc:`Attention CTE Kernel </nki/library/api/attention-cte>`: New parameter ``mm_out_dtype`` has been inserted between ``softmax_dtype`` and ``cp_offset``. Callers using positional arguments for ``cp_offset``, ``global_cp_deg``, or ``cp_strided_q_slicing`` must update to use keyword arguments.
* :doc:`Attention TKG Kernel </nki/library/api/attention-tkg>`: New parameter ``start_pos_ids`` has been inserted after ``rope_pos_ids``. Callers using positional arguments beyond ``rope_pos_ids`` must update to use keyword arguments.
* :doc:`Attention BWD Kernel </nki/library/api/attention-cte>`: New parameters ``bound_min`` and ``bound_max`` have been inserted between ``sinks_ref`` and ``use_causal_mask``. Callers using positional arguments for ``use_causal_mask`` or later parameters must update to use keyword arguments.
* :doc:`Attention Block TKG Kernel </nki/library/api/attention-block-tkg>`: The keyword-only marker (``*``) has been removed and multiple parameters have been reordered. New pre-RoPE QK-norm parameters (``rmsnorm_QK_pre_rope_W_Q``, ``rmsnorm_QK_pre_rope_W_K``) have been added. ``softmax_scale``, ``k_scale``, and ``v_scale`` have been moved to optional parameters with defaults. All callers must review their argument ordering.
* **rmsnorm_tkg** / **layernorm_tkg**: New parameter ``shard_on_h`` has been inserted before ``use_heap_memory`` and ``sbm``. Callers using positional arguments beyond ``single_core_forced`` (rmsnorm) or ``eps`` (layernorm) must update to use keyword arguments. Helper functions ``process_rmsnorm_tile``, ``rmsnorm_tkg_llama_impl``, and ``layernorm_tkg_llama_impl`` have been made private (prefixed with ``_``).
* **SbufManager** has been renamed to **BufferManager**. A backward-compatible alias ``SbufManager = BufferManager`` is provided, so existing code using ``SbufManager`` will continue to work.
* MoE TKG: Replaced boolean sharding flags (``shard_on_I``, ``shard_on_T``) with ``LNCShardingStrategy`` enum in down projection interfaces.
* MoE TKG MX quantization files restructured: ``down_projection_mx_shard_I.py`` and ``gate_up_projection_mx_shard_I.py`` replaced with ``all_expert_mx_utils.py``, ``down_projection_mx.py``, and ``gate_up_projection_mx.py``. Callers importing from the old file paths must update their imports.
* ``find_nonzero_indices`` has been moved from ``nkilib.experimental.subkernels`` to ``nkilib.core.subkernels``. A backward-compatible re-export is provided, so imports via the experimental path continue to work.
* Removed usage of ``nki.language.par_dim`` throughout the library.

Bug Fixes
~~~~~~~~~

* Fixed MLP CTE indexing in gate proj row scales.
* Fixed QKV TKG ``sb2sb_wrapper_kernel`` signature missing QK-norm parameters.
* Fixed MLP failure for FP4 quantization with specific dimension combinations (``vnc=2, h=3072, i=384``).
* Fixed ``bwmm_shard_on_H`` with explicit TensorCopy from PSUM to SBUF for NKI 0.3.0 compatibility.

Known Issues
~~~~~~~~~~~~



.. _nki-lib-2-28-0-rn:   

NKI Library (NKI-Lib) (Neuron 2.28.0 Release)
--------------------------------------------------------------------

What's New
~~~~~~~~~~

This release expands the NKI Library with 9 new kernels, bringing the total to 16 documented kernel APIs. New core kernels include RoPE, Router Top-K, MoE CTE, MoE TKG, and Cumsum. New experimental kernels include Attention Block TKG (fused attention block for token generation), Cross Entropy (forward and backward passes), Depthwise Conv1D, and Blockwise MM Backward for MoE training.

Existing kernels receive FP8 and MX quantization support across QKV, MLP, and both Output Projection kernels. Kernel utilities gain new TensorView methods, SbufManager logging improvements with tree-formatted allocation tracing, and new utilities including ``interleave_copy``, ``LncSubscriptable``, and ``rmsnorm_mx_quantize_tkg``. Note that several breaking changes affect kernel signatures and utility APIs — see the Breaking Changes section for details.

New Core Kernels
^^^^^^^^^^^^^^^^

* :doc:`RoPE Kernel </nki/library/api/rope>` — Applies Rotary Position Embedding to input embeddings with optional LNC sharding and flexible layout support (contiguous and interleaved).
* :doc:`Router Top-K Kernel </nki/library/api/router-topk>` — Computes router logits and top-K expert selection for Mixture of Experts models, with support for multiple layout configurations and sharding strategies.
* :doc:`MoE CTE Kernel </nki/library/api/moe-cte>` — Implements Mixture of Experts optimized for Context Encoding with multiple sharding strategies (block sharding, intermediate dimension sharding) and MxFP4/MxFP8 quantization.
* :doc:`MoE TKG Kernel </nki/library/api/moe-tkg>` — Implements Mixture of Experts optimized for Token Generation with all-expert and selective-expert modes, supporting FP8 and MxFP4 quantization.
* :doc:`Cumsum Kernel </nki/library/api/cumsum>` — Computes cumulative sum along the last dimension, optimized for batch sizes up to 2048.

New Experimental Kernels
^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`Attention Block TKG Kernel </nki/library/api/attention-block-tkg>` — Fused attention block for Token Generation that combines RMSNorm, QKV projection, RoPE, attention, and output projection in SBUF to minimize HBM traffic.
* :doc:`Cross Entropy Kernel </nki/library/api/cross-entropy>` — Memory-efficient cross entropy loss forward and backward passes for large vocabularies using online log-sum-exp algorithm, optimized for LNC2.
* :doc:`Depthwise Conv1D Kernel </nki/library/api/depthwise-conv1d>` — Depthwise 1D convolution using implicit GEMM algorithm with support for arbitrary stride and padding values, optimized for TRN2.
* :doc:`Blockwise MM Backward Kernel </nki/library/api/blockwise-mm-backward>` — Backward pass for blockwise matrix multiplication in MoE layers, computing gradients for all parameters with support for dropless MoE.

Improvements
~~~~~~~~~~~~~~~

* :doc:`QKV Kernel </nki/library/api/qkv>`: Added FP8 quantization support (``quantization_type``, ``qkv_w_scale``, ``qkv_in_scale``), fused FP8 KV cache quantization (``k_cache``, ``v_cache``, ``k_scale``, ``v_scale``, ``fp8_max``, ``fp8_min``, ``kv_dtype``), block-based KV cache layout (``use_block_kv``, ``block_size``, ``slot_mapping``), and MX quantization input swizzling (``is_input_swizzled``).
* :doc:`MLP Kernel </nki/library/api/mlp>`: Added FP8 quantization support (``quantization_type``, ``gate_w_scale``, ``up_w_scale``, ``down_w_scale``, ``gate_up_in_scale``, ``down_in_scale``, ``quant_clipping_bound``), gate/up projection clamping (``gate_clamp_upper_limit``, ``gate_clamp_lower_limit``, ``up_clamp_upper_limit``, ``up_clamp_lower_limit``), ``skip_gate_proj`` option, and fp16 support for TKG mode.
* :doc:`Output Projection CTE Kernel </nki/library/api/output-projection-cte>`: Added FP8 quantization support (``quantization_type``, ``input_scales``, ``weight_scales``).
* :doc:`Output Projection TKG Kernel </nki/library/api/output-projection-tkg>`: Added FP8 quantization support (``quantization_type``, ``weight_scale``, ``input_scale``) and removed 512 restriction on non-transpose path.
* :doc:`Attention CTE Kernel </nki/library/api/attention-cte>`: Added strided Q slicing for context parallelism (``cp_strided_q_slicing``).
* :doc:`RMSNorm-Quant Kernel </nki/library/api/rmsnorm-quant>`: Added input dequantization scale support (``input_dequant_scale``).

Kernel Utilities
^^^^^^^^^^^^^^^^

See :doc:`Kernel Utilities Reference </nki/library/kernel-utils/index>` for full documentation.

* :doc:`TensorView </nki/library/kernel-utils/tensor-view>`: Added ``rearrange`` method for flexible dimension reordering, ``has_dynamic_access`` for checking whether a view requires runtime-dependent addressing, and ``key_in_dict`` helper. The ``slice`` method now clamps the end index to dimension bounds instead of asserting.
* :doc:`TiledRange </nki/library/kernel-utils/tiled-range>`: ``TiledRangeIterator`` now exposes an ``end_offset`` attribute, enabling kernels to determine the end position of each tile without manual calculation.
* :doc:`SbufManager (Allocator) </nki/library/kernel-utils/allocator>`: Added ``get_total_space`` and ``get_used_space`` for querying SBUF utilization, ``set_name_prefix`` / ``get_name_prefix`` for scoped naming, and ``flush_logs`` to emit buffered allocation logs. SbufManager now uses ``TreeLogger`` to provide hierarchical, tree-formatted logs of SBUF allocation and deallocation events, making it easier to debug memory usage across nested scopes.
* **QuantizationType**: Added ``MX`` enum value for microscaling quantization (MxFP4/MxFP8).
* **common_types**: Added ``GateUpDim`` enum for distinguishing gate vs up projection dimensions.
* **rmsnorm_tkg / layernorm_tkg**: Both subkernels now accept a ``TensorView`` or ``nl.ndarray`` for input and require an explicit ``output`` tensor parameter, giving callers control over output placement.
* **New utilities**: Added ``rmsnorm_mx_quantize_tkg`` subkernel for fused RMSNorm with MX quantization in token generation, ``interleave_copy`` for interleaved tensor copy operations, ``LncSubscriptable`` for LNC-aware data access patterns, and ``TreeLogger`` for hierarchical allocation logging.

Breaking Changes
~~~~~~~~~~~~~~~~

* The open source repository source directory has been renamed from ``nkilib_standalone`` to ``nkilib_src``.
* :doc:`MLP Kernel </nki/library/api/mlp>`: The function has been renamed from ``mlp_kernel`` to ``mlp``. New parameters have been inserted in the middle of the signature; callers using positional arguments beyond ``normalization_type`` must update to use keyword arguments.
* :doc:`QKV Kernel </nki/library/api/qkv>`: New parameters (``quantization_type``, ``qkv_w_scale``, ``qkv_in_scale``) have been inserted after ``bias``; callers using positional arguments beyond ``bias`` must update to use keyword arguments.
* :doc:`Output Projection TKG Kernel </nki/library/api/output-projection-tkg>`: The ``bias`` parameter is now optional (default ``None``). New parameters (``quantization_type``, ``weight_scale``, ``input_scale``) have been inserted before ``TRANSPOSE_OUT``; callers using positional arguments beyond ``bias`` must update to use keyword arguments.
* **TiledRangeIterator**: The constructor now requires a fourth positional argument ``end_offset``.
* **TensorView**: The ``sizes`` attribute has been renamed to ``shape``.
* **rmsnorm_tkg**: The ``inp`` parameter has been renamed to ``input``. A new required ``output`` parameter has been added as the third argument. The ``output_in_sbuf`` parameter has been removed. New parameters ``hidden_dim_tp`` and ``single_core_forced`` have been added.
* **layernorm_tkg**: The ``inp`` parameter has been renamed to ``input``. A new required ``output`` parameter has been added as the third argument. The ``output_in_sbuf`` parameter has been removed.

Bug Fixes
~~~~~~~~~

* Fixed attention TKG compilation and non-determinism issues.
* Fixed incorrect v_active slice indices in attention TKG block KV path.
* Fixed batch sharding in gen_mask_tkg active mask loading.
* Fixed expert_affinities masking when ``mask_unselected_experts`` is True in MoE TKG.
* Fixed expert_index shape mismatch in MoE TKG for T > 128.
* Fixed MoE affinity mask handling for T not divisible by 128.
* Fixed MoE TKG MX weight generation x4 pack size.
* Fixed MLP CTE ``force_cte_mode`` parameter validation.
* Fixed output projection CTE mixed precision support.
* Fixed output projection TKG variable name typo.
* Fixed router_topk bias shape to satisfy NKI check requirements.
* Fixed tail iteration bug for sequences not a multiple of 128 in MoE CTE.
* Fixed reading extra partitions for last rank in MoE CTE.

Known Issues
~~~~~~~~~~~~

.. _nki-lib-2-27-0-rn:

NKI Library (NKI-Lib) (Neuron 2.27.0 Release)
--------------------------------------------------------------------

What's New
~~~~~~~~~~

This release introduces the NKI Library, which provides pre-built kernels you can use to optimize
the performance of your models. The NKI Library offers ready-to-use, pre-optimized kernels that
leverage the full capabilities of AWS Trainium hardware.

NKI Library kernels are published in the `NKI Library GitHub repository <https://github.com/aws-neuron/nki-library>`_.
In Neuron 2.27, these kernels are also shipped as part of neuronx-cc under the ``nkilib.*`` namespace.

Accessing NKI Library Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can access NKI Library kernels in two ways:

* **Shipped version**: Import from the ``nkilib.*`` namespace (included with neuronx-cc in Neuron 2.27)
* **Open source repository**: Clone and use kernels from the GitHub repository under the ``nkilib_standalone.nkilib.*`` namespace

New Kernels
~~~~~~~~~~~

This release includes the following pre-optimized kernels:

* **Attention CTE Kernel** — Implements attention with support for multiple variants and optimizations
* **Attention TKG Kernel** — Implements attention specifically optimized for token generation scenarios
* **MLP Kernel** — Implements a Multi-Layer Perceptron with optional normalization fusion and various optimizations
* **Output Projection CTE Kernel** — Computes the output projection operation optimized for Context Encoding use cases
* **Output Projection TKG Kernel** — Computes the output projection operation optimized for Token Generation use cases
* **QKV Kernel** — Performs Query-Key-Value projection with optional normalization fusion
* **RMSNorm-Quant Kernel** — Performs optional RMS normalization followed by quantization to fp8

NKI Library Kernel Migration to New nki.* Namespace in Neuron 2.28
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some NKI Library kernels currently use the legacy ``neuronxcc.nki.*`` namespace. Starting with
Neuron 2.28, all NKI Library kernels will migrate to the new ``nki.*`` namespace.

The new ``nki.*`` namespace introduces changes to NKI APIs and language constructs. Customers
using NKI Library kernels should review the migration guide for any required changes.

NKI Library Namespace Changes in Neuron 2.28
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting with Neuron 2.28, the open source repository namespace will change from
``nkilib_standalone.nkilib.*`` to ``nkilib.*``, providing a consistent namespace between
the open source repository and the shipped version.

Customers who want to add or modify NKI Library kernels can build and install them to
replace the default implementation without changing model imports.



    