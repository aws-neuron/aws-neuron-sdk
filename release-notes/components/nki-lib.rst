.. meta::
    :description: Complete release notes for the NKI Library component across all AWS Neuron SDK versions.
    :keywords: nki library, nki-lib, release notes, aws neuron sdk
    :date-modified: 10/19/2026

.. _nki-lib_rn:

Release Notes for Neuron Component: NKI Library
================================================

The release notes for the NKI Library Neuron component. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _nki-lib-2-33-0-rn:

NKI Library (NKI-Lib) (Neuron 2.33.0 Release)
--------------------------------------------------------------------

Date of Release: 10/19/2026

What's New
~~~~~~~~~~

This release adds 25 new experimental kernels. Highlights include a DeepSeek-V3.2 decode path (MLA QKV projection, sparse-gather MLA attention and its fused attention block, the sparse indexer, the group-limited router), a DeepSeek-V3.2 blockwise MXFP MoE and MX MLP for prefill, Gated DeltaNet (GDN) kernels for Qwen3.5 prefill and decode, Kimi Delta Attention (KDA) token-generation kernels, a FlashAttention-2 forward kernel and a constant-max attention kernel, 3D convolution backward, batch normalization backward, 3D max/average pooling (forward and backward), a chunked top-K for vocabularies wider than ``nisa.topk`` supports, and a GPT-OSS decode sampling-tail megakernel. Existing kernels gain sequence-packed attention in ``attention_block_tkg``, speculative decoding in ``attention_mxfp8_tkg``, fused batch norm and residual add in ``conv3d``, BF16 precision paths and compact scales in the DeepSeek MLA and sparse-indexer kernels, caller-provided outputs in ``output_projection_cte``, and expert-parallel offsets in the MX blockwise MoE kernels. A new ``models/`` folder holds model-specific kernels whose tuning only pays off for one model's geometry: Gemma4 (``attention_tkg`` with a partitioned block-K load) and GPT-OSS (self-contained C128/C512 decode GIGA kernel packages). This release also removes the deprecated ``TensorView`` module, makes packed routing required in the all-to-all-v MoE training kernels, and moves the MXFP8 MoE kernel options into config objects — see Breaking Changes.

New Experimental Kernels
^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`Attention Const-Max </nki/library/api/attention-const-max>` — Attention with a caller-supplied global softmax maximum, ``softmax(Q @ K.T * softmax_scale - softmax_max) @ V``, with sequence-major Q/K layouts and arbitrary sequence lengths.
* :doc:`Flash Attention 2 </nki/library/api/flash-attention-2>` — FlashAttention-2 forward for unmasked bidirectional attention over long key sequences (Trainium 3).
* :doc:`KDA TKG </nki/library/api/kda-tkg>` — Kimi Delta Attention token generation: advances the KDA recurrence by one token per request.
* :doc:`KDA Block TKG </nki/library/api/kda-block-tkg>` — Full Kimi Delta Attention block for token generation (input RMSNorm, projections, short convolutions, forget gate, recurrence, gated output RMSNorm, output projection) with all intermediates kept in SBUF.
* :doc:`GDN CTE </nki/library/api/gdn-cte>` — Chunked gated delta-rule (GDN) prefill with a selectable intra-chunk inverse algorithm (Schur complement by default, or Neumann series).
* :doc:`GDN TKG </nki/library/api/gdn-tkg>` — Fused single-token gated delta-rule decode step.
* :doc:`GDN Block TKG </nki/library/api/gdn-block-tkg>` — Fused Qwen3.5 GDN decode layer (input projections, conv1d, GQA expansion, gated delta-rule) in one kernel.
* :doc:`GDN Conv1D </nki/library/api/gdn-conv1d>` — Depthwise causal conv1d + SiLU + ``conv_state`` update for GDN, with prefill and decode variants.
* :doc:`MLA Decode QKV </nki/library/api/mla-decode-qkv>` — DeepSeek-V3.2 MLA query and key/value projections for one decode step, with interleaved RoPE, query absorption, and paged KV-cache write.
* :doc:`MLA Decode Attention </nki/library/api/mla-decode-attention>` — DeepSeek-V3.2 MLA decode attention over indexer-selected prior keys, followed by V up-projection and output projection.
* :doc:`MLA Attention Block Gather Decode </nki/library/api/mla-attention-block-gather-decode>` — End-to-end DeepSeek-V3.2 MLA decode attention block composing ``mla_decode_qkv`` and ``mla_decode_attention`` with the query tiles kept in SBUF.
* :doc:`DeepSeek-V3.2 Sparse Indexer </nki/library/api/deepseek-v32-sparse-indexer>` — DeepSeek Sparse Attention lightning indexer for decode (Trainium 3).
* :doc:`DeepSeek-V3 Router </nki/library/api/deepseek-v3-router>` — DeepSeek-V3.2 group-limited ``noaux_tc`` routing for one decode step.
* :doc:`Blockwise MM Block MX </nki/library/api/bwmm-block-mx>` — DeepSeek-V3.2 blockwise MXFP4/MXFP8 MoE for context encoding on a single logical core (LNC1).
* :doc:`MLP DeepSeek MX </nki/library/api/mlp-deepseek-mx>` — DeepSeek-V3.2 shared-experts and first dense MLP layers for prefill, consuming the packed MX input from ``rmsnorm_mx_prefill``.
* :doc:`Conv3D Backward </nki/library/api/conv3d-bwd>` — Any subset of the input, weight, and bias gradients of a 3D convolution, with a planner that selects the strategy from the problem geometry.
* :doc:`Batch Norm Backward </nki/library/api/batch-norm-bwd>` — Batch normalization backward over the channel dimension, with optional fused ReLU.
* :doc:`Max Pooling 3D </nki/library/api/max-pooling-3d>` and :doc:`Max Pooling 3D Backward </nki/library/api/max-pooling-3d-backward>` — 3D max pooling forward and backward (NCDHW and NDHWC layouts).
* :doc:`Average Pooling 3D </nki/library/api/avg-pooling-3d>` and :doc:`Average Pooling 3D Backward </nki/library/api/avg-pooling-3d-backward>` — 3D average pooling forward and backward (NCDHW and NDHWC layouts).
* :doc:`RMSNorm Residual Prefill </nki/library/api/rmsnorm-residual-prefill>` — Pre-norm residual add followed by RMSNorm for prefill, with the residual add performed on the DMA engines during the load.
* :doc:`Chunked Top-K </nki/library/api/chunked-topk>` — Single-core top-K over a vocabulary wider than the 65,535-element limit of ``nisa.topk``, with no collective.
* :doc:`GPT-OSS Tail Chunked Megakernel </nki/library/api/gpt-oss-tail-chunked-megakernel>` — GPT-OSS decode sampling tail (final RMSNorm, ``lm_head``, top-K, sampler) with a single all-to-all collective.
* :doc:`Build Packed Routing </nki/library/api/build-packed-routing>` — Builds the destination-major ``packed_source_indices`` consumed by ``permute_a2av`` and ``unpermute_a2av``.

Improvements
~~~~~~~~~~~~~~~

* :doc:`Attention CTE </nki/library/api/attention-cte>`: Added ``d_v`` support (a value head dimension that differs from the query/key head dimension, for MLA).
* :doc:`Attention KV-Parallel Segmented CTE </nki/library/api/attention-kv-parallel-segmented-cte>`: Added packed-FP8 KV support (``fp8_packed``, ``k_scale``, ``v_scale``).
* :doc:`Attention Block TKG </nki/library/api/attention-block-tkg>`: Added sequence-packed token-generation attention for mixed-context batches through the ``seq_packed_attention_mask``, ``seq_packed_active_blocks_table``, ``seq_packed_q_index_table``, ``seq_packed_seq_id_table``, ``seq_packed_accumulator_partial_route_table``, and ``seq_packed_accumulator_group_state_route_table`` parameters, with KV-cache block resizing.
* :doc:`QKV </nki/library/api/qkv>` (``qkv_cte``): Added ``head_split_kv_cache`` (head-major block K/V cache layout) and ``hbm_name_prefix`` (for multiple calls in one trace); added packed-FP8 KV writes in the MX path; the normalization beta bias is now supported with fused ``RMS_NORM``.
* :doc:`MLP </nki/library/api/mlp>`: Added ``use_folded_mx_scales`` so CTE MX weight scales can be supplied pre-folded into the kernel's SBUF layout and loaded with one DMA per operand.
* :doc:`Output Projection CTE </nki/library/api/output-projection-cte>`: Added ``attention_layout`` to declare the input dimension order, and ``out`` / ``out_s_offset`` to write multiple S chunks into one caller-allocated output.
* :doc:`RMSNorm MX Prefill </nki/library/api/rmsnorm-mx-prefill>`: Added ``hidden_interleaved`` for hidden states in the model's 4-way interleaved H order.
* :doc:`Router Top-K </nki/library/api/router-topk>`: Added ``expert_affinities_eager_out`` (dense ``[T, k]`` eager-affinity output) and ``skip_store_expert_affinities``.
* :doc:`MoE CTE </nki/library/api/moe-cte>`: Added ``ep_rank`` for expert-parallel slicing of packed affinities (also added to ``bwmm_shard_on_block_mx``).
* :doc:`Attention MXFP8 TKG </nki/library/api/attention-mxfp8-tkg>`: Added speculative decoding (``s_active`` in {1, 2, 4, 8}) and support for ``q_head=32``.
* :doc:`RMSNorm Router Top-K TKG </nki/library/api/rmsnorm-router-topk-tkg>`: Added ``store_eager_affi_only``.
* :doc:`Conv3D </nki/library/api/conv3d>`: Added fused batch normalization (``batch_norm_mode``, ``gamma``, ``beta``, ``momentum``, ``running_means``, ``running_variances``, ``batch_norm_eps``, ``output_pre_norm``), fused residual add (``residual_add_loc``, ``residuals_in``), and caller-provided SBUF management (``sbm``, ``use_auto_allocation``).
* :doc:`Conv3D Transpose </nki/library/api/conv3d-transpose>`: Added ``output_padding``, asymmetric 6-value ``padding``, and ``flip_filter_taps_in_kernel`` (pass filters in forward orientation and let the kernel reverse the taps).
* :doc:`MS Deformable Attention </nki/library/api/ms-deformable-attention>`: Added ``gather_method`` and ``max_indices_per_indirect`` for batched indirect gathers.
* :doc:`MS Deformable Attention Backward </nki/library/api/ms-deformable-attention-bwd>`: Added batched indirect gather/scatter controls (``gather_method``, ``max_gather_indices_per_indirect``, ``max_scatter_indices_per_indirect``, ``num_scatter_buffers``), per-gradient selection (``compute_grad_value``, ``compute_grad_sampling_locations``, ``compute_grad_attention_weights``), and optional intermediate inputs and dumps.
* :doc:`Scatter-Add </nki/library/api/scatter-add>`: Added ``unique_indices`` to skip duplicate-index accumulation when indices are known to be unique.
* :doc:`MLA QKV CTE </nki/library/api/mla-qkv-cte>`, :doc:`MLA Sparse Attention CTE </nki/library/api/mla-sparse-attention-cte>`, and :doc:`MLA V-Up O-Proj CTE </nki/library/api/mla-vup-oproj-cte>`: Added a ``precision`` selector (MX or BF16) and ``compact_scales``; added GLM configurations. ``mla_sparse_attention_cte`` adds a ``dense`` attention path, ``q_pos_offset_hbm``, ``use_sbuf_indirect``, and 1M-token context support. ``mla_vupmx_oproj_cte_kernel`` adds ``kv_lora_rank`` and supports S < 128.
* :doc:`Sparse Attention Indexer (MX + BF16 Score) </nki/library/api/sparse-attention-indexer-mx-bf16score>`: Added ``precision``, ``compact_scales``, ``rope_layout``, and ``qr_bf16_hbm``; added chunking for long context.
* :doc:`Blockwise MM Forward MXFP8 </nki/library/api/blockwise-mm-forward-mxfp8>`: Added ``num_valid_tokens`` to schedule blocks from a runtime token count; added pre-quantized weights, backward-natural weight layouts, a 1x32 PE-swizzle load mode, and a ``TRANSPOSED`` checkpoint store layout.
* :doc:`Blockwise MM Backward MXFP8 </nki/library/api/blockwise-mm-backward-mxfp8>`: Added ``num_valid_tokens``, 1x32 swizzle support, and a single-expert dense path.
* :doc:`Matmul MXFP8 </nki/library/api/matmul-mxfp8-generic-kernel>`: Added ``disable_dma_transpose``, K-slicing, flexible K (``K % 512 != 0``), and K-by-F inputs in the 1x32 PE-swizzle loader. :doc:`Matmul MXFP8 Backward </nki/library/api/matmul-mxfp8-generic-backward-kernel>` adds ``fast_dma_transpose``, ``quant_scheme``, ``disable_dma_transpose``, ``weights_is_f_by_k``, and ``input_is_f_by_k``.
* Each new kernel ships with a PyTorch reference implementation for testing and validation.

Optimizations
~~~~~~~~~~~~~

* **d_head=64 prefill**: Optimized :doc:`Attention CTE </nki/library/api/attention-cte>`, :doc:`Attention Segmented CTE </nki/library/api/attention-segmented-cte>`, :doc:`QKV </nki/library/api/qkv>`, and :doc:`Output Projection CTE </nki/library/api/output-projection-cte>` on the d_head=64 prefill path.
* :doc:`Attention CTE </nki/library/api/attention-cte>`: Chained the softmax reduce accumulators; enabled compiler buffer transforms in seven-bank PSUM mode.
* :doc:`Attention Segmented CTE </nki/library/api/attention-segmented-cte>`: Nested the ``q_head`` loop inside the dynamic loop; removed a Vector copy and balanced the V-load DMA in the K/V load.
* :doc:`Attention TKG </nki/library/api/attention-tkg>`: Increased the V-load DMA batching cap; stores the output with unit stride when ``H * s_active == 1``.
* **Gen Mask TKG**: Pooled scratch buffers so consecutive batches no longer serialize.
* :doc:`RMSNorm-Quant </nki/library/api/rmsnorm-quant>`: Shares the squared-value scratch across tiles when a core processes 8 or more tiles.
* :doc:`MLP </nki/library/api/mlp>`: Widened MLP CTE weight DMAs and decoupled prefetch depth from DMA width; I-sharded the Llama3-70B high-batch MLP TKG dispatch.
* :doc:`MoE TKG </nki/library/api/moe-tkg>`: Reduced Vector Engine work in the all-expert software quantization.
* :doc:`Cumsum </nki/library/api/cumsum>`: Uses ``tensor_scalar_cumulative`` for supported dtypes, with an int32/uint32 fallback.
* **Rotational Top-K**: Removed inter-engine stalls in the schedule.
* :doc:`RNG </nki/library/api/rng>`: Uses all RNG lanes for remainder tiles.
* :doc:`Permute A2AV </nki/library/api/permute-a2av>` and :doc:`Unpermute A2AV </nki/library/api/unpermute-a2av>`: Partitioned gather and receive work across LNC cores, gather rows from packed source indices, size the receive storage by routed rows, and optimized the receive copies.
* :doc:`MS Deformable Attention </nki/library/api/ms-deformable-attention>`: Optimized index calculation, reduced memsets, and batched indirect DMAs.
* :doc:`Blockwise MM Forward MXFP8 </nki/library/api/blockwise-mm-forward-mxfp8>`: Reuse spilled weights across blocks, store checkpoints per block, fuse the SwiGLU affinity fold, speed up the output store, split odd-N tail blocks across LNC cores, and allocate accumulators in bf16 when there is a single K block.
* :doc:`Matmul MXFP8 </nki/library/api/matmul-mxfp8-generic-kernel>`: Shards the non-pre-quantized operand for asymmetric inputs, stops reloading loop-invariant operands, uses a single bulk DMA for 1x32 PE-swizzle loads, balances PSUM drain across Vector and Scalar engines, and re-tunes autotune cache entries.
* :doc:`Quantize MXFP8 </nki/library/api/quantize-mxfp8>`: Removed a redundant bf16 buffer and DMA copy.

Breaking Changes
~~~~~~~~~~~~~~~~

* **TensorView removed**: The ``TensorView`` utility (``core/utils/tensor_view.py``) has been deleted because its view operations (slicing, permuting, reshaping, broadcasting) have been migrated upstream into ``nl.NkiTensor``. All NKI Library kernels now use the native ``nl.NkiTensor`` view methods. Callers that imported ``TensorView`` must switch to the equivalent ``nl.NkiTensor`` methods.
* :doc:`Permute A2AV </nki/library/api/permute-a2av>` and :doc:`Unpermute A2AV </nki/library/api/unpermute-a2av>`: Packed routing is now required. The ``send_indices`` parameter has been removed and replaced by the required ``packed_source_indices`` (destination-major source indices, built with :doc:`Build Packed Routing </nki/library/api/build-packed-routing>`). ``permute_a2av`` adds ``recv_capacity`` and ``unpermute_a2av`` adds ``send_counts``; both are inserted before existing parameters, so callers must switch to keyword arguments. Receive buffers are now sized by capacity (``[C, H]``) instead of ``[EP*T, H]``.
* :doc:`Blockwise MM Forward MXFP8 </nki/library/api/blockwise-mm-forward-mxfp8>` and :doc:`Blockwise MM Backward MXFP8 </nki/library/api/blockwise-mm-backward-mxfp8>`:

  * The tuning keyword arguments (including ``gate_up_weight_is_swizzled``, ``down_weight_is_swizzled``, ``gate_up_config``/``down_config``, ``phase1_config``–``phase4_config``, ``fp8_x4_dtype``, ``spill_reload``, ``use_scale_packing``, ``run_with_lnc2``, ``shard_option``, ``affinity_option``, ``compute_dtype``, ``skip_dma``, ``is_tensor_update_accumulating``, ``no_indirect_load``, ``clamp_limits``, ``activation_type``, ``bias``, ``checkpoint_config``, and ``skip_grad_initialization``) have been removed from ``blockwise_mm_fwd_mxfp8`` and ``blockwise_mm_bwd_mxfp8``. Pass them through the new ``config`` parameter (``MXFP8MOEFwdConfig`` / ``MXFP8MOEBwdConfig``) instead. ``config`` is inserted before the weight-scale parameters, so positional callers must switch to keyword arguments.
  * The config modules moved: import ``MXFP8MOEFwdConfig`` from ``experimental/moe_mxfp8/fwd/config.py`` (was ``moe_fwd_mxfp8_config.py``) and ``MXFP8MOEBwdConfig`` from ``experimental/moe_mxfp8/bwd/config.py`` (was ``moe_bwd_mxfp8_config.py``).
  * The forward ``gate_up_proj_weight`` layout changed from ``[E, 2, I_TP, H]`` to ``[E, I_TP, 2, H]`` (per intermediate channel, ``[gate(H), up(H)]`` contiguous). Callers must re-layout the weight.
  * ``MXFP8MOECheckpointConfig.save_scaled_intermediate`` and the ``scaled_intermediate_checkpoint_T`` parameter of ``blockwise_mm_fwd_dropless_mxfp8`` have been removed.
* :doc:`Matmul MXFP8 </nki/library/api/matmul-mxfp8-generic-kernel>`:

  * The ``lhs_is_f_by_k`` and ``rhs_is_f_by_k`` parameters have been replaced by ``lhs_orientation`` and ``rhs_orientation`` (``TensorOrientation``).
  * The default ``float8_dtype`` changed from ``"float8_e5m2"`` to ``"float8_e4m3fn"`` in both ``matmul_mxfp8`` and ``matmul_mxfp8_backward``. Callers relying on the old default get different numerics with no error; pass ``float8_dtype="float8_e5m2"`` to keep the previous behavior.
  * In ``MatmulMxfp8KernelConfig``, ``run_with_lnc2`` and ``lnc_2_shard_rhs`` have been replaced by ``lnc_sharding``, and ``lhs_is_swizzled``, ``rhs_is_swizzled``, and ``load_with_PE_swizzle`` have been removed. In ``TensorDescriptor``, ``is_f_by_k``, ``load_with_PE_swizzle``, and ``fast_dma_transpose`` have been replaced by ``orientation`` and ``swizzle_mode``.
* :doc:`MLA QKV CTE </nki/library/api/mla-qkv-cte>`: The ``qr_qtz_hbm`` and ``qr_scale_hbm`` parameters have been removed. The kernel now allocates and returns the qr intermediate itself, so the number of outputs changed from 4 to 6 (MX: ``qr_qtz``, ``qr_scale``) or 5 (BF16: ``qr``). Callers must update how they unpack the outputs.
* :doc:`Attention MXFP8 TKG </nki/library/api/attention-mxfp8-tkg>`: The ``mask`` input changed from the kernel's internal eviction layout ``[B * num_chunks, 128, score_free]`` to a token-sequential ``[B, H, s_active, s_prior + s_active]`` tensor. ``q``, ``k_active``, ``v_active``, and ``out_hbm`` gained an ``s_active`` axis (``[B, H, s_active, d]``, ``[B, s_active, d]``). Callers must pass the new shapes.
* :doc:`SWA Fused CTE </nki/library/api/swa-fused-cte>`: ``k_scale`` and ``v_scale`` changed from ``[1, 1]`` to ``[128, 1]`` (the per-tensor scale replicated across partitions, matching ``qkv``). Callers must replicate the scale.
* :doc:`Conv3D Transpose </nki/library/api/conv3d-transpose>`: ``padding`` is now an asymmetric 6-value tuple ``(pad_d_lo, pad_d_hi, pad_h_lo, pad_h_hi, pad_w_lo, pad_w_hi)``; 3-value tuples are no longer accepted. ``output_padding`` is inserted before ``activation_fn``, so positional callers must switch to keyword arguments.
* **Positional-argument shifts (use keyword arguments)**: These kernels inserted new parameters in the middle of their signatures. Callers passing later arguments positionally must switch to keyword arguments:

  * :doc:`MoE CTE </nki/library/api/moe-cte>`: ``ep_rank`` inserted before ``accumulation_dtype``.
  * **Blockwise MM Shard-on-Block MX** (``bwmm_shard_on_block_mx``): ``ep_rank`` inserted before ``block_size``.
  * :doc:`QKV </nki/library/api/qkv>` (``qkv_cte``): ``head_split_kv_cache`` inserted before ``block_size``.
  * **Blockwise MM Backward Dropless MXFP8** (``blockwise_mm_bwd_dropless_mxfp8``): ``output_grad_swizzle_mode``, ``d_gate_up_swizzle_mode``, ``d_gate_up_t_swizzle_mode``, ``hidden_states_t_swizzle_mode``, ``output_grad_t_swizzle_mode``, and ``scaled_intermediate_t_swizzle_mode`` inserted before ``config``.

* **Deprecation**: The ``lnc_shard`` parameter of :doc:`Conv3D </nki/library/api/conv3d>` and :doc:`Conv3D Transpose </nki/library/api/conv3d-transpose>` no longer has any effect and emits a warning when set. It will be removed in a future release.

Bug Fixes
~~~~~~~~~

* **Attention TKG**: Re-enabled the QK-swapped decode path (disabled in 2.32 as an out-of-bounds workaround) with the missing index clamp.
* **Attention Block TKG**: Corrected the Q/K transpose layout in multi-batch decode; removed an underflow in the K-cache update.
* **Attention Segmented CTE**: Guarded the indirect-DGE gathers with ``oob_mode.skip`` to prevent out-of-bounds reads.
* **Fused Segmented Attention**: Fixed an SBUF clobber of the saved prior offset.
* **Attention KV-Parallel Segmented CTE**: Fixed collective operand placement for LNC1 and LNC>1.
* **Attention Backward**: Halved the fp32 KV section length so the kernel fits in SBUF.
* **SWA Fused CTE**: Passes the Q scale as an SBUF pointer to work around a Scalar Engine erratum.
* **Ring Attention Forward**: Fixed compilation for ``bs * nheads == 1`` on LNC2, and fixed ``cp_offset_hbm`` handling.
* **QKV**: Corrected the QKV TKG SBUF reservation; fixed the QKV CTE multi-buffer degree collapsing to 0 at large H.
* **Output Projection TKG**: Corrected perf-mode access.
* **MLP**: Fixed the MLP CTE 2x-row remainder tile and skipped empty subtiles in row quantization; corrected the Llama3-70B high-batch MLP TKG gate/up weight-ring prefetch, restored static quantization, and set ``reduce_cmd`` on its activation reductions.
* **RMSNorm MX Prefill**: The kernel now works without shuffled input.
* **MoE CTE**: Fixed dynamic-block clamping and expert-parallel slicing with packed affinities in ``bwmm_shard_on_block_mx``, skipped the dummy-block output write, vectorized the token-position-to-id load, and capped ``bwmm_shard_on_I`` PSUM usage only when the grid exceeds the available banks.
* **Scatter-Add**: Duplicate indices within a tile are now accumulated instead of overwritten.
* **Unpermute A2AV**: Fixed the tail tile when the token count is not a multiple of the tile size.
* **Foreach Norm**: Guarded the cross-core reduce so the kernel supports LNC1.
* **GPSIMD Top-K**: Bounded the gather indices and folded unmatchable sort keys.
* **SSD**: Eliminated an uninitialized SBUF read in the transpose helper.
* **MLA (DeepSeek)**: Made BF16 V-up allocation names unique across S-batches; fixed PSUM allocation in ``mla_sparse_attention_cte`` and the weighted reduce in the sparse attention indexer; added native and swizzled compact scales; fixed scale-swizzle group loops.
* **MXFP8 kernels**: Fixed MXFP8 matmul autotune cache lookups and the swizzle layout for ``K % tile_k == 384``; clamped the K-by-F PE-swizzle load to the F extent; sliced the DGT vector offset for F-remainder tiles; guarded against loading 0-sized tiles; gated ``MxMatmulScalePidx`` to Trainium 3; fixed an SbufManager leak in the MXFP8 MoE forward.
* **NeuroTile**: Fixed unaligned tile extents in block grids, partial last blocks in ``BlockStream``, true extents in the remainder flow, and oversized SBUF slots in ``HBMLayout`` transpose loads.

Known Issues
~~~~~~~~~~~~


.. _nki-lib-2-32-0-rn:

NKI Library (NKI-Lib) (Neuron 2.32.0 Release)
--------------------------------------------------------------------

Date of Release: 08/17/2026

What's New
~~~~~~~~~~

This release adds 12 new experimental kernels and 1 new core kernel. Highlights include the split DeepSeek-V3.2 MLA context-encoding path (QKV projection, sparse latent attention, and V-up + output projection), the DeepSeek Sparse Attention Indexer, an MXFP8 flash-decode attention kernel, MXFP8 forward blockwise MoE and MXFP8 matmul backward, 3D transposed and temporally-unrolled convolutions, a fused GPT-OSS sliding-window-attention block, a GpSIMD-based top-K, and an all-to-all dispatch-metadata builder for MoE. A new fused RMSNorm + MX-quantize (+ optional router top-K) prefill kernel is added to core. Existing kernels gain context-parallel options across attention (``attention_tkg``, ``attention_block_tkg``, ``gen_mask_tkg``), squared-sum outputs in ``qkv``, ``skip_gate_proj``/``accumulation_dtype`` controls across the blockwise MoE kernels, expanded MX quantization options, and new precision/layout controls in the MXFP8 matmul kernel. The ``mlp_cte`` kernel's internal implementation was refactored into separate ``basic`` and MX variants with no change to its public entry point. PyTorch reference implementations were added for 22 additional kernels. This release also removes several deprecated ``neurotile`` helper APIs and legacy tensor-view utilities — see Breaking Changes.

New Experimental Kernels
^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`SWA Fused CTE </nki/library/api/swa-fused-cte>` — Fused GPT-OSS sliding-window-attention block (QKV projection, RoPE, attention with sink, and output projection); returns the attention output with the K/V caches updated in place, including a packed-FP8 KV-cache path.
* :doc:`Attention MXFP8 TKG </nki/library/api/attention-mxfp8-tkg>` — MXFP8 flash-decode attention over a quantized block KV cache with separate active/prior KV blocks and packed-Q eviction (Trainium 3).
* :doc:`Conv3D Temporal Unroll </nki/library/api/conv3d-temporal-unroll>` — 3D convolution with temporal unrolling and column tiling for small ``C_out``, plus a ``should_use_temporal_unroll`` advisory check.
* :doc:`Conv3D Transpose </nki/library/api/conv3d-transpose>` — 3D transposed convolution (ConvTranspose3d) implemented by delegating to an embedded 3D convolution core with remapped parameters.
* :doc:`Matmul MXFP8 Backward </nki/library/api/matmul-mxfp8-generic-backward-kernel>` — Backward pass for a linear layer with MXFP8 quantization, computing both input gradients (dX) and weight gradients (dW).
* :doc:`MLA QKV CTE </nki/library/api/mla-qkv-cte>` — DeepSeek MLA QKV projection (MX) for context encoding, consuming the packed MX activation from ``rmsnorm_mx_prefill`` and emitting absorbed latents.
* :doc:`MLA Sparse Attention CTE </nki/library/api/mla-sparse-attention-cte>` — Standalone sparse latent + RoPE attention (kernel A of the split DeepSeek-V3.2 sparse-MLA forward), S-sharded across cores.
* :doc:`MLA V-Up O-Proj CTE </nki/library/api/mla-vup-oproj-cte>` — Standalone MX V-up + MX output projection (kernel B of the split DeepSeek-V3.2 sparse-MLA forward), S-sharded across cores.
* :doc:`Build All2All Dispatch Metadata </nki/library/api/build-all2all-dispatch-metadata>` — Builds per-rank ``send_counts`` (with token deduplication) and ``send_displs`` from ``expert_index`` for MoE all-to-all dispatch.
* :doc:`Blockwise MM Forward MXFP8 </nki/library/api/blockwise-mm-forward-mxfp8>` — MXFP8 forward pass for blockwise (dropless) Mixture of Experts; emits the activation checkpoints consumed by the MXFP8 MoE backward, forming a validated training pair.
* :doc:`Sparse Attention Indexer (MX + BF16 Score) </nki/library/api/sparse-attention-indexer-mx-bf16score>` — DeepSeek Sparse Attention Indexer with MX projections and a BF16 score path.
* :doc:`GPSIMD Top-K </nki/library/api/gpsimd-topk>` — Top-k over the last dimension using the GpSIMD ``nisa.topk`` instruction, with a ``create_gpsimd_topk_config`` helper.

New Core Kernels
^^^^^^^^^^^^^^^^

* :doc:`RMSNorm MX Prefill </nki/library/api/rmsnorm-mx-prefill>` — Fused RMSNorm ``[T,H]`` + MX quantization (+ optional router top-K) for prefill, with optional residual add and an optional bf16 norm output.

Improvements
~~~~~~~~~~~~~~~

* :doc:`Attention TKG </nki/library/api/attention-tkg>`: Added ``cp_softmax_stats_out`` for context-parallel softmax statistics.
* :doc:`Attention KV-Parallel Segmented CTE </nki/library/api/attention-kv-parallel-segmented-cte>`: Added ``apc_mode`` and ``valid_num_prior_tokens`` parameters.
* :doc:`Attention Block TKG </nki/library/api/attention-block-tkg>`: Added context-parallel parameters ``CP``, ``CP_replica_group``, and ``CP_collective_mode``.
* **Gen Mask TKG**: Added ``transposed_out`` and ``cp_seq_offset`` parameters to ``gen_mask_tkg`` and ``gen_mask_tkg_hbm``.
* :doc:`QKV </nki/library/api/qkv>`: Added ``q_squared_sum_out``, ``k_squared_sum_out``, and ``v_squared_sum_out`` outputs (also added to the CTE variant).
* :doc:`MoE CTE </nki/library/api/moe-cte>`: Added ``accumulation_dtype`` and ``skip_gate_proj`` parameters.
* **Blockwise MM (shard-on-I / shard-on-I-MX)**: Added ``skip_gate_proj``, ``accumulation_dtype``, and ``use_block128_scales`` parameters to the intermediate-sharded blockwise MoE kernels.
* :doc:`Blockwise MM Backward </nki/library/api/blockwise-mm-backward>`: Added ``skip_gate_proj``.
* **Blockwise MM (shard-on-H)**: Added ``activation_dtype`` and ``accum_dtype`` to the hidden-sharded blockwise MoE kernel.
* :doc:`Matmul MXFP8 </nki/library/api/matmul-mxfp8-generic-kernel>`: Added ``fast_dma_transpose``, ``enable_psum_copy_in``, and ``quant_scheme`` parameters; added ``inputs_were_prequantized`` to the block-level entry and ``enable_psum_copy_in`` to the default config generator.
* **RMSNorm MX Quantize TKG**: Added ``is_static_mx`` for static MX scaling.
* **MLP CTE**: Internal implementation refactored into separate ``basic`` (NONE/ROW/STATIC) and MX (MX/STATIC_MX/ROW_MX) variants dispatched by the existing ``mlp_cte`` router; no change to the public entry point.
* Added PyTorch reference implementations for 22 additional kernels for testing and validation.

Breaking Changes
~~~~~~~~~~~~~~~~

* **Neurotile core APIs**: In ``experimental/neurotile/core``, the public helpers ``tensor_view`` (``factories.py``), ``is_identity_tensor`` (``_helpers.py``), and the shape-transform helpers ``compute_reshape``, ``compute_reshape_dim``, ``compute_permute``, ``compute_flatten_dims``, ``compute_squeeze_dim``, ``compute_expand_dim``, and ``compute_broadcast`` (``transforms.py``) have been removed. In addition, the ``root`` and ``buffer_type`` parameters were removed from ``factories.tiles`` and ``factories.blocks``. Callers must migrate off these APIs and drop the removed keyword arguments.
* **Legacy tensor-view helpers**: The tensor-view convenience helpers ``alloc_tensor_view`` and ``convert_params_to_views`` (``core/mlp/mlp_tkg/mlp_tkg_utils.py``), ``safe_tensor_view`` (``core/moe/moe_tkg/moe_tkg_utils.py``), and ``alloc_tensor_view`` (``core/moe/moe_tkg/projection_utils.py``) have been removed. Callers must use ``TensorView`` directly.
* **MXFP8 load utils**: ``load_tile_fp32_transpose`` and ``load_tile_bf16_PE_transpose`` have been removed from ``experimental/mxfp_utils/mxfp8_utils/load_apis.py``. Callers must switch to the remaining load APIs.
* **Rotational Top-K**: The public functions ``prepare_rotational_constants`` and ``cleanup_rotational_constants`` (``core/topk/rotational_topk.py``) and the ``RotationalConstants`` class (``core/topk/rotational_topk_utils.py``) have been removed without an alias. Callers depending on these must migrate.
* **MoE TKG down projection**: The ``is_row_quant`` parameter has been removed from ``down_projection_mx`` (``core/moe/moe_tkg/down_projection_mx.py``). Callers using this keyword argument will get a ``TypeError`` and must remove it.
* **RMSNorm MX Quantize TKG (validation)**: The ``has_gate_up_in_scale`` parameter has been removed from ``validate_rmsnorm_mx_quantize_tkg`` (``core/subkernels/norm_tkg_utils.py``).
* **Positional-argument shifts (use keyword arguments)**: Several kernels inserted new parameters in the middle of their signatures, shifting later positional arguments. Callers passing these arguments positionally must switch to keyword arguments:

  * :doc:`Attention TKG </nki/library/api/attention-tkg>`: ``cp_softmax_stats_out`` inserted before ``DBG_TENSORS``.
  * :doc:`QKV </nki/library/api/qkv>` (``qkv_cte``): ``q_squared_sum_out``/``k_squared_sum_out``/``v_squared_sum_out`` inserted before ``strided_input_config``.
  * **Gate-Up Projection MX** (``gate_up_projection_mx_tp``): ``is_packed_moving_scale`` inserted before ``activation_op``.
  * **RMSNorm MX Quantize TKG** (``rmsnorm_mx_quantize_tkg``): ``is_static_mx`` inserted before ``output_row_dequant_scale``.

Bug Fixes
~~~~~~~~~

* **SWA Fused CTE**: Guarded the indirect-DGE and block-table K/V gathers with ``oob_mode.skip`` so ``-1``-padded ``block_tables`` and packed-FP8 padded prefill do not read out of bounds; unified the fused projection tile width to 256 so TP8 compiles on Trainium 3; corrected the V-cache layout in the packed-FP8 KV cache.
* **Attention TKG**: Removed ``no_reorder`` on attention TKG MM1.
* **Attention CTE**: Share one GpSimd register for V-tile indirect offsets; disabled row-tiled attention with manual allocation.
* **QKV CTE**: Guard ``flatten_dims`` for a 1-D ``slot_mapping``.
* **Gen Mask TKG**: Thorough cleanup of gen-mask-tkg.
* **Segmented Prefill**: Added packed-FP8 support for V in the segmented prefill kernel.
* **Ring Attention**: Handle fully masked ring-attention rows.
* **RoPE**: Group-tile ``(B, n_heads)`` so the SBUF working set fits.
* **MoE (Blockwise)**: Weight-skipping budget fix for ``bwmm_shard_on_block_mx``; prevent invalid gate reads in the blockwise MoE backward.
* **MoE TKG**: Dropped redundant ``TensorView`` wrapping in the MX loaders; materialize ``TensorView`` to ``NkiTensor`` before ``dma_copy``.
* **MLP TKG**: Made the LayerNorm reduction non-shard-on-H to remove an LNC2 race.
* **GPSIMD / Rotational Top-K**: Corrected ``gpsimd_topk`` small-tile races and padding masking; restricted reads to valid partitions to eliminate uninitialized reads in LNC-2 rotational top-K.
* **MX Quantize**: Zero-initialize the MX scale buffer (memset) to avoid uninitialized reads; avoid uninitialized SBUF reads in ``pad_compute``.
* **Transformer TKG**: Fixed timeout errors in the separation pass.
* **MXFP8 kernels**: Added ``with_active_sbm`` cleanup to the MXFP8 production kernels; fixed an incorrect output initialization that caused end-to-end numeric failures.

Known Issues
~~~~~~~~~~~~

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

* **TensorView**: Added ``rearrange`` method for flexible dimension reordering, ``has_dynamic_access`` for checking whether a view requires runtime-dependent addressing, and ``key_in_dict`` helper. The ``slice`` method now clamps the end index to dimension bounds instead of asserting.
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



    