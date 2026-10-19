.. meta::
    :description: Reference for the pre-built NKI Library kernels included with the AWS Neuron SDK.
    :date-modified: 10/19/2026

.. _nkl_api_ref_home:

NKI Library Supported Kernel Reference
======================================

The NKI Library provides pre-built reference kernels you can use directly in your model development with the AWS Neuron SDK and NKI. These kernels provide the default classes, functions, and parameters you can use to integrate the NKI Library kernels into your models.

**Source code for these kernel APIs can be found at**: https://github.com/aws-neuron/nki-library

Core Kernels
-------------

Normalization and Quantization Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`RMSNorm-Quant </nki/library/api/rmsnorm-quant>`
     - Performs optional RMS normalization followed by quantization to ``fp8``.
   * - :doc:`RMSNorm MX Prefill </nki/library/api/rmsnorm-mx-prefill>`
     - Fused RMSNorm + MX quantization (+ optional router top-K) for prefill, with optional residual add.

QKV Projection Kernels
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`QKV </nki/library/api/qkv>`
     - Performs Query-Key-Value projection with optional normalization and RoPE fusion.

Attention Kernels
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Attention CTE </nki/library/api/attention-cte>`
     - Implements attention optimized for Context Encoding (prefill) use cases.
   * - :doc:`Attention Segmented CTE </nki/library/api/attention-segmented-cte>`
     - Segmented attention with block-based KV cache and prefix caching for decode.
   * - :doc:`Attention TKG </nki/library/api/attention-tkg>`
     - Implements attention optimized for Token Generation (decode) use cases with small active sequence lengths.
   * - :doc:`Attention KV-Parallel Segmented CTE </nki/library/api/attention-kv-parallel-segmented-cte>`
     - KV-parallel segmented prefill attention with paged KV cache, sliding-window, and round-robin KV distribution.

Rotary Position Embedding (RoPE) Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`RoPE </nki/library/api/rope>`
     - Applies Rotary Position Embedding to input embeddings with flexible layout support.

Multi-Layer Perceptron (MLP) Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`MLP </nki/library/api/mlp>`
     - Implements Multi-Layer Perceptron with optional normalization fusion and quantization support.

Output Projection Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Output Projection CTE </nki/library/api/output-projection-cte>`
     - Computes output projection optimized for Context Encoding use cases.
   * - :doc:`Output Projection TKG </nki/library/api/output-projection-tkg>`
     - Computes output projection optimized for Token Generation use cases.

Mixture of Experts (MoE) Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Router Top-K </nki/library/api/router-topk>`
     - Computes router logits, applies activation functions, and performs top-K selection for MoE models.
   * - :doc:`MoE CTE </nki/library/api/moe-cte>`
     - Implements Mixture of Experts MLP operations optimized for Context Encoding use cases.
   * - :doc:`MoE TKG </nki/library/api/moe-tkg>`
     - Implements Mixture of Experts MLP operations optimized for Token Generation use cases.

Quantization Kernels
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`FP8 Quantize </nki/library/api/fp8-quantize>`
     - Static and row-wise dynamic FP8 quantization with pre-combined dequantization scale support.

Cumulative Sum Kernels
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Cumsum </nki/library/api/cumsum>`
     - Computes cumulative sum along the last dimension with optimized tiling.

Core Subkernels
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Find Nonzero Indices </nki/library/api/find-nonzero-indices>`
     - Finds indices of nonzero elements along the T dimension using GpSimd ``nonzero_with_count`` ISA.

Experimental Kernels
---------------------

.. note::
   Experimental kernels are under active development and their APIs may change in future releases.

Attention Kernels
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Attention Block TKG </nki/library/api/attention-block-tkg>`
     - Fused attention block for Token Generation that keeps all intermediate tensors in SBUF to minimize HBM traffic.
   * - :doc:`Ring Attention Forward </nki/library/api/ring-attention-fwd>`
     - Ring attention forward pass for context parallelism across multiple workers.
   * - :doc:`Ring Attention Backward </nki/library/api/ring-attention-bwd>`
     - Ring attention backward pass SPMD kernel for context parallelism.
   * - :doc:`Ring Attention Unpermute </nki/library/api/ring-attention-unpermute>`
     - Reorders striped ring-attention output back to contiguous sequence order.
   * - :doc:`QKV CTE MLA </nki/library/api/qkv-cte-mla>`
     - DeepSeek Multi-head Latent Attention (MLA) QKV projection with MX quantization for context encoding.
   * - :doc:`SWA Fused CTE </nki/library/api/swa-fused-cte>`
     - Fused GPT-OSS sliding-window-attention block with QKV projection, RoPE, attention with sink, and output projection; updates the K/V caches in place.
   * - :doc:`Attention MXFP8 TKG </nki/library/api/attention-mxfp8-tkg>`
     - MXFP8 flash-decode attention over a quantized block KV cache with packed-Q eviction (Trainium 3).
   * - :doc:`Sparse Attention Indexer (MX + BF16 Score) </nki/library/api/sparse-attention-indexer-mx-bf16score>`
     - DeepSeek Sparse Attention Indexer with MX projections and a BF16 score path.
   * - :doc:`MLA QKV CTE </nki/library/api/mla-qkv-cte>`
     - DeepSeek MLA QKV projection (MX) for context encoding, emitting absorbed latents.
   * - :doc:`MLA Sparse Attention CTE </nki/library/api/mla-sparse-attention-cte>`
     - Standalone sparse latent + RoPE attention (kernel A of the split DeepSeek-V3.2 sparse-MLA forward).
   * - :doc:`MLA V-Up O-Proj CTE </nki/library/api/mla-vup-oproj-cte>`
     - Standalone MX V-up + MX output projection (kernel B of the split DeepSeek-V3.2 sparse-MLA forward).
   * - :doc:`Attention Const-Max </nki/library/api/attention-const-max>`
     - Attention with a caller-supplied global softmax maximum, supporting sequence-major Q/K layouts.
   * - :doc:`Flash Attention 2 </nki/library/api/flash-attention-2>`
     - FlashAttention-2 forward for unmasked bidirectional attention over long key sequences (Trainium 3).
   * - :doc:`MLA Decode QKV </nki/library/api/mla-decode-qkv>`
     - DeepSeek-V3.2 MLA query and key/value projections for decode, with RoPE, query absorption, and paged KV-cache write.
   * - :doc:`MLA Decode Attention </nki/library/api/mla-decode-attention>`
     - DeepSeek-V3.2 MLA decode attention over indexer-selected keys, with V up-projection and output projection.
   * - :doc:`MLA Attention Block Gather Decode </nki/library/api/mla-attention-block-gather-decode>`
     - End-to-end DeepSeek-V3.2 MLA decode attention block (``mla_decode_qkv`` + ``mla_decode_attention``).
   * - :doc:`DeepSeek-V3.2 Sparse Indexer </nki/library/api/deepseek-v32-sparse-indexer>`
     - DeepSeek Sparse Attention lightning indexer for decode (Trainium 3).

Linear Attention Kernels
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`KDA TKG </nki/library/api/kda-tkg>`
     - Kimi Delta Attention token generation: advances the KDA recurrence by one token per request.
   * - :doc:`KDA Block TKG </nki/library/api/kda-block-tkg>`
     - Full Kimi Delta Attention block for token generation with all intermediates kept in SBUF.
   * - :doc:`GDN CTE </nki/library/api/gdn-cte>`
     - Chunked gated delta-rule (GDN) prefill with a selectable intra-chunk inverse algorithm.
   * - :doc:`GDN TKG </nki/library/api/gdn-tkg>`
     - Fused single-token gated delta-rule decode step.
   * - :doc:`GDN Block TKG </nki/library/api/gdn-block-tkg>`
     - Fused Qwen3.5 GDN decode layer (input projections, conv1d, GQA, gated delta-rule).
   * - :doc:`GDN Conv1D </nki/library/api/gdn-conv1d>`
     - Depthwise causal conv1d + SiLU + ``conv_state`` update for GDN prefill and decode.

Deformable Attention Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`MS Deformable Attention </nki/library/api/ms-deformable-attention>`
     - Multi-scale deformable attention using bilinear interpolation and indirect DMA transpose.
   * - :doc:`MS Deformable Attention Backward </nki/library/api/ms-deformable-attention-bwd>`
     - Backward pass for multi-scale deformable attention.

Transformer Kernels
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Transformer TKG </nki/library/api/transformer-tkg>`
     - Multi-layer transformer forward pass megakernel for token generation.
   * - :doc:`GPT-OSS Tail Chunked Megakernel </nki/library/api/gpt-oss-tail-chunked-megakernel>`
     - GPT-OSS decode sampling tail (RMSNorm, ``lm_head``, top-K, sampler) with a single all-to-all collective.

Convolution Kernels
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Conv1D </nki/library/api/conv1d>`
     - 1D convolution using tensor engine with replication strategy.
   * - :doc:`Conv3D </nki/library/api/conv3d>`
     - 3D convolution using tensor engine with K-replication strategy and W-contiguous tiling.
   * - :doc:`Depthwise Conv1D </nki/library/api/depthwise-conv1d>`
     - Implements depthwise 1D convolution using implicit GEMM algorithm.
   * - :doc:`Conv3D Temporal Unroll </nki/library/api/conv3d-temporal-unroll>`
     - 3D convolution with temporal unrolling and column tiling for small ``C_out``.
   * - :doc:`Conv3D Transpose </nki/library/api/conv3d-transpose>`
     - 3D transposed convolution (ConvTranspose3d) via an embedded 3D convolution core with remapped parameters.
   * - :doc:`Conv3D Backward </nki/library/api/conv3d-bwd>`
     - Input, weight, and bias gradients of a 3D convolution with a geometry-driven planner.

Normalization Kernels
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Batch Norm Backward </nki/library/api/batch-norm-bwd>`
     - Batch normalization backward over the channel dimension, with optional fused ReLU.
   * - :doc:`RMSNorm Residual Prefill </nki/library/api/rmsnorm-residual-prefill>`
     - Pre-norm residual add followed by RMSNorm for prefill, with the add performed during the DMA load.

Pooling Kernels
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Max Pooling 3D </nki/library/api/max-pooling-3d>`
     - 3D max pooling over (D, H, W) in NCDHW or NDHWC layout.
   * - :doc:`Max Pooling 3D Backward </nki/library/api/max-pooling-3d-backward>`
     - Backward pass of 3D max pooling.
   * - :doc:`Average Pooling 3D </nki/library/api/avg-pooling-3d>`
     - 3D average pooling over (D, H, W) in NCDHW or NDHWC layout.
   * - :doc:`Average Pooling 3D Backward </nki/library/api/avg-pooling-3d-backward>`
     - Backward pass of 3D average pooling.

Collective Communication Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Fine-Grained All-Gather </nki/library/api/fg-allgather>`
     - Ring-based all-gather for TRN2 with double-buffered collective permute.
   * - :doc:`FGCC (All-Gather + Matmul) </nki/library/api/fgcc>`
     - Fused all-gather and matrix multiplication for TRN2.
   * - :doc:`SBUF-to-SBUF All-Gather </nki/library/api/sb2sb-allgather>`
     - SBUF-to-SBUF all-gather with variants for small and large tensors.
   * - :doc:`Collective Communication Kernels </nki/library/api/collectives>`
     - HBM-based all-reduce, all-gather, reduce-scatter, and all-to-all kernels.
   * - :doc:`Permute A2AV </nki/library/api/permute-a2av>`
     - MoE training dispatch: permute tokens by destination EP rank and exchange via all-to-all-v.
   * - :doc:`Unpermute A2AV </nki/library/api/unpermute-a2av>`
     - MoE training combine: all-to-all-v exchange and unpermute to original token order.
   * - :doc:`Build Packed Routing </nki/library/api/build-packed-routing>`
     - Builds the destination-major ``packed_source_indices`` consumed by ``permute_a2av`` and ``unpermute_a2av``.
   * - :doc:`QKV Batch Shard </nki/library/api/batch-shard>`
     - Q projection transition from tensor-parallel to tensor-and-data-parallel layout for batch-sharded attention.

Foreach Kernels
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Foreach Elementwise </nki/library/api/foreach-elementwise>`
     - Suite of fused elementwise operations (add, sub, mul, div, addcdiv, addcmul, lerp, sqrt) with SPMD tiling.
   * - :doc:`Foreach Norm </nki/library/api/foreach-norm>`
     - L1, L2, and Linf norm computation kernels with SPMD parallelization.

Matmul and MLP MXFP8 Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Matmul MXFP8 </nki/library/api/matmul-mxfp8-generic-kernel>`
     - Generic matrix multiplication with MXFP8 quantization, supporting BF16 and pre-quantized inputs.
   * - :doc:`MLP Forward MXFP8 </nki/library/api/mlp-fwd-mxfp8-kernel>`
     - MXFP8 SwiGLU MLP forward pass with optional activation checkpointing.
   * - :doc:`MLP Backward MXFP8 </nki/library/api/mlp-bwd-mxfp8-kernel>`
     - MXFP8 SwiGLU MLP backward pass with 4-phase gradient computation.
   * - :doc:`Matmul MXFP8 Backward </nki/library/api/matmul-mxfp8-generic-backward-kernel>`
     - Backward pass for a linear layer with MXFP8 quantization, computing both input (dX) and weight (dW) gradients.
   * - :doc:`MLP DeepSeek MX </nki/library/api/mlp-deepseek-mx>`
     - DeepSeek-V3.2 shared-experts and first dense MLP layers for prefill with MXFP8 weights and packed MX input.

MoE Kernels
~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`MX MoE Block TKG Wrapper </nki/library/api/mx-moe-block-tkg-wrapper>`
     - Wrapper that bitcasts unsigned integer weights to MX x4 dtype for MoE block.
   * - :doc:`Blockwise MM Shard-on-Block </nki/library/api/bwmm-shard-on-block-v2>`
     - Block-sharded blockwise matrix multiplication for context-encoding MoE layers.
   * - :doc:`RMSNorm Router Top-K A2AV </nki/library/api/rmsnorm-router-topk-a2av>`
     - Fused RMSNorm + router top-K for MoE token generation feeding the all-to-all-v dispatch path.
   * - :doc:`RMSNorm Router Top-K TKG </nki/library/api/rmsnorm-router-topk-tkg>`
     - Fused RMSNorm (+ optional MX quantize) + router top-K for MoE token generation.
   * - :doc:`Blockwise MM Forward MXFP8 </nki/library/api/blockwise-mm-forward-mxfp8>`
     - MXFP8 forward pass for blockwise (dropless) MoE; emits activation checkpoints for the MXFP8 MoE backward.
   * - :doc:`Build All2All Dispatch Metadata </nki/library/api/build-all2all-dispatch-metadata>`
     - Builds per-rank send counts (with token deduplication) and displacements from ``expert_index`` for MoE all-to-all dispatch.
   * - :doc:`DeepSeek-V3 Router </nki/library/api/deepseek-v3-router>`
     - DeepSeek-V3.2 group-limited ``noaux_tc`` routing for one decode step.
   * - :doc:`Blockwise MM Block MX </nki/library/api/bwmm-block-mx>`
     - DeepSeek-V3.2 blockwise MXFP4/MXFP8 MoE for context encoding on a single logical core (LNC1).

Indexing Kernels
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Gather </nki/library/api/gather>`
     - Gather rows from a 2D input by a 1D index using indirect DMA load.
   * - :doc:`Scatter-Add </nki/library/api/scatter-add>`
     - Scatter-add rows from src into a 2D input by a 1D index.

Top-K Kernels
~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`GPSIMD Top-K </nki/library/api/gpsimd-topk>`
     - Top-k over the last dimension using the GpSIMD ``nisa.topk`` instruction.
   * - :doc:`Chunked Top-K </nki/library/api/chunked-topk>`
     - Single-core top-K over a vocabulary wider than the ``nisa.topk`` limit, with no collective.

Optimizer Kernels
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Fused Adam/AdamW </nki/library/api/fused-adam>`
     - Fused Adam (L2 regularization) and AdamW (decoupled weight decay) optimizer kernels.

Padding Kernels
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Pad </nki/library/api/pad>`
     - Multi-mode tensor padding (constant, replicate, reflect, circular) following PyTorch semantics.

Quantization Kernels
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Quantize MXFP8 </nki/library/api/quantize-mxfp8>`
     - Block-wise BF16-to-MXFP8 quantization kernel with packed scale support.

RNG Kernels
~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`RNG </nki/library/api/rng>`
     - Random number generation kernels using GPSIMD engine with state management.

Scan Kernels (State Space Models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Linear Scan </nki/library/api/linear-scan>`
     - First-order linear recurrence computation along the last dimension.
   * - :doc:`Selective Scan </nki/library/api/selective-scan>`
     - Selective scan (SSM) as in Mamba models.
   * - :doc:`SSD </nki/library/api/ssd>`
     - State Space Duality scan for Mamba-2 models.

MoE Subkernels
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Top-K Reduce </nki/library/api/topk-reduce>`
     - MoE Top-K reduction across sparse all-to-all collective output.
   * - :doc:`Argsort Unstable </nki/library/api/argsort-unstable>`
     - Unstable argsort on 1D input buffer for MoE routing.
   * - :doc:`Build All-to-All-V Metadata </nki/library/api/build-all-to-all-v-metadata>`
     - Builds metadata buffer for all_to_all_v collective from MoE routing decisions.
   * - :doc:`Permute Routed Tokens </nki/library/api/permute-routed-tokens>`
     - Sorts tokens by expert and packs hidden states for MoE dispatch.

Dynamic Shape Kernels
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Dynamic Elementwise Add </nki/library/api/dynamic-elementwise-add>`
     - Elementwise addition with runtime-variable M-dimension tiling.

Loss Kernels
~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Cross Entropy </nki/library/api/cross-entropy>`
     - Memory-efficient cross entropy loss forward and backward passes using online log-sum-exp algorithm.

MoE Backward Kernels
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Blockwise MM Backward </nki/library/api/blockwise-mm-backward>`
     - Computes backward pass for blockwise matrix multiplication in Mixture of Experts layers.
   * - :doc:`Blockwise MM Backward MXFP8 </nki/library/api/blockwise-mm-backward-mxfp8>`
     - MXFP8-quantized backward pass for blockwise Mixture of Experts.

.. toctree::
    :maxdepth: 1
    :hidden:

    Argsort Unstable <argsort-unstable>
    Attention Block TKG <attention-block-tkg>
    Attention Const-Max <attention-const-max>
    Attention CTE <attention-cte>
    Attention KV-Parallel Segmented CTE <attention-kv-parallel-segmented-cte>
    Attention MXFP8 TKG <attention-mxfp8-tkg>
    Attention Segmented CTE <attention-segmented-cte>
    Attention TKG <attention-tkg>
    Average Pooling 3D <avg-pooling-3d>
    Average Pooling 3D Backward <avg-pooling-3d-backward>
    Batch Norm Backward <batch-norm-bwd>
    Blockwise MM Backward <blockwise-mm-backward>
    Blockwise MM Backward MXFP8 <blockwise-mm-backward-mxfp8>
    Blockwise MM Block MX <bwmm-block-mx>
    Blockwise MM Forward MXFP8 <blockwise-mm-forward-mxfp8>
    Blockwise MM Shard-on-Block <bwmm-shard-on-block-v2>
    Build All-to-All-V Metadata <build-all-to-all-v-metadata>
    Build All2All Dispatch Metadata <build-all2all-dispatch-metadata>
    Build Packed Routing <build-packed-routing>
    Chunked Top-K <chunked-topk>
    Collective Communication Kernels <collectives>
    Conv1D <conv1d>
    Conv3D <conv3d>
    Conv3D Backward <conv3d-bwd>
    Conv3D Temporal Unroll <conv3d-temporal-unroll>
    Conv3D Transpose <conv3d-transpose>
    Cross Entropy <cross-entropy>
    Cumsum <cumsum>
    DeepSeek-V3 Router <deepseek-v3-router>
    DeepSeek-V3.2 Sparse Indexer <deepseek-v32-sparse-indexer>
    Depthwise Conv1D <depthwise-conv1d>
    Dynamic Elementwise Add <dynamic-elementwise-add>
    FGCC <fgcc>
    Find Nonzero Indices <find-nonzero-indices>
    Fine-Grained All-Gather <fg-allgather>
    Flash Attention 2 <flash-attention-2>
    Foreach Elementwise <foreach-elementwise>
    Foreach Norm <foreach-norm>
    FP8 Quantize <fp8-quantize>
    Fused Adam <fused-adam>
    Gather <gather>
    GDN Block TKG <gdn-block-tkg>
    GDN Conv1D <gdn-conv1d>
    GDN CTE <gdn-cte>
    GDN TKG <gdn-tkg>
    GPSIMD Top-K <gpsimd-topk>
    GPT-OSS Tail Chunked Megakernel <gpt-oss-tail-chunked-megakernel>
    KDA Block TKG <kda-block-tkg>
    KDA TKG <kda-tkg>
    Linear Scan <linear-scan>
    Matmul MXFP8 <matmul-mxfp8-generic-kernel>
    Matmul MXFP8 Backward <matmul-mxfp8-generic-backward-kernel>
    Max Pooling 3D <max-pooling-3d>
    Max Pooling 3D Backward <max-pooling-3d-backward>
    MLA Attention Block Gather Decode <mla-attention-block-gather-decode>
    MLA Decode Attention <mla-decode-attention>
    MLA Decode QKV <mla-decode-qkv>
    MLA QKV CTE <mla-qkv-cte>
    MLA Sparse Attention CTE <mla-sparse-attention-cte>
    MLA V-Up O-Proj CTE <mla-vup-oproj-cte>
    MLP <mlp>
    MLP Backward MXFP8 <mlp-bwd-mxfp8-kernel>
    MLP DeepSeek MX <mlp-deepseek-mx>
    MLP Forward MXFP8 <mlp-fwd-mxfp8-kernel>
    MoE CTE <moe-cte>
    MoE TKG <moe-tkg>
    MS Deformable Attention <ms-deformable-attention>
    MS Deformable Attention Backward <ms-deformable-attention-bwd>
    MX MoE Block TKG Wrapper <mx-moe-block-tkg-wrapper>
    Output Projection CTE <output-projection-cte>
    Output Projection TKG <output-projection-tkg>
    Pad <pad>
    Permute A2AV <permute-a2av>
    Permute Routed Tokens <permute-routed-tokens>
    QKV <qkv>
    QKV Batch Shard <batch-shard>
    QKV CTE MLA <qkv-cte-mla>
    Quantize MXFP8 <quantize-mxfp8>
    Ring Attention Backward <ring-attention-bwd>
    Ring Attention Forward <ring-attention-fwd>
    Ring Attention Unpermute <ring-attention-unpermute>
    RMSNorm MX Prefill <rmsnorm-mx-prefill>
    RMSNorm Residual Prefill <rmsnorm-residual-prefill>
    RMSNorm Router Top-K A2AV <rmsnorm-router-topk-a2av>
    RMSNorm Router Top-K TKG <rmsnorm-router-topk-tkg>
    RMSNorm-Quant <rmsnorm-quant>
    RNG <rng>
    RoPE <rope>
    Router Top-K <router-topk>
    SBUF-to-SBUF All-Gather <sb2sb-allgather>
    Scatter-Add <scatter-add>
    Selective Scan <selective-scan>
    Sparse Attention Indexer MX BF16Score <sparse-attention-indexer-mx-bf16score>
    SSD <ssd>
    SWA Fused CTE <swa-fused-cte>
    Top-K Reduce <topk-reduce>
    Transformer TKG <transformer-tkg>
    Unpermute A2AV <unpermute-a2av>
