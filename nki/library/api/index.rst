.. meta::
    :description: Reference for the pre-built NKI Library kernels included with the AWS Neuron SDK.
    :date-modified: 05/21/2026

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
   * - :doc:`KV-Parallel Segmented Prefill </nki/library/api/kv-parallel-segmented-prefill>`
     - KV-parallel segmented prefill attention kernel.

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

Transformer Kernels
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Transformer TKG </nki/library/api/transformer-tkg>`
     - Multi-layer transformer forward pass megakernel for token generation.

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

MoE Kernels
~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`MX MoE Block TKG Wrapper </nki/library/api/mx-moe-block-tkg-wrapper>`
     - Wrapper that bitcasts unsigned integer weights to MX x4 dtype for MoE block.

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

.. toctree::
    :maxdepth: 1
    :hidden:

    Argsort Unstable <argsort-unstable>
    Attention Block TKG <attention-block-tkg>
    Attention CTE <attention-cte>
    Attention Segmented CTE <attention-segmented-cte>
    Attention TKG <attention-tkg>
    Blockwise MM Backward <blockwise-mm-backward>
    Build All-to-All-V Metadata <build-all-to-all-v-metadata>
    Conv1D <conv1d>
    Conv3D <conv3d>
    Cross Entropy <cross-entropy>
    Cumsum <cumsum>
    Depthwise Conv1D <depthwise-conv1d>
    Dynamic Elementwise Add <dynamic-elementwise-add>
    FGCC <fgcc>
    Find Nonzero Indices <find-nonzero-indices>
    Fine-Grained All-Gather <fg-allgather>
    Foreach Elementwise <foreach-elementwise>
    Foreach Norm <foreach-norm>
    FP8 Quantize <fp8-quantize>
    Fused Adam <fused-adam>
    KV-Parallel Segmented Prefill <kv-parallel-segmented-prefill>
    Linear Scan <linear-scan>
    Matmul MXFP8 <matmul-mxfp8-generic-kernel>
    MLP <mlp>
    MLP Backward MXFP8 <mlp-bwd-mxfp8-kernel>
    MLP Forward MXFP8 <mlp-fwd-mxfp8-kernel>
    MoE CTE <moe-cte>
    MoE TKG <moe-tkg>
    MX MoE Block TKG Wrapper <mx-moe-block-tkg-wrapper>
    Output Projection CTE <output-projection-cte>
    Output Projection TKG <output-projection-tkg>
    Pad <pad>
    Permute Routed Tokens <permute-routed-tokens>
    QKV <qkv>
    Quantize MXFP8 <quantize-mxfp8>
    Ring Attention Backward <ring-attention-bwd>
    Ring Attention Forward <ring-attention-fwd>
    RMSNorm-Quant <rmsnorm-quant>
    RNG <rng>
    RoPE <rope>
    Router Top-K <router-topk>
    SBUF-to-SBUF All-Gather <sb2sb-allgather>
    Selective Scan <selective-scan>
    SSD <ssd>
    Top-K Reduce <topk-reduce>
    Transformer TKG <transformer-tkg>
