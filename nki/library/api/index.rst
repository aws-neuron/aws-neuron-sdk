.. meta::
    :description: API reference for the pre-built NKI Library kernels included with the AWS Neuron SDK.
    :date-modified: 02/13/2026

.. _nkl_api_ref_home:

NKI Library Kernel API Reference
=================================

The NKI Library provides pre-built reference kernels you can use directly in your model development with the AWS Neuron SDK and NKI. These kernel APIs provide the default classes, functions, and parameters you can use to integrate the NKI Library kernels into your models.

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
   * - :doc:`Attention TKG </nki/library/api/attention-tkg>`
     - Implements attention optimized for Token Generation (decode) use cases with small active sequence lengths.

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

Cumulative Sum Kernels
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Cumsum </nki/library/api/cumsum>`
     - Computes cumulative sum along the last dimension with optimized tiling.

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

Convolution Kernels
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Depthwise Conv1D </nki/library/api/depthwise-conv1d>`
     - Implements depthwise 1D convolution using implicit GEMM algorithm.

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

    Attention Block TKG <attention-block-tkg>
    Attention CTE <attention-cte>
    Attention TKG <attention-tkg>
    Blockwise MM Backward <blockwise-mm-backward>
    Cross Entropy <cross-entropy>
    Cumsum <cumsum>
    Depthwise Conv1D <depthwise-conv1d>
    MLP <mlp>
    MoE CTE <moe-cte>
    MoE TKG <moe-tkg>
    Output Projection CTE <output-projection-cte>
    Output Projection TKG <output-projection-tkg>
    QKV <qkv>
    RMSNorm-Quant <rmsnorm-quant>
    RoPE <rope>
    Router Top-K <router-topk>
