.. meta::
    :description: Hands-on tutorials for AWS Neuron Kernel Interface (NKI), covering matrix operations, normalization techniques, advanced kernels, and distributed computing patterns.
    :keywords: NKI, AWS Neuron, Tutorials, Matrix Multiplication, Normalization, SPMD
    :date-modified: 12/01/2025

.. _nki-tutorials:

NKI Tutorials
==============

.. toctree::
   :maxdepth: 1
   :hidden:

   matrix_multiplication
   layernorm
   rmsnorm
   average_pool2d
   transpose2d
   fused-self-attn
   fused_mamba
   SPMD Tensor Addition <spmd_tensor_addition>
   Multi-core SPMD Addition <spmd_multiple_nc_tensor_addition>

This section provides hands-on tutorials for the Neuron Kernel Interface (NKI), demonstrating how to write custom kernels for AWS Trainium and Inferentia instances. These tutorials cover fundamental operations, advanced techniques, and distributed computing patterns using NKI.

The full source code of the following tutorials can be also viewed on the 
`nki-samples <https://github.com/aws-neuron/nki-samples/tree/main/src/nki_samples/tutorials/>`_ repository on GitHub.

Basic Operations
----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: matrix_multiplication
      :link-type: doc

      **Matrix Multiplication**
      ^^^
      Learn the fundamentals of implementing matrix multiplication in your NKI kernels.

   .. grid-item-card::
      :link: transpose2d
      :link-type: doc

      **2D Transpose**
      ^^^
      Implement efficient 2D matrix transpose operations using NKI

   .. grid-item-card::
      :link: average_pool2d
      :link-type: doc

      **Average Pooling 2D**
      ^^^
      Create custom 2D average pooling kernels for computer vision workloads

Normalization Techniques
------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: layernorm
      :link-type: doc

      **Layer Normalization**
      ^^^
      Implement layer normalization kernels for transformer models

   .. grid-item-card::
      :link: rmsnorm
      :link-type: doc

      **RMS Normalization**
      ^^^
      Build RMS normalization kernels for modern neural network architectures

Advanced Kernels
----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: fused-self-attn
      :link-type: doc

      **Fused Self-Attention**
      ^^^
      Advanced tutorial on implementing fused self-attention mechanisms

   .. grid-item-card::
      :link: fused_mamba
      :link-type: doc

      **Fused Mamba**
      ^^^
      Implement fused Mamba state space model kernels

Distributed Computing
---------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: spmd_tensor_addition
      :link-type: doc

      **SPMD Tensor Addition**
      ^^^
      Single Program Multiple Data tensor addition across multiple cores

   .. grid-item-card::
      :link: spmd_multiple_nc_tensor_addition
      :link-type: doc

      **Multi-Core SPMD Addition**
      ^^^
      Advanced SPMD tensor operations across multiple NeuronCores
