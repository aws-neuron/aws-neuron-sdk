.. meta::
    :description: Guides for AWS Neuron Kernel Interface (NKI), including architectures, tutorials for implementing and optimizing kernels, and how to use kernels with common frameworks.
    :keywords: NKI, AWS Neuron, Guides, Tutorials, how-to
    :date-modified: 12/14/2025

.. _nki-guides:

NKI Guides
===========

This section provides hands-on tutorials for the Neuron Kernel Interface (NKI), demonstrating how to write custom kernels for AWS Trainium and Inferentia instances. These tutorials cover fundamental operations, advanced techniques, and distributed computing patterns using NKI.

Tutorials
---------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: nki-matrix-multiplication
      :link-type: ref

      **Matrix Multiplication**
      ^^^
      Learn the fundamentals of implementing matrix multiplication in your NKI kernels.

   .. grid-item-card::
      :link: nki-transpose2d
      :link-type: ref

      **Transpose 2D**
      ^^^
      Implement efficient 2D matrix transpose operations using NKI

   .. grid-item-card::
      :link: nki-averagepool2d
      :link-type: ref

      **Average Pooling 2D**
      ^^^
      Create custom 2D average pooling kernels for computer vision workloads

   .. grid-item-card::
      :link: nki-fused-mamba
      :link-type: ref

      **Fused Mamba**
      ^^^
      Implement fused Mamba state space model kernels

   .. grid-item-card::
      :link: nki-tutorial-spmd-tensor-addition
      :link-type: ref

      **SPMD Tensor Addition**
      ^^^
      Single Program Multiple Data tensor addition

   .. grid-item-card::
      :link: nki_spmd_multiple_nc_tensor_addition
      :link-type: ref

      **Multi-Core SPMD Addition**
      ^^^
      Advanced SPMD tensor addition across multiple NeuronCores

Architecture Guides
-------------------

Neuron recommends new NKI developers start with :doc:`Trainium/Inferentia2 Architecture Guide </nki/guides/architecture/trainium_inferentia2_arch>` before exploring newer NeuronDevice architecture.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Trainium/Inferentia2 Architecture Guide
      :link: trainium_inferentia2_arch
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Foundational architecture guide for understanding NeuronDevice basics.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Trainium2 Architecture Guide
      :link: trainium2_arch
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Architecture enhancements and improvements in the Trainium2 generation.

   .. grid-item-card:: Trainium3 Architecture Guide
      :link: trainium3_arch
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Latest architecture features and capabilities in Trainium3 devices.

How-To Guides
-------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: How to Insert NKI Kernels into Models
      :link: nki_framework_custom_op
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      How to insert a NKI kernel as a custom operator into a PyTorch or JAX model using simple code examples.

.. toctree::
   :maxdepth: 1
   :hidden:

   Tutorials </nki/guides/tutorials/index>
   Architecture </nki/guides/architecture/index>
   Insert NKI Kernels into Models </nki/guides/framework_custom_op>