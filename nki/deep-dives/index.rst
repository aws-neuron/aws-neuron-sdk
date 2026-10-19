.. _nki_deep-dives_home:

.. meta::
    :description: Documentation home for the AWS Neuron SDK NKI Deep Dives and other advanced materials.
    :keywords: NKI, AWS Neuron, Deep Dives, Advanced Programming
    :date-modified: 12/01/2025

NKI Deep Dives
==============

This section provides in-depth technical documentation and guides for advanced users of the Neuron Kernel Interface (NKI). These deep dives offer detailed explanations of NKI concepts, programming patterns, and best practices to help you maximize the performance and capabilities of your NKI code on AWS Neuron devices.

Optimizing a NKI Kernel
-----------------------

.. grid:: 2
   :margin: 4 1 0 0

   .. grid-item-card:: NKI Performance Optimizations
      :link: nki_perf_guide
      :link-type: ref
      :class-body: sphinx-design-class-title-small

Advanced NKI Programming
------------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: MXFP4/8 Matrix Multiplication Guide
      :link: mxfp-matmul
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Perform matrix multiplication using MXFP8 data types in NKI kernels, including data layout, quantization, and tiling strategies.

   .. grid-item-card:: NKI Compiler
      :link: nki_compiler_about
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Learn about the NKI Compiler.

   .. grid-item-card:: NKI Loops
      :link: nki-loops
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Learn about different types of loop constructs in NKI.

   .. grid-item-card:: Descriptor Generation Engine (DGE)
      :link: dge-documentation
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Control how DMA descriptors are generated: pre-computed, software (GpSimd), or hardware DGE.

   .. grid-item-card:: DMA Bandwidth Guide
      :link: nki-dma-bandwidth-guide
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Guidelines for maximizing DMA bandwidth with large contiguous payloads.

   .. grid-item-card:: NKI Scalar Values
      :link: nki-scalar
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Read integers on device with nki.isa.scalar_load, and use them as dynamic offsets and loop bounds.

   .. grid-item-card:: NKI Access Patterns
      :link: nki-aps
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Learn about Access Patterns (AP) to directly specify how the Trainium hardware accesses tensors.


Additional NKI Information
--------------------------

.. toctree::
    :maxdepth: 1
    :hidden:

    Performance Optimizations <nki_perf_guide>
    MXFP8/4 Matrix Multiplication <mxfp-matmul>
    NKI Access Patterns <nki-aps>
    NKI Loops <nki-loops>
    NKI Scalar Values <nki-scalar>
    Descriptor Generation Engine (DGE) <nki-dge>
    DMA Bandwidth Guide <nki-dma-bandwidth-guide>
    nki-compiler
