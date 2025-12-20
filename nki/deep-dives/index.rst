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

   .. grid-item-card:: Profiling a NKI Kernel with Neuron Explorer
      :link: /nki/deep-dives/use-neuron-profile
      :link-type: doc
      :class-body: sphinx-design-class-title-small

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

   .. grid-item-card:: NKI Access Patterns
      :link: nki-aps
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Learn about Access Patterns (AP) to directly specify how the Trainium hardware accesses tensors.

   .. grid-item-card:: NKI Block Dimension Migration Guide
      :link: /nki/deep-dives/nki_block_dimension_migration_guide 
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Migrate NKI kernels to use block dimensions for improved performance and resource utilization on Trainium devices.


Additional NKI Information
--------------------------

.. grid:: 2
   :margin: 4 1 0 0

   .. grid-item-card:: NKI Beta Versions
      :link: nki-beta-versions
      :link-type: doc
      :class-body: sphinx-design-class-title-small
   
   .. grid-item-card:: NKI Beta Migration Guide
      :link: nki-migration-guide
      :link-type: doc
      :class-body: sphinx-design-class-title-small

.. toctree::
    :maxdepth: 1
    :hidden:

    Profile a NKI Kernel <use-neuron-profile>
    Performance Optimizations <nki_perf_guide>
    MXFP8/4 Matrix Multiplication <mxfp-matmul>
    Migrating Kernels to NKI Beta 2 <nki-migration-guide>
    NKI Access Patterns <nki-aps>
    Block Dimension Migration Guide <nki_block_dimension_migration_guide>
    nki-compiler
    nki-beta-versions
