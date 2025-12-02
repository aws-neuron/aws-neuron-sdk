.. meta::
   :description: Comprehensive NKI developer guides for AWS Neuron Kernel Interface. Learn NKI programming, performance optimization, architecture guides, and custom operator development for Trainium and Inferentia.
   :date-modified: 12-02-2025

.. _nki_developer_guide:

NKI Developer How-To Guides
============================

Comprehensive guides for developing high-performance kernels with the Neuron Kernel Interface (NKI). These resources cover everything from basic NKI concepts to advanced performance optimization techniques for AWS Trainium and Inferentia accelerators.

For details on programming with NKI APIs and syntax, see :doc:`/nki/deep-dives/programming_model`.

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: NKI Kernel Optimization Guide
      :link: kernel-optimization
      :link-type: doc
      :class-card: sd-border-2

      Learn the basics of NKI kernel optimization with this code-backed guide.

.. grid:: 2
   :gutter: 2


   .. grid-item-card:: Framework Custom Operators
      :link: framework_custom_op
      :link-type: doc
      :class-card: sd-border-2

      Integrate NKI kernels as custom operators in PyTorch and JAX frameworks

   .. grid-item-card:: How to Profile a NKI Kernel
      :link: use-neuron-profile
      :link-type: doc
      :class-card: sd-border-2

      Learn how to profile a NKI kernel with :doc:`Neuron Explorer </tools/neuron-explorer/index>`.

   .. grid-item-card:: Profiling NKI Kernels (Legacy Guide)
      :link: neuron_profile_for_nki
      :link-type: doc
      :class-card: sd-border-2

      Performance analysis and debugging techniques using Neuron Profile tools

   .. grid-item-card:: NKI Performance Guide
      :link: nki_perf_guide
      :link-type: doc
      :class-card: sd-border-2

      Advanced optimization strategies for maximizing kernel performance and efficiency

   .. grid-item-card:: Direct Allocation Guide
      :link: nki_direct_allocation_guide
      :link-type: doc
      :class-card: sd-border-2

      Manual memory management techniques for fine-grained control over data placement

   .. grid-item-card:: Block Dimension Migration
      :link: nki_block_dimension_migration_guide
      :link-type: doc
      :class-card: sd-border-2

      Migration guide for updating kernels to use new block dimension features

.. toctree::
      :maxdepth: 1
      :hidden:

      kernel-optimization
      framework_custom_op
      use-neuron-profile
      neuron_profile_for_nki
      nki_perf_guide
      nki_direct_allocation_guide
      nki_block_dimension_migration_guide
