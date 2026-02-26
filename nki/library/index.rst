.. meta::
    :description: Home page for the NKI Library  documentation. NKI Library provides pre-built NKI kernels you can use in model development with Neuron.
    :date-modified: 12/02/2025

.. _nkl_home:

NKI Library Documentation
==========================

The NKI Library is a collection of pre-built kernels optimized for AWS Neuron-powered devices. These kernels are designed to accelerate machine learning workloads by providing efficient implementations of common operations used in deep learning models.

**NKI Library GitHub repository**: https://github.com/aws-neuron/nki-library

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: **NKI Library Kernel Design Specs**
      :class-card: sd-border-1
      :link: /nki/library/specs/index
      :link-type: doc

      Review the formal specifications for the pre-built NKI kernels available in the NKI Library.

   .. grid-item-card:: **NKI Library Kernel API Reference**
      :class-card: sd-border-1
      :link: /nki/library/api/index
      :link-type: doc

      Use the Kernel API reference to understand the functions, parameters, and usage of the pre-built NKI kernels in the NKI Library.

.. grid:: 1 
   :gutter: 3

   .. grid-item-card:: **NKI Library Kernel Utilities**
      :class-card: sd-border-1
      :link: /nki/library/kernel-utils/index
      :link-type: doc

      Utility modules for memory management, tensor views, and iteration helpers used in NKI kernel development.

   .. grid-item-card:: **NKI Library Release Notes**
      :class-card: sd-border-1
      :link: /release-notes/components/nki-lib
      :link-type: doc

      Release notes for the NKI Library kernels and APIs.

.. toctree::
   :maxdepth: 1
   :hidden:

   Overview <about/index>
   Kernel Design Specs <specs/index>
   Kernel API Reference <api/index>
   Kernel Utilities <kernel-utils/index>
   Release Notes </release-notes/components/nki-lib>

