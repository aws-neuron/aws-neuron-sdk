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

   .. grid-item-card:: **Tutorial: Use a Pre-built NKI Library Kernel**
      :class-card: sd-border-1
      :link: /nki/library/tutorial-use-a-prebuilt-kernel
      :link-type: doc

      Learn how to integrate an optimized multi-layer perceptron (MLP) kernel into your PyTorch model as an example use case.

   .. grid-item-card:: **NKI Library Kernel Design Specs**
      :class-card: sd-border-1
      :link: /nki/library/specs/index
      :link-type: doc

      Review the formal specifications for the pre-built NKI kernels available in the NKI Library.

.. grid:: 1 
   :gutter: 3

   .. grid-item-card:: **NKI Library Kernel API Reference**
      :class-card: sd-border-2

      Use the Kernel API reference to understand the functions, parameters, and usage of the pre-built NKI kernels in the NKI Library .

      * :doc:`Kernel API Reference <api/index>`


.. toctree::
   :maxdepth: 1
   :hidden:

   Overview <about/index>
   Tutorial: Use a NKI Library Kernel <tutorial-use-a-prebuilt-kernel>
   Kernel Design Specs <specs/index>
   Kernel API Reference <api/index>

