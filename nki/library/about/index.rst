.. meta::
    :description: Overviews and conceptual docs for the NKI Library . NKI Library provides pre-built NKI kernels you can use in model development with Neuron.
    :date-modified: 12/02/2025

.. _nkl_overviews_home:

About the NKI Library 
======================================

Learn about the NKI Library and the pre-built kernels it provides to accelerate the performance of your models.

What is the NKI Library?
-----------------------------------

The NKI Library is a collection of pre-built NKI kernels optimized for AWS Neuron-powered devices. These kernels are designed to accelerate machine learning workloads by providing efficient implementations of common operations used in deep learning models. NKI kernels are commonly used to implement custom PyTorch operators that run on NeuronCores, enabling developers to optimize performance-critical operations beyond what the Neuron Compiler generates automatically.

How do I use the NKI Library?
------------------------------

The NKI Library kernels are shipped directly with the Neuron Compiler on supported DLAMIs and container environments, so the easiest way to get started is to simply import them from the ``nkilib``namespace. For example, you can use ``from nkilib.kernels import <kernel_name>`` to integrate an optimized kernel into your model. 

Alternatively, if you want to browse the source code, contribute, or pin to a specific version, the kernels are also available in a public GitHub repository that you can clone and integrate into your Neuron-based model development workflow.

.. admonition:: NKI Library GitHub repository
    * **NKI Library repository**: https://github.com/aws-neuron/nki-library

    To get started using NKI Library kernels in your model development, clone or fork the repo and follow the instructions in the `README <https://github.com/aws-neuron/nki-library/blob/main/README.md>`_ file.

Resources
---------

* :doc:`NKI Library Kernel API Reference </nki/library/api/index>`
* :doc:`NKI Library Kernel Design Specifications </nki/library/specs/index>`
* :doc:`NKI Documentation </nki/index>`

    