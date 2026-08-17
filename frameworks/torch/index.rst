.. meta::
   :description: PyTorch support on AWS Neuron SDK - TorchNeuron Native for eager execution and torch.compile on Trainium and Inferentia, with torch-neuronx XLA-based support for training and inference.
   :keywords: PyTorch, TorchNeuron, torch-neuronx, AWS Neuron, Trainium, Inferentia, deep learning, torch.compile, eager mode
   :date-modified: 01/22/2026

.. _neuron-pytorch:
.. _pytorch-neuronx-main:

PyTorch Support on Neuron
==========================

PyTorch running on Neuron unlocks high-performance and cost-effective deep learning acceleration on AWS Trainium-based and AWS Inferentia-based Amazon EC2 instances.

The PyTorch plugin for Neuron architecture enables native PyTorch models to be accelerated on Neuron devices, so you can use your existing framework application and get started easily with minimal code changes.

PyTorch Neuron support is available at three levels:

* **TorchNeuron Native** *(recommended)*: The newest native PyTorch backend providing eager execution, ``torch.compile``, and standard distributed APIs (FSDP, DTensor, DDP, Tensor Parallelism) for Trainium and Inferentia. This is the recommended starting point for new workloads.
* **PyTorch NeuronX (torch-neuronx)** *(supported)*: The XLA-based PyTorch integration supporting NeuronCores v2 architecture (Trn1, Trn2, Inf2, Trn1n). Provides full capabilities for both training and inference workloads. (This library is not included in Neuron 2.32.0 and later versions.)
* **PyTorch Neuron (torch-neuron)** *(archived)*: The legacy PyTorch integration for NeuronCores v1 architecture (Inf1 only). This package is no longer actively developed. See :doc:`/archive/torch-neuron/index` for reference documentation.

.. admonition:: Which Neuron framework for PyTorch should I select?

   For help selecting a framework type for inference, see:
   
   *  :doc:`/frameworks/torch/about/index`
   *  :ref:`torch-neuron_vs_torch-neuronx`

.. toctree::
    :maxdepth: 1
    :hidden:
    
    About PyTorch on Neuron </frameworks/torch/about/index>
    Native PyTorch </frameworks/torch/pytorch-native-overview>
    PyTorch Setup </frameworks/torch/torch-setup>
    Inference </frameworks/torch/inference-torch-neuronx>
    torch-neuron v. torch-neuronx </frameworks/torch/guide-torch-neuron-vs-torch-neuronx-inference>
    Release Notes </release-notes/components/pytorch>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: torch-neuronx references

    Supported operators </frameworks/torch/torch-neuronx/pytorch-neuron-supported-operators>
    Profiling API </frameworks/torch/torch-neuronx/api-reference-guide/torch-neuronx-profiling-api>
    Profiling developer guide </frameworks/torch/torch-neuronx/programming-guide/torch-neuronx-profiling-dev-guide>
    neuron_parallel_compile CLI </frameworks/torch/torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile>
    Multi-node EFA setup </frameworks/torch/torch-neuronx/setup-trn1-multi-node-execution>

Get Started
------------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: TorchNeuron Native Backend Overview
        :link: /frameworks/torch/pytorch-native-overview
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        **Recommended for new workloads** — Learn about the native PyTorch backend with eager execution, ``torch.compile`` support, and standard distributed APIs for Trainium and Inferentia.

    .. grid-item-card:: Setup Guide
        :link: /frameworks/torch/torch-setup
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        Install and configure PyTorch NeuronX for your environment.

Inference
---------------------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Inference on Inf2, Trn1, and Trn2
        :link: inference-torch-neuronx
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Deploy inference workloads using PyTorch NeuronX.

Release Notes
--------------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: PyTorch Neuron Component Release Notes
        :link: /release-notes/components/pytorch
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        Review the PyTorch Neuron release notes for all versions of the Neuron SDK.

.. note::

   Looking for torch-neuron (Inf1) documentation? The torch-neuron package has been
   archived. See :doc:`/archive/torch-neuron/index` for legacy Inf1 documentation.
