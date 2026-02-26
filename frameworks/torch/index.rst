.. meta::
   :description: PyTorch support on AWS Neuron SDK - PyTorch NeuronX for training and inference on Trn1, Trn2, and Inf2, and PyTorch Neuron for inference on Inf1 instances.
   :keywords: PyTorch, torch-neuronx, torch-neuron, AWS Neuron, Trainium, Inferentia, deep learning
   :date-modified: 01/22/2026

.. _neuron-pytorch:
.. _pytorch-neuronx-main:

PyTorch Support on Neuron
==========================

PyTorch running on Neuron unlocks high-performance and cost-effective deep learning acceleration on AWS Trainium-based and AWS Inferentia-based Amazon EC2 instances.

The PyTorch plugin for Neuron architecture enables native PyTorch models to be accelerated on Neuron devices, so you can use your existing framework application and get started easily with minimal code changes. 

PyTorch Neuron is available in two versions to support different AWS ML accelerator architectures:

* **PyTorch NeuronX (torch-neuronx)**: The next-generation PyTorch integration supporting NeuronCores v2 architecture (Trn1, Trn2, Inf2, Trn1n). This version provides enhanced capabilities for both training and inference workloads with support for the latest PyTorch features.
* **PyTorch Neuron (torch-neuron)**: The original PyTorch integration supporting NeuronCores v1 architecture (Inf1). This version is optimized for inference workloads on Inf1 instances.

For help selecting a framework type for inference, see :ref:`torch-neuron_vs_torch-neuronx`. 

.. admonition:: Introducing TorchNeuron, a native backend for AWS Trainium

    At re:Invent '25, AWS Neuron announced their new PyTorch package, "TorchNeuron", which includes the ``torch-neuronx`` library and initial support for a native PyTorch backend (TorchDynamo) with eager execution, ``torch.compile``, and standard distributed APIs. 

    For more details on what is coming with TorchNeuron and PyTorch eager mode support, see :doc:`pytorch-native-overview`.

.. grid:: 1 
   :gutter: 3

   .. grid-item-card:: PyTorch Neuron Component Release Notes
      :link: /release-notes/components/pytorch
      :link-type: doc

      Review the PyTorch Neuron release notes for all versions of the Neuron SDK.

.. tab-set::

   .. tab-item:: PyTorch NeuronX for Trn1, Trn2 & Inf2

      .. grid:: 1 
         :gutter: 3

         .. grid-item-card:: Setup Guide
            :link: /frameworks/torch/torch-setup
            :link-type: doc

            Install and configure PyTorch NeuronX for your environment

         .. grid-item-card:: PyTorch Native Backend Overview
            :link: /frameworks/torch/pytorch-native-overview
            :link-type: doc

            Learn about the new native PyTorch backend with eager execution and torch.compile support

         .. grid-item-card:: Training on Trn1 and Trn2
            :link: training-torch-neuronx
            :link-type: ref

            Train models using PyTorch NeuronX on Trainium instances

         .. grid-item-card:: Inference on Inf2, Trn1, and Trn2
            :link: inference-torch-neuronx
            :link-type: ref

            Deploy inference workloads using PyTorch NeuronX

   .. tab-item:: PyTorch Neuron for Inf1

      .. grid:: 1 
         :gutter: 3

         .. grid-item-card:: Setup Guide
            :link: /frameworks/torch/torch-setup
            :link-type: doc

            Install and configure PyTorch Neuron for Inf1 instances

         .. grid-item-card:: Inference on Inf1
            :link: inference-torch-neuron
            :link-type: ref

            Deploy inference workloads using PyTorch Neuron on Inf1 instances

.. toctree::
    :maxdepth: 1
    :hidden:
    
    /frameworks/torch/torch-setup
    /frameworks/torch/pytorch-native-overview
    Training </frameworks/torch/training-torch-neuronx>
    Inference (Inf2, Trn1, Trn2) </frameworks/torch/inference-torch-neuronx>
    Inference (Inf1) </frameworks/torch/inference-torch-neuron>
    Release Notes </release-notes/components/pytorch>
