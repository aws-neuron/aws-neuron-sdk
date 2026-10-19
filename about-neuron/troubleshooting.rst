.. meta::
   :description: Find troubleshooting guides for the AWS Neuron SDK on Trainium and Inferentia, including TorchNeuron, TorchTitan, vLLM Neuron, vLLM Omni Neuron, NKI, the Neuron compiler and runtime, Neuron Explorer, Neuron Observer, and containers.
   :keywords: AWS Neuron, troubleshooting, TorchNeuron, torch-neuronx, TorchTitan, vLLM Neuron, vLLM Omni Neuron, NKI, neuronx-cc, Neuron Runtime, Neuron Explorer, Neuron Observer, containers
   :content-type: navigation-hub
   :date-modified: 2026-10-13

.. _general-troubleshooting:

Troubleshooting guide
=====================

Use this page to find the troubleshooting guide for the Neuron component that you use. Each guide lists symptoms, causes, and fixes for that component.

.. contents:: Table of contents
   :local:
   :depth: 1

Setup and installation
----------------------

* :doc:`Installation troubleshooting </setup/troubleshooting>` - Package installation, import errors, and driver issues

Training
--------

* :ref:`Troubleshooting: TorchNeuron <torch-troubleshooting>` - Installation, eager mode, ``torch.compile``, and distributed training issues with native PyTorch on Trainium
* :ref:`Troubleshooting: TorchTitan Neuron <torchtitan-neuron-troubleshooting>` - Common TorchTitan errors and the TorchTitan profiler interface

Inference
---------

* :doc:`Debugging model code in vLLM Neuron </vllm-neuron/docs/model-dev/debugging>` - Debugging techniques for models that run on vLLM Neuron
* :doc:`Debugging accuracy issues in vLLM Neuron </vllm-neuron/docs/model-dev/accuracy-debugging-guide>` - Find where accuracy drift starts in a vLLM Neuron deployment
* :doc:`Accuracy evaluation and debugging in vLLM Omni Neuron </vllm-omni-neuron/docs/model-dev/accuracy-evaluation-debugging>` - Evaluate output quality and isolate accuracy issues in diffusion models

Kernels (NKI)
-------------

* :ref:`NKI FAQ <nki_faq>` - Common questions and issues when you write and run NKI kernels

Compiler
--------

* :ref:`Neuron compiler error codes <ncc-errors-home>` - Causes and fixes for ``neuronx-cc`` compiler errors

Runtime
-------

* :ref:`Neuron Runtime troubleshooting <nrt-troubleshooting>` - Driver installation, model load, memory, hardware, and collective communication errors
* :ref:`nrtpy troubleshooting <nrtpy-troubleshooting>` - Common issues with the ``nrtpy`` Python runtime layer
* :ref:`Neuron Runtime core dumps <runtime-core-dump-quickstart>` - Capture and inspect core dumps when a model fails on the device

Developer tools
---------------

* :ref:`Neuron Explorer troubleshooting and FAQ <neuron-explorer-troubleshooting>` - Profile capture, processing, and missing data issues
* :ref:`Neuron Observer troubleshooting and FAQs <neuron-observer-troubleshooting-faq>` - Installation and metrics export issues on Amazon EKS

Containers and deployment
-------------------------

* :ref:`Troubleshooting Neuron containers <container-troubleshooting>` - Runtime, driver, and device access errors in Neuron Deep Learning Containers
* :ref:`Neuron containers FAQ <container-faq>` - Common questions about Neuron containers

JAX
---

* :ref:`JAX NeuronX known issues <jax-neuron-known-issues>` - Known issues and limitations for JAX on Neuron

Legacy components
-----------------

* :ref:`NxD Inference troubleshooting <nxdi-troubleshooting>` - Troubleshooting for the legacy NxD Inference library

Troubleshooting guides for retired components, such as torch-neuron, TensorFlow Neuron, MXNet Neuron, NeuronPerf, and NxD Training, are in the :doc:`archive </archive/index>`.

Get help
--------

If your issue is not covered in these guides, submit a request with `AWS Neuron GitHub issues <https://github.com/aws-neuron/aws-neuron-sdk/issues>`_ or contact AWS Support.
