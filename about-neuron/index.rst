.. meta::
   :description: About the AWS Neuron SDK for AWS Trainium and Inferentia — TorchNeuron native PyTorch, JAX, TorchTitan, vLLM Neuron, vLLM Omni Neuron, NKI, the Neuron compiler and runtime, Neuron Explorer, and Neuron Observer.
   :keywords: AWS Neuron, Neuron SDK, Trainium, Inferentia, TorchNeuron, torch-neuronx, JAX, TorchTitan, vLLM Neuron, vLLM Omni Neuron, NKI, Neuron Explorer, Neuron Observer, NeuronX
   :content-type: navigation-hub
   :date-modified: 2026-10-13

.. _about-neuron:

About the AWS Neuron SDK
========================

AWS Neuron is the software development kit (SDK) for machine learning on AWS Trainium and AWS Inferentia, the machine learning accelerators that AWS designs. Use Neuron to train, serve, profile, and monitor deep learning and generative AI workloads on Trainium and Inferentia instances, with the frameworks and libraries that you already use.

The AWS Neuron SDK includes:

* **TorchNeuron (native PyTorch)** - Runs PyTorch on Trainium as a native device, with eager mode, ``torch.compile``, and standard distributed APIs such as FSDP, DDP, and DTensor
* **TorchTitan Neuron** - Pre-trains large models on Trainium with TorchTitan and TorchNeuron
* **JAX support** - Runs JAX workloads on Neuron through a PJRT plugin
* **vLLM Neuron** - Serves large language models (LLMs) with the standard ``vllm serve`` command and an OpenAI-compatible API
* **vLLM Omni Neuron** - Runs diffusion models for image and video generation with vLLM Omni
* **Neuron Kernel Interface (NKI)** - Writes custom kernels with direct access to the NeuronCore instruction set, plus pre-optimized kernels in the NKI Library
* **Neuron Compiler** - Compiles models from ML frameworks into optimized code for Neuron devices
* **Neuron Runtime** - Loads and runs compiled models on NeuronCores, and provides collective communication for distributed workloads
* **Neuron Explorer** - Profiles workloads at the framework, model, and NeuronCore level to find and fix performance bottlenecks
* **Neuron Observer** - Collects device health, hardware fault, and performance metrics from Trainium clusters
* **Neuron system tools** - Command-line tools for device monitoring and administration
* **Deployment support** - Deep Learning AMIs (DLAMIs), Deep Learning Containers (DLCs), and integration with Amazon EC2, Amazon EKS, Amazon ECS, AWS Batch, and Amazon SageMaker

For an overview of each component, see :ref:`what-is-neuron`. To run your first workload, see the :ref:`Neuron quickstarts <neuron-quickstart>`.

About the "NeuronX" name
------------------------

You see "NeuronX" (or "neuronx") in the names of the Neuron packages for Inf2, Trn1, Trn2, and Trn3 instances, such as the ``neuronx-cc`` compiler, the ``torch-neuronx`` package for TorchNeuron, and the ``aws-neuronx-*`` runtime, driver, and tools packages. Packages without the "x", such as ``neuron-cc`` and ``torch-neuron``, are for first-generation Inf1 instances. Inf1 is no longer supported, and its documentation is in the :doc:`archive </archive/index>`.

Catch up on the latest Neuron news
-----------------------------------

.. grid:: 1
   :gutter: 2

   .. grid-item-card::
      :link: /about-neuron/whats-new
      :link-type: doc
      :class-card: sd-border-1

      **What's New in Neuron**
      ^^^
      Read about the latest releases and features of the Neuron SDK

Learn about AWS Neuron
----------------------

.. grid:: 1
   :gutter: 2

   .. grid-item-card::
      :link: /about-neuron/what-is-neuron
      :link-type: doc
      :class-card: sd-border-1

      **What is AWS Neuron?**
      ^^^
      Short overview of the AWS Neuron SDK and its components

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: /about-neuron/arch/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron architecture**
      ^^^
      Understand the Neuron hardware and software architecture

   .. grid-item-card::
      :link: /about-neuron/arch/neuron-features/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron features**
      ^^^
      Overviews of model development features provided by Neuron

   .. grid-item-card::
      :link: /torch/index
      :link-type: doc
      :class-card: sd-border-1

      **TorchNeuron (native PyTorch)**
      ^^^
      Run PyTorch on Trainium with eager mode, ``torch.compile``, and standard distributed APIs

   .. grid-item-card::
      :link: /jax/index
      :link-type: doc
      :class-card: sd-border-1

      **JAX on Neuron**
      ^^^
      Run JAX workloads on Neuron with the PJRT plugin

   .. grid-item-card::
      :link: /training
      :link-type: doc
      :class-card: sd-border-1

      **Training on Neuron**
      ^^^
      Train models with TorchNeuron and pre-train at scale with TorchTitan

   .. grid-item-card::
      :link: /inference
      :link-type: doc
      :class-card: sd-border-1

      **Inference on Neuron**
      ^^^
      Choose an inference path: vLLM Neuron, vLLM Omni Neuron, or native PyTorch

   .. grid-item-card::
      :link: /vllm-neuron/docs/index
      :link-type: doc
      :class-card: sd-border-1

      **vLLM Neuron**
      ^^^
      Serve LLMs on Trainium with vLLM and an OpenAI-compatible API

   .. grid-item-card::
      :link: /vllm-omni-neuron/docs/index
      :link-type: doc
      :class-card: sd-border-1

      **vLLM Omni Neuron**
      ^^^
      Generate images and video with diffusion models on Trainium

   .. grid-item-card::
      :link: /nki/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron Kernel Interface (NKI)**
      ^^^
      Write custom kernels with direct access to the NeuronCore, or use the NKI Library

   .. grid-item-card::
      :link: /compiler/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron Compiler**
      ^^^
      The Neuron compiler optimizes models for Neuron hardware

   .. grid-item-card::
      :link: /neuron-runtime/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron Runtime**
      ^^^
      Runtime for executing compiled models on Neuron devices

   .. grid-item-card::
      :link: /tools/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron developer tools**
      ^^^
      Profile with Neuron Explorer, monitor cluster health with Neuron Observer, and use system tools

   .. grid-item-card::
      :link: /deploy/environments/index
      :link-type: doc
      :class-card: sd-border-1

      **Deep Learning AMIs and Containers**
      ^^^
      Start from preconfigured Neuron DLAMIs and DLCs

   .. grid-item-card::
      :link: /deploy/index
      :link-type: doc
      :class-card: sd-border-1

      **Deploy on AWS**
      ^^^
      Deploy Neuron workloads on Amazon EC2, Amazon EKS, Amazon ECS, AWS Batch, and more

Resources
---------

* :ref:`Neuron quickstarts <neuron-quickstart>`
* :ref:`Setup guide <setup-guide-index>`
* :ref:`Release notes <neuron_release_notes>`
* :ref:`Neuron open source repositories <neuron-oss>`
* :doc:`Term glossary <arch/glossary>`
* :ref:`Neuron FAQ <neuron_faq>`
* :doc:`Contributing guidelines FAQ <faq/contributing-faq>`
* :doc:`Legacy inference frameworks (NxD Inference and NxD Core) </inference-legacy/index>`
* :doc:`Archived documentation </archive/index>`

Support
-------

* :ref:`AWS Neuron SDK maintenance policy <sdk-maintenance-policy>`
* :doc:`Security disclosures <security>`

.. _contact-us:

Contact us
----------

For support, submit a request with AWS Neuron `GitHub issues <https://github.com/aws-neuron/aws-neuron-sdk/issues>`_.

If you want to request a feature or report a critical issue, you can contact us directly at ``aws-neuron-support@amazon.com``.

.. toctree::
   :maxdepth: 1
   :hidden:

   Contributing Guidelines FAQ <faq/contributing-faq>
   Neuron FAQ <faq>
   Neuron Features </about-neuron/arch/neuron-features/index>
   Open Source </about-neuron/oss/index>
   SDK Maintenance Policy <sdk-policy>
   Security <security>
   Term Glossary <arch/glossary>
   Troubleshooting <troubleshooting>
   What is AWS Neuron? <what-is-neuron>
