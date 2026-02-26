.. meta::
   :description: NxD Training (NeuronX Distributed Training) is a PyTorch library for end-to-end distributed training on AWS Trainium instances, offering turnkey workflows for pre-training, fine-tuning, and PEFT.
   :keywords: NxD Training, NeuronX Distributed Training, AWS Neuron SDK, Distributed Training, PyTorch Lightning, Tensor Parallelism, Pipeline Parallelism, ZeRO-1, LoRA, PEFT, Model Training
   :date-modified: 01/22/2026

.. _nxdt:

NxD Training
============

This section contains the technical documentation specific to the NxD Training library included with the Neuron SDK.

.. toctree::
    :maxdepth: 1
    :hidden:

    Overview </libraries/nxd-training/overview>
    Setup </libraries/nxd-training/general/installation_guide>
    Tutorials  </libraries/nxd-training/tutorials/index>
    Developer Guides  </libraries/nxd-training/developer_guides/index>
    API Reference Guide </libraries/nxd-training/api-reference-guide>
    App Notes </libraries/nxd-training/app_notes>
    Release Notes </release-notes/components/nxd-training>
    Misc  </libraries/nxd-training/misc>

What is NxD Training?
---------------------

NxD Training (NeuronX Distributed Training) is a PyTorch library for end-to-end distributed training on AWS Trainium instances. It combines ease-of-use with powerful features built on top of the NxD Core library, offering turnkey support for model pre-training, supervised fine-tuning (SFT), and parameter-efficient fine-tuning (PEFT) using LoRA.

With NxD Training, developers can:

* Train large-scale models with turnkey workflows for pre-training, SFT, and PEFT (LoRA)
* Leverage distributed strategies including Data Parallelism, Tensor Parallelism, Sequence Parallelism, Pipeline Parallelism, and ZeRO-1
* Use PyTorch Lightning integration for organized training code
* Access ready-to-use model samples based on HuggingFace and Megatron-LM formats
* Manage experiments with integrated checkpointing, logging, and S3 storage support
* Choose from three usage interfaces: YAML configuration files, PyTorch Lightning APIs, or NxD Core primitives

NxD Training is compatible with training platforms like NVIDIA's NeMo (except for Trainium-specific features) and is available on GitHub as both pip wheel and source code.

Usage Interfaces
----------------

NxD Training provides three interfaces to meet different developer needs:

* **YAML Configuration Files**: High-level access for distributed training with minimal code changes
* **PyTorch Lightning APIs**: Standardized training workflows with NxD Core primitives
* **NxD Core Primitives**: Low-level APIs for custom model integration and advanced use cases


NxD Training documentation
---------------------------

.. grid:: 1 1 2 2
    :gutter: 3
    
    .. grid-item-card:: Overview
        :link: /libraries/nxd-training/overview
        :link-type: doc
        :class-card: sd-rounded-3
        
        Learn about NxD Training architecture, key features, and usage interfaces for distributed training on AWS Trainium.

    .. grid-item-card:: Setup
        :link: /libraries/nxd-training/general/installation_guide
        :link-type: doc
        :class-card: sd-rounded-3
        
        Step-by-step instructions for installing and configuring NxD Training on Trainium instances.

    .. grid-item-card:: Tutorials
        :link: /libraries/nxd-training/tutorials/index
        :link-type: doc
        :class-card: sd-rounded-3
        
        Hands-on tutorials for training various models including Llama, GPT, and BERT with different parallelism strategies.

    .. grid-item-card:: Developer Guides
        :link: /libraries/nxd-training/developer_guides/index
        :link-type: doc
        :class-card: sd-rounded-3
        
        In-depth guides for model integration, YAML configuration, migration from NeMo/NNM, and advanced training workflows.

    .. grid-item-card:: API Reference
        :link: /libraries/nxd-training/api-reference-guide
        :link-type: doc
        :class-card: sd-rounded-3
        
        Comprehensive API documentation for NxD Training modules, configuration options, and programming interfaces.

    .. grid-item-card:: Application Notes
        :link: /libraries/nxd-training/app_notes
        :link-type: doc
        :class-card: sd-rounded-3
        
        Detailed application notes on distributed strategies, optimization techniques, and best practices for training.

    .. grid-item-card:: Misc Resources
        :link: /libraries/nxd-training/misc
        :link-type: doc
        :class-card: sd-rounded-3
        
        Known issues, troubleshooting guides, and other helpful resources for working with NxD Training.

    .. grid-item-card:: NxD Training Release Notes
        :link: /release-notes/components/nxd-training
        :link-type: doc
        :class-card: sd-rounded-3
        
        Review the latest updates, new features, and bug fixes in NxD Training releases.