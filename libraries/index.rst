.. meta::
   :description: AWS NeuronX distributed libraries - High-performance distributed training and inference libraries for AWS Trainium and Inferentia, including NxD Core, NxD Inference, NxD Training, and third-party integrations.

.. _libraries-neuron-sdk:

AWS NxD libraries
==================

Accelerate your machine learning workloads with Neuron's distributed libraries. Our libraries provide high-level abstractions and optimized implementations for distributed training and inference on AWS Trainium and Inferentia.

What are NeuronX Distributed libraries?
----------------------------------------

NeuronX Distributed (NxD) libraries are a comprehensive suite of PyTorch-based libraries designed to enable scalable machine learning on AWS Neuron hardware. The NxD ecosystem provides a layered architecture where foundational distributed primitives support higher-level training and inference workflows.

**The NxD Stack:**

* **NxD Core**: The foundational layer providing distributed primitives, model sharding techniques, and XLA-optimized implementations
* **NxD Training**: High-level training library built on NxD Core, offering turnkey distributed training workflows with NeMo compatibility
* **NxD Inference**: Production-ready inference library with advanced features like continuous batching, speculative decoding, and vLLM integration

Together, these libraries enable developers to scale from prototype to production while leveraging the full performance potential of AWS Trainium and Inferentia instances.

.. grid:: 2 2 2 2
    :gutter: 3
    :class-container: library-grid

    .. grid-item-card:: NxD Core
        :link: neuronx-distributed-index
        :link-type: ref
        :class-header: bg-primary text-white
        :class-body: library-card-body
        
        Core distributed training and inference mechanisms for Neuron devices with XLA-friendly implementations.
        
        * Tensor Parallel (TP) sharding
        * Pipeline Parallel (PP) support
        * Model partitioning across devices
        * XLA-optimized distributed operations
        * Foundation for other NxD libraries

    .. grid-item-card:: NxD Inference
        :link: nxdi-index
        :link-type: ref
        :class-header: bg-success text-white
        :class-body: library-card-body
        
        PyTorch-based inference library for deploying large models on Inferentia and Trainium.
        
        * Large Language Model (LLM) inference
        * Disaggregated inference architecture
        * vLLM integration and compatibility
        * Model sharding and parallelism
        * Performance optimization tools

    .. grid-item-card:: NxD Training
        :link: nxdt
        :link-type: ref
        :class-header: bg-info text-white
        :class-body: library-card-body
        
        PyTorch library for distributed training with NeMo-compatible YAML interface.
        
        * Large-scale model training
        * NeMo YAML configuration support
        * HuggingFace and Megatron-LM models
        * Experiment management
        * Advanced parallelism strategies

    .. grid-item-card::  Transformers NeuronX
        :link: transformers_neuronx_readme
        :link-type: ref
        :class-header: bg-warning text-dark
        :class-body: library-card-body

        **Legacy Library (Archived)**
        
        Original transformer inference library - now superseded by NxD Inference.
        
        * **Status**: Support ended 9/16/2025
        * Migration path to NxD Inference
        * Archived documentation available
        * Legacy workload support
        * **Recommended**: Migrate to NxD Inference

Hardware compatibility
----------------------

.. list-table::
   :header-rows: 1
   :class: compatibility-matrix

   * - Library
     - Inf1
     - Inf2
     - Trn1/Trn1n
     - Trn2
     - Inference
     - Training
   * - **NxD Core**
     - ❌
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **NxD Inference**
     - ❌
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
   * - **NxD Training**
     - ❌
     - ❌
     - ✅
     - ✅
     - ❌
     - ✅

Third-Party partner libraries
-----------------------------

.. grid:: 2 2 2 2
    :gutter: 2

    .. grid-item-card:: 🤗 Hugging Face Optimum Neuron
        :class-body: text-center

        Standard Hugging Face APIs for Trainium and Inferentia with SageMaker support.

    .. grid-item-card:: ⚡ PyTorch Lightning
        :class-body: text-center

        Professional AI framework with maximal flexibility and NxD integration.

    .. grid-item-card:: 🔬 AXLearn
        :class-body: text-center

        JAX-based library for distributed training with AWS Trainium integration.
