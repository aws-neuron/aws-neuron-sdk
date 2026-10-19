.. meta::
   :description: Developer guides for NxD Inference (neuronx-distributed-inference) on AWS Inferentia and AWS Trainium, covering model deployment, optimization, quantization, and integration with vLLM.
   :keywords: AWS Neuron, NxD Inference, neuronx-distributed-inference, LLM inference, model deployment, AWS Inferentia, AWS Trainium, model optimization, quantization, vLLM integration
   :author: AWS Neuron Team

.. _nxdi-dev-ref-index:

Developer Guides
================

Comprehensive guides for using NxD Inference (neuronx-distributed-inference) to deploy and optimize machine learning models on AWS Inferentia and AWS Trainium accelerators. These guides cover model onboarding, performance optimization, quantization techniques, integration with vLLM, and other advanced features to help you maximize the performance of your models on AWS Neuron hardware.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Developer Guides
    
    Accuracy Evaluation </inference-legacy/nxd-inference/developer_guides/accuracy-eval-with-datasets>
    Custom Quantization </inference-legacy/nxd-inference/developer_guides/custom-quantization>
    Disaggregated Inference </inference-legacy/nxd-inference/developer_guides/disaggregated-inference>
    Feature Guide </inference-legacy/nxd-inference/developer_guides/feature-guide>
    Using Pipeline Execution Mode </inference-legacy/nxd-inference/developer_guides/how-to-use-fpem>
    LLM Benchmarking </inference-legacy/nxd-inference/developer_guides/llm-inference-benchmarking-guide>
    Migrate from TNX </inference-legacy/nxd-inference/developer_guides/migrate-from-tnx-to-nxdi>
    Model Reference </inference-legacy/nxd-inference/developer_guides/model-reference>
    MoE Architecture </inference-legacy/nxd-inference/developer_guides/moe-arch-deep-dive>
    Migrate from NxD Core </inference-legacy/nxd-inference/developer_guides/nxd-examples-migration-guide>
    Onboarding Models </inference-legacy/nxd-inference/developer_guides/onboarding-models>
    Performance Benchmarking CLI </inference-legacy/nxd-inference/developer_guides/performance-cli-params>
    vLLM Guide (Legacy) </inference-legacy/nxd-inference/developer_guides/vllm-user-guide>
    vLLM Guide v1 </inference-legacy/nxd-inference/developer_guides/vllm-user-guide-v1>
    Weights Sharding </inference-legacy/nxd-inference/developer_guides/weights-sharding-guide>
    Writing Tests </inference-legacy/nxd-inference/developer_guides/writing-tests>

Use the NxD Inference (``neuronx-distributed-inference``) Developer Guides to learn how to use NxD Inference.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Accuracy Evaluation with Datasets
        :link: /inference-legacy/nxd-inference/developer_guides/accuracy-eval-with-datasets
        :link-type: doc
        
        Guide for evaluating model accuracy using datasets to ensure model quality and performance.

    .. grid-item-card:: Custom Quantization
        :link: /inference-legacy/nxd-inference/developer_guides/custom-quantization
        :link-type: doc
        
        Guide for implementing custom quantization techniques to optimize model size and performance.

    .. grid-item-card:: Disaggregated Inference
        :link: /inference-legacy/nxd-inference/developer_guides/disaggregated-inference
        :link-type: doc

        Guide for using disaggregated inference architecture that separates prefill and decode phases for improved performance.

    .. grid-item-card:: Feature Guide
        :link: /inference-legacy/nxd-inference/developer_guides/feature-guide
        :link-type: doc
        
        Overview of NxD Inference features and configuration options for optimizing model deployment.

    .. grid-item-card:: How to Use FPEM
        :link: /inference-legacy/nxd-inference/developer_guides/how-to-use-fpem
        :link-type: doc
        
        Guide for using Fast Parameter-Efficient Module (FPEM) for efficient model fine-tuning.

    .. grid-item-card:: LLM Inference Benchmarking Guide
        :link: /inference-legacy/nxd-inference/developer_guides/llm-inference-benchmarking-guide
        :link-type: doc
        
        Guide for benchmarking LLM inference performance to optimize deployment configurations.

    .. grid-item-card:: Migrate from TNX to NxDI
        :link: /inference-legacy/nxd-inference/developer_guides/migrate-from-tnx-to-nxdi
        :link-type: doc
        
        Guide for migrating from Transformers NeuronX to NxD Inference with step-by-step instructions.

    .. grid-item-card:: Model Reference
        :link: /inference-legacy/nxd-inference/developer_guides/model-reference
        :link-type: doc
        
        Reference for production-ready models supported by NxD Inference and their configuration options.

    .. grid-item-card:: MoE Architecture Deep Dive
        :link: /inference-legacy/nxd-inference/developer_guides/moe-arch-deep-dive
        :link-type: doc
        
        Deep dive into Mixture of Experts (MoE) architecture implementation in NxD Inference.

    .. grid-item-card:: NxD Examples Migration Guide
        :link: /inference-legacy/nxd-inference/developer_guides/nxd-examples-migration-guide
        :link-type: doc
        
        Guide for migrating examples to NxD Inference from other frameworks or previous versions.

    .. grid-item-card:: Onboarding Models
        :link: /inference-legacy/nxd-inference/developer_guides/onboarding-models
        :link-type: doc
        
        Guide for onboarding new models to NxD Inference with detailed implementation steps.

    .. grid-item-card:: Performance CLI Parameters
        :link: /inference-legacy/nxd-inference/developer_guides/performance-cli-params
        :link-type: doc
        
        Guide for performance tuning using command-line interface parameters for optimal model execution.

    .. grid-item-card:: vLLM User Guide (Legacy)
        :link: /inference-legacy/nxd-inference/developer_guides/vllm-user-guide
        :link-type: doc
        
        Guide for using vLLM v0.x with NxD Inference (Legacy version) for LLM inference and serving.

    .. grid-item-card:: vLLM User Guide v1
        :link: /inference-legacy/nxd-inference/developer_guides/vllm-user-guide-v1
        :link-type: doc
        
        Guide for using vLLM v1.x with NxD Inference for efficient LLM inference and serving.

    .. grid-item-card:: Weights Sharding Guide
        :link: /inference-legacy/nxd-inference/developer_guides/weights-sharding-guide
        :link-type: doc
        
        Guide for implementing weights sharding to distribute model parameters across multiple devices.

    .. grid-item-card:: Writing Tests
        :link: /inference-legacy/nxd-inference/developer_guides/writing-tests
        :link-type: doc
        
        Guide for writing tests for NxD Inference models to ensure accuracy and performance.

