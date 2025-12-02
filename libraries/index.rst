.. meta::
   :description: AWS NeuronX distributed libraries - High-performance distributed training and inference libraries for AWS Trainium and Inferentia, including NxD Core, NxD Inference, NxD Training, and third-party integrations.

.. _libraries-neuron-sdk:

Work with training and inference libraries
===========================================

Accelerate your machine learning workloads with Neuron's distributed libraries. Our libraries provide high-level abstractions and optimized implementations for distributed training and inference on AWS Trainium and Inferentia.

What are NeuronX Distributed libraries?
----------------------------------------

NeuronX Distributed (NxD) libraries are a comprehensive suite of PyTorch-based libraries designed to enable scalable machine learning on AWS Neuron hardware. The NxD ecosystem provides a layered architecture where foundational distributed primitives support higher-level training and inference workflows.

**The NxD Stack:**

* **NxD Core**: The foundational layer providing distributed primitives, model sharding techniques, and XLA-optimized implementations
* **NxD Training**: High-level training library built on NxD Core, offering turnkey distributed training workflows with NeMo compatibility
* **NxD Inference**: Production-ready inference library with advanced features like continuous batching, speculative decoding, and vLLM integration

Together, these libraries enable developers to scale from prototype to production while leveraging the full performance potential of AWS Trainium and Inferentia instances.

About NxD Core Libraries
------------------------
        
NxD Core libraries provide distributed training and inference mechanisms for Neuron devices with XLA-friendly implementations. This includes:
        
* :doc:`Tensor Parallel (TP) sharding </libraries/neuronx-distributed/ptl_developer_guide>` (:doc:`Overview </libraries/neuronx-distributed/tensor_parallelism_overview>`)
* :doc:`Pipeline Parallel (PP) support </libraries/neuronx-distributed/pp_developer_guide>` (:doc:`Overview </libraries/neuronx-distributed/pipeline_parallelism_overview`)
* :doc:`Model activation memory reduction support </libraries/neuronx-distributed/activation_memory_reduction_developer_guide>` (:doc:`Overview </libraries/neuronx-distributed/activation_memory_reduction>`)
* Model partitioning across devices
* XLA-optimized distributed operations
* Foundation for other NxD libraries

The NxD Training and Inference documentation below provides documentation for NxD Core libraries in the context of of training and inference models respectively.

NxD Training and Inference Libraries 
-------------------------------------

.. grid:: 1
  :gutter: 3
  :class-container: library-grid

  .. grid-item-card:: NxD Inference
      :link: /libraries/nxd-inference/index
      :link-type: doc
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

      PyTorch library for end-to-end distributed training with Neuron.
        
       * Large-scale model training
       * NeMo YAML configuration support
       * HuggingFace and Megatron-LM models   
       * Experiment management
       * Advanced parallelism strategies

Other Libraries
----------------

.. grid:: 1 1 2 2
  :gutter: 3
  :class-container: library-grid

  .. grid-item-card:: Hugging Face Transformers (legacy)
      :link: /libraries/transformers-neuronx/index
      :link-type: doc
      :class-header: bg-success text-white
      :class-body: library-card-body
          
  .. grid-item-card:: NeMo Megatron
      :link: /libraries/nemo-megatron/index
      :link-type: doc
      :class-header: bg-success text-white
      :class-body: library-card-body

Hardware Compatibility
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
     - N/A
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **NxD Inference**
     - N/A
     - ✅
     - ✅
     - ✅
     - ✅
     - N/A
   * - **NxD Training**
     - N/A
     - N/A
     - ✅
     - ✅
     - N/A
     - ✅

.. _third-party-libraries:

Third-party libraries
-----------------------

AWS Neuron integrates with multiple third-party partner products that alow you to run deep learning workloads on Amazon EC2 
instances powered by AWS Trainium and AWS Inferentia chips. The following list gives an overview of the third-party libraries 
working with AWS Neuron.

Hugging Face Optimum Neuron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimum Neuron bridges Hugging Face Transformers and the AWS Neuron SDK, providing standard Hugging Face APIs for 
`AWS Trainium <https://aws.amazon.com/ai/machine-learning/trainium/>`_ and `AWS Inferentia <https://aws.amazon.com/ai/machine-learning/inferentia/>`_. 
It offers solutions for both training and inference, including support for large-scale model training and deployment for AI workflows. 
Supporting Amazon SageMaker and pre-built Deep Learning Containers, Optimum Neuron simplifies the use of Trainium and Inferentia 
for machine learning. This integration allows developers to work with familiar Hugging Face interfaces while leveraging Trainium 
and Inferentia for their transformer-based projects.

`Optimum Neuron documentation <https://huggingface.co/docs/optimum-neuron/en/index>`_

PyTorch Lightning
^^^^^^^^^^^^^^^^^^^

PyTorch Lightning is a deep learning framework for professional AI researchers and machine learning engineers who need maximal 
flexibility without sacrificing performance at scale. Lightning organizes PyTorch code to remove boilerplate and unlock scalability.

`Get Started with Lightning  <https://lightning.ai/lightning-ai/studios/finetune-llama-90-cheaper-on-aws-trainium~01hh3kj60fs8b8x91rv9n9fn2j?section=featured>`_

Use PyTorch Lightning Trainer with :ref:`NeuronX Distributed <pytorch-lightning>`. 


AXLearn
^^^^^^^^^

AXLearn is an open-source JAX-based library used by AWS Neuron for training deep learning models on AWS Trainium. Integrates with JAX ecosystem and supports distributed training.

Check out the `AXLearn Github repository <https://github.com/apple/axlearn>`_.


Additional libraries
---------------------

NeMo 
^^^^^

:ref:`NxD Training <nxd-training-overview>` offers a `NeMo <https://github.com/NVIDIA/NeMo>`_-compatible YAML interface for training 
PyTorch models on AWS Trainium chips. The library supports both Megatron-LM and HuggingFace model classes through its model hub. 
NxD Training leverages key NeMo components, including Experiment Manager for tracking ML experiments and data loaders for efficient 
data processing. This library simplifies the process of training deep learning models on AWS Trainium while providing compatibility 
with familiar NeMo YAML Interface.

.. toctree::
   :hidden:
   :maxdepth: 1

   HF Transformers </libraries/transformers-neuronx/index>
   NeMo Megatron </libraries/nemo-megatron/index>

  
