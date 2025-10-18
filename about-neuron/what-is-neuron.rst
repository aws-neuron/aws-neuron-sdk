.. _what-is-neuron:

.. meta::
   :description: AWS Neuron is a software development kit for high-performance machine learning on AWS Inferentia and Trainium, enabling developers to compile, optimize, and deploy deep learning models at scale.

What is AWS Neuron?
===================

AWS Neuron is a software development kit (SDK) that enables high-performance deep learning acceleration using AWS Inferentia and Trainium, AWS's custom-designed machine learning accelerators. Neuron provides developers with the tools needed to compile, optimize, and deploy machine learning workloads on accelerated EC2 instances such as ``Inf1``, ``Inf2``, ``Trn1``, ``Trn1n``, and ``Trn2``.

For more details, see the detailed documentation under :ref:`About the AWS Neuron SDK <about-neuron>`.

Core Components
---------------

**Neuron Compiler**
    Optimizes machine learning models for AWS Inferentia and Trainium chips, converting models from popular frameworks into efficient executable formats.

**Neuron Kernel Interface (NKI)**
    Provides low-level access to Neuron hardware capabilities, enabling advanced optimizations and custom operations.

**Neuron Runtime**
    Manages model execution on Neuron devices, handling memory allocation, scheduling, and inter-chip communication for maximum throughput.

**Neuron Tools**
    Debug and profiling utilities including:
    
    * Neuron Monitor for real-time performance monitoring
    * Neuron Profiler (``neuron-profile``) for detailed performance analysis

**Neuron distributed libraries**
    Libraries for distributed training and inference, enabling scalable ML workloads across multiple Neuron devices.

**Framework integration**
    Pre-integrated support for popular machine learning frameworks:
    
    * PyTorch
    * JAX

Supported Hardware
------------------

**AWS Inferentia**
    Purpose-built for high-performance inference workloads:
    
    * ``Inf1`` instances - First-generation Inferentia chips
    * ``Inf2`` instances - Second-generation with improved performance and efficiency

**AWS Trainium**
    Designed for distributed training of large models:
    
    * ``Trn1`` instances - High-performance training acceleration
    * ``Trn1n`` instances - Enhanced networking for large-scale distributed training
    * ``Trn2`` instances - Next-generation Trainium with superior performance
    * ``Trn2`` UltraServer - High-density Trainium servers for massive training workloads

Why use AWS Neuron?
-------------------

**High Performance**
    Delivers up to 2.3x better price-performance compared to GPU-based instances for inference workloads.

**Cost Optimization**
    Reduces inference costs through efficient model compilation and optimized hardware utilization.

**Seamless Integration**
    Works with existing ML workflows through native framework support and familiar APIs.

**Scalability**
    Supports both single-chip and multi-chip deployments for various workload sizes.

What can I use AWS Neuron for?
------------------------------

**Natural Language Processing**
    * Large language model inference
    * Text classification and sentiment analysis
    * Machine translation

**Computer Vision**
    * Image classification and object detection
    * Video analysis and processing
    * Medical imaging applications

**Recommendation Systems**
    * Real-time personalization
    * Content recommendation engines
    * Ad targeting and optimization

**Training Workloads**
    * Large-scale model training on Trainium
    * Distributed training across multiple chips
    * Fine-tuning of pre-trained models

How do I get more information?
------------------------------

* Review the comprehensive documentation and follow the tutorials on this site
* Check the Neuron GitHub repositories for code examples. GitHub repos include:

  * `Neuron SDK code samples <https://github.com/aws-neuron/aws-neuron-samples>`_
  * `Neuron NKI ML kernel samples <https://github.com/aws-neuron/nki-samples>`_
  * `Neuron container confirguations <https://github.com/aws-neuron/deep-learning-containers>`_
  * `Helm charts for Kubernetes deployment <https://github.com/aws-neuron/neuron-helm-charts>`_
  * `NeuronX Distributed Core library sources <https://github.com/aws-neuron/neuronx-distributed>`_
  * `NeuronX Distributed Training library sources <https://github.com/aws-neuron/neuronx-distributed-training>`_
  * `NeuronX Distributed Inference library sources <https://github.com/aws-neuron/neuronx-distributed-inference>`_
  * `Linux kernel driver sources <https://github.com/aws-neuron/aws-neuron-driver>`_
  * `Neuron workshop model samples <https://github.com/aws-neuron/neuron-workshops>`_

* Visit the `AWS Neuron support forum <https://forums.aws.amazon.com/forum.jspa?forumID=355>`_ for community assistance
