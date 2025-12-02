.. _what-is-neuron:

.. meta::
   :description: AWS Neuron is a software development kit for high-performance machine learning on AWS Inferentia and Trainium, enabling developers to compile, optimize, and deploy deep learning models at scale.

What is AWS Neuron?
===================

AWS Neuron is the software stack for running deep learning and generative AI workloads on AWS Trainium and AWS Inferentia. Built on an open source foundation, Neuron enables developers to build, deploy and explore natively with PyTorch and JAX frameworks and with ML libraries such as Hugging Face, vLLM, PyTorch Lightning, and others without modifying your code.  It includes a compiler, runtime, training and inference libraries, and developer tools for monitoring, profiling, and debugging. Neuron supports your end-to-end machine learning (ML) development lifecycle from building and deploying deep learning and AI models, optimizing to achieve highest performance and lowest cost, and getting deeper insights into model behavior.

Neuron enables rapid experimentation, production scale training of frontier models, low level performance optimization through the Neuron Kernel Interface (NKI) for custom kernels, cost optimized inference deployment for agentic AI and reinforcement learning workloads, and comprehensive profiling and debugging with Neuron Explorer.

For more details, see the detailed documentation under :ref:`About the AWS Neuron SDK <about-neuron>`.

Who is AWS Neuron for?
-----------------------

* **ML engineers** can use Neuron's vLLM integration to migrate their models to Trainium for improved performance and without code modifications. They can
* **Performance engineers** can use NKI and our Developer Tools to create new ML kernels and optimize existing ones.
* **ML researchers** can use their existing PyTorch experience and ecosystem tools to experiment freely on Trainium using our native PyTorch implementatio, without having to learn new frameworks or APIs

What is AWS Neuron used for?
-----------------------------

**Research and Development**: Neuron provides native PyTorch execution on Trainium with full Eager mode compatibility. The stack supports standard distributed training patterns including FSDP, DDP, and DTensor for model sharding across devices and nodes. torch.compile integration enables graph optimization, while existing frameworks like TorchTitan and HuggingFace Transformers run without code modifications. JAX support includes XLA compilation targeting Inferentia and Trainium hardware. 

**Production Inference**: Neuron implements vLLM V1 API compatibility on Trainium and Inferentia with optimizations for large-scale inference workloads. The runtime supports Expert Parallelism for MoE models, disaggregated inference architectures, and speculative decoding. Optimized kernels from the NKI Library provide hardware-specific implementations. Training workflows integrate with HuggingFace Optimum Neuron, PyTorch Lightning, and TorchTitan, with seamless deployment through standard vLLM interfaces. 

**Performance Engineering**: Neuron Kernel Interface (NKI) provides direct access to Trainium instruction set architecture with APIs for memory management, execution scheduling, and low-level kernel development. The NKI Compiler, built on MLIR, offers full visibility into the compilation pipeline from high-level operations to hardware instructions. The NKI Library contains optimized kernel implementations with source code and performance benchmarks. Neuron Explorer enables comprehensive profiling from application code to hardware execution, supporting both single-node and distributed workload analysis with detailed performance metrics and optimization recommendations.

AWS Neuron Core Components
----------------------------

**vLLM**
    Neuron enables production inference deployment with standard frameworks and APIs on Trainium and Inferentia. Use Neuron's vLLM integration with standard APIs to deliver high-performance model serving with optimized kernels from the NKI Library. 

    It provides:

    * **Standard vLLM APIs**: Full compatibility with vLLM V1 APIs, enabling customers to use familiar vLLM interfaces on Neuron hardware without code changes
    * **Advanced Inference Features**: Support for Expert Parallelism for MoE models, disaggregated inference for flexible deployment architectures, and speculative decoding for improved latency
    * **Optimized Performance**: Pre-optimized kernels from the NKI Library for peak performance across dense, MoE, and multimodal models
    * **Open Source**: Source code released under the vLLM project organization with source code on GitHub, enabling community contributions

**Native PyTorch**
    Neuron provides native integration with PyTorch, enabling researchers and ML developers to run existing code unchanged on Trainium. Train models with familiar workflows and tools, from pre-training to post-training with reinforcement learning, while leveraging Trainium's performance and cost advantages for both experimentation and production scale training.

    It provides:

    * **Native Device Support**: Neuron registers as a native device type in PyTorch with standard device APIs like ``torch.tensor([1,2,3], device='neuron')`` and ``.to('neuron')``
    * **Standard Distributed Training APIs**: Support for FSDP, DTensor, DDP, tensor parallelism, context parallelism, and distributed checkpointing
    * **Eager Mode Execution**: Immediate operation execution for interactive development and debugging in notebook environments
    * **torch.compile Integration**: Support for ``torch.compile`` for optimized performance
    * **Open Source**: Released as an open source package on GitHub under Apache 2.0, enabling community contributions.  

**Neuron Kernel Interface (NKI)**
    For performance engineers seeking maximum hardware efficiency, Neuron provides complete control through the Neuron Kernel Interface (NKI), with direct access to the NeuronISA (NISA) instruction set, memory allocation, and execution scheduling. Developers can create new operations not available in standard frameworks and optimize performance critical code with custom kernels. 

    It includes:

    * The NKI Compiler, built on MLIR, which provides greater transparency into the kernel compilation process
    * The NKI Library , which provides pre-built kernels you can use to optimize the performance of your models

**Neuron Tools**
    Debug and profiling utilities including:
    
    * Neuron Monitor for real-time performance monitoring
    * Neuron Explorer, built on the Neuron Profiler (``neuron-profile``), for detailed performance analysis

    Neuron Explorer provides:

    * **Hierarchical Profiling**: Top-down visualization from framework layers through HLO operators to hardware instructions, enabling developers to understand execution at any level of the stack
    * **Code Linking**: Direct navigation between PyTorch, JAX, and NKI source code and performance timeline with automatic annotations showing metrics for specific code lines
    * **IDE Integration**: VSCode extension for profile visualization and analysis directly within the development environment
    * **Device Profiling**: Unified interface for comprehensive view of system-wide metrics and device-specific execution details

**Neuron Compiler**
    Optimizes machine learning models for AWS Inferentia and Trainium chips, converting models from popular frameworks into efficient executable formats.

**Neuron Runtime**
    Manages model execution on Neuron devices, handling memory allocation, scheduling, and inter-chip communication for maximum throughput.

**AWS DLAMIs and DLCs**
    Orchestrate and deploy your models using Deep Learning AWS Machine Images (DLAMIs) and Deep Learning Containers (DLCs).

    Neuron DLAMIs come pre-configured with the Neuron SDK, popular frameworks, and helpful libraries, allowing you to quickly begin training and running inference on AWS Inferentia. Or, quickly deploy models using pre-configured AWS Neuron Deep Learning Containers (Neuron DLCs) with optimized frameworks for AWS Trainium and Inferentia.  

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
    * ``Trn3`` UltraServer -- The next generation of Trainium servers for massive training workloads


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
