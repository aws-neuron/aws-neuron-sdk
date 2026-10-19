.. _what-is-neuron:

.. meta::
   :description: AWS Neuron is a software development kit for high-performance machine learning on AWS Inferentia and Trainium, enabling developers to compile, optimize, and deploy deep learning models at scale.
   :date-modified: 2026-10-13

What is AWS Neuron?
===================

AWS Neuron is the software stack for running deep learning and generative AI workloads on AWS Trainium and AWS Inferentia. Built on an open source foundation, Neuron enables developers to build, deploy and explore natively with PyTorch and JAX frameworks and with ML libraries such as Hugging Face, vLLM, and others without modifying your code.  It includes a compiler, runtime, training and inference libraries, and developer tools for monitoring, profiling, and debugging. Neuron supports your end-to-end machine learning (ML) development lifecycle from building and deploying deep learning and AI models, optimizing to achieve highest performance and lowest cost, and getting deeper insights into model behavior.

Neuron enables rapid experimentation, production scale training of frontier models, low level performance optimization through the Neuron Kernel Interface (NKI) for custom kernels, cost optimized inference deployment for agentic AI and reinforcement learning workloads, and comprehensive profiling and debugging with Neuron Explorer.

For more details, see the detailed documentation under :ref:`About the AWS Neuron SDK <about-neuron>`.

Who is AWS Neuron for?
-----------------------

* **ML engineers** can use the vLLM Neuron and vLLM Omni Neuron plugins to serve LLMs and diffusion models on Trainium without code modifications.
* **Performance engineers** can use NKI and our Developer Tools to create new ML kernels and optimize existing ones.
* **ML researchers** can use their existing PyTorch experience and ecosystem tools to experiment freely on Trainium using our native PyTorch implementation, without having to learn new frameworks or APIs.

What is AWS Neuron used for?
-----------------------------

**Research and Development**: For model training, the recommended path is native PyTorch (TorchNeuron) together with TorchTitan for scaled pre-training. Neuron provides native PyTorch execution on Trainium with full Eager mode compatibility and supports standard distributed training patterns including FSDP, DDP, and DTensor for model sharding across devices and nodes. torch.compile integration enables graph optimization, while existing frameworks like TorchTitan, HuggingFace Optimum Neuron, and HuggingFace Transformers run without code modifications. JAX support includes XLA compilation targeting Inferentia and Trainium hardware. 

**Production Inference**: For LLM serving, the recommended path is the vLLM Neuron plugin. It implements the vLLM V1 APIs on Trainium (Trn2/Trn3) and supports expert parallelism for MoE models, disaggregated inference, and speculative decoding. For image and video generation with diffusion models, use the vLLM Omni Neuron plugin. Both plugins build on native PyTorch (TorchNeuron) and use optimized kernels from the NKI Library. 

**Performance Engineering**: Neuron Kernel Interface (NKI) provides direct access to Trainium instruction set architecture with APIs for memory management, execution scheduling, and low-level kernel development. The NKI Compiler, built on MLIR, compiles NKI kernels for Neuron hardware. The NKI Library contains optimized kernel implementations with source code and performance benchmarks. Neuron Explorer enables comprehensive profiling from application code to hardware execution, supporting both single-node and distributed workload analysis with detailed performance metrics and optimization recommendations.

AWS Neuron Core Components
----------------------------

**vLLM Neuron**
    vLLM Neuron is the Neuron plugin for vLLM and the recommended path for serving large language models (LLMs) on Trainium (Trn2 and Trn3). You serve models with the standard ``vllm serve`` command and an OpenAI-compatible API. vLLM Neuron compiles models with ``torch.compile`` through TorchNeuron, the native PyTorch backend, and uses optimized kernels from the NKI Library.

    It provides:

    * **Standard vLLM APIs**: Use the familiar vLLM V1 interfaces, including ``vllm serve``, the ``vllm.LLM`` Python API, and OpenAI-compatible endpoints for chat, completions, and embeddings
    * **Advanced inference features**: Continuous batching, prefix caching, speculative decoding (EAGLE3 and DFlash), disaggregated inference, expert parallelism for mixture-of-experts (MoE) models, structured outputs, and FP8 and MXFP8 quantization
    * **Broad model support**: Dense, MoE, multimodal, embedding, and speech recognition models, with a model recipe for each supported model family
    * **Open source**: Source code in the `vLLM Neuron GitHub repository <https://github.com/vllm-project/vllm-neuron>`_ under the vLLM project organization

    To get started, see :doc:`vLLM Neuron </vllm-neuron/docs/index>`.

**vLLM Omni Neuron**
    vLLM Omni Neuron is the Neuron plugin for `vLLM Omni <https://docs.vllm.ai/projects/vllm-omni/en/latest/>`_, the vLLM project library for multimodal generation. Use it to run diffusion models for image and video generation, such as Wan2.2, on Trainium (Trn2 and Trn3). Upstream vLLM does not serve diffusion models, so vLLM Neuron does not support them.

    It provides:

    * **Neuron-optimized model implementations**: Reference implementations with NKI kernels, custom compilation, and hardware-aware scheduling for models whose upstream implementations live in vLLM Omni
    * **Offline and online generation**: Offline generation from Python and online serving through the vLLM Omni OpenAI-compatible video API
    * **Parallelism for diffusion inference**: Tensor parallelism, context parallelism, and classifier-free guidance (CFG) parallelism
    * **Native PyTorch foundation**: The same native PyTorch (``torch.compile``) infrastructure that vLLM Neuron uses
    * **Open source**: Source code in the `vLLM Omni Neuron GitHub repository <https://github.com/aws-neuron/vllm-omni-neuron>`_

    To get started, see :doc:`vLLM Omni Neuron </vllm-omni-neuron/docs/index>`.

**Native PyTorch**
    Neuron provides native integration with PyTorch, enabling researchers and ML developers to run existing code unchanged on Trainium. Train models with familiar workflows and tools, from pre-training to post-training with reinforcement learning, while taking advantage of Trainium's performance and cost advantages for both experimentation and production scale training.

    It provides:

    * **Native Device Support**: Neuron registers as a native device type in PyTorch with standard device APIs like ``torch.tensor([1,2,3], device='neuron')`` and ``.to('neuron')``
    * **Standard Distributed Training APIs**: Support for FSDP, DTensor, DDP, tensor parallelism, context parallelism, and distributed checkpointing
    * **Eager Mode Execution**: Immediate operation execution for interactive development and debugging in notebook environments
    * **torch.compile Integration**: Support for ``torch.compile`` for optimized performance
    * **Open Source**: Released as an open source package on GitHub under Apache 2.0, enabling community contributions.  

**Neuron Kernel Interface (NKI)**
    For performance engineers seeking high hardware efficiency, Neuron provides complete control through the Neuron Kernel Interface (NKI), with direct access to the NeuronISA (NISA) instruction set, memory allocation, and execution scheduling. Developers can create new operations not available in standard frameworks and optimize performance critical code with custom kernels. 

    It includes:

    * The NKI Compiler, built on MLIR, which compiles NKI kernels for Neuron hardware
    * The NKI Library , which provides pre-built kernels you can use to optimize the performance of your models

**Neuron Tools**
    Debug and profiling utilities including:
    
    * Neuron Monitor for real-time performance monitoring
    * Neuron Explorer, the recommended profiler, for detailed performance analysis. Neuron Explorer replaced Neuron Profiler, which is no longer supported starting with Neuron 2.30.

    Neuron Explorer provides:

    * **Hierarchical Profiling**: Top-down visualization from framework layers through HLO operators to hardware instructions, enabling developers to understand execution at any level of the stack
    * **Code Linking**: Direct navigation between PyTorch, JAX, and NKI source code and performance timeline with automatic annotations showing metrics for specific code lines
    * **IDE Integration**: VSCode extension for profile visualization and analysis directly within the development environment
    * **Device Profiling**: Unified interface for comprehensive view of system-wide metrics and device-specific execution details

**Neuron Observer**
    Neuron Observer monitors the health of Trainium clusters. It collects accelerator and system telemetry from each node, reports device health and hardware faults, and identifies the node and device that a fault affects. Neuron Observer runs on Trn2 and Trn3 instances on Amazon EKS.

    It provides:

    * **Device health state**: A health state for each device: ``healthy``, ``unhealthy``, or ``unknown``
    * **Fault localization**: Device health and fault events that identify the affected node and device
    * **Performance telemetry**: Utilization, latency, EFA, ECC, runtime, and device metrics for performance investigations
    * **Flexible export**: Metrics for Prometheus and Grafana, Amazon Managed Service for Prometheus, or Amazon CloudWatch, plus an API and the ``nobs`` CLI for schedulers, control planes, and debugging workflows

    To get started, see :ref:`What is Neuron Observer? <neuron-observer-index>`.

**Neuron Compiler**
    Optimizes machine learning models for AWS Inferentia and Trainium chips, converting models from popular frameworks into efficient executable formats.

**Neuron Runtime**
    Manages model execution on Neuron devices, handling memory allocation, scheduling, and inter-chip communication for high throughput.

**AWS DLAMIs and DLCs**
    Orchestrate and deploy your models using Deep Learning AWS Machine Images (DLAMIs) and Deep Learning Containers (DLCs).

    Neuron DLAMIs come pre-configured with the Neuron SDK, popular frameworks, and helpful libraries, allowing you to quickly begin training and running inference. Or, quickly deploy models using pre-configured AWS Neuron Deep Learning Containers (Neuron DLCs) with optimized frameworks for AWS Trainium and Inferentia.  

Supported Hardware
------------------

**AWS Inferentia**
    Purpose-built for high-performance inference workloads:
    
    * ``Inf1`` instances - First-generation Inferentia chips (no longer supported)
    * ``Inf2`` instances - Second-generation with improved performance and efficiency

**AWS Trainium**
    Designed for distributed training of large models:
    
    * ``Trn1`` instances - High-performance training acceleration
    * ``Trn1n`` instances - Enhanced networking for large-scale distributed training
    * ``Trn2`` instances - Second-generation Trainium with upgraded performance
    * ``Trn3`` instances - Next-generation Trainium with further upgrades for modern model support
    * ``Trn2`` UltraServer - High-density Trainium servers for massive training workloads
    * ``Trn3`` UltraServer - The next generation of Trainium servers for massive training workloads


How do I get more information?
------------------------------

* Review the comprehensive documentation and follow the tutorials on this site
* Find source code, samples, and contribution guidelines in the :ref:`Neuron open source repositories <neuron-oss>`

* Submit questions and issues with AWS Neuron `GitHub issues <https://github.com/aws-neuron/aws-neuron-sdk/issues>`_
