.. meta::
    :description: History and evolution of PyTorch support on AWS Neuron across Inferentia and Trainium platforms
    :keywords: pytorch, torch-neuron, torch-neuronx, torchneuron, neuron, inferentia, trainium
    :date-modified: 02/26/2026

About PyTorch on AWS Neuron
===========================

This topic provides an overview of PyTorch support in Neuron for AWS ``Inf*`` (Inferentia-based) and ``Trn*`` (Trainium-based) ML platforms. 

Throughout the past 5 years, AWS Neuron has evolved its PyTorch support to match the capabilities and architectures of successive generations of AWS ML accelerators, delivering three distinct PyTorch implementations optimized for different hardware platforms and use cases:

* **torch-neuron** (2019): Graph-based inference for Inferentia (Inf1)
* **torch-neuronx** (2022): XLA-based training and inference for Inferentia2 (Inf2) and Trainium (Trn1/Trn2)
* **TorchNeuron** (2025): Native PyTorch backend for Trainium (Trn2/Trn3) with eager mode and ``torch.compile``

Overview
--------

AWS Neuron's PyTorch support has evolved through three major implementations, each designed to leverage the unique capabilities of AWS ML accelerators:

1. **torch-neuron** (2019-2026): The original PyTorch integration for AWS Inferentia (Inf1), focused on inference workloads with a graph-based compilation approach
2. **torch-neuronx** (2022-): An XLA-based PyTorch implementation for AWS Inferentia2 (Inf2) and Trainium (Trn1/Trn2/Trn3), supporting both training and inference with distributed computing capabilities
3. **TorchNeuron** (2025-): A native PyTorch backend for Trainium that provides eager mode execution, ``torch.compile`` support, and standard PyTorch distributed APIs without requiring XLA

Each implementation represents a significant architectural evolution, reflecting advances in both AWS ML accelerator hardware and PyTorch framework capabilities.

torch-neuron for Inf1
---------------------

The first Neuron library supporting PyTorch, ``torch-neuron``, was initially released in December 2019 alongside the launch of AWS Inferentia. This implementation introduced PyTorch developers to AWS's purpose-built ML inference accelerators.

``torch-neuron`` uses a graph-based compilation approach where PyTorch models are traced and compiled into optimized Neuron Executable File Format (NEFF) binaries. The library integrates with PyTorch through custom operators and provides APIs for model compilation (``torch.neuron.trace``) and execution on Inferentia NeuronCores.

Key characteristics of torch-neuron:

* **Target Platform**: AWS Inferentia (Inf1 instances)
* **Primary Use Case**: Inference workloads
* **Compilation Approach**: Ahead-of-time (AOT) graph compilation via ``torch.neuron.trace``
* **Supported Models**: Computer vision models (ResNet, VGG, EfficientNet, YOLO variants), NLP models (BERT, RoBERTa, DistilBERT, MarianMT), and other inference-optimized architectures
* **Integration Method**: Custom PyTorch operators and tracing API

When to choose torch-neuron
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose ``torch-neuron`` when:

* Deploying inference workloads on AWS Inferentia (Inf1) instances
* Working with models that can be traced and compiled ahead of time
* Optimizing for inference latency and throughput on first-generation Inferentia hardware
* Requiring compatibility with existing Inf1-based infrastructure


torch-neuronx for Inf2 and Trn1
-------------------------------

In October 2022, AWS introduced Inferentia2 and Trainium, second-generation ML accelerators with enhanced capabilities for both training and inference. To support these platforms, Neuron delivered ``torch-neuronx``, a new PyTorch implementation built on PyTorch/XLA.

``torch-neuronx`` represents a significant architectural shift from torch-neuron, leveraging the XLA (Accelerated Linear Algebra) compiler infrastructure to enable both training and inference workloads. This XLA-based approach provides support for dynamic shapes, control flow, distributed training primitives, and advanced parallelism strategies.

Key characteristics of torch-neuronx:

* **Target Platforms**: AWS Inferentia2 (Inf2 instances) and AWS Trainium (Trn1, Trn1n, Trn2, Trn3 instances)
* **Primary Use Cases**: Both training and inference workloads
* **Compilation Approach**: XLA-based compilation with support for dynamic shapes and control flow
* **Distributed Computing**: Native support for data parallelism, tensor parallelism, pipeline parallelism, sequence parallelism, and Zero Redundancy Optimizer (ZeRO)
* **Training Capabilities**: Full support for large-scale model training including LLMs (Llama, GPT, BERT families), with gradient accumulation, mixed precision training, and distributed checkpointing
* **Inference Capabilities**: Support for large language model inference with features like continuous batching, speculative decoding, and quantization
* **Integration Method**: PyTorch/XLA device backend (``xla`` device type)

The XLA-based architecture enables torch-neuronx to support advanced training techniques and distributed strategies that were not possible with the original torch-neuron implementation. This includes support for frameworks like NeuronX Distributed (NxD) for training and inference, Transformers NeuronX for LLM inference, and integration with popular ML libraries like HuggingFace Transformers and PyTorch Lightning.

When to choose torch-neuronx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose ``torch-neuronx`` when:

* Training models on AWS Trainium (Trn1, Trn1n, Trn2) instances
* Running inference on AWS Inferentia2 (Inf2) instances
* Requiring distributed training capabilities with tensor parallelism, pipeline parallelism, or data parallelism
* Working with large language models or other models requiring multi-device training
* Needing dynamic shape support or control flow in your models
* Using PyTorch versions 2.5 through 2.9 (XLA-based implementation)

**Note**: PyTorch 2.9 is the last version of torch-neuronx based on PyTorch/XLA. Starting with PyTorch 2.10 support (planned for a future Neuron release), torch-neuronx will transition to the native PyTorch implementation (TorchNeuron).


TorchNeuron (Native PyTorch integration)
----------------------------------------

**TorchNeuron**, the latest evolution of PyTorch support for Neuron, was announced in December 2025 at AWS re:Invent and shipped its initial version as part of Neuron release 2.27.0. While it retains the same Python package name as its predecessor (``torch-neuronx``), TorchNeuron is an entirely new native PyTorch backend developed specifically for Trainium platforms.

TorchNeuron represents a fundamental architectural shift from XLA-based compilation to native PyTorch integration through the PrivateUse1 device backend mechanism. This native integration enables PyTorch code to run on Trainium with minimal modifications, supporting both eager mode execution for rapid iteration and ``torch.compile`` for production optimization.

Key characteristics of TorchNeuron:

* **Target Platforms**: AWS Trainium (Trn2, Trn3 instances)
* **Primary Use Cases**: Training and inference workloads with native PyTorch workflows
* **Execution Modes**: 
  
  * **Eager Mode**: Immediate operation execution for interactive development and debugging
  * **torch.compile**: Just-in-time (JIT) compilation via TorchDynamo for optimized performance

* **Distributed APIs**: Native support for standard PyTorch distributed primitives:
  
  * Fully Sharded Data Parallel (FSDP)
  * Distributed Tensor (DTensor)
  * Distributed Data Parallel (DDP)
  * Tensor Parallelism (TP)

* **Integration Method**: Native PyTorch backend via PrivateUse1 mechanism (``neuron`` device type)
* **Ecosystem Compatibility**: Works with TorchTitan, HuggingFace Transformers, and other PyTorch ecosystem tools with minimal code changes
* **Custom Kernels**: Integration with Neuron Kernel Interface (NKI) for performance-critical operations
* **Open Source**: Available on GitHub under Apache 2.0 license

TorchNeuron's native integration eliminates the need for XLA-specific APIs and enables researchers and ML developers to use familiar PyTorch patterns. The eager mode support provides immediate feedback during development, while ``torch.compile`` delivers production-grade performance through hardware-specific optimizations.

The implementation includes Adaptive Eager Execution, which applies optimizations like operator fusion while maintaining functional accuracy and debuggability. This approach provides a balance between development velocity and runtime performance.

When to choose TorchNeuron
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose **TorchNeuron** (native PyTorch) when:

* Training models on AWS Trainium (Trn2, Trn3) instances with PyTorch 2.10 or later
* Requiring eager mode execution for interactive development and debugging
* Using standard PyTorch distributed training APIs (FSDP, DTensor, DDP)
* Working with PyTorch ecosystem tools like TorchTitan or HuggingFace Transformers
* Needing minimal code changes to run existing PyTorch code on Trainium
* Leveraging ``torch.compile`` for production optimization
* Developing custom kernels with Neuron Kernel Interface (NKI)

**Migration Note**: Starting with PyTorch 2.10 support (planned for a future Neuron release), AWS Neuron will transition from PyTorch/XLA to native PyTorch support via TorchNeuron. Users on PyTorch 2.9 or earlier will need to update their scripts when upgrading to PyTorch 2.10 or later. See :ref:`native-pytorch-trainium` for complete migration guidance.


Read More
---------

**Training Resources**

* :doc:`Training with torch-neuronx [archived content] </archive/nxd-training/index>` - Training guides and tutorials for Trainium
* :doc:`PyTorch Neuron Programming Guide [archived content] </archive/nxd-training/pytorch-neuron-programming-guide>` - Core concepts for training on Neuron
* :doc:`NeuronX Distributed (NxD) Training [archived content] </archive/nxd-training/index>` - Distributed training library for large-scale models
* :doc:`PyTorch Training Tutorials [archived content] </archive/nxd-training/index>` - Step-by-step training examples

**Inference Resources**

* :doc:`Inference with torch-neuronx </frameworks/torch/inference-torch-neuronx>` - Inference guides for Inf2 and Trn1/Trn2
* :doc:`Inference with torch-neuron </archive/torch-neuron/inference-torch-neuron>` - Inference guides for Inf1
* :doc:`NeuronX Distributed Inference (NxDI) </libraries/nxd-inference/index>` - Inference library for large language models
* :ref:`torch-neuron vs torch-neuronx Comparison <torch-neuron_vs_torch-neuronx>` - Detailed comparison for inference workloads

**Architecture and Hardware**

* :doc:`AWS Inferentia Architecture </about-neuron/arch/neuron-hardware/inferentia>` - Inf1 hardware architecture
* :doc:`AWS Inferentia2 Architecture </about-neuron/arch/neuron-hardware/inferentia2>` - Inf2 hardware architecture
* :doc:`AWS Trainium Architecture </about-neuron/arch/neuron-hardware/trainium>` - Trn1 hardware architecture
* :doc:`AWS Trainium2 Architecture </about-neuron/arch/neuron-hardware/trainium2>` - Trn2 hardware architecture
* :doc:`AWS Trainium3 Architecture </about-neuron/arch/neuron-hardware/trainium3>` - Trn3 hardware architecture


