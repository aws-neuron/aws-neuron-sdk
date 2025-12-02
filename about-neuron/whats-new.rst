.. meta::
    :description: Latest features and updates for AWS Neuron SDK including Trainium3 UltraServer support, native PyTorch integration, and enhanced NKI capabilities
    :date-modified: 12/02/2025

AWS Neuron Expands with Trainium3, Native PyTorch, Faster NKI, and Open Source at re:Invent 2025
=================================================================================================

**Last updated**: 12/02/2025

.. image:: /images/NeuronStandalone_white_small.png
   :alt: AWS Neuron Logo
   :align: right
   :width: 120px



At re:Invent 2025, AWS Neuron introduces support for `Trainium3 UltraServer <https://aws.amazon.com/ai/machine-learning/trainium/>`__ with expanded open source components and enhanced developer experience. These updates enable standard frameworks to run unchanged on Trainium, removing barriers for researchers to experiment and innovate. For developers requiring deeper control, the enhanced Neuron Kernel Interface (NKI) provides direct access to hardware-level optimizations, enabling customers to scale AI workloads with improved performance.


**Expanded capabilities and enhancements include**:


* :doc:`Trainium3 UltraServer support </about-neuron/arch/neuron-hardware/trn3-arch>`: Enabling customers to scale AI workloads with improved performance
* :doc:`Native PyTorch support </frameworks/torch/pytorch-native-overview>`: Standard PyTorch runs unchanged on Trainium without platform-specific modifications
* :doc:`Enhanced Neuron Kernel Interface (NKI) </nki/about/index>` with open source :doc:`NKI Compiler </nki/compiler/about/index>`: Improved programming capabilities with direct access to Trainium hardware instructions and fine-grained optimization control, compiler built on MLIR
* :doc:`NKI Library </nki/library/index>`: Open source collection of optimized, ready-to-use kernels for common ML operations
* :doc:`Neuron Explorer </tools/neuron-explorer/index>`: Tools suite to support developers and performance engineers in their performance optimization journey from framework operations to hardware instructions
* :doc:`Neuron DRA for Kubernetes </containers/neuron-dra>`: Kubernetes-native resource management eliminating custom scheduler extensions
* :doc:`Expanded open source components </about-neuron/oss/index>``: Open sourcing more components including NKI Compiler, Native PyTorch, NKI Library, and more released under Apache 2.0


AI development requires rapid experimentation, hardware optimization, and production scale workloads. These updates enable researchers to experiment with novel architectures using familiar workflows, ML developers to build AI applications using standard frameworks, and performance engineers to optimize workloads using low-level hardware optimization.


Native PyTorch Support
-----------------------

**Private Preview**

AWS Neuron now natively supports PyTorch through TorchNeuron, an open source native PyTorch backend for Trainium. TorchNeuron integrates with PyTorch through the PrivateUse1 device backend mechanism, registering Trainium as a native device alongside other backends and allowing researchers and ML developers to run their code without modifications.

TorchNeuron provides eager mode execution for interactive development and debugging, native distributed APIs including FSDP and DTensor for distributed training, and torch.compile support for optimization. TorchNeuron enables compatibility with minimal code changes with ecosystem tools like TorchTitan and HuggingFace Transformers.

Use TorchNeuron to run your PyTorch research and training workloads on Trainium without platform-specific code changes.

**Learn more**: `Native PyTorch blog post <https://pytorch.org/blog/torchneuron-native-pytorch-backend/>`__, :doc:`documentation </frameworks/torch/pytorch-native-overview>`, and `TorchNeuron GitHub repository <https://github.com/aws-neuron/torch-neuronx>`.

**Access**: Contact your AWS account team for access.


Enhanced NKI
-------------

**Public Preview**

The enhanced Neuron Kernel Interface (NKI) provides developers with complete hardware control through advanced APIs for fine-grained scheduling and allocation. The enhanced NKI enables instruction-level programming, memory allocation control, and execution scheduling with direct access to the Trainium ISA. 

We are also releasing the NKI Compiler as open source under Apache 2.0, built on MLIR to enable transparency and collaboration with the broader compiler community. NKI integrates with PyTorch and JAX, enabling developers to use custom kernels within their training workflows.

Use Enhanced NKI to innovate and build optimized kernels on Trainium. Explore the NKI Compiler source code to inspect and contribute to the MLIR-based compilation pipeline. 

.. note::
  The NKI Compiler source code is currently in **Private Preview**, while the NKI programming interface is in **Public Preview**.

**Learn more**: :doc:`NKI home page </nki/index>`, :doc:`NKI Language Guide </nki/deep-dives/nki-language-guide>`, and `NKI Compiler GitHub repository <https://github.com/aws-neuron/nki-compiler>`__.


NKI Library
------------

**Public Preview**

The NKI Library provides an open source collection of optimized, ready-to-use kernels for common ML operations. The library includes kernels for dense transformer operations, MoE-specific operations, and attention mechanisms, all with complete source code, documentation, and benchmarks.

Use NKI Library kernels directly in your models to improve performance, or explore the implementations as reference for best practices of performance optimizations on Trainium.

**Learn more**: `GitHub repository <https://github.com/aws-neuron/nki-library>`__ and :doc:`API documentation </nki/library/api/index>`.


Neuron Explorer
----------------

**Public Preview**

Neuron Explorer is a tools suite that supports developers and performance engineers in their performance optimization journey. It provides capabilities to inspect and optimize code from framework operations down to hardware instructions with hierarchical profiling, source code linking, IDE integration, and AI-powered recommendations for optimization insights.

Use Neuron Explorer to understand and optimize your model performance on Trainium, from high-level framework operations to low-level hardware execution.

**Learn more**: :doc:`Neuron Explorer documentation </tools/neuron-explorer/index>`.


Kubernetes-Native Resource Management with Neuron DRA
------------------------------------------------------

**Private Preview**

Neuron Dynamic Resource Allocation (DRA) provides Kubernetes-native resource management for Trainium, eliminating custom scheduler extensions. DRA enables topology-aware scheduling using the default Kubernetes scheduler, atomic UltraServer allocation, and flexible per-workload configuration.

Neuron DRA supports EKS, SageMaker HyperPod, and UltraServer configurations. The driver is open source with container images in AWS ECR public gallery.

Use Neuron DRA to simplify Kubernetes resource management for your Trainium workloads with native scheduling and topology-aware allocation.

**Learn more**: :doc:`Neuron DRA documentation </containers/neuron-dra>`.

**Access**: Contact your AWS account team to participate in the Private Preview.


Resources and Additional Information
--------------------------------------

For more information visit the `AWS Trainium official page <https://aws.amazon.com/ai/machine-learning/trainium/>`__, the :doc:`AWS Neuron Documentation </index>`, and :doc:`the AWS Neuron GitHub repositories </about-neuron/oss/index>`.




