.. _main_whats-new:

.. meta::
    :description: Blog posts for the latest features and updates for the AWS Neuron SDK
    :date-modified: 05/01/2026

What's New in the AWS Neuron SDK
================================

.. toctree::
   :hidden:
   :maxdepth: 1

   Release Notes </release-notes/index>

*Explore detailed posts about the latest releases, updates, and upcoming changes to the AWS Neuron SDK.*

.. grid:: 1
    :gutter: 2

    .. grid-item-card:: Neuron Release Notes
        :link: /release-notes/index
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        **Latest release**: 2.29.1 (05/01/2026)

----

.. _whats-new-2026-04-30-neuron-agentic-development:

Neuron Agentic Development: AI coding agents and skills to develop on Neuron
-----------------------------------------------------------------------------

**Posted on**: April 30, 2026

Today we are releasing Neuron Agentic Development, an open-source suite of AI capabilities starting with agents and skills that author, debug, profile, and analyze Neuron Kernel Interface (NKI) kernels for AWS Trainium and AWS Inferentia. The agents and skills run inside agentic coding environments such as Claude Code and Kiro, letting you drive kernel development with natural language or reference implementations in PyTorch or NumPy.

This first release focuses on the NKI kernel development lifecycle and includes:

* **Kernel authoring**: Translate PyTorch, NumPy, or natural-language specifications into NKI kernel code.
* **Compilation debugging**: Compile on-device, interpret compiler errors, and iteratively apply fixes.
* **Profile capture and querying**: Capture NEFF and NTFF execution profiles and query engine utilization and execution behavior in the agent conversation.
* **Profile analysis**: Detect intermediate-data spilling and redundant TensorEngine transposes, and attribute them to known inefficiency patterns.
* **Documentation lookup**: Route questions to the right NKI API references, tutorials, and compiler error-code pages from within the agent.

Skills chain automatically, so a single prompt such as "profile my kernel and tell me what's slow" can trigger the full capture, metric-extraction, and analysis sequence. You can also call individual skills directly for targeted tasks.

Writing and documentation skills work in any supported IDE without hardware. Debugging, profiling, and analysis skills require access to Trainium or Inferentia hardware because they compile and execute on the NeuronCores.

For an overview of capabilities, supported IDEs, and prerequisites, see :ref:`neuron-agentic-development-overview`. For installation, the current catalog of agents and skills, and contribution guidance, visit the `Neuron Agentic Development repository on GitHub <https://github.com/aws-neuron/neuron-agentic-development>`_.

----

.. _whats-new-2026-04-02-v2_29:

AWS Neuron SDK 2.29.0: NKI Exits Beta, CPU Simulator, and Expanded NKI Library
-------------------------------------------------------------------------------

**Posted on**: April 09, 2026

Today we are releasing AWS Neuron SDK 2.29.0. This release brings NKI 0.3.0 out of Beta into Stable, featuring the new NKI Standard Library and an experimental CPU Simulator for local kernel development without Trainium hardware. The NKI Library adds 7 new experimental kernels including Conv1D, a Transformer TKG megakernel, and fused communication-compute primitives, along with improvements to existing attention, MLP, and MoE kernels. NxD Inference delivers performance gains for Qwen2 VL, Qwen3 VL, and Flux.1 models. Neuron Runtime introduces new APIs for collective stream management and network proxy tuning. Neuron Explorer is now out of Beta and Stable, with full Device widget support in the System Trace Viewer and availability on the VS Code Extension Marketplace. The Neuron Driver adds support for new Trn3 Gen2 Ultraserver configurations.


Neuron Kernel Interface (NKI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AWS Neuron SDK 2.29.0 introduces NKI 0.3.0, the latest update to the Neuron Kernel Interface. NKI 0.3.0 is now out of Beta and Stable. It features the NKI Standard Library (``nki-stdlib``), which provides developer-visible code for all NKI APIs and native language objects (such as ``NkiTensor``). This release provides new exposed Trainium capabilities and features in the NKI API and introduces ``nki.language`` APIs.

**NKI CPU Simulator (Experimental)**: NKI 0.3.0 includes a CPU Simulator, which executes NKI kernels entirely on CPU and allows for a fast development cycle on inexpensive CPUs and compute instances to validate kernel correctness, using standard Python step-by-step debugging tools and instrumentation to print results for every line of kernel code. Activate it with ``NKI_SIMULATOR=1`` or use ``nki.simulate(kernel)``.

**New Language APIs (Experimental)**: Introduced ``nki.language`` high-level convenience wrappers including ``nl.load``, ``nl.store``, ``nl.copy``, ``nl.matmul``, ``nl.transpose``, and ``nl.softmax``. 

**New ISA and Hardware Features**: Added the ability to set DMA priority of DMA operations and collectives operations for Trn3 (NeuronCore-v4). A dedicated ``nki.isa.exponential`` instruction is optimized for vectorising exponents (``exp``) with VectorE. Matmul accumulation control is added via the ``accumulate`` parameter on ``nc_matmul`` and ``nc_matmul_mx``. Variable-length all-to-all collectives are now available via ``nki.collectives.all_to_all_v``.

**Breaking Changes**: NKI 0.3.0 includes several API breaking changes that improve correctness and consistency. All kernels must be updated to NKI 0.3.0; mixing with NKI 0.2.0 kernels in the same model is not supported. For the full list of changes and migration examples, see the :doc:`NKI 0.3.0 Update Guide </nki/migration/nki-0-3-0-update-guide>`.

For more details, see :ref:`nki-2-29-0-rn`.


NKI Library
^^^^^^^^^^^

**New Experimental Kernels (7 added)**: Conv1D provides 1D convolution with stride, padding, dilation, bias, activation fusion, and LNC sharding. Transformer TKG is a multi-layer transformer forward pass megakernel for token generation. Fine-Grained All-Gather and FGCC (All-Gather + Matmul) enable ring-based communication with compute overlap on Trn2. SBUF-to-SBUF All-Gather provides two variants for small and large tensors. Top-K Reduce supports MoE output gathering with LNC sharding. Dynamic Elementwise Add handles runtime-variable M-dimension tiling. The ``find_nonzero_indices`` subkernel is promoted from experimental to core.

**Key Improvements to Existing Kernels**: Attention CTE increases max batch size from 32 to 512 and max sequence length from 36,864 to 131,072 with sequence packing support. Attention Block TKG adds fused QK-norm before RoPE and KVDP attention sharding. MLP adds BufferManager support and MXFP4/MXFP8 quantization paths. MoE TKG introduces a dynamic all-expert algorithm with ``block_size``. QKV adds flexible weight layout support. PyTorch reference implementations are added for 22 kernels.

**Breaking Changes**: Multiple kernel signatures have changed with new parameters inserted mid-signature; callers using positional arguments must switch to keyword arguments. ``SbufManager`` is renamed to ``BufferManager``. MoE TKG replaces boolean sharding flags with ``LNCShardingStrategy`` enum. For the full list of breaking changes, see :ref:`nki-lib-2-29-0-rn`.

For more details, see :ref:`nki-lib-2-29-0-rn`.


Inference Updates
^^^^^^^^^^^^^^^^^

**NxD Inference 0.9.17155**: Qwen2 VL gains vision data parallelism with 7% QPS improvement for image-heavy workloads. Qwen3 VL adds text-model sequence parallelism with 2.2x QPS throughput improvement. Flux.1 adds CFG parallelism with 19% end-to-end latency improvement and 23% instance throughput improvement.

**vLLM Neuron Plugin 0.5.0**: Updated alongside NxD Inference with model performance improvements.

**Hardware Support Change**: NxD Inference no longer supports Trn1/Inf2. Only Trn2 and newer hardware is supported. Pin to Neuron SDK 2.28 for Trn1/Inf2 support.

For more details, see :ref:`nxd-inference-2-29-0-rn`.


Runtime and Driver
^^^^^^^^^^^^^^^^^^

**Neuron Runtime Library 2.31**: New ``nrt_cc_create_stream`` API creates a collective stream to be used by host-initiated collectives, replacing the previous environment variable approach. New ``nrt_get_attached_efa_bdf`` API returns the BDF string of the EFA device for optimal network interface selection. New environment variables ``NEURON_RT_ONE_THREAD_PER_CORE`` (up to 2x improvement in collective communication latency) and ``NEURON_RT_RANKS_PER_NETWORK_PROXY`` provide fine-grained control over network proxy threading. RDMA support extends to Trn3. Collectives XU gains profiling support, context caching with up to 90% performance improvement, and removal of the 512 queue set instance limit. The async API version is bumped from 2.x to 3.0; applications using the async API must be recompiled.

**Neuron Driver 2.27**: Adds support for new Trn3 Gen2 Ultraserver configurations: US3 (2-node), US4 (4-node), US16 (4-node), and US18 (4-node). Top-level DMA reset support is added during TPB reset on Trn3 and later platforms.

**Neuron Collectives 2.31**: EFA device processing is restructured to per-stream granularity for improved stability. Fixed incorrect interface selection in multi-ultraserver collectives and crash on channel initialization failures.

For more details, see :ref:`runtime-2-29-0-rn`.


Neuron Explorer
^^^^^^^^^^^^^^^

Neuron Explorer is now out of Beta and Stable. The System Trace Viewer now supports the full suite of Device widgets, enabling multi-device profile analysis across all linked Device Profiles within a single System Profile. The Summary Viewer includes system-level profile data for both system and device profiles. New System Timeline HBM Usage shows device HBM usage with memory allocation breakdown by category. Box Selection Summary enables viewing aggregated device profile information for a selected region in the trace viewer. Neuron Explorer for VS Code is now available on the Visual Studio Code Extension Marketplace and Open VSX, enabling simpler installation and automatic updates.

For more details, see :ref:`dev-tools-2-29-0-rn`.


PyTorch Framework
^^^^^^^^^^^^^^^^^

PyTorch 2.7 and 2.8 have reached end of support starting with this release. Use PyTorch 2.9 on Ubuntu 24.04. Starting with PyTorch 2.10 support (planned for a future Neuron release), AWS Neuron will transition from PyTorch/XLA to native PyTorch support via TorchNeuron.

For more details, see :ref:`pytorch-2-29-0-rn`.


End of Support and Migration Notices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Effective this release:**

* PyTorch 2.7 and 2.8 have reached end of support. Pin to Neuron SDK 2.28 if required.
* NeuronX Distributed Training (NxDT) and NxD Core training APIs reach end of support; DLCs and DLAMI virtual environments pinned to SDK 2.28.0.
* ``neuron-profile analyze`` subcommand is no longer supported. Migrate to Neuron Explorer.
* Ubuntu 22.04 Multi-Framework DLAMI is no longer published. Use Ubuntu 24.04.

**Hardware support:**

* NxD Inference no longer supports Trn1/Inf2. Pin to Neuron SDK 2.28 for continued support.

**NKI namespace migration:**

* Removal of ``neuronxcc.nki.*`` namespace postponed to a future release. Both ``neuronxcc.nki.*`` and ``nki.*`` namespaces continue to work. Migration to ``nki.*`` is encouraged.

**Effective with PyTorch 2.10 support:**

* PyTorch/XLA will be replaced by TorchNeuron.

* Read the :doc:`Neuron 2.29.0 component release notes </release-notes/prev/2.29.0>` for specific Neuron component improvements and details.

----

.. _whats-new-2026-03-13-v2_28_1:

AWS Neuron SDK 2.28.1 Patch Available
--------------------------------------

**Posted on**: March 13, 2026

AWS Neuron provides a patch version, 2.28.1, to address a Neuron Driver compatibility issue with Linux kernel 6.18. 

.. _whats-new-2026-02-26-v2_28:

AWS Neuron SDK 2.28.0: Enhanced Profiling, Vision Language Models, and Expanded NKI Capabilities
--------------------------------------------------------------------------------------------------

**Posted on**: February 26, 2026

Today we are releasing AWS Neuron SDK 2.28.0. This release enhances Neuron Explorer with system profiling, Tensor Viewer, and Database Viewer for comprehensive performance analysis. NxD Inference adds support for Qwen2/Qwen3 VL vision language models, Flux.1 inpainting capabilities, and Eagle3 speculative decoding. The NKI Library expands with 9 new kernels including RoPE, MoE operations, and experimental kernels for attention and cross entropy. NKI 0.2.0 introduces LNC multi-core support with intra-LNC collectives and new APIs. Kubernetes users gain Neuron DRA Driver support for advanced resource allocation.


Developer Tools and Profiling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Neuron Explorer Enhancements** - Added system profiling support with drill-down navigation to device profiles. New Tensor Viewer helps identify memory bottlenecks by displaying tensor names, shapes, sizes, and memory usage. Database Viewer provides an interactive interface for querying profiling data using SQL or natural language. Profile Manager now supports tag-based organization and search. A migration guide from Neuron Profiler/Profiler 2.0 is now available.

**nccom-test Improvements** - Enhanced data integrity checks use pseudo-random data patterns for better corruption detection. Added support for ``alltoallv`` collective operation for benchmarking variable-sized all-to-all communication patterns.

For more details, see :ref:`dev-tools-2-28-0-rn`.

Inference Updates
^^^^^^^^^^^^^^^^^

**NxD Inference 0.8.16251** - Added support for vision language models including Qwen2 VL (Qwen2-VL-7B-Instruct) and Qwen3 VL (Qwen3-VL-8B-Thinking) for processing text and image inputs (Beta). Pixtral model support improved with batch size 32 and sequence length 10240 on Trn2 with vLLM V1. Flux.1 model gains new functionality for in-paint, out-paint, canny edge detection, and depth-based image generation (Beta).

**vLLM Neuron Plugin 0.4.1** - Multi-LoRA serving enhancements enable streaming LoRA adapters via vLLM's ``load_adapter`` API with dynamic runtime loading. Users can now run the base model alone when multi-LoRA serving is enabled. Added Eagle3 speculative decoding support for Llama 3.1 8B. Updated to support vLLM v0.13.0 and PyTorch 2.9.

For more details, see :ref:`nxd-inference-2-28-0-rn`.

NKI Library
^^^^^^^^^^^

**9 New Kernels** - The NKI Library expands from 7 to 16 documented kernel APIs. New core kernels include RoPE (Rotary Position Embedding), Router Top-K (expert selection for MoE), MoE CTE (Context Encoding), MoE TKG (Token Generation), and Cumsum. New experimental kernels include Attention Block TKG (fused attention for token generation), Cross Entropy (forward and backward passes), Depthwise Conv1D, and Blockwise MM Backward (for MoE training).

**Enhanced Quantization Support** - Existing kernels receive FP8 and MX quantization support across QKV, MLP, and Output Projection kernels. QKV kernel adds fused FP8 KV cache quantization and block-based KV cache layout. MLP kernel adds gate/up projection clamping and fp16 support for TKG mode. Attention CTE kernel adds strided Q slicing for context parallelism.

**Improved Utilities** - TensorView gains ``rearrange`` method for dimension reordering and ``has_dynamic_access`` for runtime-dependent addressing checks. SbufManager provides hierarchical tree-formatted allocation logging with new query methods for SBUF utilization. New utilities include ``rmsnorm_mx_quantize_tkg``, ``interleave_copy``, ``LncSubscriptable``, and ``TreeLogger``.

For more details, see :ref:`nki-lib-2-28-0-rn`.

Neuron Kernel Interface (NKI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**NKI 0.2.0** - This release includes LNC multi-core support for LNC=2, enabling kernels to leverage multiple NeuronCores within a logical NeuronCore. The compiler now tracks ``shared_hbm`` tensors and canonicalizes LNC kernel outputs. Users can declare tensors private to a single NeuronCore using ``private_hbm`` memory type.

**New nki.collectives Module** - Enables collective communication across multiple NeuronCores with operations including ``all_reduce``, ``all_gather``, ``reduce_scatter``, ``all_to_all``, ``collective_permute`` variants, and ``rank_id``.

**New APIs and Features** - New ``nki.isa`` APIs include ``nonzero_with_count`` for sparse computation and ``exponential`` for element-wise operations. New ``float8_e4m3fn`` dtype supports FP8 workloads. Language features include ``no_reorder`` blocks for instruction ordering control, ``__call__`` special method support, ``tensor.view`` method for reshaping, and shared constants as string arguments.

**API Improvements** - ``dma_transpose`` now supports indirect addressing, ``dma_copy`` adds the ``unique_indices`` parameter, and ``register_alloc`` accepts optional tensor arguments for pre-filling. The compiler no longer truncates diagnostic output.

For more details, see :ref:`nki-2-28-0-rn`.

Kubernetes Support
^^^^^^^^^^^^^^^^^^

**Neuron DRA Driver** - Introduced Neuron Dynamic Resource Allocation (DRA) Driver enabling advanced resource allocation using the Kubernetes DRA API for flexible and efficient Neuron device management. The DRA API provides topology-aware scheduling, atomic resource allocation, and per-workload configuration. Neuron Helm Charts now include DRA Driver support.

For more details, see :ref:`containers-2-28-0-rn`.

PyTorch Framework (torch-neuronx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Transition to Native PyTorch Support** - Starting with PyTorch 2.10 support (planned for a future Neuron release), AWS Neuron will transition from PyTorch/XLA to native PyTorch support via TorchNeuron. PyTorch 2.9 is the last version using PyTorch/XLA. Users will need to update their scripts when upgrading to PyTorch 2.10 or later. See :ref:`native-pytorch-trainium` for migration guidance.

For more details, see :ref:`pytorch-2-28-0-rn`.

* Read the :doc:`Neuron 2.28.1 component release notes </release-notes/prev/2.28.1>` for specific Neuron component improvements and details.

.. _whats-new-2025-12-19-v2_27:

AWS Neuron SDK 2.27.0: Trainium3 Support, Enhanced NKI, and Unified Profiling with Neuron Explorer
---------------------------------------------------------------------------------------------------

**Posted on**: December 19, 2025

Today we are releasing AWS Neuron SDK 2.27.0. This release adds support for Trainium3 (``Trn3``) instances. Enhanced NKI with new NKI Compiler introduces the ``nki.*`` namespace with updated APIs and language constructs. The NKI Library provides pre-optimized kernels for common model operations including attention, MLP, and normalization. Neuron Explorer delivers a unified profiling suite with AI-driven optimization recommendations. vLLM V1 integration is now available through the vLLM-Neuron Plugin. Deep Learning Containers and AMIs are updated with vLLM V1, PyTorch 2.9, JAX 0.7, Ubuntu 24.04, and Python 3.12.

In addition to this release, we are introducing new capabilities and features in private beta access (see Private Beta Access section). We are also announcing our transition to PyTorch native support starting with PyTorch 2.10 in Neuron 2.28, plans to simplify NxDI in upcoming releases, and other important updates.

Neuron Kernel Interface (NKI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**NKI Compiler** - The new ``nki.*`` namespace replaces the legacy ``neuronxcc.nki.*`` namespace. Top-level kernel functions now require the ``@nki.jit`` annotation. Neuron 2.27 supports both namespaces side by side; the legacy namespace will be removed in Neuron 2.28. A kernel migration guide is available in the documentation.

For more details, see :ref:`neuron-2-27-0-nki`.

NKI Library
^^^^^^^^^^^

The NKI Library provides pre-optimized kernels: Attention CTE, Attention TKG, MLP, Output Projection CTE, Output Projection TKG, QKV, and RMSNorm-Quant. Kernels are accessible via the ``nkilib.*`` namespace in neuronx-cc or from the GitHub repository.

For more details, see :ref:`neuron-2-27-0-nkilib`.

Developer Tools
^^^^^^^^^^^^^^^

**Neuron Explorer** - A a suite of tools designed to support ML engineers throughout their development journey on AWS Trainium. This release features improved performance and user expereince for device profiling, with four core viewers to provide insights into model performance:

* **Hierarchy Viewer**: Visualizes model structure and component interactions
* **AI Recommendation Viewer**: Delivers AI-driven optimization recommendations
* **Source Code Viewer**: Links profiling data directly to source code
* **Summary Viewer**: Displays high-level performance metrics

Neuron Explorer is available through UI, CLI, and VSCode IDE integration. Existing NTFF files are compatible but require reprocessing for new features.

New tutorials cover profiling NKI kernels, multi-node training jobs, and vLLM inference workloads. The ``nccom-test`` tool now includes fine-grained collective communication support.

For more details, see :ref:`neuron-2-27-0-tools`.

Inference Updates
^^^^^^^^^^^^^^^^^

**vLLM V1** - The vLLM-Neuron Plugin enables vLLM V1 integration for inference workloads. vLLM V0 support ends in Neuron 2.28.

**NxD Inference** - Model support expands with beta releases of Qwen3 MoE (Qwen3-235B-A22B) for multilingual text and Pixtral (Pixtral-Large-Instruct-2411) for image understanding. Both models use HuggingFace checkpoints and are supported on ``Trn2`` and ``Trn3`` instances.

For more details, see :ref:`neuron-2-27-0-nxd-inference`.

Neuron Graph Compiler
^^^^^^^^^^^^^^^^^^^^^

Default accuracy settings are now optimized for precision. The ``--auto-cast`` flag defaults to ``none`` (previously ``matmul``), and ``--enable-mixed-precision-accumulation`` is enabled by default. FP32 models may see performance impacts; restore previous behavior with ``--auto-cast=matmul`` and ``--disable-mixed-precision-accumulation``. Python 3.10 or higher is now required.

For more details, see :ref:`neuron-2-27-0-compiler`.

Runtime Improvements
^^^^^^^^^^^^^^^^^^^^

**Neuron Runtime Library 2.29** adds support for Trainium3 (``Trn3``) instances and delivers performance improvements for Collectives Engine overhead, NeuronCore branch overhead, NEFF program startup, and all-gather latency.

For more details, see :ref:`neuron-2-27-0-runtime`.

Deep Learning AMIs and Containers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Platform Updates** - All DLCs are updated to Ubuntu 24.04 and Python 3.12. DLAMIs add Ubuntu 24.04 support for base, single framework, and multi-framework configurations.

**Framework Updates**:

* vLLM V1 single framework DLAMI and multi-framework virtual environments
* PyTorch 2.9 single framework DLAMIs and multi-framework virtual environments (Amazon Linux 2023, Ubuntu 22.04, Ubuntu 24.04)
* JAX 0.7 single framework DLAMI and multi-framework virtual environments

**New Container** - The ``pytorch-inference-vllm-neuronx`` 0.11.0 DLC provides a complete vLLM inference environment with PyTorch 2.8 and all dependencies.

For more details, see :ref:`neuron-2-27-0-dlami` and :ref:`neuron-2-27-0-dlc`.


End of Support and Migration Notices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Effective this release:**

* :ref:`announcement-python-3-9-eol`
* :ref:`announcement-end-of-support-pytorch-2-6`
* :ref:`announce-no-support-tensorflow2-10`
* :ref:`announce-eos-inf1-virtual-environments`
* :ref:`announcement-end-of-support-parallel-model-trace`
* :ref:`announce-eos-tensorboard-tools`

**Effective Neuron 2.28:**

* :ref:`announcement-end-of-support-neuronxcc-nki`
* :ref:`announcement-nki-library-namespace-changes`
* :ref:`announcement-nki-library-kernel-migration`
* :ref:`announcement-end-of-support-vllm-v0`

**Effective with PyTorch 2.10 support:**

* :ref:`announce-transition-pytorch-trainium`
* :ref:`announcement-end-of-support-nxdt-nxd-core`

**Future Releases:**

* :ref:`announce-nxdi-changes`
* :ref:`announce-eos-dlami-ubuntu-22-04`
* :ref:`announce-eos-pytorch-profling-api`
* :ref:`announce-eos-neuron-profiler`

Detailed Release Notes
^^^^^^^^^^^^^^^^^^^^^^^

* Read the :doc:`Neuron 2.27.0 component release notes </release-notes/prev/2.27.0/index>` for specific Neuron component improvements and details.

----

.. _whats-new-2025-12-02-riv:

AWS Neuron Expands with Trainium3, Native PyTorch, Faster NKI, and Open Source at re:Invent 2025
------------------------------------------------------------------------------------------------

**Posted on**: 12/02/2025

.. image:: /images/NeuronStandalone_white_small.png
   :alt: AWS Neuron Logo
   :align: right
   :width: 120px

At re:Invent 2025, AWS Neuron introduces support for `Trainium3 UltraServer <https://aws.amazon.com/ai/machine-learning/trainium/>`__ with expanded open source components and enhanced developer experience. These updates enable standard frameworks to run unchanged on Trainium, removing barriers for researchers to experiment and innovate. For developers requiring deeper control, the enhanced Neuron Kernel Interface (NKI) provides direct access to hardware-level optimizations, enabling customers to scale AI workloads with improved performance.

**Expanded capabilities and enhancements include**:

* :doc:`Trainium3 UltraServer support </about-neuron/arch/neuron-hardware/trn3-arch>`: Enabling customers to scale AI workloads with improved performance
* :doc:`Native PyTorch support </frameworks/torch/pytorch-native-overview>`: Standard PyTorch runs unchanged on Trainium without platform-specific modifications
* :doc:`Enhanced Neuron Kernel Interface (NKI) </nki/get-started/about/index>` with open source :doc:`NKI Compiler </nki/deep-dives/nki-compiler>`: Improved programming capabilities with direct access to Trainium hardware instructions and fine-grained optimization control, compiler built on MLIR
* :doc:`NKI Library </nki/library/index>`: Open source collection of optimized, ready-to-use kernels for common ML operations
* :doc:`Neuron Explorer </tools/neuron-explorer/index>`: Tools suite to support developers and performance engineers in their performance optimization journey from framework operations to hardware instructions
* :doc:`Neuron DRA for Kubernetes </containers/neuron-dra>`: Kubernetes-native resource management eliminating custom scheduler extensions
* :doc:`Expanded open source components </about-neuron/oss/index>`: Open sourcing more components including NKI Compiler, Native PyTorch, NKI Library, and more released under Apache 2.0


AI development requires rapid experimentation, hardware optimization, and production scale workloads. These updates enable researchers to experiment with novel architectures using familiar workflows, ML developers to build AI applications using standard frameworks, and performance engineers to optimize workloads using low-level hardware optimization.

.. admonition:: Looking to try out our Beta features?

   Submit your beta access request through `this form <https://pulse.aws/survey/NZU6MQGW?p=0>`__ and the Neuron Product team will get back to you.

Native PyTorch Support
^^^^^^^^^^^^^^^^^^^^^^

**Private Preview**

AWS Neuron now natively supports PyTorch through TorchNeuron, an open source native PyTorch backend for Trainium. TorchNeuron integrates with PyTorch through the PrivateUse1 device backend mechanism, registering Trainium as a native device alongside other backends and allowing researchers and ML developers to run their code without modifications.

TorchNeuron provides eager mode execution for interactive development and debugging, native distributed APIs including FSDP and DTensor for distributed training, and torch.compile support for optimization. TorchNeuron enables compatibility with minimal code changes with ecosystem tools like TorchTitan and HuggingFace Transformers.

Use TorchNeuron to run your PyTorch research and training workloads on Trainium without platform-specific code changes.

**Learn more**: :doc:`documentation </frameworks/torch/pytorch-native-overview>`, and `TorchNeuron GitHub repository <https://github.com/aws-neuron/torch-neuronx>`__.

**Access**: Contact your AWS account team for access.


Enhanced NKI
^^^^^^^^^^^^

**Public Preview**

The enhanced Neuron Kernel Interface (NKI) provides developers with complete hardware control through advanced APIs for fine-grained scheduling and allocation. The enhanced NKI enables instruction-level programming, memory allocation control, and execution scheduling with direct access to the Trainium ISA. 

We are also releasing the NKI Compiler as open source under Apache 2.0, built on MLIR to enable transparency and collaboration with the broader compiler community. NKI integrates with PyTorch and JAX, enabling developers to use custom kernels within their training workflows.

Use Enhanced NKI to innovate and build optimized kernels on Trainium. Explore the NKI Compiler source code to inspect and contribute to the MLIR-based compilation pipeline. 

.. note::
  The NKI Compiler source code is currently in **Private Preview**, while the NKI programming interface is in **Public Preview**.

**Learn more**: :doc:`NKI home page </nki/index>` and :doc:`NKI Language Guide </nki/get-started/nki-language-guide>`.

NKI Library
^^^^^^^^^^^

**Public Preview**

The NKI Library provides an open source collection of optimized, ready-to-use kernels for common ML operations. The library includes kernels for dense transformer operations, MoE-specific operations, and attention mechanisms, all with complete source code, documentation, and benchmarks.

Use NKI Library kernels directly in your models to improve performance, or explore the implementations as reference for best practices of performance optimizations on Trainium.

**Learn more**: `GitHub repository <https://github.com/aws-neuron/nki-library>`__ and :doc:`API documentation </nki/library/api/index>`.


Neuron Explorer
^^^^^^^^^^^^^^^

**Public Preview**

Neuron Explorer is a tools suite that supports developers and performance engineers in their performance optimization journey. It provides capabilities to inspect and optimize code from framework operations down to hardware instructions with hierarchical profiling, source code linking, IDE integration, and AI-powered recommendations for optimization insights.

Use Neuron Explorer to understand and optimize your model performance on Trainium, from high-level framework operations to low-level hardware execution.

**Learn more**: :doc:`Neuron Explorer documentation </tools/neuron-explorer/index>`.


Kubernetes-Native Resource Management with Neuron DRA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Private Preview**

Neuron Dynamic Resource Allocation (DRA) provides Kubernetes-native resource management for Trainium, eliminating custom scheduler extensions. DRA enables topology-aware scheduling using the default Kubernetes scheduler, atomic UltraServer allocation, and flexible per-workload configuration.

Neuron DRA supports EKS, SageMaker HyperPod, and UltraServer configurations. The driver is open source with container images in AWS ECR public gallery.

Use Neuron DRA to simplify Kubernetes resource management for your Trainium workloads with native scheduling and topology-aware allocation.

**Learn more**: :doc:`Neuron DRA documentation </containers/neuron-dra>`.

**Access**: Contact your AWS account team to participate in the Private Preview.


Resources and Additional Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more information visit the `AWS Trainium official page <https://aws.amazon.com/ai/machine-learning/trainium/>`__, the :doc:`AWS Neuron Documentation </index>`, and :doc:`the AWS Neuron GitHub repositories </about-neuron/oss/index>`.




