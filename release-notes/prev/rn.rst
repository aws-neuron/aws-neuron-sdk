.. _prev-rn:

Previous release notes (Neuron 2.x)
====================================

.. toctree::
   :maxdepth: 1
   :hidden:

   Neuron 2.27.1 </release-notes/prev/2.27.1> 
   Neuron 2.27.0 </release-notes/prev/2.27.0/index>
   Neuron 2.26.1 </release-notes/prev/2.26.1>
   Neuron 2.26.0 </release-notes/prev/2.26.0/index>
   Neuron 2.25.0 </release-notes/prev/2.25.0/index>
   Component release notes </release-notes/components/index>

* **The latest Neuron release is 2.28.0, released on 2/26/2026.** Read the :doc:`2.28.0 release notes </release-notes/2.28.0>` or :doc:`the individual Neuron component release notes </release-notes/components/index>` for more details.
  
.. contents:: Table of contents
   :local:
   :depth: 1

.. grid:: 1 
        :gutter: 2

        .. grid-item-card::
                :link: /release-notes/components/index
                :link-type: doc
                :class-card: sd-border-1
        
                **Neuron component release notes**
                ^^^
                Release notes by component for prior Neuron SDK versions.

Neuron 2.27.0 (12/19/2025)
--------------------------

See :ref:`neuron-2-27-0-whatsnew` for the full Neuron 2.27.0 release notes or :doc:`the individual Neuron component release notes </release-notes/components/index>`.

* Neuron 2.27.1 was released as a patch for 2.27.0 on 1/26/2026. See the :doc:`2.27.0 release notes </release-notes/prev/2.27.1>` for details.

Neuron 2.26.1 (10/29/2025)
--------------------------

See :doc:`2.26.1` for the updated Neuron 2.26.1 release notes or :doc:`the individual Neuron component release notes </release-notes/components/index>`.

Neuron 2.26.0 (09/18/2025)
--------------------------

See :ref:`neuron-2-26-0-whatsnew` for the full Neuron 2.26.0 release notes or :doc:`the individual Neuron component release notes </release-notes/components/index>`.


Neuron 2.25.0 (07/31/2025)
--------------------------

See :ref:`neuron-2-25-0-whatsnew` for the full Neuron 2.25.0 release notes or :doc:`the individual Neuron component release notes </release-notes/components/index>`.

.. _neuron-2-24-1-whatsnew:

Neuron 2.24.1 (06/30/2025)
--------------------------

Neuron version 2.24.1 resolves an installation issue that could prevent NeuronX Distributed Training from being installed successfully.

.. _neuron-2-24-0-whatsnew:

Neuron 2.24.0 (06/24/2025)
--------------------------

Neuron version 2.24 introduces new inference capabilities including prefix caching, disaggregated inference (Beta), and context parallelization support (Beta). This release also includes NKI language enhancements and enhanced profiling visualizations for improved debugging and performance analysis. Neuron 2.24 adds support for PyTorch 2.7 and JAX 0.6, updates existing DLAMIs and DLCs, and introduces a new vLLM inference container.

.. contents:: Table of contents
   :local:
   :depth: 1

What's New
^^^^^^^^^^

NxD Inference (NxDI) includes the following enhancements:

- **Prefix caching**: Improves Time To First Token (TTFT) by up to 3x when processing common shared prompts across requests.
- **Disaggregated inference (Beta)**: Uses 1P1D (1 Prefill, 1 Decode) architecture to reduce prefill-decode interference and improve goodput.
- **Context parallelism (Beta)**: Improves TTFT for longer sequence lengths by processing context encoding in parallel across multiple NeuronCores.
- **Model support**: Added beta support for Qwen 2.5 text models.
- **NxD Inference Library**: Upgraded to support PyTorch 2.7 and Transformers 4.48.

Hugging Face Optimum Neuron 0.2.0 now supports PyTorch-based NxD Core backend for LLM inference, simplifying the implementation of new PyTorch model architectures. Models including Llama 3.1-8B and Llama-3.3-70B have migrated from Transformers NeuronX to the NxD backend.

Training
^^^^^^^^

**Library Upgrades**

- **NxD Training  (NxDT) Library**: Upgraded to support PyTorch 2.7 and Transformers 4.48.
- **JAX Training Support**: Upgraded to JAX 0.6.0.

Neuron Kernel Interface (NKI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **New nki.language.gather_flattened**: Provides efficient parallel tensor element gathering.
- **Enhanced accuracy**: Improved valid range of ``nki.language.sqrt`` and ``nki.isa.activation(nl.sqrt)`` 
- **Advanced indexing**: Improved performance for ``nki.isa.nc_match_replace8``.

Neuron Tools
^^^^^^^^^^^^

**Neuron Profiler Enhancements**

- **Framework stack traces**: Maps device instructions to model source code.
- **Scratchpad memory usage visualization**: Shows tensor-level memory usage over time with HLO name association.
- **On-device collectives barriers**: Identifies synchronization overhead.
- **HBM throughput visualization**: Tracks data movement involving High Bandwidth Memory (HBM) over time.

**NCCOM-TEST Improvements**

- Added ``--report-to-json-file`` flag: Outputs results in JSON format.
- Added ``--show-input-output-size`` flag: Explicitly displays input and output sizes based on operations.

Neuron Deep Learning Containers (DLCs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Updated containers with PyTorch 2.7 support for inference and training.
- Added new inference container with NxD Inference and vLLM with FastAPI.
- JAX DLCs now support JAX 0.6.0 training.

Neuron Deep Learning AMIs (DLAMIs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Updated MultiFramework DLAMIs to include PyTorch 2.7 and JAX 0.6.0.
- Added new Single Framework DLAMIs for PyTorch 2.7 and JAX 0.6.0.

Neuron 2.24 Feature Release Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - NxD Core (neuronx-distributed) 
     - * :ref:`nxd-core_rn`   
     - ``Trn1`` / ``Trn1n``, ``Trn2``

   * - NxD Inference (neuronx-distributed-inference)
     - * :ref:`nxd-inference_rn` 
     - ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

   * - NxD Training (neuronx-distributed-training)
     - * :ref:`nxd-training_rn` 
     - ``Trn1`` / ``Trn1n``, ``Trn2``

   * - PyTorch NeuronX (torch-neuronx)
     - * :ref:`pytorch_rn`
     - ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

   * - Neuron Compiler (neuronx-cc)
     - * :ref:`compiler_rn`
     - ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

   * - Neuron Kernel Interface (NKI)
     - * :ref:`nki_rn`
     - ``Inf2``, ``Trn1``/ ``Trn1n``

   * - Neuron Tools
     - * :ref:`dev-tools_rn`
     - ``Inf1``, ``Inf2``, ``Trn1``/ ``Trn1n``

   * - Neuron Runtime
     - * :ref:`runtime_rn`
     - ``Inf1``, ``Inf2``, ``Trn1``/ ``Trn1n``

   * - Transformers NeuronX (transformers-neuronx) for Inference
     - * :ref:`nxd-inference_rn` 
     - ``Inf2``, ``Trn1`` / ``Trn1n``

   * - Neuron Deep Learning AMIs (DLAMIs)
     - * :ref:`neuron-dlami-overview`
     - ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``

   * - Neuron Deep Learning Containers (DLCs)
     - * :ref:`containers_rn`
     - ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``

   * - Release Announcements
     - * :ref:`announce-no-longer-support-beta-pytorch-neuroncore-placement-apis`
       * :ref:`announce-eos-block-dimension-nki`
       * :ref:`announce-eos-tensorflow-tutorial`
       * :ref:`announce-eos-tnx`
       * :ref:`announce-eos-longer-support-xla-bf16-vars`
       * :ref:`announce-eos-block-dimension-nki`
       * :ref:`announce-no-longer-support-llama-32-meta-checkpoint`
       * :ref:`announce-no-longer-support-nki-jit`
       * See more at :ref:`announcements-main`.
     - ``Inf1``, ``Inf2``, ``Trn1``/ ``Trn1n``

.. _neuron-2.23.0-whatsnew:

Neuron 2.23.0 (05/20/2025)
---------------------------

.. contents:: Table of contents
   :local:
   :depth: 1

What's New
^^^^^^^^^^

With the Neuron 2.23 release, we move NxD Inference (NxDI) library out of beta. It is now recommended for all multi-chip inference use-cases. In addition, Neuron has new training capabilities, including Context Parallelism and ORPO, NKI improvements (new operators and ISA features), and new Neuron Profiler debugging and performance analysis optimizations. Finally, Neuron now supports :ref:`PyTorch 2.6 <introduce-pytorch-2-6>` and JAX 0.5.3.

Inference: NxD Inference (NxDI) moves from beta to GA. NxDI now supports Persistent Cache to reduce compilation times, and optimizes model loading with improved weight sharding performance.

Training: NxD Training (NxDT) added Context Parallelism support (beta) for Llama models, enabling sequence lengths up to 32K. NxDT now supports model alignment, ORPO, using DPO-style datasets. NxDT has upgraded supports for 3rd party libraries, specifically: PyTorch Lightning 2.5, Transformers 4.48, and NeMo 2.1.

Neuron Kernel Interface (NKI): New support for 32-bit integer nki.language.add and nki.language.multiply on GPSIMD Engine. NKI.ISA improvements include range_select for Trainium2, fine-grained engine control, and enhanced tensor operations. New performance tuning API ``no_reorder`` has been added to enable user-scheduling of instructions. When combined with allocation, this enables software pipelining. Language consistency has been improved for arithmetic operators (``+=``, ``-=``, ``/=``, ``*=``) across loop types, PSUM, and SBUF.

Neuron Profiler: Profiling performance has improved, allowing users to view profile results 5x times faster on average. New features include timeline-based error tracking and JSON error event reporting, supporting execution and OOB error detection. Additionally, this release improves multiprocess visualization with Perfetto. 

Neuron Monitoring: Added Kubernetes context information (pod_name, namespace, and container_name) to neuron monitor prometheus output, enabling resource utilization tracking by pod, namespace, and container.

Neuron DLCs: This release updates containers with PyTorch 2.6 support for inference and training. For JAX DLC, this release adds JAX 0.5.0 training support.

Neuron DLAMIs: This release updates MultiFramework AMIs to include PyTorch 2.6, JAX 0.5, and TensorFlow 2.10 and Single Framework AMIs for PyTorch 2.6 and JAX 0.5.

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - NxD Core (neuronx-distributed) 
     - * :ref:`nxd-core_rn`   
     - Trn1/Trn1n,Trn2

   * - NxD Inference (neuronx-distributed-inference)
     - * :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n,Trn2

   * - NxD Training (neuronx-distributed-training)
     - * :ref:`nxd-training_rn` 
     - Trn1/Trn1n,Trn2

   * - PyTorch NeuronX (torch-neuronx)
     - * :ref:`pytorch_rn`
     - Trn1/Trn1n,Inf2,Trn2

   * - Neuron Compiler (neuronx-cc)
     - * :ref:`compiler_rn`
     - Trn1/Trn1n,Inf2,Trn2

   * - Neuron Kernel Interface (NKI)
     - * :ref:`nki_rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Tools
     - * :ref:`dev-tools_rn`
     - Inf1,Inf2,Trn1/Trn1n,Trn2

   * - Neuron Runtime
     - * :ref:`runtime_rn`
     - Inf1,Inf2,Trn1/Trn1n,Trn2

   * - Transformers NeuronX (transformers-neuronx) for Inference
     - * :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n

   * - Neuron Deep Learning AMIs (DLAMIs)
     - * :ref:`neuron-dlami-overview`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Deep Learning Containers (DLCs)
     - * :ref:`containers_rn`
     - Inf1,Inf2,Trn1/Trn1n

   * - Release Announcements
     - * :ref:`announce-eos-block-dimension-nki`
       * :ref:`announce-eos-mllama-checkpoint`
       * :ref:`announce-eos-torch-neuronx-nki-jit`
       * :ref:`announce-eos-xla-bf`
       * :ref:`announce-eos-jax-neuronx-features`
       * :ref:`announce-no-support-nemo-megatron`
       * :ref:`announce-no-support-tensorflow-eos`
       * :ref:`announce-u20-base-no-support`
       * :ref:`announce-tnx-maintenance`
       * :ref:`announce-eol-nxd-examples`
       * See more at :ref:`announcements-main`
     - Inf1, Inf2, Trn1/Trn1n

For detailed release artifiacts, see :ref:`Release Artifacts <latest-neuron-release-artifacts>`.



.. _neuron-2.22.1-whatsnew:

Neuron 2.22.1 (05/12/2025)
---------------------------

Neuron 2.22.1 release includes a Neuron Driver update that resolves DMA abort errors on Trainium2 devices. These errors were previously occurring in the Neuron Runtime during specific workload executions.


.. _neuron-2.22.0-whatsnew:

Neuron 2.22.0 (04/03/2025)
---------------------------

.. contents:: Table of contents
   :local:
   :depth: 1

What's New
^^^^^^^^^^

The Neuron 2.22 release includes performance optimizations, enhancements and new capabilities across the Neuron software stack. 

For inference workloads, the NxD Inference library now supports Llama-3.2-11B model and supports multi-LoRA serving, allowing customers to load and serve multiple LoRA adapters. Flexible quantization features have been added, enabling users to specify which model layers or NxDI modules to quantize. Asynchronous inference mode has also been introduced, improving performance by overlapping Input preparation with model execution.

For training, we added LoRA supervised fine-tuning to NxD Training to enable additional model customization and adaptation.

Neuron Kernel Interface (NKI): This release adds new APIs in nki.isa, nki.language, and nki.profile. These enhancements provide customers with greater flexibility and control.

The updated Neuron Runtime includes optimizations for reduced latency and improved device memory footprint. On the tooling side, the Neuron Profiler 2.0 (beta) has added UI enhancements and new event type support.

Neuron DLCs: this release reduces DLC image size by up to 50% and enables faster build times with updated Dockerfiles structure. On the Neuron DLAMI side, new PyTorch 2.5 single framework DLAMIs have been added for Ubuntu 22.04 and Amazon Linux 2023, along with several new virtual environments within the Neuron Multi Framework DLAMIs.


More release content can be found in the table below and each component release notes.

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - NxD Core (neuronx-distributed) 
     - * :ref:`nxd-core_rn`   
     - Trn1/Trn1n,Trn2

   * - NxD Inference (neuronx-distributed-inference)
     - * :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n,Trn2

   * - NxD Training (neuronx-distributed-training)
     - * :ref:`nxd-training_rn` 
     - Trn1/Trn1n,Trn2

   * - PyTorch NeuronX (torch-neuronx)
     - * :ref:`pytorch_rn`
     - Trn1/Trn1n,Inf2,Trn2

   * - NeuronX Nemo Megatron for Training
     - * `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_  and  :ref:`neuronx-nemo-rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * :ref:`compiler_rn`
     - Trn1/Trn1n,Inf2,Trn2

   * - Neuron Kernel Interface (NKI)
     - * :ref:`nki_rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Tools
     - * :ref:`dev-tools_rn`
     - Inf1,Inf2,Trn1/Trn1n,Trn2

   * - Neuron Runtime
     - * :ref:`runtime_rn`
     - Inf1,Inf2,Trn1/Trn1n,Trn2

   * - Transformers NeuronX (transformers-neuronx) for Inference
     - * :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n

   * - Neuron Deep Learning AMIs (DLAMIs)
     - * :ref:`neuron-dlami-overview`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Deep Learning Containers (DLCs)
     - * :ref:`containers_rn`
     - Inf1,Inf2,Trn1/Trn1n

   * - Release Announcements
     - * :ref:`announce-eos-neuron-det`
       * :ref:`announce-eos-nxd-examples`
       * :ref:`announce-python-eos`
       * :ref:`announce-eos-pytorch-eos-113`
       * :ref:`announce-eos-pytorch-2-1`
       * :ref:`announce-u20-dlami-dlc-eos`
       * :ref:`announce-no-support-torch-neuron`
       * See more at :ref:`announcements-main`
     - Inf1, Inf2, Trn1/Trn1n

For detailed release artifacts, see :ref:`Release Artifacts <latest-neuron-release-artifacts>`.

.. _neuron-2.21.1-whatsnew:

Neuron 2.21.1 (01/14/2025)
---------------------------

Neuron 2.21.1 release pins Transformers NeuronX dependency to transformers<4.48 and fixes DMA abort errors on Trn2.

Additionally, this release addresses NxD Core and Training improvements, including fixes for sequence parallel support in quantized models and a new flag for dtype control in Llama3/3.1 70B configurations. See :ref:`NxD Training Release Notes <nxd-training_rn>` (neuronx-distributed-training) for details.

NxD Inference update includes minor bug fixes for sampling parameters. See :ref:`NxD Inference Release Notes <nxd-inference_rn>`.

Neuron supported DLAMIs and DLCs have been updated to Neuron 2.21.1 SDK. Users should be aware of an incompatibility between Tensorflow-Neuron 2.10 (Inf1) and Neuron Runtime 2.21 in DLAMIs, which will be addressed in the next minor release. See :ref:`Neuron DLAMI Release Notes <dlamis_rn>`.

The Neuron Compiler includes bug fixes and performance enhancements specifically targeting the Trn2 platform.

.. _neuron-2.21.0-whatsnew:

Neuron 2.21.0 (12/20/2024)
---------------------------

.. contents:: Table of contents
   :local:
   :depth: 1

What's New
^^^^^^^^^^

**Overview**: Neuron 2.21.0 introduces support for :ref:`AWS Trainium2 <trainium2-arch>` and
:ref:`Trn2 instances <aws-trn2-arch>`, including the trn2.48xlarge instance type and Trn2
UltraServer (Preview). The release adds new capabilities in both training and
inference of large-scale models. It introduces :ref:`NxD Inference (beta) <introduce-nxd-inference>`, a
PyTorch-based library for deployment, :ref:`Neuron Profiler 2.0 (beta) <neuron-profiler-2-0-guide>`, and
:ref:`PyTorch 2.5 <introduce-pytorch-2-5>` support across the Neuron SDK, and :ref:`Logical NeuronCore
Configuration (LNC) <logical-neuroncore-config>` for optimizing NeuronCore allocation. The release
enables :ref:`Llama 3.1 405B model inference <nxdi-trn2-llama3.1-405b-tutorial>` on a single trn2.48xlarge
instance.

**NxD Inference**: :ref:`NxD Inference (beta) <nxdi-overview>` is a new PyTorch-based inference library for
deploying large-scale models on AWS Inferentia and Trainium instances.
It enables PyTorch model onboarding with minimal code changes and
integrates with :doc:`vLLM </libraries/nxd-inference/developer_guides/vllm-user-guide>`. NxDI supports various model architectures,
including Llama versions for text processing (Llama 2, Llama 3, Llama
3.1, Llama 3.2, and Llama 3.3), and Mixture-of-Experts (MoE) model architectures including
Mixtral and DBRX. The library supports quantization methods, includes
dynamic sampling, and is compatible with HuggingFace checkpoints and
generate() API. NxDI also supports distributed strategies including tensor parallelism and incorporates speculative decoding techniques (Draft model and EAGLE). The
release includes :ref:`Llama 3.1 405B model sample <nxdi-trn2-llama3.1-405b-tutorial>`, :ref:`Llama 3.3 70B model sample <nxdi-trn2-llama3.3-70b-tutorial>` 
and :ref:`Llama 3.1 405B model with speculative decoding <nxdi-trn2-llama3.1-405b-speculative-tutorial>` for inference on a single trn2.48xlarge instance.

For more information, see :ref:`NxD Inference documentation <nxdi-overview>` and check the NxD
Inference Github repository: `aws-neuron/neuronx-distributed-inference <https://github.com/aws-neuron/neuronx-distributed-inference>`_

**Transformers NeuronX (TNx)**: This release introduces several new features, including flash decoding support for speculative decoding, and on-device generation in speculative decoding flows. It adds :ref:`Eagle speculative decoding <cb-eagle-speculative-decoding>` with greedy and lossless sampling, as well as support for :ref:`CPU compilation <transformers_neuronx_readme>` and sharded model saving. Performance improvements include optimized MLP and QKV for Llama models with sequence parallel norm and control over concurrent compilation workers.

**Training Highlights:** NxD Training in this release adds support for
HuggingFace :ref:`Llama3/3.1 70B <hf_llama3_70B_pretraining>` on trn2 instances, introduces :doc:`DPO support </libraries/nxd-training/tutorials/hf_llama3_8B_DPO_ORPO>` for
post-training model alignment, and adds support for Mixture-of-Experts
(MoE) models including Mixtral 7B. The release includes improved
:ref:`checkpoint conversion <checkpoint_conversion>` capabilities and supports MoE with Tensor,
Sequence, Pipeline, and Expert parallelism.

**ML Frameworks:** Neuron 2.21.0 adds support for :ref:`PyTorch 2.5 <introduce-pytorch-2-5>` and 
JAX 0.4.35.

.. note::
  The CVEs
  `CVE-2024-31583 <https://github.com/advisories/GHSA-pg7h-5qx3-wjr3>`__
  and
  `CVE-2024-31580 <https://github.com/advisories/GHSA-5pcm-hx3q-hm94>`__
  affect PyTorch versions 2.1 and earlier. Based on Amazon’s analysis,
  executing models on Trainium and Inferentia is not exposed to either of
  these vulnerabilities. We recommend upgrading to the new version of
  Torch-NeuronX by following the Neuron setup instructions.

**Logical NeuronCore Configuration (LNC)**: This release introduces :ref:`LNC <logical-neuroncore-config>`
for Trainium2 instances, optimizing NeuronCore allocation for ML
applications. LNC offers two configurations: default (LNC=2) combining
two physical cores, and alternative (LNC=1) mapping each physical core
individually. This feature allows users to efficiently manage resources
for large-scale model training and deployment through runtime variables
and compiler flags.

**Neuron Profiler 2.0:** The new :ref:`profiler <neuron-profiler-2-0-guide>` provides system and
device-level profiling, timeline annotations, container integration, and
support for distributed workloads. It includes trace export capabilities
for Perfetto visualization and integration with JAX and PyTorch
profilers, and support for :ref:`Logical NeuronCore
Configuration (LNC) <logical-neuroncore-config>`.

**Neuron Kernel Interface (NKI)**: NKI now supports Trainium2 including
:ref:`Logical NeuronCore Configuration (LNC) <logical-neuroncore-config>`, adds SPMD capabilities for
multi-core operations, and includes new modules and APIs including
support for float8_e5m2 datatype.

**Deep Learning Containers (DLAMIs)**: This release expands support for
JAX 0.4 within the :ref:`Multi Framework DLAMI <neuron-dlami-overview>`. It also introduces NxD Training, NxD Inference, and NxD Core with
:ref:`PyTorch 2.5 <introduce-pytorch-2-5>` support. Additionally, a new Single Framework DLAMI for
TensorFlow 2.10 on Ubuntu 22 is now available.

**Deep Learning Containers (DLCs):** This release introduces new DLCs
for :doc:`JAX 0.4 </setup/jax-neuronx>` training and PyTorch 2.5.1 inference and training. All DLCs
have been updated to Ubuntu 22, and the pytorch-inference-neuronx DLC
now supports both NxD Inference and TNx libraries.

**Documentation**: Documentation updates include architectural details
about Trainium2 and :ref:`NeuronCore-v3 <neuroncores-v3-arch>`, along with specifications and
topology information for the trn2.48xlarge instance type and Trn2
UltraServer.

**Software Maintenance**: This release includes the following  :ref:`announcements <announcements-main>`:

-  Announcing migration of NxD Core examples from NxD Core repository to NxD Inference repository in next release
-  Announcing end of support for Neuron DET tool starting next release
-  PyTorch Neuron versions 1.9 and 1.10 no longer supported
-  Announcing end of support for PyTorch 2.1 for Trn1, Trn2 and Inf2 starting next release 
-  Announcing end of support for PyTorch 1.13 for Trn1 and Inf2 starting next release
-  Announcing end of support for Python 3.8 in future releases
-  Announcing end of support for Ubuntu20 DLCs and DLAMIs

**Amazon Q**: `Use Q Developer <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/amazonq-getstarted.html#amazon-q-dev>`__
as your Neuron Expert for general technical guidance and to jumpstart your NKI kernel development.


More release content can be found in the table below and each component release notes.

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - Known Issues and Limitations
     - * See :ref:`neuron-2.21.0-known-issues`
     - Trn1/Trn1n , Inf2, Inf1

   * - Transformers NeuronX (transformers-neuronx) for Inference
     - * Flash decoding support for speculative decoding
       * Added support for EAGLE speculative decoding with greedy and lossless sampling
       * Enabled on-device generation support in speculative decoding flows
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n, Trn2


   * - NxD Core (neuronx-distributed) 
     - **Training:**

       * Added support for HuggingFace Llama3 70B with Trn2 instances
       * Added DPO support for post-training model alignment
       * See more at :ref:`nxd-core_rn`   
     - Trn1/Trn1n,Trn2

   * - NxD Inference (neuronx-distributed-inference)
     - * Introduced new NxD Inference Library. See :ref:`introduce-nxd-inference`
       * Added Llama3.1 405B Inference Example on Trn2. See :ref:`nxdi-trn2-llama3.1-405b-tutorial`
       * Added support for vLLM integration for NxD Inference. See :ref:`nxdi-vllm-user-guide-v1`
       * Introduced Open Source Github repository for NxD Inference. See `aws-neuron/neuronx-distributed-inference <https://github.com/aws-neuron/neuronx-distributed-inference>`_
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n,Trn2

   * - NxD Training (neuronx-distributed-training)
     - * Added support for HuggingFace Llama3/3.1 70B with Trn2 instances
       * Added support for Mixtral 8x7B Megatron and HuggingFace models
       * Added support for custom pipeline parallel cuts in HuggingFace Llama3
       * Added support for DPO post-training model alignment
       * See more at :ref:`nxd-training_rn` 
     - Trn1/Trn1n,Trn2

   * - PyTorch NeuronX (torch-neuronx)
     - * Introduced PyTorch 2.5 support 
       * See more at :ref:`pytorch_rn`
     - Trn1/Trn1n,Inf2,Trn2

   * - NeuronX Nemo Megatron for Training
     - * Added support for HuggingFace to NeMo checkpoint conversion when virtual pipeline parallel is enabled.
       * Added collective compute coalescing for ZeRO-1 optimizer
       * See more at `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_  and  :ref:`neuronx-nemo-rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * Minor bug fixes and performance enhancements for the Trn2 platform.
       * See more at :ref:`compiler_rn`
     - Trn1/Trn1n,Inf2,Trn2
  
   * - Neuron Kernel Interface (NKI)
     - * Added ``nki.compiler`` module with Allocation Control and Kernel decorators
       * Added new nki.isa APIs. 
       * Added new nki.language APIs. 
       * Added new kernels (``allocated_fused_self_attn_for_SD_small_head_size``, ``allocated_fused_rms_norm_qkv``).
       * See more at :ref:`nki_rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Deep Learning AMIs (DLAMIs)
     - * Added support for Trainium2 chips within the Neuron Multi Framework DLAMI.
       * Added support for JAX 0.4 to Neuron Multi Framework DLAMI.
       * Added NxD Training (NxDT), NxD Inference (NxDI) and NxD Core PyTorch 2.5 support within the Neuron Multi Framework DLAMI.
       * See more at :ref:`neuron-dlami-overview`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Deep Learning Containers (DLCs)
     - * Added new pytorch-inference-neuronx 2.5.1 and pytorch-training-neuronx 2.5.1 DLCs
       * Added new jax-training-neuronx 0.4 Training DLC
       * See more at :ref:`containers_rn`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Tools
     - * Introduced Neuron Profiler 2.0. See :ref:`neuron-profiler-2-0-guide`
       * See more at :ref:`dev-tools_rn`
     - Inf1,Inf2,Trn1/Trn1n,Trn2

   * - Neuron Runtime
     - * Added runtime support to fail in case of out-of-bound memory access when DGE is enabled.
       * Added support for 4-rank replica group on adjacent Neuron cores on TRN1/TRN1N
       * See more at :ref:`runtime_rn`
     - Inf1,Inf2,Trn1/Trn1n,Trn2

   * - Release Annoucements
     - * :ref:`announce-eos-neuron-det`
       * :ref:`announce-eos-nxd-examples`
       * :ref:`announce-python-eos`
       * :ref:`announce-eos-pytorch-eos-113`
       * :ref:`announce-eos-pytorch-2-1`
       * :ref:`announce-u20-dlami-dlc-eos`
       * :ref:`announce-no-support-torch-neuron`
       * See more at :ref:`announcements-main`
     - Inf1, Inf2, Trn1/Trn1n

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1, Trn2

.. _neuron-2.21.0-known-issues:

2.21.0 Known Issues and Limitations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* See component release notes below for any additional known issues.


.. _neuron-2.21.0.beta-whatsnew:

Neuron 2.21.0 Beta (12/03/2024)
--------------------------------

.. note::
  This release (Neuron 2.21 Beta) was only tested with Trn2 instances. The next release (Neuron 2.21) will support all instances (Inf1, Inf2, Trn1, and Trn2).

  For access to this release (Neuron 2.21 Beta), please contact your account manager.

This release (Neuron 2.21 beta) introduces support for :ref:`AWS Trainium2 <trainium2-arch>` and :ref:`Trn2 instances <aws-trn2-arch>`, including the trn2.48xlarge instance type and Trn2 UltraServer. The release showcases Llama 3.1 405B model inference using NxD Inference on a single trn2.48xlarge instance, and FUJI 70B model training using the AXLearn library across eight trn2.48xlarge instances.

:ref:`NxD Inference <nxdi-index>`, a new PyTorch-based library for deploying large language models and multi-modality models, is introduced in this release. It integrates with vLLM and enables PyTorch model onboarding with minimal code changes. The release also adds support for `AXLearn <https://github.com/apple/axlearn>`_ training for JAX models.

The new :ref:`Neuron Profiler 2.0 <neuron-profiler-2-0-guide>` introduced in this release offers system and device-level profiling, timeline annotations, and container integration. The profiler supports distributed workloads and provides trace export capabilities for Perfetto visualization.

The documentation has been updated to include architectural details about :ref:`Trainium2 <trainium2-arch>` and :ref:`NeuronCore-v3 <neuroncores-v3-arch>`, along with specifications and topology information for the trn2.48xlarge instance type and Trn2 UltraServer.

:ref:`Use Q Developer <amazon-q-dev>` as your Neuron Expert for general technical guidance and to jumpstart your NKI kernel development.

.. note::
  For the latest release that supports Trn1, Inf2 and Inf1 instances, please see :ref:`Neuron Release 2.20.2 <neuron-2-20-2-whatsnew>`



.. _neuron-2-20-2-whatsnew:

Neuron 2.20.2 (11/20/2024)
---------------------------

Neuron 2.20.2 release fixes a stability issue in Neuron Scheduler Extension that previously caused crashes in Kubernetes (K8) deployments. See :ref:`containers_rn`.

This release also addresses a security patch update to Neuron Driver that fixes a kernel address leak issue. 
See more on :ref:`runtime_rn` and :ref:`runtime_rn`.

Addtionally, Neuron 2.20.2 release updates ``torch-neuronx`` and ``libneuronxla`` packages to add support for ``torch-xla`` 2.1.5 package 
which fixes checkpoint loading issues with Zero Redundancy Optimizer (ZeRO-1). See :ref:`pytorch_rn` and :ref:`libneuronxla-rn`.

Neuron supported DLAMIs and DLCs are updated with this release (Neuron 2.20.2 SDK). The Training DLC is also updated to address the 
version dependency issues in NxD Training library. See :ref:`containers_rn`.

NxD Training library in Neuron 2.20.2 release is updated to transformers 4.36.0 package. See :ref:`nxd-training_rn`.


Neuron 2.20.1 (10/25/2024)
---------------------------

Neuron 2.20.1 release addresses an issue with the Neuron Persistent Cache that was brought forth in 2.20 release. In the 2.20 release, the Neuron persistent cache issue resulted in a cache-miss scenario when attempting to load a previously compiled Neuron Executable File Format (NEFF) from a different path or Python environment than the one used for the initial Neuron SDK installation and NEFF compilation. This release resolves the cache-miss problem, ensuring that NEFFs can be loaded correctly regardless of the path or Python environment used to install the Neuron SDK, as long as they were compiled using the same Neuron SDK version.

This release also addresses the excessive lock wait time issue during neuron_parallel_compile graph extraction for large cluster training. See :ref:`pytorch_rn` and :ref:`libneuronxla-rn`.

Additionally, Neuron 2.20.1 introduces new Multi Framework DLAMI for Amazon Linux 2023 (AL2023) that customers can use to easily get started with latest Neuron SDK on multiple frameworks that Neuron supports. See :ref:`dlamis_rn`.

Neuron 2.20.1 Training DLC is also updated to pre-install the necessary dependencies and support NxD Training library out of the box. See :ref:`containers_rn`

.. _neuron-2.20-whatsnew:

Neuron 2.20.0 (09/16/2024)
---------------------------
.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

**Overview**: Neuron 2.20 release introduces usability improvements and new capabilities across training and inference workloads. A key highlight is the introduction of :ref:`Neuron Kernel Interface (beta) <neuron-nki>`. NKI, pronounced 'Nicky', is enabling developers to build optimized custom compute kernels for Trainium and Inferentia. Additionally, this release introduces :ref:`NxD Training (beta) <nxdt>`, a PyTorch-based library enabling efficient distributed training, with a user-friendly interface compatible with NeMo. This release also introduces the support for the :ref:`JAX framework (beta) <jax-neuron-main>`.

Neuron 2.20 also adds inference support for Pixart-alpha and Pixart-sigma Diffusion-Transformers (DiT) models, and adds support for Llama 3.1 8B, 70B and 405B models inference supporting up to 128K context length.

**Neuron Kernel Interface**: NKI is a programming interface enabling developers to build optimized compute custom kernels on top of Trainium and Inferentia. NKI empowers developers to enhance deep learning models with new capabilities, performance optimizations, and scientific innovation. It natively integrates with PyTorch and JAX, providing a Python-based programming environment with Triton-like syntax and tile-level semantics, offering a familiar programming experience for developers. 
All of our NKI work is shared as open source, enabling the community developers to collaborate and use these kernels in their projects, improve existing kernels, and contribute new NKI kernels. The list of kernels we are introducing includes Optimized Flash Attention NKI kernel (``flash_attention``), a NKI kernel with an optimized implementation of Mamba model architecture (``mamba_nki_kernels``) and Optimized Stable Diffusion Attention kernel (``fused_sd_attention_small_head``). In addition to NKI kernel samples for ``average_pool2d``, ``rmsnorm``, ``tensor_addition``, ``layernorm``, ``transpose_2d``, and ``matrix_multiplication``.

For more information see :ref:`NKI section <neuron-nki>` and check the NKI samples Github repository: https://github.com/aws-neuron/nki-samples

**NxD Training (NxDT)**: NxDT is a PyTorch-based library that adds support for user-friendly distributed training experience through a YAML configuration file compatible with NeMo,, allowing users to easily set up their training workflows. At the same time, NxDT maintains flexibility, enabling users to choose between using the YAML configuration file, PyTorch Lightning Trainer, or writing their own custom training script using the NxD Core.
The library supports PyTorch model classes including Hugging Face and Megatron-LM. Additionally, it leverages NeMo's data engineering and data science modules enabling end-to-end training workflows on NxDT, and providing compatability with NeMo through minimal changes to the YAML configuration file for models that are already supported in NxDT. Furthermore, the functionality of the Neuron NeMo Megatron (NNM) library is now part of NxDT, ensuring a smooth migration path from NNM to NxDT.

For more information see :ref:`NxD Training (beta) <nxdt>` and check the NxD Training Github repository: https://github.com/aws-neuron/neuronx-distributed-training 

**Training Highlights**: This release adds support for Llama 3.1 8B and 70B model training up to 32K sequence length (beta). It also adds support for torch.autocast() for native PyTorch mixed precision support and PEFT LoRA model training.

**Inference Highlights**: Neuron 2.20 adds support for Llama 3.1 models (405b, 70b, and 8b variants) and introduces new features like on-device top-p sampling for improved performance, support for up to 128K context length through Flash Decoding, and multi-node inference for large models like Llama-3.1-405B.
Furthermore, this release improves model loading in Transformers Neuronx for models like Llama-3 by loading the pre-sharded or pre-transformed weights and adds support to Diffusion-Transformers (DiT) models such as Pixart-alpha and Pixart-sigma.

**Compiler**: This release introduces Neuron Compiler support for RMSNorm and RMSNormDx operators, along with enhanced performance for the sort operator. 

**System Tools**: As for the Neuron Tools, it enables NKI profiling support in the Neuron Profiler and introduces improvements to the Neuron Profiler UI.

**Neuron Driver**: This release adds support for the Rocky Linux 9.0 operating system. 

**Neuron Containers**: This release introduces Neuron Helm Chart, which helps streamline the deployment of AWS Neuron components on Amazon EKS. See Neuron Helm Chart Github repository: https://github.com/aws-neuron/neuron-helm-charts. 
Additionaly, this release adds ECS support for the "Neuron Node Problem Detector and Recovery" artifact. See :ref:`ecs-neuron-problem-detector-and-recovery`.

**Neuron DLAMIs and DLCs**: This release includes the addition of the NxDT package to various Neuron DLAMIs (Multi-Framework Neuron DLAMI, PyTorch 1.13 Neuron DLAMI, and PyTorch 2.1 Neuron DLAMI) and the inclusion of NxDT in the PyTorch 1.13 Training Neuron DLC and PyTorch 2.1 Training Neuron DLC.

**Software Maintenance Policy**: This release also updates Neuron SDK software maintenance poclicy, For more information see :ref:`sdk-maintenance-policy`


More release content can be found in the table below and each component release notes.

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - Known Issues and Limitations
     - * See :ref:`neuron-2.20.0-known-issues`
     - Trn1/Trn1n , Inf2, Inf1

   * - Transformers NeuronX (transformers-neuronx) for Inference
     - * Support for on-device sampling (Top P) and dynamic sampling (per request parameters) with Continuous batching. See :ref:`developer guide <transformers_neuronx_readme>`
       * Support for Flash Decoding to enable inference for higher sequence lengths of upto 128K. See :ref:`developer guide <transformers_neuronx_readme>`.
       * Support for multi-node inference for large models like ``Llama-3.1-405B``. See :ref:`developer guide <transformers_neuronx_readme>`.
       * Support for bucketing, multi-node inference , on-device sampling and other improvements in Neuron vLLM integration. See :ref:`developer guide <transformers_neuronx_readme>` 
       * Support for Llama 3.1 models (405B, 70B, and 8B variants). See samples for `Llama-3.1-405B <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-405b-multinode-16k-sampling.ipynb>`_ , `Llama-3.1-70B <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-70b-64k-sampling.ipynb>`_  and  `Llama-3.1-8B <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-8b-128k-sampling.ipynb>`_
       * Support for improved model loading for models like Llama-3 by loading the pre-sharded or pre-transformed weights. See :ref:`serialization support in developer guide <transformers_neuronx_readme>`. 
       * Support for ROPE scaling for Llama 3 and Llama 3.1 models. 
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n


   * - NxD Core (neuronx-distributed) 
     - **Training:**

       * Support for LoRA finetuning
       * Support for Mixed precision enhancements

       **Inference:**
       
       * Suppport for DBRX and Mixtral inference samples. See  samples for `DBRX <https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/dbrx>`_ and `Mixtral <https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/mixtral>`_
       * Support for sequence length autobucketing to improve inference performance.
       * Support for improved tracing in the inference samples.
       * See more at :ref:`nxd-core_rn`   
     - Trn1/Trn1n


   * - NxD Training (neuronx-distributed-training)
     - * First release of NxD Training (beta)
       * See more at :ref:`nxd-training_rn` 
     - Trn1/Trn1n


   * - PyTorch NeuronX (torch-neuronx)
     - * Support for inference of Diffusion-Transformers (DiT) models such as ``Pixart-alpha`` and ``Pixart-sigma``. See samples for `Pixart-alpha <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_pixart_alpha_inference_on_inf2.ipynb>`_ and `Pixart-sigma <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_pixart_sigma_inference_on_inf2.ipynb>`_.
       * Support for inference of ``wav2vec2-conformer`` models.  See samples for inference of ``wav2vec2-conformer`` with `relative position embeddings <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_wav2vec2_conformer_relpos_inference_on_inf2.ipynb>`_ and `rotary position embeddings <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_wav2vec2_conformer_rope_inference_on_inf2.ipynb>`_
       * See more at :ref:`pytorch_rn`
     - Trn1/Trn1n,Inf2

   * - NeuronX Nemo Megatron for Training
     - * Fixed issue with linear warmup with cosine annealing
       * Fixed indexing issues with MPI job checkpoint conversion.
       * Fixed pipeline parallel bug for NeMo to HF checkpoint conversion       
       * See more at `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_  and  :ref:`neuronx-nemo-rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * Memory optimization that will reduce the generated compiler artifacts size (i.e., NEFFs)
       * See more at :ref:`compiler_rn`
     - Trn1/Trn1n,Inf2
  
   * - Neuron Kernel Interface (NKI)
     - * First Release on Neuron Kernel Interface (NKI)
       * See more at :ref:`nki_rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Deep Learning AMIs (DLAMIs)
     - * Support for ``neuronx-distributed-training`` library in PyTorch Neuron DLAMI virtual enviornments. See :ref:`neuron-dlami-overview`
       * Updated existing Neuron supported DLAMIs with Neuron 2.20 SDK release.
       * See more at :ref:`Neuron DLAMI Release Notes <neuron-dlami-overview>`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Deep Learning Containers (DLCs)
     - * Updated existing PyTorch Neuron DLCs with Neuron 2.20 SDK release.
       * Support for ``neuronx-distributed-training`` library in `pytorch-training-neuronx DLCs <https://github.com/aws-neuron/deep-learning-containers/tree/main?tab=readme-ov-file#pytorch-training-neuronx>`_. 
       * See more at :ref:`containers_rn`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Tools
     - * Improvements in Neuron Profile
       * See more at :ref:`dev-tools_rn`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Runtime
     - * Introduced a sysfs memory usage counter for DMA rings (:ref:`reference <neuron-sysfs-ug>`)
       * See more at :ref:`runtime_rn`
     - Inf1,Inf2,Trn1/Trn1n

   * - Release Annoucements
     - * :ref:`announce-component-name-change-nxdcore`
       * :ref:`eos-neurondevice`
       * :ref:`eos-neuron-device-version`
       * :ref:`announce-tfx-no-support`
       * :ref:`announce-torch-neuron-eos`
       * :ref:`eos-al2`
       * See more at :ref:`announcements-main`
     - Inf1, Inf2, Trn1/Trn1n

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

.. _neuron-2.20.0-known-issues:

2.20.0 Known Issues and Limitations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Known issues when using ``on_device_generation`` flag in Transformers NeuronX config for Llama models. Customers are advised not to use the flag when they see an issue. See more at :ref:`nxd-inference_rn`  
* See component release notes below for any additional known issues.

Neuron Components Release Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inf1, Trn1/Trn1n and Inf2 common packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size


   * - Component
     - Instance/s
     - Package/s
     - Details


   * - Neuron Runtime
     - Trn1/Trn1n, Inf1, Inf2
     - * Trn1/Trn1n: ``aws-neuronx-runtime-lib`` (.deb, .rpm)

       * Inf1: Runtime is linked into the ML frameworks packages
       
     - * :ref:`runtime_rn`

   * - Neuron Runtime Driver
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-dkms``  (.deb, .rpm)

     - * :ref:`runtime_rn`

   * - Neuron System Tools
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-tools``  (.deb, .rpm)
     - * :ref:`dev-tools_rn`



   * - Containers
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-k8-plugin`` (.deb, .rpm)

       * ``aws-neuronx-k8-scheduler`` (.deb, .rpm)
       
       * ``aws-neuronx-oci-hooks`` (.deb, .rpm)

     - * :ref:`containers_rn`

       * :ref:`containers_rn`

   * - NeuronPerf (Inference only)
     - Trn1/Trn1n, Inf1, Inf2
     - * ``neuronperf`` (.whl)
     - * :ref:`dev-tools_rn`

   * - TensorFlow Model Server Neuron
     - Trn1/Trn1n, Inf1, Inf2
     - * ``tensorflow-model-server-neuronx`` (.deb, .rpm)
     - * :ref:`tensorflow-modeslserver-neuronx-rn`



Trn1/Trn1n and Inf2 only packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   
   * - Component
     - Instance/s
     - Package/s
     - Details


   * - PyTorch Neuron
     - Trn1/Trn1n, Inf2
     - * ``torch-neuronx`` (.whl)
     - * :ref:`pytorch_rn`
       * :ref:`pytorch-neuron-supported-operators`
       

   * - TensorFlow Neuron
     - Trn1/Trn1n, Inf2
     - * ``tensorflow-neuronx`` (.whl)
     - * :ref:`tensorflow-neuronx-release-notes`

 
   * - Neuron Compiler (Trn1/Trn1n, Inf2 only)
     - Trn1/Trn1n, Inf2
     - * ``neuronx-cc`` (.whl)
     - * :ref:`compiler_rn`


   * - Neuron Kernel Interface (NKI) Compiler (Trn1/Trn1n, Inf2 only)
     - Trn1/Trn1n, Inf2
     - * Supported within ``neuronx-cc`` (.whl)
     - * :ref:`nki_rn`

   * - Collective Communication library
     - Trn1/Trn1n, Inf2    
     - * ``aws-neuronx-collective`` (.deb, .rpm)
     - * :ref:`runtime_rn`


   * - Neuron Custom C++ Operators
     - Trn1/Trn1n, Inf2
  
     - * ``aws-neuronx-gpsimd-customop`` (.deb, .rpm)
  
       * ``aws-neuronx-gpsimd-tools`` (.deb, .rpm)
  
     - * :ref:`gpsimd-customop-lib-rn`

       * :ref:`gpsimd-customop-tools-rn`


   * - Transformers Neuron
     - Trn1/Trn1n, Inf2
     - * ``transformers-neuronx`` (.whl)
     - * :ref:`nxd-inference_rn`

   * - NxD Training
     - Trn1/Trn1n, Inf2
     - * ``neuronx-distributed-training`` (.whl)
     - * :ref:`nxd-training_rn`


   * - NxD Core
     - Trn1/Trn1n, Inf2
     - * ``neuronx-distributed`` (.whl)
     - * :ref:`nxd-core_rn`

   * - AWS Neuron Reference for NeMo Megatron
     - Trn1/Trn1n
     - * `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - * :ref:`neuronx-nemo-rn`




Inf1 only packages
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   

   * - Component
     - Instance/s
     - Package/s
     - Details


   * - PyTorch Neuron
     - Inf1
     - * ``torch-neuron`` (.whl)
     - * :ref:`pytorch-neuron-rn`

       * :ref:`neuron-cc-ops-pytorch`


   * - TensorFlow Neuron
     - Inf1
     - * ``tensorflow-neuron`` (.whl)
     - * :ref:`tensorflow-neuron-rn`

       * :ref:`neuron-cc-ops-tensorflow`
       
       * :ref:`tensorflow-neuron-rn-v2` 



   * - Apache MXNet
     - Inf1
     - * ``mx_neuron`` (.whl)
     - * :ref:`mxnet-neuron-rn`

       * :ref:`neuron-cc-ops-mxnet`


   * - Neuron Compiler (Inf1 only)
     - Inf1
     - * ``neuron-cc`` (.whl)
     - * :ref:`neuron-cc-rn`

       * :ref:`neuron-supported-operators`

.. _neuron-2.19.0-whatsnew:

Neuron 2.19.1 (07/19/2024)
---------------------------

This release (Neuron 2.19.1) addresses an issue with the Neuron Persistent Cache that was introduced in the previous release, Neuron 2.19. The issue resulted in a cache-miss scenario when attempting to load a previously compiled Neuron Executable File Format (NEFF) from a different path or Python environment than the one used for the initial Neuron SDK installation and NEFF compilation. This release resolves the cache-miss problem, ensuring that NEFFs can be loaded correctly regardless of the path or Python environment used to install the Neuron SDK, as long as they were compiled using the same Neuron SDK version.



Neuron 2.19.0 (07/03/2024)
---------------------------
.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

Neuron 2.19 release adds Llama 3 training support and introduces Flash Attention kernel support to enable LLM training and inference for
large sequence lengths. Neuron 2.19 also introduces new features and performance
improvements to LLM training, improves LLM inference performance for Llama 3 model by upto 20%, and adds tools for monitoring, problem detection and recovery in Kubernetes (EKS) environments, improving efficiency and reliability.

**Training highlights**: LLM model training user experience using
NeuronX Distributed (NxD) is improved by support for Flash Attention to
enable training with longer sequence lengths >= 8K. Neuron 2.19 adds support for Llama 3 model training. This release also
adds support for Interleaved pipeline parallelism to reduce idle time
(bubble size) and enhance training efficiency and resource utilization for large cluster sizes.

**Inference highlights**: Flash Attention kernel support in the Transformers NeuronX library enables LLM inference for context lengths of up to 32k. This release also adds [Beta] support for continuous batching with ``mistralai/Mistral-7B-v0.2`` in Transformers NeuronX.

**Tools and Neuron DLAMI/DLC highlights**: This release introduces the new Neuron Node
Problem Detector and Recovery plugin in EKS supported Kubernetes
environments:a tool to monitor the health of Neuron instances and
triggers automatic node replacement upon detecting an unrecoverable
error. Neuron 2.19 introduces the new Neuron Monitor container to
enable easy monitoring of Neuron metrics in Kubernetes, and adds monitoring support with Prometheus and Grafana.
This release also introduces new PyTorch 2.1 and PyTorch 1.13 single framework DLAMIs for Ubuntu 22. Neuron DLAMIs and Neuron DLCs are also updated to support this release (Neuron 2.19).

More release content can be found in the table below and each component release notes.

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - Known Issues and Limitations
     - * See :ref:`neuron-2.19.0-known-issues`
     - Trn1/Trn1n , Inf2, Inf1

   * - Transformers NeuronX (transformers-neuronx) for Inference
     - * Support for Flash Attention kernel in Llama models to enable inference for higher sequence lengths. See :ref:`developer guide <transformers_neuronx_readme>`.
       * Support for running Top-K sampling on Neuron device for generation in Mixtral models. See ``Mixtral-8x7b`` `sample <https://github.com/aws-neuron/transformers-neuronx/blob/main/src/transformers_neuronx/mixtral/model.py>`__.
       * [Beta] Support for Continuous batching with ``mistralai/Mistral-7B-Instruct-v0.2`` model inference. See :ref:`developer guide <transformers_neuronx_readme>`.
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Training
     - * Support for Interleaved pipeline parallelism to reduce idle time (bubble size) and enhance training efficiency and resource utilization for large cluster sizes. See :ref:`api guide <api_guide>` , :ref:`developer guide <pp_developer_guide>`
       * Support for Flash Attention kernel to enable training with longer sequence lengths.
       * See more at :ref:`nxd-core_rn` 
     - Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Inference
     - * Support for Flash Attention kernel for longer sequence length inference. See :pytorch-neuron-src:`[CodeLlama-13b Inference with 16k sequence length] <neuronx_distributed/llama/codellama_16k_inference.ipynb>`
       * [Beta] Support for speculative decoding. See :ref:`developer guide <neuronx_distributed_inference_developer_guide>`.
       * See more at :ref:`nxd-core_rn` 
     - Inf2,Trn1/Trn1n

   * - PyTorch NeuronX (torch-neuronx)
     - * Support for FP32 master weights and BF16 all-gather during Zero1 training to enhance training efficiency.
       * Support to add custom SILU activation functions by configuring NEURON_CUSTOM_SILU variable
       * See more at :ref:`pytorch_rn`
     - Trn1/Trn1n,Inf2

   * - NeuronX Nemo Megatron for Training
     - * Support for FP32 gradient accumulation enhancing accuracy for large model training.
       * Support for Zero1 training with master weights
       * Support for Flash Attention kernel to train with longer sequence lengths (greater than 8K)
       * See more at `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_  and  :ref:`neuronx-nemo-rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * Support for Flash Attention kernel to enable usage of long sequence lengths during training and inference.
       * See more at :ref:`compiler_rn`
     - Trn1/Trn1n,Inf2

   * - Neuron DLAMI and DLC
     - * Neuron DLAMIs are updated with latest 2.19 Neuron SDK. See :ref:`neuron-dlami-overview`
       * New Neuron Single Framework DLAMIs with PyTorch-2.1 and PyTorch-1.13 for Ubuntu 22. See :ref:`neuron-dlami-overview`
       * New Base Deep Learning AMI (DLAMI) for Ubuntu 22. See :ref:`neuron-dlami-overview`
       * PyTorch 1.13 and PyTorch 2.1 Inference and Training DLCs are updated with latest 2.19 Neuron SDK. See :ref:`neuron_containers`
       * PyTorch 1.13 Inference and PyTorch 2.1 Inference DLCs are updated with TorchServe v0.11.0. See :ref:`neuron_containers`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Tools
     - * Support for new Neuron Node Problem Detector and Recovery plugin in EKS supported kubernetes environments that monitors health of Neuron instances and triggers automatic node replacement upon detecting an unrecoverable error. See :doc:`configuration </containers/tutorials/k8s-neuron-problem-detector-and-recovery-irsa>` and :ref:`tutorial <k8s-neuron-problem-detector-and-recovery>`.
       * Support for new Neuron Monitor container to enable easy monitoring of Neuron metrics in Kubernetes. Supports monitoring with Prometheus and Grafana. See :ref:`tutorial <k8s-neuron-monitor>`
       * Support for Neuron scheduler extension to enforce allocation of contiguous Neuron Devices for the pods based on the Neuron instance type. See :ref:`tutorial <neuron_scheduler>`
       * Neuron Profiler bugfixes and UI updates, including improvements to visualizing collective operations and to the consistency of information being displayed
       * Added memory usage metrics and device count information to neuron-monitor 
       * See more at :ref:`dev-tools_rn`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Runtime
     - * Support for dynamic Direct Memory Access (DMA) that reduces memory usage during runtime.
       * Runtime Enhancements that improve collectives performance
       * See more at :ref:`runtime_rn`
     - Inf1,Inf2,Trn1/Trn1n
  
   * - Other Documentation Updates
     - * Announced maintenance mode of MxNet. See :ref:`announce-mxnet-maintenance`
       * Announced End of support of Neuron TensorFlow 1.x (Inf1). See :ref:`announce-tfx-eos`
       * Announce End of support for AL2. See :ref:`announce-eos-al2`
       * --
     - Inf1, Inf2, Trn1/Trn1n

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

.. _neuron-2.19.0-known-issues:

2.19.0 Known Issues and Limitations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Known issues when using ``on_device_generation`` flag in Transformers NeuronX config for Llama models. Customers are advised not to use the flag when they see an issue. See more at :ref:`nxd-inference_rn`  
* See component release notes below for any additional known issues.


Neuron Components Release Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inf1, Trn1/Trn1n and Inf2 common packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size


   * - Component
     - Instance/s
     - Package/s
     - Details


   * - Neuron Runtime
     - Trn1/Trn1n, Inf1, Inf2
     - * Trn1/Trn1n: ``aws-neuronx-runtime-lib`` (.deb, .rpm)

       * Inf1: Runtime is linked into the ML frameworks packages
       
     - * :ref:`runtime_rn`

   * - Neuron Runtime Driver
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-dkms``  (.deb, .rpm)

     - * :ref:`runtime_rn`

   * - Neuron System Tools
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-tools``  (.deb, .rpm)
     - * :ref:`dev-tools_rn`

   * - Neuron DLAMI
     - Trn1/Trn1n, Inf1, Inf2
     - * 
     - * `Neuron DLAMI Release Notes <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/dlami/index.html>`_.

   * - Neuron DLC
     - Trn1/Trn1n, Inf1, Inf2
     - *
     - * :ref:`containers_rn`

   * - Containers
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-k8-plugin`` (.deb, .rpm)

       * ``aws-neuronx-k8-scheduler`` (.deb, .rpm)
       
       * ``aws-neuronx-oci-hooks`` (.deb, .rpm)

     - * :ref:`containers_rn`

       * :ref:`containers_rn`

   * - NeuronPerf (Inference only)
     - Trn1/Trn1n, Inf1, Inf2
     - * ``neuronperf`` (.whl)
     - * :ref:`dev-tools_rn`

   * - TensorFlow Model Server Neuron
     - Trn1/Trn1n, Inf1, Inf2
     - * ``tensorflow-model-server-neuronx`` (.deb, .rpm)
     - * :ref:`tensorflow-modeslserver-neuronx-rn`

Trn1/Trn1n and Inf2 only packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   
   * - Component
     - Instance/s
     - Package/s
     - Details


   * - PyTorch Neuron
     - Trn1/Trn1n, Inf2
     - * ``torch-neuronx`` (.whl)
     - * :ref:`pytorch_rn`
       * :ref:`pytorch-neuron-supported-operators`
       

   * - TensorFlow Neuron
     - Trn1/Trn1n, Inf2
     - * ``tensorflow-neuronx`` (.whl)
     - * :ref:`tensorflow-neuronx-release-notes`

 
   * - Neuron Compiler (Trn1/Trn1n, Inf2 only)
     - Trn1/Trn1n, Inf2
     - * ``neuronx-cc`` (.whl)
     - * :ref:`compiler_rn`

   * - Collective Communication library
     - Trn1/Trn1n, Inf2    
     - * ``aws-neuronx-collective`` (.deb, .rpm)
     - * :ref:`runtime_rn`


   * - Neuron Custom C++ Operators
     - Trn1/Trn1n, Inf2
  
     - * ``aws-neuronx-gpsimd-customop`` (.deb, .rpm)
  
       * ``aws-neuronx-gpsimd-tools`` (.deb, .rpm)
  
     - * :ref:`gpsimd-customop-lib-rn`

       * :ref:`gpsimd-customop-tools-rn`


   * - Transformers Neuron
     - Trn1/Trn1n, Inf2
     - * ``transformers-neuronx`` (.whl)
     - * :ref:`nxd-inference_rn`

   * - Neuron Distributed
     - Trn1/Trn1n, Inf2
     - * ``neuronx-distributed`` (.whl)
     - * :ref:`nxd-core_rn`

   * - AWS Neuron Reference for NeMo Megatron
     - Trn1/Trn1n
     - * `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - * :ref:`neuronx-nemo-rn`



.. note::

   In next releases ``aws-neuronx-tools`` and ``aws-neuronx-runtime-lib`` will add support for Inf1.


Inf1 only packages
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   

   * - Component
     - Instance/s
     - Package/s
     - Details


   * - PyTorch Neuron
     - Inf1
     - * ``torch-neuron`` (.whl)
     - * :ref:`pytorch-neuron-rn`

       * :ref:`neuron-cc-ops-pytorch`


   * - TensorFlow Neuron
     - Inf1
     - * ``tensorflow-neuron`` (.whl)
     - * :ref:`tensorflow-neuron-rn`

       * :ref:`neuron-cc-ops-tensorflow`
       
       * :ref:`tensorflow-neuron-rn-v2` 



   * - Apache MXNet
     - Inf1
     - * ``mx_neuron`` (.whl)
     - * :ref:`mxnet-neuron-rn`

       * :ref:`neuron-cc-ops-mxnet`


   * - Neuron Compiler (Inf1 only)
     - Inf1
     - * ``neuron-cc`` (.whl)
     - * :ref:`neuron-cc-rn`

       * :ref:`neuron-supported-operators`


.. _neuron-2.18.0-whatsnew:


Neuron 2.18.2 (04/25/2024)
--------------------------
Patch release with minor Neuron Compiler bug fixes and enhancements. See more in  :ref:`compiler_rn`



Neuron 2.18.1 (04/10/2024)
--------------------------

Neuron 2.18.1 release introduces :ref:`Continuous batching(beta) <transformers_neuronx_readme>` and Neuron vLLM integration(beta) support in Transformers NeuronX library that improves LLM inference throughput. This release also fixes hang issues related to Triton Inference Server as well as updating Neuron DLAMIs and DLCs with this release(2.18.1). 
See more in  :ref:`nxd-inference_rn` and :ref:`compiler_rn` 



Neuron 2.18.0 (04/01/2024)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

Neuron 2.18 release introduces stable support (out of beta) for PyTorch 2.1, introduces new features and performance improvements to LLM training and inference, and updates Neuron DLAMIs and Neuron DLCs to support this release (Neuron 2.18).

**Training highlights**: LLM model training user experience using NeuronX Distributed (NxD) is improved by introducing asynchronous checkpointing. This release also adds support for auto partitioning pipeline parallelism in NxD and introduces Pipeline Parallelism in PyTorch Lightning Trainer (beta).

**Inference highlights**: Speculative Decoding support (beta) in TNx library improves LLM inference throughput and output token latency(TPOT) by up to 25% (for LLMs such as Llama-2-70B). TNx also improves weight loading performance by adding support for SafeTensor checkpoint format. Inference using Bucketing in PyTorch NeuronX and NeuronX Distributed is improved by introducing auto-bucketing feature.
This release also adds a new sample for ``Mixtral-8x7B-v0.1`` and ``mistralai/Mistral-7B-Instruct-v0.2`` in TNx.

**Neuron DLAMI and Neuron DLC support highlights**: This release introduces new Multi Framework DLAMI for Ubuntu 22 that customers can use to easily get started with latest Neuron SDK on multiple frameworks that Neuron supports as well as SSM parameter support for DLAMIs to automate the retrieval of latest DLAMI ID in cloud automation flows. Support for new Neuron Training and Inference Deep Learning containers (DLCs) for PyTorch 2.1, as well as a new dedicated GitHub repository to host Neuron container dockerfiles and a public Neuron container registry to host Neuron container images.

More release content can be found in the table below and each component release notes.


.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances


   * - Transformers NeuronX (transformers-neuronx) for Inference
     - * [Beta] Support for Speculative Decoding API. See :ref:`developer guide <transformers_neuronx_readme>` 
       * Support for SafeTensors checkpoint format with improved weight loading performance.  See :ref:`developer guide <transformers_neuronx_readme>` 
       * Support for running  Top-K sampling on Neuron Device for improved performance.  See :ref:`developer guide <transformers_neuronx_readme>` 
       * Code Llama model inference sample with 16K input seq length. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/codellama-13b-16k-sampling.ipynb>`__
       * [Beta] Support for streaming API and stopping criteria API. See :ref:`developer guide <transformers_neuronx_readme>`
       * Support for ``Mixtral-8x7B-v0.1`` model inference. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/mixtral-8x7b-sampling.ipynb>`__
       * [Beta] Support for ``mistralai/Mistral-7B-Instruct-v0.2`` model inference. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/mistralai-Mistral-7b-Instruct-v0.2.ipynb>`__
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Training
     - * Support for Pipeline Parallelism training using PyTorch Lightning. See :ref:`api guide <api_guide>` , :ref:`developer guide <ptl_developer_guide>` and :doc:`tutorial </archive/tutorials/training_llama2_tp_pp_ptl>`
       * Support for auto partitioning pipeline parallel stages when training large models.  See :ref:`api guide <api_guide>` and :ref:`pp_developer_guide`
       * Support for asynchronous checkpointing to improve the time it takes to save the checkpoint.  See :ref:`api guide <api_guide>` , :ref:`save_load_developer_guide` and :doc:`tutorial </archive/tutorials/training_llama2_tp_pp_ptl>`
       * Tutorial to fine-tune Llama-2-7B model using PyTorch Lightning and running evaluation on the fine-tuned model using Hugging Face optimum-neuron. See :ref:`tutorial <llama2_7b_tp_zero1_ptl_finetune_tutorial>`
       * ``codegen25-7b-mono`` model training tutorial and script. See :ref:`codegen25_7b_tp_zero1_tutorial` 
       * See more at :ref:`nxd-core_rn` 
     - Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Inference
     - * Support for auto bucketing in inference using a custom bucket kernel that can be passed as a bucket configuration to Tracing API. See :ref:`api guide <api_guide>` and :ref:`neuronx_distributed_inference_developer_guide`
       * Support for inference with bf16 data type using XLA_USE_BF16=1 flag.
       * See more at :ref:`nxd-core_rn` 
     - Inf2,Trn1/Trn1n

   * - PyTorch NeuronX (torch-neuronx)
     - * PyTorch 2.1 support is now stable (out of beta). 
       * Support for auto bucketing in inference using a custom bucket kernel that can be passed as a bucket configuration to Tracing API. See :ref:`torch-neuronx-autobucketing-devguide`
       * See more at :ref:`pytorch_rn`
     - Trn1/Trn1n,Inf2

   * - NeuronX Nemo Megatron for Training
     - * Support for LoRa finetuning. See `sample script <https://github.com/aws-neuron/neuronx-nemo-megatron/tree/main/nemo/examples/nlp/language_modeling/test_llama_lora.sh>`__
       * Support for Mistral-7B training. See `sample script <https://github.com/aws-neuron/neuronx-nemo-megatron/tree/main/nemo/examples/nlp/language_modeling/test_mistral.sh>`__
       * Support for asynchronous checkpointing to improve the time it takes to save the checkpoint.
       * See more at `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_  and  :ref:`neuronx-nemo-rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * New ``--enable-mixed-precision-accumulation`` compiler option to perform intermediate computations of an operation in FP32 regardless of the operation's defined datatype. See :ref:`neuron-compiler-cli-reference-guide`
       * See more at :ref:`compiler_rn`
     - Trn1/Trn1n,Inf2

   * - Neuron DLAMI and DLC
     - * New Neuron Multi Framework Deep Learning AMI (DLAMI) for Ubuntu 22 with separate virtual environments for PyTorch 2.1, PyTorch 1.13, Transformers NeuronX and Tensorflow 2.10.  See :ref:`setup guide <setup-ubuntu22-multi-framework-dlami>` and :ref:`neuron-dlami-overview`
       * Neuron Multi Framework Deep Learning AMI (DLAMI) is now the default Neuron AMI in QuickStart AMI list when launching Neuron instances for Ubuntu through AWS console. See :ref:`setup guide <setup-ubuntu22-multi-framework-dlami>`
       * Neuron DLAMIs for PyTorch 1.13 and Tensorflow 2.10 are updated with 2.18 Neuron SDK for both Ubuntu 20 and AL2. See :ref:`neuron-dlami-overview`
       * SSM parameter support for Neuron DLAMIs to find the DLAMI id with latest Neuron release SDK. See :ref:`neuron-dlami-overview`
       * New Neuron Deep Learning Containers(DLCs) for PyTorch 2.1 Inference and Training.  See :ref:`neuron_containers`
       * PyTorch 1.13 Inference and Training DLCs are updated with latest 2.18 Neuron SDK and now also comes with pre-installed NeuronX Distributed library. See :ref:`neuron_containers`
       * Neuron DLCs are now hosted both in public Neuron ECR and as private images. Private images are only needed when using with Sagemaker. See :ref:`neuron_containers`
       * New Neuron Github Repository to host dockerfiles for Neuron DLCs. See `neuron deep learning containers github repo <https://github.com/aws-neuron/deep-learning-containers>`_
     - Inf1,Inf2,Trn1/Trn1n
  
   * - Other Documentation Updates
     - * App Note on snapshotting models with PyTorch NeuronX 2.1 to support dumping debug information. See :ref:`pytorch-neuronx-debug`
       * Added announcement for Maintenance mode of TensorFlow 1.x. See :ref:`announce-tfx-maintenance`
       * --
     - Inf1, Inf2, Trn1/Trn1n
   
   * - Known Issues and Limitations
     - * See :ref:`neuron-2.18.0-known-issues`
     - Trn1/Trn1n , Inf2, Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1


.. _neuron-2.18.0-known-issues:

2.18.0 Known Issues and Limitations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* For PyTorch 2.1 (NeuronX), slow convergence for LLaMA-2 70B training when using Zero Redundancy Optimizer (ZeRO1) can be resolved by removing all compiler flags.
* For PyTorch 2.1 (NeuronX), torch-xla 2.1 is incompatible with the default GLibC on AL2. Users are advised to migrate to Amazon Linux 2023 , Ubuntu 22 or Ubuntu 20 Operating systems.
* See component release notes below for any additional known issues.


.. _neuron-2.17.0-whatsnew:


Neuron 2.17.0 (02/13/2024)
--------------------------

What's New
^^^^^^^^^^

Neuron 2.17 release improves small collective communication operators (smaller than 16MB) by up to 30%, which improves large language model (LLM) Inference performance by up to 10%.
This release also includes improvements in :ref:`Neuron Profiler <neuron-profile-ug>` and other minor enhancements and bug fixes.


.. _neuron-2.16.0-whatsnew:



Neuron 2.16.1 (01/18/2024)
--------------------------
Patch release with compiler bug fixes, updates to :ref:`Neuron Device Plugin and Neuron Kubernetes Scheduler <containers_rn>` .


Neuron 2.16.0 (12/21/2023)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

Neuron 2.16 adds support for Llama-2-70B training and inference, upgrades to PyTorch 2.1 (beta) and adds new support for PyTorch Lightning Trainer (beta) as well as performance improvements and adding Amazon Linux 2023 support.

**Training highlights**: NeuronX Distributed library LLM models training performance is improved by up to 15%. LLM model training user experience is improved by introducing support of PyTorch Lightning Trainer (beta), and a new model optimizer wrapper which will minimize the amount of changes needed to partition models using NeuronX Distributed primitives.  

**Inference highlights**: PyTorch inference now allows to dynamically swap different fine-tuned weights for an already loaded model, as well as overall improvements of LLM inference throughput and latency with Transformers NeuronX. Two new reference model samples for LLama-2-70b and Mistral-7b model inference.

**User experience**: This release introduces two new capabilities: A new tool, Neuron Distributed Event Tracing (NDET) which improves debuggability, and the support of profiling collective communication operators in the Neuron Profiler tool.

More release content can be found in the table below and each component release notes.



.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances


   * - Transformers NeuronX (transformers-neuronx) for Inference
     - * [Beta] Support for Grouped Query Attention(GQA). See :ref:`developer guide <transformers_neuronx_readme>` 
       * [Beta] Support for ``Llama-2-70b`` model inference using ``Grouped Query Attention``. See `tutorial <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-70b-sampling.ipynb>`__ 
       * [Beta] Support for ``Mistral-7B-Instruct-v0.1`` model inference. See :ref:`sample code <mistral_gqa_code_sample>`
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Training
     - * [Beta] Support for ``PyTorch Lightning``  to train models using ``tensor parallelism`` and ``data parallelism`` . See :ref:`api guide <api_guide>` , :ref:`developer guide <ptl_developer_guide>` and tutorial
       * Support for Model and Optimizer Wrapper training API that handles the parallelization. See :ref:`api guide <api_guide>` and :ref:`model_optimizer_wrapper_developer_guide`
       * New ``save_checkpoint``  and ``load_checkpoint`` APIs to save/load checkpoints during distributed training. See :ref:`save_load_developer_guide`
       * Support for a new ``Query-Key-Value(QKV)`` module that provides the ability to replicate the Key Value heads and adds flexibility to use higher Tensor parallel degree during Training. See :ref:`api guide <api_guide>` and :doc:`tutorial </archive/tutorials/training_llama2_tp_pp_ptl>`
       * See more at :ref:`nxd-core_rn` 
     - Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Inference
     - * Support weight-deduplication amongst TP shards by giving ability to save weights separately than in NEFF files.  See developer guide
       * See more at :ref:`nxd-core_rn` and  :ref:`api_guide`
     - Inf2,Trn1/Trn1n

   * - PyTorch NeuronX (torch-neuronx)
     - * [Beta]Support for] ``PyTorch 2.1``. See PyTorch 2.1 support documentation. See  `llama-2-13b inference <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb>`_ sample.
       * Support to separate out model weights from NEFF files and new ``replace_weights`` API to replace the separated weights. See :ref:`torch_neuronx_replace_weights_api` and :ref:`torch_neuronx_trace_api`
       * [Beta] Script for training ``stabilityai/stable-diffusion-2-1-base`` and  ``runwayml/stable-diffusion-v1-5`` models . See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/stable_diffusion/>`__ 
       * [Beta] Script for training ``facebook/bart-large`` model. See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_summarization/BartLarge.ipynb>`__ 
       * [Beta] Script for ``stabilityai/stable-diffusion-2-inpainting`` model inference.  See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sd2_inpainting_936_624_inference.ipynb>`__ 
     - Trn1/Trn1n,Inf2

   * - Neuron Tools
     - * New ``Neuron Distributed Event Tracing (NDET) tool`` to help visualize execution trace logs and diagnose errors in multi-node workloads.
       * Support for multi-worker jobs in ``neuron-profile`` . See :ref:`neuron-profile-ug`
       * See more at :ref:`dev-tools_rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Documentation Updates
     - * Added setup guide instructions for ``AL2023`` OS. See :ref:`setup-guide-index`
       * Added announcement for name change of Neuron Components. See :ref:`announce-component-name-change`
       * Added announcement for End of Support for ``PyTorch 1.10`` . See :ref:`announce-eos_pytorch110`
       * Added announcement for End of Support for ``PyTorch 2.0`` Beta. See :ref:`announce-eos_pytorch2`
       * --
     - Inf1, Inf2, Trn1/Trn1n
   
   * - Known Issues and Limitations
     - * See :ref:`neuron-2.16.0-known-issues`
     - Trn1/Trn1n , Inf2, Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1


.. _neuron-2.16.0-known-issues:

2.16.0 Known Issues and Limitations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* We recommend running multi-node training jobs on AL2023 using Amazon EKS. Parallel Cluster currently does not support AL2023.
* There are known compiler issues impacting inference accuracy of certain model configurations of ``Llama-2-13b`` when ``amp = fp16`` is used. If this issue is observed, ``amp=fp32`` should be used as a work around.  This issue will be addressed in future Neuron releases.
* Execution time reported in ``neuron-profile`` tool is sometimes in-accurate due to a bug in how the time is captured.  The bug will be addressed in upcoming Neuron releases.
* See component release notes below for any additional known issues.



.. _neuron-2.15.0-whatsnew:


Neuron 2.15.2 (11/17/2023)
--------------------------
Patch release that fixes compiler issues related to performance when training using ``neuronx-nemo-megatron`` library.


Neuron 2.15.1 (11/09/2023)
--------------------------
Patch release to fix execution overhead issues in Neuron Runtime that were inadvertently introduced in 2.15 release.



Neuron 2.15.0 (10/26/2023)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release adds support for PyTorch 2.0 (Beta), increases performance for both training and inference workloads, adding ability to train models like ``Llama-2-70B`` using ``neuronx-distributed``. With this release, we are also adding pipeline parallelism support for ``neuronx-distributed`` enabling full 3D parallelism support to easily scale training to large model sizes.
Neuron 2.15 also introduces support for training ``resnet50``, ``milesial/Pytorch-UNet`` and ``deepmind/vision-perceiver-conv`` models using ``torch-neuronx``, as well as new sample code for ``flan-t5-xl`` model inference using ``neuronx-distributed``, in addition to other performance optimizations, minor enhancements and bug fixes.

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - Neuron Distributed (neuronx-distributed) for Training
     - * Pipeline parallelism support. See :ref:`api_guide` , :ref:`pp_developer_guide` and :ref:`pipeline_parallelism_overview`
       * ``Llama-2-70B`` model training script  (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/llama2/tp_pp_llama2_70b_hf_pretrain>`__) (tutorial)
       * Mixed precision support. See :ref:`pp_developer_guide`
       * Support serialized checkpoint saving and loading using ``save_xser`` and ``load_xser`` parameters. See :ref:`api_guide` 
       * See more at :ref:`nxd-core_rn` 
     - Trn1/Trn1n

   * - Neuron Distributed (neuronx-distributed) for Inference
     - * ``flan-t5-xl`` model inference script (:pytorch-neuron-src:`tutorial <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`)
       * See more at :ref:`nxd-core_rn` and  :ref:`api_guide`
     - Inf2,Trn1/Trn1n

   * - Transformers Neuron (transformers-neuronx) for Inference
     - * Serialization support for ``Llama``, ``Llama-2``, ``GPT2`` and ``BLOOM`` models . See :ref:`developer guide <transformers_neuronx_readme>`
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n

   * - PyTorch Neuron (torch-neuronx)
     - * Introducing ``PyTorch 2.0`` Beta support. See PyTorch 2.0 support documentation. See `bert training <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/dp_bert_hf_pretrain>`_ and  `t5-3b inference <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/neuronx_distributed/t5-inference/t5-inference-tutorial.html>`_ samples.
       * Scripts for training `resnet50[Beta] <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/resnet50>`_ ,
         `milesial/Pytorch-UNet[Beta] <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/unet_image_segmentation>`_ and `deepmind/vision-perceiver-conv[Beta] <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_image_classification/VisionPerceiverConv.ipynb>`_ models.
     - Trn1/Trn1n,Inf2

   * - AWS Neuron Reference for Nemo Megatron library (``neuronx-nemo-megatron``)
     - * ``Llama-2-70B`` model training sample using pipeline parallelism and tensor parallelism ( `tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`__)
       * ``GPT-NeoX-20B`` model training using pipeline parallelism and tensor parallelism 
       * See more at :ref:`neuronx-nemo-rn` and `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - Trn1/Trn1n

   * - Neuron Compiler (neuronx-cc)
     - * New ``llm-training`` option argument to ``--distribution_strategy`` compiler option for optimizations related to distributed training. See more at :ref:`neuron-compiler-cli-reference-guide`
       * See more at :ref:`compiler_rn`
     - Inf2/Trn1/Trn1n

   * - Neuron Tools
     - * ``alltoall`` Collective Communication operation for intra node(with in the instance), previously released in Neuron Collectives v2.15.13, was added as a testable operation in ``nccom-test``. See :ref:`nccom-test`
       * See more at :ref:`dev-tools_rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Documentation Updates
     - * New :ref:`App Note <activation_memory_reduction>` and :ref:`Developer Guide <activation_memory_reduction_developer_guide>` about Activation memory reduction using ``sequence parallelism`` and ``activation recomputation`` in ``neuronx-distributed``
       * Added a new Model Samples and Tutorials summary page. See :ref:`model_samples_tutorials`
       * Added Neuron SDK Classification guide. See :ref:`sdk-classification`
       * --
     - Inf1, Inf2, Trn1/Trn1n
   
   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1








.. _neuron-2.14.0-whatsnew:


Neuron 2.14.1 (09/26/2023)
--------------------------

This is a patch release that fixes compiler issues in certain configurations of ``Llama`` and ``Llama-2`` model inference using ``transformers-neuronx``.

.. note::

   There is still a known compiler issue for inference of some configurations of ``Llama`` and ``Llama-2`` models that will be addressed in future Neuron release.
   Customers are advised to use ``--optlevel 1 (or -O1)`` compiler flag to mitigate this known compiler issue.  
    
   See :ref:`neuron-compiler-cli-reference-guide` on the usage of ``--optlevel 1`` compiler flag. Please see more on the compiler fix and known issues in :ref:`compiler_rn` and :ref:`nxd-inference_rn` 
   



Neuron 2.14.0 (09/15/2023)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release introduces support for ``Llama-2-7B`` model training and ``T5-3B`` model inference using ``neuronx-distributed``. It also adds support for  ``Llama-2-13B`` model training using ``neuronx-nemo-megatron``. Neuron 2.14 also adds support for ``Stable Diffusion XL(Refiner and Base)`` model inference using ``torch-neuronx`` . This release also introduces other new features, performance optimizations, minor enhancements and bug fixes.
This release introduces the following:

.. note::
   This release deprecates ``--model-type=transformer-inference`` compiler flag. Users are highly encouraged to migrate to the ``--model-type=transformer`` compiler flag.


.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - AWS Neuron Reference for Nemo Megatron library (``neuronx-nemo-megatron``)
     - * ``Llama-2-13B`` model training support ( `tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`__ )
       * ZeRO-1 Optimizer support  that works with tensor parallelism and pipeline parallelism
       * See more at :ref:`neuronx-nemo-rn` and `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - Trn1/Trn1n
   
   * - Neuron Distributed (neuronx-distributed) for Training
     - * ``pad_model`` API to pad attention heads that do not divide by the number of NeuronCores, this will allow users to use any supported tensor-parallel degree. See  :ref:`api_guide`
       * ``Llama-2-7B`` model training support  (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/tp_zero1_llama2_7b_hf_pretrain>`__)
       * See more at :ref:`nxd-core_rn` and  :ref:`api_guide`
     - Trn1/Trn1n

   * - Neuron Distributed (neuronx-distributed) for Inference
     - * ``T5-3B`` model inference support (:pytorch-neuron-src:`tutorial <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`)
       * ``pad_model`` API to pad attention heads that do not divide by the number of NeuronCores, this will allow users to use any supported tensor-parallel degree. See  :ref:`api_guide` 
       * See more at :ref:`nxd-core_rn` and  :ref:`api_guide`
     - Inf2,Trn1/Trn1n

   * - Transformers Neuron (transformers-neuronx) for Inference
     - * Introducing ``--model-type=transformer`` compiler flag that deprecates ``--model-type=transformer-inference`` compiler flag. 
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n

   * - PyTorch Neuron (torch-neuronx)
     - * Performance optimizations in ``torch_neuronx.analyze`` API. See :ref:`torch_neuronx_analyze_api`
       * ``Stable Diffusion XL(Refiner and Base)`` model inference support  ( `sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sdxl_base_and_refiner_1024_inference.ipynb>`__)
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * New  ``--optlevel``(or ``-O``) compiler option that enables different optimizations with tradeoff between faster model compile time and faster model execution. See more at :ref:`neuron-compiler-cli-reference-guide`
       * See more at :ref:`compiler_rn`
     - Inf2/Trn1/Trn1n

   * - Neuron Tools
     - * Neuron SysFS support for showing connected devices on ``trn1.32xl``, ``inf2.24xl`` and ``inf2.48xl`` instances. See :ref:`neuron-sysfs-ug`
       * See more at :ref:`dev-tools_rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Documentation Updates
     - * Neuron Calculator now supports multiple model configurations for Tensor Parallel Degree computation. See :ref:`neuron_calculator`
       * Announcement to deprecate ``--model-type=transformer-inference`` flag. See :ref:`announce-end-of-support-transformer-flag`
       * --
     - Inf1, Inf2, Trn1/Trn1n
   
   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1




.. _neuron-2.13.0-whatsnew:

Neuron 2.13.2 (09/01/2023)
---------------------------

This is a patch release that fixes issues in Kubernetes (K8) deployments related to Neuron Device Plugin crashes and other pod scheduling issues. This release also adds support for zero-based Neuron Device indexing in K8 deployments, see the :ref:`Neuron K8 release notes <containers_rn>` for more details on the specific bug fixes.

Updating to latest Neuron Kubernetes components and Neuron Driver is highly encouraged for customers using Kubernetes.

Please :ref:`follow these instructions in setup guide <setup-guide-index>` to upgrade to latest Neuron release.


Neuron 2.13.1 (08/29/2023)
--------------------------
This release adds support for ``Llama 2`` model training (`tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`__) using `neuronx-nemo-megatron <https://github.com/aws-neuron/neuronx-nemo-megatron>`_ library, and adds support for ``Llama 2`` model inference using ``transformers-neuronx`` library (`tutorial <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb>`__) . 

Please :ref:`follow these instructions in setup guide <setup-guide-index>` to upgrade to latest Neuron release.

.. note::

   Please install  ``transformers-neuronx`` from https://pip.repos.neuron.amazonaws.com to get latest features and improvements.
   
   This release does not support LLama 2 model with Grouped-Query Attention


Neuron 2.13.0 (08/28/2023)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release introduces support for ``GPT-NeoX`` 20B model training in ``neuronx-distributed`` including Zero-1 optimizer capability. It also adds support for ``Stable Diffusion XL`` and ``CLIP`` models inference in ``torch-neuronx``. Neuron 2.13 also introduces `AWS Neuron Reference for Nemo Megatron <https://github.com/aws-neuron/neuronx-nemo-megatron>`_ library supporting distributed training of LLMs like ``GPT-3 175B``. This release also introduces other new features, performance optimizations, minor enhancements and bug fixes.
This release introduces the following:



.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - AWS Neuron Reference for Nemo Megatron library
     - * Modified versions of the open-source packages `NeMo <https://github.com/NVIDIA/NeMo>`_ and `Apex <https://github.com/NVIDIA/apex>`_ that have been adapted for use with AWS Neuron and AWS EC2 Trn1 instances.
       * ``GPT-3`` model training support ( `tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-gpt-job.md>`__ )
       * See more at `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - Trn1/Trn1n

   * - Transformers Neuron (transformers-neuronx) for Inference
     - * Latency optimizations for  ``Llama`` and ``GPT-2`` models inference.
       * Neuron Persistent Cache support (:ref:`developer guide <transformers_neuronx_readme>`)
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n
   
   * - Neuron Distributed (neuronx-distributed) for Training
     - * Now Stable, removed beta support
       * ZeRO-1 Optimizer support with tensor parallel. (:ref:`tutorial <gpt_neox_tp_zero1_tutorial>`)
       * Sequence Parallel support. (:ref:`api guide <api_guide>`)
       * See more at :ref:`nxd-core_rn` and  :ref:`api_guide`
     - Trn1/Trn1n

   * - Neuron Distributed (neuronx-distributed) for Inference
     - * KV Cache Support for LLM Inference (:ref:`release notes <nxd-core_rn>`)
     - Inf2,Trn1/Trn1n


   * - PyTorch Neuron (torch-neuronx)
     - * Seedable dropout enabled by default for training
       * KV Cache inference support ( :pytorch-neuron-src:`tutorial <torch-neuronx/t5-inference-tutorial.ipynb>` )
       * ``camembert-base`` training script. (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_text_classification/CamembertBase.ipynb>`__)
       * New models inference support that include `Stable Diffusion XL <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sdxl_1024_inference.ipynb>`_ , CLIP (`clip-vit-base-patch32 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_clip_base_inference_on_inf2.ipynb>`_ , `clip-vit-large-patch14 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_clip_large_inference_on_inf2.ipynb>`_ ) , `Vision Perceiver <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_perceiver_vision_inference.ipynb>`_ , `Language Perceiver <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_perceiver_language_inference.ipynb>`_ and :pytorch-neuron-src:`T5 <torch-neuronx/t5-inference-tutorial.ipynb>`
     - Trn1/Trn1n,Inf2


   * - Neuron Tools
     - * New data types support for Neuron Collective Communication Test Utility (NCCOM-TEST)  --check option: fp16, bf16, (u)int8, (u)int16, and (u)int32 
       * Neuron SysFS support for FLOP count(flop_count) and connected Neuron Device ids (connected_devices).  See :ref:`neuron-sysfs-ug`
       * See more at :ref:`dev-tools_rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Neuron Runtime 
     - * Runtime version and Capture Time support to NTFF
       * Async DMA copies support to improve Neuron Device copy times for all instance types
       * Logging and error messages improvements for Collectives timeouts and when loading NEFFs.
       * See more at :ref:`runtime_rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - End of Support Announcements and Documentation Updates 
     - * Announcing End of support for ``AWS Neuron reference for Megatron-LM`` starting Neuron 2.13. See more at :ref:`announce-eol-megatronlm`
       * Announcing end of support for ``torch-neuron`` version 1.9 starting Neuron 2.14. See more at :ref:`announce-eol-pytorch19`
       * Added TensorFlow 2.x (``tensorflow-neuronx``) analyze_model API section. See more at :ref:`tensorflow-ref-neuron-analyze_model-api`
       * Upgraded ``numpy`` version to ``1.21.6`` in various training scripts for `Text Classification <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training>`_
       * Updated ``bert-japanese`` training Script to use ``multilingual-sentiments`` dataset. See `hf-bert-jp <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_bert_jp>`_
       * --
     - Inf1, Inf2, Trn1/Trn1n
   
   * - Known Issues and Limitations
     - * See :ref:`neuron-2.13.0-known-issues`
     - Trn1/Trn1n , Inf2, Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1



.. _neuron-2.13.0-known-issues:

2.13.0 Known Issues and Limitations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Currently we see a NaN generated when the model implementation uses torch.dtype(float32.min) or torch.dtype(float32.max) along with XLA_USE_BF16/XLA_DOWNCAST_BF16. This is because, float32.min or float32.max gets downcasted to Inf in bf16 thereby producing a NaN. Short term fix is that we can use a small/large fp32 number instead of using float32.min/float32.max. Example, for mask creation, we can use -/+1e4 instead of min/max values. The issue will be addressed in future Neuron releases.   



.. _neuron-2.12.0-whatsnew:


Neuron 2.12.2 (08/19/2023)
--------------------------
Patch release to fix a jemalloc conflict for all Neuron customers that use Ubuntu 22.  The previous releases shipped with a dependency on jemalloc that may lead to compilation failures in Ubuntu 22 only.  
Please :ref:`follow these instructions in setup guide<setup-guide-index>` to upgrade to latest Neuron release.


Neuron 2.12.1 (08/09/2023)
--------------------------
Patch release to improve reliability of Neuron Runtime when running applications on memory constrained instances. The Neuron Runtime has reduced the contiguous memory requirement for initializing the Neuron Cores associated with applications.
This reduction allows bringup when only small amounts of contiguous memory remain on an instance.  Please :ref:`upgrade to latest Neuron release<setup-guide-index>` to use the latest Neuron Runtime.


Neuron 2.12.0 (07/19/2023)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release introduces  ZeRO-1 optimizer for model training in ``torch-neuronx`` , introduces beta support for ``GPT-NeoX``, ``BLOOM`` , ``Llama`` and ``Llama 2(coming soon)`` models in ``transformers-neuronx``. This release also adds support for model inference serving on Triton Inference Server for Inf2 & Trn1 instances, ``lazy_load`` API and ``async_load`` API for model loading in ``torch-neuronx``, as well as other new features,
performance optimizations, minor enhancements and bug fixes. This release introduces the following:


.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - ZeRO-1 optimizer for model training in ``torch-neuronx``
     - * Support of ZeRO-Stage-1 optimizer ( ZeroRedundancyOptimizer() API) for training models using ``torch-neuronx``
       * See tutorial at  :ref:`zero1-gpt2-pretraining-tutorial`
     - Inf2, Trn1/Trn1n

   * - Support for new models and Enhancements in ``transformers-neuronx``
     - * [Beta] Support for inference of ``GPT-NeoX``, ``BLOOM`` and ``Llama`` models. 
       * [Beta] Support for ``Llama 2`` coming soon. Please monitor the `transformers-neuronx repository <https://github.com/aws-neuron/transformers-neuronx/tree/main/src/transformers_neuronx>`_ for updates.
       * Removed constraints on ``tp_degree`` in tensor-parallel configurations for ``GPT2``, ``OPT``, and ``BLOOM`` . See more at :ref:`nxd-inference_rn`
       * Added multi-query / multi-group attention support for ``GPT2``.
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n
   
   * - Support for Inf2 and Trn1 instances on Triton Inference Server
     - * Support for Model Inference serving on Triton for Inf2 and Trn1 instances. See more at `Triton Server Python Backend <https://github.com/triton-inference-server/python_backend/tree/main/inferentia#using-triton-with-inferentia-2-or-trn1>`_
       * See tutorial at `Triton on SageMaker - Deploying on Inf2 <https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-triton/inferentia2>`_
     - Inf2, Trn1

   * - Support for new computer vision models 
     - * Performance optimizations in Stable Diffusion 2.1 model script and added [beta] support for Stable Diffusion 1.5 models.
       * [Beta] Script for training CLIP model for Image Classification.
       * [Beta] Script for inference of Multimodal perceiver model
       * Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`__
     - Inf2, Trn1/Trn1n

   * - New Features in ``neuronx-distributed`` for training
     - * Added parallel cross entropy loss function.
       * See more at tensor parallelism API guide
     - Trn1/Trn1n

   * - ``lazy_load`` and ``async_load`` API for model loading in inference and performance enhancements in ``torch-neuronx`` 
     - * Added ``lazy_load`` and ``async_load`` API to accelerate model loading for Inference. See more at :ref:`torch_neuronx_lazy_async_load_api`
       * Optimize DataParallel API to load onto multiple cores simultaneously when device IDs specified are consecutive.
       * See more at :ref:`pytorch_rn`
     - Inf2, Trn1/Trn1n
  
   * - [Beta] Asynchronous Execution support and Enhancements in Neuron Runtime 
     - * Added beta asynchronous execution feature which can reduce latency by roughly 12% for training workloads. See more at :ref:`nrt-configuration`
       * AllReduce with All-to-all communication pattern enabled for 16 ranks on TRN1/TRN1N within the instance (intranode)
       * See more at :ref:`runtime_rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - Support for ``distribution_strategy`` compiler option in ``neuronx-cc``
     - * Support for optional ``--distribution_strategy`` compiler option to enable compiler specific optimizations based on distribution strategy used.
       * See more at :ref:`neuron-compiler-cli-reference-guide`
     - Inf2, Trn1/Trn1n

   * - New Micro Benchmarking Performance User Guide and Documentation Updates 
     - * Added best practices user guide for benchmarking performance of Neuron devices. See more at `Benchmarking Guide and Helper scripts <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/microbenchmark>`_
       * Announcing end of support for Ubuntu 18. See more at :ref:`announce-eol-ubuntu18`
       * Removed support for Distributed Data Parallel(DDP) Tutorial.
       * Improved sidebar navigation in Documentation.
       * --
     - Inf1, Inf2, Trn1/Trn1n
   
   * - Known Issues and Limitations
     - * See :ref:`neuron-2.12.0-known-issues`
     - Trn1/Trn1n , Inf2, Inf1
  
   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1



.. _neuron-2.12.0-known-issues:

2.12.0 Known Issues and Limitations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Known Issues in Ubuntu 22 Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Several Vision and NLP models on Ubuntu 22 are not supported due to Compilation issues. Issues will be addressed in upcoming releases.
* CustomOp feature failing with seg fault on Ubuntu 22.  Issue will be addressed in upcoming releases.
  
Known issues in certain resnet models on Ubuntu 20
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Known issue with support for resnet-18, resnet-34, resnet-50, resnet-101 and resnet-152 models on Ubuntu 20. Issues will be addressed in upcoming releases.



.. _neuron-2.11.0-whatsnew:

Neuron 2.11.0 (06/14/2023)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release introduces Neuron Distributed, a new python library to simplify training and inference of large models, improving usability with features like S3 model caching, standalone profiler tool, support for Ubuntu22, as well as other new features,
performance optimizations, minor enhancements and bug fixes. This release introduces the following:


.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

  
   * - New Features and Performance Enhancements in ``transformers-neuronx``
     - * Support for ``int8`` inference. See example at :ref:`int8_weight_storage_support`
       * Improved prompt context encoding performance. See more at :ref:`transformers_neuronx_developer_guide`
       * Improved collective communications performance for Tensor Parallel inference on Inf2 and Trn1.
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n

   * - Neuron Profiler Tool 
     - * Profiling and visualization of model execution on Trainium and Inferentia devices now supported as a stand-alone tool.
       * See more at :ref:`neuron-profile-ug`
     - Inf1, Inf2, Trn1/Trn1n

   * - Neuron Compilation Cache through S3
     - * Support for sharing compiled models across Inf2 and Trn1 nodes through S3
       * See more at :ref:`pytorch-neuronx-parallel-compile-cli`
     - Inf2, Trn1/Trn1n

   * - New script to scan a model for supported/unsupported operators
     - * Script to scan a model for supported/unsupported operators before training, scan output includes supported and unsupported operators at both XLA operators and PyTorch operators level.
       * See a sample tutorial at :ref:`torch-analyze-for-training-tutorial`
     - Inf2, Trn1/Trn1n

   * - Neuron Distributed Library [Beta]
     - * New Python Library based on PyTorch enabling distributed training and inference of large models.
       * Initial support for tensor-parallelism.
       * See more at :doc:`NeuronX Distributed </libraries/neuronx-distributed/index-training>`
     - Inf2, Trn1/Trn1n

   * - Neuron Calculator and Documentation Updates  
     - * New :ref:`neuron_calculator` Documentation section to help determine number of Neuron Cores needed for LLM Inference.
       * Added App Note :ref:`neuron_llm_inference`
       * --
     - Inf1, Inf2, Trn1/Trn1n

   * - Enhancements to Neuron SysFS
     - * Support for detailed breakdown of memory usage across the NeuronCores
       * See more at :ref:`neuron-sysfs-ug`
     - Inf1, Inf2, Trn1/Trn1n

   * - Support for Ubuntu 22
     - * See more at :ref:`setup-guide-index` for setup instructions on Ubuntu22
     - Inf1, Inf2, Trn1/Trn1n

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1




.. _neuron-2.10.0-whatsnew:

Neuron 2.10.0 (05/01/2023)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release introduces new features, performance optimizations, minor enhancements and bug fixes. This release introduces the following:

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances


   * - Initial support for computer vision models inference
     - * Added Stable Diffusion 2.1 model script for Text to Image Generation
       * Added VGG model script for Image Classification Task
       * Added UNet model script for Image Segmentation Task
       * Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`__
     - Inf2, Trn1/Trn1n

   * - Profiling support in PyTorch Neuron(``torch-neuronx``) for Inference with TensorBoard
     - * See more at :ref:`torch-neuronx-profiling-with-tb`
     - Inf2, Trn1/Trn1n
  
   * - New Features and Performance Enhancements in transformers-neuronx
     - * Support for the HuggingFace generate function. 
       * Model Serialization support for GPT2 models. (including model saving, loading, and weight swapping)
       * Improved prompt context encoding performance.
       * See :ref:`transformers_neuronx_readme` for examples and usage
       * See more at :ref:`nxd-inference_rn` 
     - Inf2, Trn1/Trn1n

   * - Support models larger than 2GB in TensorFlow 2.x Neuron (``tensorflow-neuronx``) 
     - * See :ref:`tensorflow-neuronx-special-flags` for details. (``tensorflow-neuronx``) 
     - Trn1/Trn1n, Inf2

   * - Support models larger than 2GB in TensorFlow 2.x Neuron (``tensorflow-neuron``) 
     - * See :ref:`Special Flags <tensorflow-ref-neuron-tracing-api>` for details. (``tensorflow-neuron``)
     - Inf1
  
   * - Performance Enhancements in PyTorch C++ Custom Operators (Beta)
     - * Support for using multiple GPSIMD Cores in Custom C++ Operators
       * See :ref:`custom-ops-api-ref-guide`
     - Trn1/Trn1n
   
   * - Weight Deduplication Feature (Inf1) 
     - * Support for Sharing weights when loading multiple instance versions of the same model on different NeuronCores.
       * See more at :ref:`nrt-configuration`
     - Inf1

   * - ``nccom-test`` - Collective Communication Benchmarking Tool
     - * Supports enabling benchmarking sweeps on various Neuron Collective Communication operations. See :ref:`nccom-test` for more details.
     - Trn1/Trn1n , Inf2

   * - Announcing end of support for tensorflow-neuron 2.7 & mxnet-neuron 1.5 versions
     - * See :ref:`announce-eol-tf-before-2-7`
       * See :ref:`announce-eol-mxnet-before-1-5`
     - Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

.. _neuron-2.9.0-whatsnew:


Neuron 2.9.1 (04/19/2023)
-------------------------
Minor patch release to add support for deserialized torchscript model compilation and support for multi-node training in EKS. Fixes included in this release are critical to enable training
and deploying models with Amazon Sagemaker or Amazon EKS.


Neuron 2.9.0 (03/28/2023)
-------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release adds support for EC2 Trn1n instances, introduces new features, performance optimizations, minor enhancements and bug fixes. This release introduces the following:

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - Support for EC2 Trn1n instances
     - * Updated Neuron Runtime for Trn1n instances     
      
       * Overall documentation update to include Trn1n instances
     - Trn1n

   * - New Analyze API in PyTorch Neuron (``torch-neuronx``)  
     - * A new API that return list of supported and unsupported PyTorch operators for a model. See :ref:`torch_neuronx_analyze_api`
     - Trn1, Inf2
  
   * - Support models that are larger than 2GB in PyTorch Neuron (``torch-neuron``) on Inf1
     - * See ``separate_weights`` flag to :func:`torch_neuron.trace` to support models that are larger than 2GB
     - Inf1

   * - Performance Improvements
     - * Up to 10% higher throughput when training GPT3 6.7B model on multi-node
     - Trn1

   * - Dynamic Batching support in TensorFlow 2.x Neuron (``tensorflow-neuronx``)
     - * See :ref:`tensorflow-neuronx-special-flags` for details.
     - Trn1, Inf2

   * - NeuronPerf support for Trn1/Inf2 instances
     - * Added Trn1/Inf2 support for PyTorch Neuron (``torch-neuronx``) and TensorFlow 2.x Neuron (``tensorflow-neuronx``)
     - Trn1, Inf2

   * - Hierarchical All-Reduce and Reduce-Scatter collective communication
     - * Added support for hierarchical All-Reduce and Reduce-Scatter in Neuron Runtime to enable better scalability of distributed workloads .
     - Trn1, Inf2
  
   * - New Tutorials added
     - * :ref:`Added tutorial to fine-tune T5 model <torch-hf-t5-finetune>`
       * Added tutorial to demonstrate use of Libtorch with PyTorch Neuron (``torch-neuronx``) for inference :ref:`[html] <pytorch-tutorials-libtorch>`
     - Trn1, Inf2

   * - Release included packages
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1, Inf2, Inf1
.. _neuron-2.8.0-whatsnew:

Neuron 2.8.0 (02/24/2023)
-------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release adds support for `EC2 Inf2 <https://aws.amazon.com/ec2/instance-types/inf2/>`_ instances, introduces initial inference support with TensorFlow 2.x Neuron (``tensorflow-neuronx``) on Trn1 and Inf2, and introduces minor enhancements and bug fixes.

This release introduces the following:

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details

   * - Support for `EC2 Inf2 <https://aws.amazon.com/ec2/instance-types/inf2/>`_ instances
     - * Inference support for Inf2 instances in PyTorch Neuron (``torch-neuronx``)      
    
       * Inference support for Inf2 instances in TensorFlow 2.x Neuron (``tensorflow-neuronx``)
        
       * Overall documentation update to include Inf2 instances
  

   * - TensorFlow 2.x Neuron (``tensorflow-neuronx``) support
     - * This releases introduces initial inference support with TensorFlow 2.x Neuron (``tensorflow-neuronx``) on Trn1 and Inf2


   * - New Neuron GitHub samples
     - * New sample scripts for deploying LLM models with ``transformer-neuronx`` under       `aws-neuron-samples <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference>`__  GitHub repository.
      
       * New sample scripts for deploying models with ``torch-neuronx`` under `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`__  GitHub repository.

   * - Release included packages
     - * see :ref:`latest-neuron-release-artifacts`

.. _neuron-2.7.0-whatsnew:

Neuron 2.7.0 (02/08/2023)
-------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release introduces new capabilities and libraries, as well as features and tools that improves usability. This release introduces the following:

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details

   * - PyTorch 1.13
     - Support of PyTorch 1.13 version for PyTorch Neuron (``torch-neuronx``). For resources see :ref:`pytorch-neuronx-main`

   * - PyTorch DistributedDataParallel (DDP) API
     - Support of PyTorch DistributedDataParallel (DDP) API in PyTorch Neuron (``torch-neuronx``). For resources how to use PyTorch DDP API with Neuron, please check the DDP tutorial.

   * - Inference support in ``torch-neuronx``
     - For more details, see Neuron Inference samples `<https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_ in the ``aws-neuron-samples`` GitHub repo.     

   * - Neuron Custom C++ Operators[Beta]
     - Initial support for Neuron Custom C++ Operators [Beta] , with Neuron Custom C++ Operators (“CustomOps”) you can now write CustomOps that run on NeuronCore-v2 chips. For more resources please check :ref:`neuron_c++customops` section.


   * - ``transformers-neuronx`` [Beta] 
     - ``transformers-neuronx``  is a new library enabling LLM model inference. It contains models that are checkpoint-compatible with HuggingFace Transformers, and currently supports Transformer Decoder models like GPT2, GPT-J and OPT. Please check `aws-neuron-samples repository <https://github.com/aws-neuron/transformers-neuronx>`__  


   * - Neuron sysfs filesystem
     - Neuron sysfs filesystem exposes Neuron Devices under ``/sys/devices/virtual/neuron_device`` providing visibility to Neuron Driver and Runtime at the system level. By performing several simple CLIs such as reading or writing to a sysfs file, you can get information such as Neuron Runtime status, memory usage, Driver info etc. For resources about Neuron sysfs filesystem visit :ref:`neuron-sysfs-ug`.


   * - TFLOPS support in Neuron System Tools
     - Neuron System Tools now also report model actual TFLOPs rate in both ``neuron-monitor`` and ``neuron-top``. More details can be found in the :ref:`Neuron Tools documentation <neuron-tools>`.

   * - New sample scripts for training
     - This release adds multiple new sample scripts for training models with ``torch-neuronx``, Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`__

   * - New sample scripts for inference
     - This release adds multiple new sample scripts for deploying models with ``torch-neuronx``, Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`__

   * - Neuron GitHub samples repository for Amazon EKS
     - A new AWS Neuron GitHub samples repository for Amazon EKS, Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-eks-samples>`__

.. _neuron-2.6.0-whatsnew:

Neuron 2.6.0 (12/12/2022)
-------------------------

This release introduces the support of PyTorch 1.12 version, and introduces PyTorch Neuron (``torch-neuronx``) profiling through Neuron Plugin for TensorBoard. Pytorch Neuron (``torch-neuronx``) users can now profile their models through the following TensorBoard views:

* Operator Framework View
* Operator HLO View
* Operator Trace View

This release introduces the support of LAMB optimizer for FP32 mode, and adds support for :ref:`capturing snapshots <torch-neuronx-snapshotting>` of inputs, outputs and graph HLO for debugging.

In addition, this release introduces the support of new operators and resolves issues that improve stability for Trn1 customers.

.. _neuron-2.5.0-whatsnew:

Neuron 2.5.0 (11/23/2022)
-------------------------

Neuron 2.5.0 is a major release which introduces new features and resolves issues that improve stability for Inf1 customers.

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size


   * - Component
     - New in this release

   * - PyTorch Neuron ``(torch-neuron)``
     - * PyTorch 1.12 support
       
       * Python 3.8 support
     
       * :ref:`LSTM <torch_neuron_lstm_support>` support on Inf1

       * :ref:`R-CNN <torch-neuron-r-cnn-app-note>` support on Inf1

       * Support for new :doc:`API for core placement </frameworks/torch/torch-neuron/api-core-placement>`
      
       * Support for :ref:`improved logging <pytorch-neuron-rn>` 
        
       * Improved :func:`torch_neuron.trace` performance when using large graphs
      
       * Reduced host memory usage of loaded models in ``libtorchneuron.so``
      
       * :ref:`Additional operators <neuron-cc-ops-pytorch>` support
       

   * - TensorFlow Neuron ``(tensorflow-neuron)``
     - * ``tf-neuron-auto-multicore`` tool to enable automatic data parallel on multiple NeuronCores.
      
       * Beta support for tracing models larger than 2GB using ``extract-weights`` flag (TF2.x only), see :ref:`tensorflow-ref-neuron-tracing-api`

       * ``tfn.auto_multicore`` Python API to enable automatic data parallel (TF2.x only)
    

This Neuron release is the last release that will include ``torch-neuron`` :ref:`versions 1.7 and 1.8 <announce-eol-pt-before-1-8>`, and that will include ``tensorflow-neuron`` :ref:`versions 2.5 and 2.6 <announce-eol-tf-before-2-5>`.

In addition, this release introduces changes to the Neuron packaging and installation instructions for Inf1 customers, see :ref:`neuron250-packages-changes` for more information.

.. _neuron-2.4.0-whatsnew:

Neuron 2.4.0 (10/27/2022)
-------------------------

This release introduces new features and resolves issues that improve stability. The release introduces "memory utilization breakdown" feature in both :ref:`Neuron Monitor <neuron-monitor-ug>` and :ref:`Neuron Top <neuron-top-ug>` system tools. The release introduces support for "NeuronCore Based Sheduling" capability to the Neuron Kubernetes Scheduler and introduces new operators support in :ref:`Neuron Compiler <neuronx-cc-index>` and :ref:`PyTorch Neuron <pytorch_rn>`. This release introduces also additional eight (8) samples of models' fine tuning using PyTorch Neuron. The new samples can be found in the `AWS Neuron Samples GitHub <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_ repository.
