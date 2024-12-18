.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.21.0-whatsnew:

Neuron 2.21.0 (12/20/2024)
---------------------------

.. contents:: Table of contents
   :local:
   :depth: 1

What's New
^^^^^^^^^^

**Overview**: Neuron 2.21.0 introduces support for :ref:`AWS Trainium 2 <trainium2-arch>` and
:ref:`Trn2 instances <aws-trn2-arch>`, including the trn2.48xlarge instance type and Trn2
UltraServer. The release adds new capabilities in both training and
inference of large-scale models. It introduces :ref:`NxD Inference (beta) <introduce-nxd-inference>`, a
PyTorch-based library for deployment, :ref:`Neuron Profiler 2.0 (beta) <neuron-profiler-2-0-guide>`, and
:ref:`PyTorch 2.5 <introduce-pytorch-2-5>` support across the Neuron SDK, and :ref:`Logical NeuronCore
Configuration (LNC) <logical-neuroncore-config>` for optimizing NeuronCore allocation. The release
enables :ref:`Llama 3.1 405B model inference <nxdi-trn2-llama3.1-405b-tutorial>` on a single trn2.48xlarge
instance.

**NxD Inference**: :ref:`NxD Inference (beta) <nxdi-overview>` is a new PyTorch-based inference library for
deploying large-scale models on AWS Inferentia and Trainium instances.
It enables PyTorch model onboarding with minimal code changes and
integrates with :ref:`vLLM <nxdi-vllm-user-guide>`. NxDI supports various model architectures,
including Llama versions for text processing (Llama 2, Llama 3, Llama
3.1, Llama 3.2, and Llama 3.3), :ref:`Llama 3.2 multimodal for multimodal
tasks <nxdi-llama3.2-multimodal-tutorial>`, and Mixture-of-Experts (MoE) model architectures including
Mixtral and DBRX. The library supports quantization methods, includes
dynamic sampling, and is compatible with HuggingFace checkpoints and
generate() API. NxDI also supports distributed strategies including tensor parallelism and incorporates speculative decoding techniques (Draft model and EAGLE). The
release includes :ref:`Llama 3.1 405B model sample <nxdi-trn2-llama3.1-405b-tutorial>` and :ref:`Llama 3.3 70B model sample <nxdi-trn2-llama3.3-70b-tutorial>` for inference on a single trn2.48xlarge
instance.

For more information, see :ref:`NxD Inference documentation <nxdi-overview>` and check the NxD
Inference Github repository: `aws-neuron/neuronx-distributed-inference <https://github.com/aws-neuron/neuronx-distributed-inference>`_

**Transformers NeuronX (TNx)**: This release introduces several new features, including flash decoding support for speculative decoding, and on-device generation in speculative decoding flows. It adds :ref:`Eagle speculative decoding <cb-eagle-speculative-decoding>` with greedy and lossless sampling, as well as support for :ref:`CPU compilation <transformers_neuronx_developer_guide>` and sharded model saving. Performance improvements include optimized MLP and QKV for Llama models with sequence parallel norm and control over concurrent compilation workers.

**Training Highlights:** NxD Training in this release adds support for
HuggingFace :ref:`Llama3/3.1 70B <hf_llama3_70B_pretraining>` on trn2 instances, introduces :ref:`DPO support <hf_llama3_8B_DPO>` for
post-training model alignment, and adds support for Mixture-of-Experts
(MoE) models including Mixtral 7B. The release includes improved
:ref:`checkpoint conversion <checkpoint_conversion>` capabilities and supports MoE with Tensor,
Sequence, Pipeline, and Expert parallelism.

**ML Frameworks:** Neuron 2.21.0 adds :ref:`PyTorch 2.5 <introduce-pytorch-2-5>` coming with improved
support for eager mode, FP8, and Automatic Mixed Precision capabilities.
JAX support extends to version 0.4.35, including support for JAX caching
APIs.

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
JAX 0.4 within the :ref:`Multi Framework DLAMI <neuron-dlami-overview>`. It also introduces NeuronX
Distributed Training (NxDT), Inference (NxDI), and Core (NxD) with
:ref:`PyTorch 2.5 <introduce-pytorch-2-5>` support. Additionally, a new Single Framework DLAMI for
TensorFlow 2.10 on Ubuntu 22 is now available.

**Deep Learning Containers (DLCs):** This release introduces new DLCs
for :ref:`JAX 0.4 <jax-neuronx-setup>` training and PyTorch 2.5.1 inference and training. All DLCs
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

**Amazon Q**: `Use Q Developer <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/amazonq-getstarted.html#amazon-q-dev>`__
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
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n, Trn2


   * - NxD Core (neuronx-distributed) 
     - **Training:**

       * Added support for HuggingFace Llama3 70B with Trn2 instances
       * Added DPO support for post-training model alignment
       * See more at :ref:`neuronx-distributed-rn`   
     - Trn1/Trn1n,Trn2

   * - NxD Inference (neuronx-distributed-inference)
     - * Introduced new NxD Inference Library. See :ref:`introduce-nxd-inference`
       * Added Llama3.1 405B Inference Example on Trn2. See :ref:`nxdi-trn2-llama3.1-405b-tutorial`
       * Added Llama 3.2 Multimodal inference sample. See :ref:`nxdi-llama3.2-multimodal-tutorial`
       * Added support for vLLM integration for NxD Inference. See :ref:`nxdi-vllm-user-guide`
       * Introduced Open Source Github repository for NxD Inference. See `aws-neuron/neuronx-distributed-inference <https://github.com/aws-neuron/neuronx-distributed-inference>`_
       * See more at :ref:`neuronx-distributed-inference-rn` 
     - Inf2, Trn1/Trn1n,Trn2

   * - NxD Training (neuronx-distributed-training)
     - * Added support for HuggingFace Llama3/3.1 70B with Trn2 instances
       * Added support for Mixtral 8x7B Megatron and HuggingFace models
       * Added support for custom pipeline parallel cuts in HuggingFace Llama3
       * Added support for DPO post-training model alignment
       * See more at :ref:`neuronx-distributed-training-rn` 
     - Trn1/Trn1n,Trn2

   * - PyTorch NeuronX (torch-neuronx)
     - * Introduced PyTorch 2.5 support 
       * See more at :ref:`torch-neuronx-rn`
     - Trn1/Trn1n,Inf2,Trn2

   * - NeuronX Nemo Megatron for Training
     - * Added support for HuggingFace to NeMo checkpoint conversion when virtual pipeline parallel is enabled.
       * Added collective compute coalescing for ZeRO-1 optimizer
       * See more at `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_  and  :ref:`neuronx-nemo-rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * Minor bug fixes and performance enhancements for the Trn2 platform.
       * See more at :ref:`neuronx-cc-rn`
     - Trn1/Trn1n,Inf2,Trn2
  
   * - Neuron Kernel Interface (NKI)
     - * Added :doc:`api/nki.compiler` module with Allocation Control and Kernel decorators
       * Added new nki.isa APIs. See :doc:`api/nki.isa`
       * Added new nki.language APIs. See :doc:`api/nki.language`
       * Added new kernels (``allocated_fused_self_attn_for_SD_small_head_size``, ``allocated_fused_rms_norm_qkv``). See :doc:`api/nki.kernels` 
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
       * See more at :ref:`neuron-dlc-release-notes`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Tools
     - * Introduced Neuron Profiler 2.0. See :ref:`neuron-profiler-2-0-guide`
       * See more at :ref:`neuron-tools-rn`
     - Inf1,Inf2,Trn1/Trn1n,Trn2

   * - Neuron Runtime
     - * Added runtime support to fail in case of out-of-bound memory access when DGE is enabled.
       * Added support for 4-rank replica group on adjacent Neuron cores on TRN1/TRN1N
       * See more at :ref:`neuron-runtime-rn`
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

   * - Documentation Updates
     - * See :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n, Trn2
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1, Trn2

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
  For the latest release that supports Trn1, Inf2 and Inf1 instances, please see :ref:`Neuron Release 2.20.2 <neuron-2.20.0-whatsnew>`



.. _latest-neuron-release-artifacts:


Release Artifacts
-----------------

.. contents:: Table of contents
   :local:
   :depth: 1

Trn2 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.21.0

Trn1 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.21.0

Inf2 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.21.0

Inf1 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.21.0

Supported Python Versions for Inf1 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.21.0

Supported Python Versions for Inf2/Trn1/Trn2 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.21.0

Supported Numpy Versions
^^^^^^^^^^^^^^^^^^^^^^^^
Neuron supports versions >= 1.21.6 and <= 1.22.2

Supported HuggingFace Transformers Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
+----------------------------------+----------------------------------+
| Package                          | Supported HuggingFace            |
|                                  | Transformers Versions            |
+==================================+==================================+
| torch-neuronx                    | < 4.35 and >=4.37.2              |
+----------------------------------+----------------------------------+
| transformers-neuronx             | >= 4.36.0                        |
+----------------------------------+----------------------------------+
| neuronx-distributed - Llama      | 4.31                             |
| model class                      |                                  |
+----------------------------------+----------------------------------+
| neuronx-distributed - GPT NeoX   | 4.26                             |
| model class                      |                                  |
+----------------------------------+----------------------------------+
| neuronx-distributed - Bert model | 4.26                             |
| class                            |                                  |
+----------------------------------+----------------------------------+
| nemo-megatron                    | 4.31.0                           |
+----------------------------------+----------------------------------+











Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`
