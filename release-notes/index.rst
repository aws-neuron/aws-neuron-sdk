.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.20.0-whatsnew:


Neuron 2.20.1 (10/25/2024)
---------------------------

Neuron 2.20.1 release addresses an issue with the Neuron Persistent Cache that was brought forth in 2.20 release. In the 2.20 release, the Neuron persistent cache issue resulted in a cache-miss scenario when attempting to load a previously compiled Neuron Executable File Format (NEFF) from a different path or Python environment than the one used for the initial Neuron SDK installation and NEFF compilation. This release resolves the cache-miss problem, ensuring that NEFFs can be loaded correctly regardless of the path or Python environment used to install the Neuron SDK, as long as they were compiled using the same Neuron SDK version.

This release also addresses the excessive lock wait time issue during neuron_parallel_compile graph extraction for large cluster training. See :ref:`torch-neuronx-rn` and :ref:`libneuronxla-rn`.

Additionally, Neuron 2.20.1 introduces new Multi Framework DLAMI for Amazon Linux 2023 (AL2023) that customers can use to easily get started with latest Neuron SDK on multiple frameworks that Neuron supports. See :ref:`neuron-dlami-release-notes`.

Neuron 2.20.1 Training DLC is also updated to pre-install the necessary dependencies and support NxD Training library out of the box. See :ref:`neuron-dlc-release-notes`


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
     - * Support for on-device sampling (Top P) and dynamic sampling (per request parameters) with Continuous batching. See :ref:`developer guide <transformers_neuronx_developer_guide>`
       * Support for Flash Decoding to enable inference for higher sequence lengths of upto 128K. See :ref:`developer guide <transformers_neuronx_developer_guide>` and `Llama-3.1-8B model sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-8b-128k-sampling.ipynb>`_.
       * Support for multi-node inference for large models like ``Llama-3.1-405B``. See :ref:`developer guide <transformers_neuronx_developer_guide>` and `Llama-3.1-405B model sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-405b-multinode-16k-sampling.ipynb>`_.
       * Support for bucketing, multi-node inference , on-device sampling and other improvements in Neuron vLLM integration. See :ref:`developer guide <transformers_neuronx_developer_guide_for_cb>` 
       * Support for Llama 3.1 models (405B, 70B, and 8B variants). See samples for `Llama-3.1-405B <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-405b-multinode-16k-sampling.ipynb>`_ , `Llama-3.1-70B <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-70b-64k-sampling.ipynb>`_  and  `Llama-3.1-8B <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-8b-128k-sampling.ipynb>`_
       * Support for improved model loading for models like Llama-3 by loading the pre-sharded or pre-transformed weights. See :ref:`serialization support in developer guide <transformers_neuronx_developer_guide>`. 
       * Support for ROPE scaling for Llama 3 and Llama 3.1 models. 
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n


   * - NxD Core (neuronx-distributed) 
     - **Training:**

       * Support for LoRA finetuning
       * Support for Mixed precision enhancements

       **Inference:**
       
       * Suppport for DBRX and Mixtral inference samples. See  samples for `DBRX <https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/dbrx>`_ and `Mixtral <https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/mixtral>`_
       * Support for sequence length autobucketing to improve inference performance.
       * Support for improved tracing in the inference samples.
       * See more at :ref:`neuronx-distributed-rn`   
     - Trn1/Trn1n


   * - NxD Training (neuronx-distributed-training)
     - * First release of NxD Training (beta)
       * See more at :ref:`neuronx-distributed-training-rn` 
     - Trn1/Trn1n


   * - PyTorch NeuronX (torch-neuronx)
     - * Support for inference of Diffusion-Transformers (DiT) models such as ``Pixart-alpha`` and ``Pixart-sigma``. See samples for `Pixart-alpha <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_pixart_alpha_inference_on_inf2.ipynb>`_ and `Pixart-sigma <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_pixart_sigma_inference_on_inf2.ipynb>`_.
       * Support for inference of ``wav2vec2-conformer`` models.  See samples for inference of ``wav2vec2-conformer`` with `relative position embeddings <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_wav2vec2_conformer_relpos_inference_on_inf2.ipynb>`_ and `rotary position embeddings <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_wav2vec2_conformer_rope_inference_on_inf2.ipynb>`_
       * See more at :ref:`torch-neuronx-rn`
     - Trn1/Trn1n,Inf2

   * - NeuronX Nemo Megatron for Training
     - * Fixed issue with linear warmup with cosine annealing
       * Fixed indexing issues with MPI job checkpoint conversion.
       * Fixed pipeline parallel bug for NeMo to HF checkpoint conversion       
       * See more at `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_  and  :ref:`neuronx-nemo-rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * Memory optimization that will reduce the generated compiler artifacts size (i.e., NEFFs)
       * See more at :ref:`neuronx-cc-rn`
     - Trn1/Trn1n,Inf2
  
   * - Neuron Kernel Interface (NKI)
     - * First Release on Neuron Kernel Interface (NKI)
       * See more at :ref:`nki_rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Deep Learning AMIs (DLAMIs)
     - * Support for ``neuronx-distributed-training`` library in PyTorch Neuron DLAMI virtual enviornments. See :ref:`neuron-dlami-overview`
       * Updated existing Neuron supported DLAMIs with Neuron 2.20 SDK release.
       * See more at :ref:`Neuron DLAMI Release Notes <neuron-dlami-overview>`_
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Deep Learning Containers (DLCs)
     - * Updated existing PyTorch Neuron DLCs with Neuron 2.20 SDK release.
       * Support for ``neuronx-distributed-training`` library in `pytorch-training-neuronx DLCs <https://github.com/aws-neuron/deep-learning-containers/tree/main?tab=readme-ov-file#pytorch-training-neuronx>`_. 
       * See more at :ref:`neuron-dlc-release-notes`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Tools
     - * Improvements in Neuron Profile
       * See more at :ref:`neuron-tools-rn`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Runtime
     - * Introduced a sysfs memory usage counter for DMA rings (:ref:`reference <neuron-sysfs-ug>`)
       * See more at :ref:`neuron-runtime-rn`
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

   * - Documentation Updates
     - * See :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

.. _neuron-2.20.0-known-issues:

2.20.0 Known Issues and Limitations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Known issues when using ``on_device_generation`` flag in Transformers NeuronX config for Llama models. Customers are advised not to use the flag when they see an issue. See more at :ref:`transformers-neuronx-rn`  
* See component release notes below for any additional known issues.


.. _components-rn:

Neuron Components Release Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inf1, Trn1/Trn1n and Inf2 common packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
       
     - * :ref:`neuron-runtime-rn`

   * - Neuron Runtime Driver
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-dkms``  (.deb, .rpm)

     - * :ref:`neuron-driver-release-notes`

   * - Neuron System Tools
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-tools``  (.deb, .rpm)
     - * :ref:`neuron-tools-rn`



   * - Containers
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-k8-plugin`` (.deb, .rpm)

       * ``aws-neuronx-k8-scheduler`` (.deb, .rpm)
       
       * ``aws-neuronx-oci-hooks`` (.deb, .rpm)

     - * :ref:`neuron-k8-rn`

       * :ref:`neuron-containers-release-notes`

   * - NeuronPerf (Inference only)
     - Trn1/Trn1n, Inf1, Inf2
     - * ``neuronperf`` (.whl)
     - * :ref:`neuronperf_rn`

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
     - * :ref:`torch-neuronx-rn`
       * :ref:`pytorch-neuron-supported-operators`
       

   * - TensorFlow Neuron
     - Trn1/Trn1n, Inf2
     - * ``tensorflow-neuronx`` (.whl)
     - * :ref:`tensorflow-neuronx-release-notes`

 
   * - Neuron Compiler (Trn1/Trn1n, Inf2 only)
     - Trn1/Trn1n, Inf2
     - * ``neuronx-cc`` (.whl)
     - * :ref:`neuronx-cc-rn`


   * - Neuron Kernel Interface (NKI) Compiler (Trn1/Trn1n, Inf2 only)
     - Trn1/Trn1n, Inf2
     - * Supported within ``neuronx-cc`` (.whl)
     - * :ref:`nki_rn`

   * - Collective Communication library
     - Trn1/Trn1n, Inf2    
     - * ``aws-neuronx-collective`` (.deb, .rpm)
     - * :ref:`neuron-collectives-rn`


   * - Neuron Custom C++ Operators
     - Trn1/Trn1n, Inf2
  
     - * ``aws-neuronx-gpsimd-customop`` (.deb, .rpm)
  
       * ``aws-neuronx-gpsimd-tools`` (.deb, .rpm)
  
     - * :ref:`gpsimd-customop-lib-rn`

       * :ref:`gpsimd-customop-tools-rn`


   * - Transformers Neuron
     - Trn1/Trn1n, Inf2
     - * ``transformers-neuronx`` (.whl)
     - * :ref:`transformers-neuronx-rn`

   * - NxD Training
     - Trn1/Trn1n, Inf2
     - * ``neuronx-distributed-training`` (.whl)
     - * :ref:`neuronx-distributed-training-rn`


   * - NxD Core
     - Trn1/Trn1n, Inf2
     - * ``neuronx-distributed`` (.whl)
     - * :ref:`neuronx-distributed-rn`

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


.. _latest-neuron-release-artifacts:

Release Artifacts
-------------------

.. contents:: Table of contents
   :local:
   :depth: 1

Trn1 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.20.1

Inf2 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.20.1

Inf1 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.20.1

Supported Python Versions for Inf1 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.20.1

Supported Python Versions for Inf2/Trn1 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.20.1

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

Supported Probuf Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^
+----------------------------------+----------------------------------+
| Package                          | Supported Probuf versions        |
+==================================+==================================+
| neuronx-cc                       | > 3                              |
+----------------------------------+----------------------------------+
| torch-neuronx                    | >= 3.20                          |
+----------------------------------+----------------------------------+
| torch-neuron                     | < 3.20                           |
+----------------------------------+----------------------------------+
| transformers-neuronx             | >= 3.20                          |
+----------------------------------+----------------------------------+
| neuronx-distributed              | >= 3.20                          |
+----------------------------------+----------------------------------+
| tensorflow-neuronx               | < 3.20                           |
+----------------------------------+----------------------------------+
| tensorflow-neuron                | < 3.20                           |
+----------------------------------+----------------------------------+

Supported Linux Kernel Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Neuron Driver (``aws-neuronx-dkms``) supports Linux kernel versions >= 5.10

Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`
