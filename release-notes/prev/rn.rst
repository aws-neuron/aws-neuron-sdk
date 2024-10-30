.. _prev-rn:

Previous Releases Notes (Neuron 2.x)
====================================

.. contents:: Table of contents
   :local:
   :depth: 1


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
     - * Support for Flash Attention kernel in Llama models to enable inference for higher sequence lengths. See :ref:`developer guide <transformers_neuronx_developer_guide>` and `Llama-3-8B model sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3-8b-32k-sampling.ipynb>`_.
       * Support for running Top-K sampling on Neuron device for generation in Mixtral models. See ``Mixtral-8x7b`` `sample <https://github.com/aws-neuron/transformers-neuronx/blob/main/src/transformers_neuronx/mixtral/model.py>`_.
       * [Beta] Support for Continuous batching with ``mistralai/Mistral-7B-Instruct-v0.2`` model inference. See :ref:`developer guide <transformers_neuronx_developer_guide_for_cb>`.
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Training
     - * Support for Interleaved pipeline parallelism to reduce idle time (bubble size) and enhance training efficiency and resource utilization for large cluster sizes. See :ref:`api guide <api_guide>` , :ref:`developer guide <pp_developer_guide>`
       * Support for Flash Attention kernel to enable training with longer sequence lengths. See :ref:`Llama-3 sample with 8K sequence length training <llama2_7b_tp_zero1_tutorial>`.
       * See more at :ref:`neuronx-distributed-rn` 
     - Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Inference
     - * Support for Flash Attention kernel for longer sequence length inference. See :pytorch-neuron-src:`[CodeLlama-13b Inference with 16k sequence length] <neuronx_distributed/llama/codellama_16k_inference.ipynb>`
       * [Beta] Support for speculative decoding. See :ref:`developer guide <neuronx_distributed_inference_developer_guide>`.
       * See more at :ref:`neuronx-distributed-rn` 
     - Inf2,Trn1/Trn1n

   * - PyTorch NeuronX (torch-neuronx)
     - * Support for FP32 master weights and BF16 all-gather during Zero1 training to enhance training efficiency.
       * Support to add custom SILU activation functions by configuring NEURON_CUSTOM_SILU variable
       * See more at :ref:`torch-neuronx-rn`
     - Trn1/Trn1n,Inf2

   * - NeuronX Nemo Megatron for Training
     - * Support for FP32 gradient accumulation enhancing accuracy for large model training.
       * Support for Zero1 training with master weights
       * Support for Flash Attention kernel to train with longer sequence lengths (greater than 8K)
       * See more at `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_  and  :ref:`neuronx-nemo-rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * Support for Flash Attention kernel to enable usage of long sequence lengths during training and inference.
       * See more at :ref:`neuronx-cc-rn`
     - Trn1/Trn1n,Inf2

   * - Neuron DLAMI and DLC
     - * Neuron DLAMIs are updated with latest 2.19 Neuron SDK. See :ref:`neuron-dlami-overview`
       * New Neuron Single Framework DLAMIs with PyTorch-2.1 and PyTorch-1.13 for Ubuntu 22. See :ref:`neuron-dlami-overview`
       * New Base Deep Learning AMI (DLAMI) for Ubuntu 22. See :ref:`neuron-dlami-overview`
       * PyTorch 1.13 and PyTorch 2.1 Inference and Training DLCs are updated with latest 2.19 Neuron SDK. See :ref:`neuron_containers`
       * PyTorch 1.13 Inference and PyTorch 2.1 Inference DLCs are updated with TorchServe v0.11.0. See :ref:`neuron_containers`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Tools
     - * Support for new Neuron Node Problem Detector and Recovery plugin in EKS supported kubernetes environments that monitors health of Neuron instances and triggers automatic node replacement upon detecting an unrecoverable error. See :ref:`configuration < k8s-neuron-problem-detector-and-recovery-irsa>` and :ref:`tutorial <k8s-neuron-problem-detector-and-recovery>`.
       * Support for new Neuron Monitor container to enable easy monitoring of Neuron metrics in Kubernetes. Supports monitoring with Prometheus and Grafana. See :ref:`tutorial <k8s-neuron-monitor>`
       * Support for Neuron scheduler extension to enforce allocation of contiguous Neuron Devices for the pods based on the Neuron instance type. See :ref:`tutorial <neuron_scheduler>`
       * Neuron Profiler bugfixes and UI updates, including improvements to visualizing collective operations and to the consistency of information being displayed
       * Added memory usage metrics and device count information to neuron-monitor 
       * See more at :ref:`neuron-tools-rn`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Runtime
     - * Support for dynamic Direct Memory Access (DMA) that reduces memory usage during runtime.
       * Runtime Enhancements that improve collectives performance
       * See more at :ref:`neuron-runtime-rn`
     - Inf1,Inf2,Trn1/Trn1n
  
   * - Other Documentation Updates
     - * Announced maintenance mode of MxNet. See :ref:`announce-mxnet-maintenance`
       * Announced End of support of Neuron TensorFlow 1.x (Inf1). See :ref:`announce-tfx-eos`
       * Announce End of support for AL2. See :ref:`announce-eos-al2`
       * See more at :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

.. _neuron-2.19.0-known-issues:

2.19.0 Known Issues and Limitations 
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

   * - Neuron DLAMI
     - Trn1/Trn1n, Inf1, Inf2
     - * 
     - * `Neuron DLAMI Release Notes <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/dlami/index.html>`_.

   * - Neuron DLC
     - Trn1/Trn1n, Inf1, Inf2
     - *
     - * :ref:`neuron-dlc-release-notes`

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


   * - Neuron Documentation
     - Trn1/Trn1n, Inf1, Inf2
     - * 
     - * :ref:`neuron-documentation-rn`


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

   * - Neuron Distributed
     - Trn1/Trn1n, Inf2
     - * ``neuronx-distributed`` (.whl)
     - * :ref:`neuronx-distributed-rn`

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
Patch release with minor Neuron Compiler bug fixes and enhancements. See more in  :ref:`neuronx-cc-rn`



Neuron 2.18.1 (04/10/2024)
--------------------------

Neuron 2.18.1 release introduces :ref:`Continuous batching(beta) <transformers_neuronx_developer_guide_for_cb>` and Neuron vLLM integration(beta) support in Transformers NeuronX library that improves LLM inference throughput. This release also fixes hang issues related to Triton Inference Server as well as updating Neuron DLAMIs and DLCs with this release(2.18.1). 
See more in  :ref:`transformers-neuronx-rn` and :ref:`neuronx-cc-rn` 



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
     - * [Beta] Support for Speculative Decoding API. See :ref:`developer guide <transformers_neuronx_developer_guide>` and  `Llama-2-70B sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/speculative_sampling.ipynb>`_ 
       * Support for SafeTensors checkpoint format with improved weight loading performance.  See :ref:`developer guide <transformers_neuronx_developer_guide>` 
       * Support for running  Top-K sampling on Neuron Device for improved performance.  See :ref:`developer guide <transformers_neuronx_developer_guide>` 
       * Code Llama model inference sample with 16K input seq length. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/codellama-13b-16k-sampling.ipynb>`_
       * [Beta] Support for streaming API and stopping criteria API. See :ref:`developer guide <transformers_neuronx_developer_guide>`
       * Support for ``Mixtral-8x7B-v0.1`` model inference. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/mixtral-8x7b-sampling.ipynb>`_
       * [Beta] Support for ``mistralai/Mistral-7B-Instruct-v0.2`` model inference. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/mistralai-Mistral-7b-Instruct-v0.2.ipynb>`_
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Training
     - * Support for Pipeline Parallelism training using PyTorch Lightning. See :ref:`api guide <api_guide>` , :ref:`developer guide <ptl_developer_guide>` and :ref:`tutorial <llama2_tp_pp_ptl_tutorial>`
       * Support for auto partitioning pipeline parallel stages when training large models.  See :ref:`api guide <api_guide>` and :ref:`pp_developer_guide`
       * Support for asynchronous checkpointing to improve the time it takes to save the checkpoint.  See :ref:`api guide <api_guide>` , :ref:`save_load_developer_guide` and :ref:`llama2_tp_pp_tutorial`
       * Tutorial to fine-tune Llama-2-7B model using PyTorch Lightning and running evaluation on the fine-tuned model using Hugging Face optimum-neuron. See :ref:`tutorial <llama2_7b_tp_zero1_ptl_finetune_tutorial>`
       * ``codegen25-7b-mono`` model training tutorial and script. See :ref:`codegen25_7b_tp_zero1_tutorial` 
       * See more at :ref:`neuronx-distributed-rn` 
     - Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Inference
     - * Support for auto bucketing in inference using a custom bucket kernel that can be passed as a bucket configuration to Tracing API. See :ref:`api guide <api_guide>` and :ref:`neuronx_distributed_inference_developer_guide`
       * Support for inference with bf16 data type using XLA_USE_BF16=1 flag. See sample (:ref:`[html] </src/examples/pytorch/neuronx_distributed/llama/llama2_inference.ipynb>` :pytorch-neuron-src:`[notebook] <neuronx_distributed/llama/llama2_inference.ipynb>`)
       * See more at :ref:`neuronx-distributed-rn` 
     - Inf2,Trn1/Trn1n

   * - PyTorch NeuronX (torch-neuronx)
     - * PyTorch 2.1 support is now stable (out of beta).  See updated :ref:`App Note <introduce-pytorch-2-1>` and :ref:`release notes <torch-neuronx-rn>` for known issues.
       * Support for auto bucketing in inference using a custom bucket kernel that can be passed as a bucket configuration to Tracing API. See :ref:`torch-neuronx-autobucketing-devguide`
       * See more at :ref:`torch-neuronx-rn`
     - Trn1/Trn1n,Inf2

   * - NeuronX Nemo Megatron for Training
     - * Support for LoRa finetuning. See `sample script <https://github.com/aws-neuron/neuronx-nemo-megatron/tree/main/nemo/examples/nlp/language_modeling/test_llama_lora.sh>`_
       * Support for Mistral-7B training. See `sample script <https://github.com/aws-neuron/neuronx-nemo-megatron/tree/main/nemo/examples/nlp/language_modeling/test_mistral.sh>`_
       * Support for asynchronous checkpointing to improve the time it takes to save the checkpoint.
       * See more at `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_  and  :ref:`neuronx-nemo-rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * New ``--enable-mixed-precision-accumulation`` compiler option to perform intermediate computations of an operation in FP32 regardless of the operation's defined datatype. See :ref:`neuron-compiler-cli-reference-guide`
       * See more at :ref:`neuronx-cc-rn`
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
       * See more at :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1
   
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

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

To learn about the model architectures currently supported on Inf1, Inf2, Trn1 and Trn1n instances, please see :ref:`model_architecture_fit`.





.. _neuron-2.16.0-whatsnew:



Neuron 2.16.1 (01/18/2024)
--------------------------
Patch release with compiler bug fixes, updates to :ref:`Neuron Device Plugin and Neuron Kubernetes Scheduler <neuron-k8-rn>` .


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
     - * [Beta] Support for Grouped Query Attention(GQA). See :ref:`developer guide <transformers_neuronx_developer_guide>` 
       * [Beta] Support for ``Llama-2-70b`` model inference using ``Grouped Query Attention``. See `tutorial <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-70b-sampling.ipynb>`_ 
       * [Beta] Support for ``Mistral-7B-Instruct-v0.1`` model inference. See :ref:`sample code <mistral_gqa_code_sample>`
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Training
     - * [Beta] Support for ``PyTorch Lightning``  to train models using ``tensor parallelism`` and ``data parallelism`` . See :ref:`api guide <api_guide>` , :ref:`developer guide <ptl_developer_guide>` and :ref:`tutorial <llama2_7b_tp_zero1_ptl_tutorial>`
       * Support for Model and Optimizer Wrapper training API that handles the parallelization. See :ref:`api guide <api_guide>` and :ref:`model_optimizer_wrapper_developer_guide`
       * New ``save_checkpoint``  and ``load_checkpoint`` APIs to save/load checkpoints during distributed training. See :ref:`save_load_developer_guide`
       * Support for a new ``Query-Key-Value(QKV)`` module that provides the ability to replicate the Key Value heads and adds flexibility to use higher Tensor parallel degree during Training. See :ref:`api guide <api_guide>` and :ref:`tutorial <llama2_tp_pp_tutorial>`
       * See more at :ref:`neuronx-distributed-rn` 
     - Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Inference
     - * Support weight-deduplication amongst TP shards by giving ability to save weights separately than in NEFF files.  See :ref:`developer guide<nxd_inference_developer_guide>`
       * ``Llama-2-7B`` model inference script (:ref:`[html] </src/examples/pytorch/neuronx_distributed/llama/llama2_inference.ipynb>` :pytorch-neuron-src:`[notebook] <neuronx_distributed/llama/llama2_inference.ipynb>`)
       * See more at :ref:`neuronx-distributed-rn` and  :ref:`api_guide`
     - Inf2,Trn1/Trn1n

   * - PyTorch NeuronX (torch-neuronx)
     - * [Beta]Support for] ``PyTorch 2.1``. See :ref:`introduce-pytorch-2-1` . See  `llama-2-13b inference <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb>`_ sample.
       * Support to separate out model weights from NEFF files and new ``replace_weights`` API to replace the separated weights. See :ref:`torch_neuronx_replace_weights_api` and :ref:`torch_neuronx_trace_api`
       * [Beta] Script for training ``stabilityai/stable-diffusion-2-1-base`` and  ``runwayml/stable-diffusion-v1-5`` models . See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/stable_diffusion/>`_ 
       * [Beta] Script for training ``facebook/bart-large`` model. See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_summarization/BartLarge.ipynb>`_ 
       * [Beta] Script for ``stabilityai/stable-diffusion-2-inpainting`` model inference.  See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sd2_inpainting_936_624_inference.ipynb>`_ 
     - Trn1/Trn1n,Inf2

   * - Neuron Tools
     - * New ``Neuron Distributed Event Tracing (NDET) tool`` to help visualize execution trace logs and diagnose errors in multi-node workloads. See :ref:`neuron-det-ug` 
       * Support for multi-worker jobs in ``neuron-profile`` . See :ref:`neuron-profile-ug`
       * See more at :ref:`neuron-tools-rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Documentation Updates
     - * Added setup guide instructions for ``AL2023`` OS. See :ref:`setup-guide-index`
       * Added announcement for name change of Neuron Components. See :ref:`announce-component-name-change`
       * Added announcement for End of Support for ``PyTorch 1.10`` . See :ref:`announce-eos_pytorch110`
       * Added announcement for End of Support for ``PyTorch 2.0`` Beta. See :ref:`announce-eos_pytorch2`
       * See more at :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1
   
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
       * ``Llama-2-70B`` model training script  (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/llama2/tp_pp_llama2_70b_hf_pretrain>`_) (:ref:`tutorial <llama2_70b_tp_pp_tutorial>`)
       * Mixed precision support. See :ref:`pp_developer_guide`
       * Support serialized checkpoint saving and loading using ``save_xser`` and ``load_xser`` parameters. See :ref:`api_guide` 
       * See more at :ref:`neuronx-distributed-rn` 
     - Trn1/Trn1n

   * - Neuron Distributed (neuronx-distributed) for Inference
     - * ``flan-t5-xl`` model inference script (:pytorch-neuron-src:`tutorial <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`)
       * See more at :ref:`neuronx-distributed-rn` and  :ref:`api_guide`
     - Inf2,Trn1/Trn1n

   * - Transformers Neuron (transformers-neuronx) for Inference
     - * Serialization support for ``Llama``, ``Llama-2``, ``GPT2`` and ``BLOOM`` models . See :ref:`developer guide <transformers_neuronx_developer_guide>` and `tutorial <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb>`_
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n

   * - PyTorch Neuron (torch-neuronx)
     - * Introducing ``PyTorch 2.0`` Beta support. See :ref:`introduce-pytorch-2-0` . See  :ref:`llama-2-7b training <llama2_7b_tp_zero1_tutorial>` , `bert training <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/dp_bert_hf_pretrain>`_ and  `t5-3b inference <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/neuronx_distributed/t5-inference/t5-inference-tutorial.html>`_ samples.
       * Scripts for training `resnet50[Beta] <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/resnet50>`_ ,
         `milesial/Pytorch-UNet[Beta] <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/unet_image_segmentation>`_ and `deepmind/vision-perceiver-conv[Beta] <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_image_classification/VisionPerceiverConv.ipynb>`_ models.
     - Trn1/Trn1n,Inf2

   * - AWS Neuron Reference for Nemo Megatron library (``neuronx-nemo-megatron``)
     - * ``Llama-2-70B`` model training sample using pipeline parallelism and tensor parallelism ( `tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`_ )
       * ``GPT-NeoX-20B`` model training using pipeline parallelism and tensor parallelism 
       * See more at :ref:`neuronx-nemo-rn` and `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - Trn1/Trn1n

   * - Neuron Compiler (neuronx-cc)
     - * New ``llm-training`` option argument to ``--distribution_strategy`` compiler option for optimizations related to distributed training. See more at :ref:`neuron-compiler-cli-reference-guide`
       * See more at :ref:`neuronx-cc-rn`
     - Inf2/Trn1/Trn1n

   * - Neuron Tools
     - * ``alltoall`` Collective Communication operation for intra node(with in the instance), previously released in Neuron Collectives v2.15.13, was added as a testable operation in ``nccom-test``. See :ref:`nccom-test`
       * See more at :ref:`neuron-tools-rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Documentation Updates
     - * New :ref:`App Note <activation_memory_reduction>` and :ref:`Developer Guide <activation_memory_reduction_developer_guide>` about Activation memory reduction using ``sequence parallelism`` and ``activation recomputation`` in ``neuronx-distributed``
       * Added a new Model Samples and Tutorials summary page. See :ref:`model_samples_tutorials`
       * Added Neuron SDK Classification guide. See :ref:`sdk-classification`
       * See more at :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1
   
   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

To learn about the model architectures currently supported on Inf1, Inf2, Trn1 and Trn1n instances, please see :ref:`model_architecture_fit`.






.. _neuron-2.14.0-whatsnew:


Neuron 2.14.1 (09/26/2023)
--------------------------

This is a patch release that fixes compiler issues in certain configurations of ``Llama`` and ``Llama-2`` model inference using ``transformers-neuronx``.

.. note::

   There is still a known compiler issue for inference of some configurations of ``Llama`` and ``Llama-2`` models that will be addressed in future Neuron release.
   Customers are advised to use ``--optlevel 1 (or -O1)`` compiler flag to mitigate this known compiler issue.  
    
   See :ref:`neuron-compiler-cli-reference-guide` on the usage of ``--optlevel 1`` compiler flag. Please see more on the compiler fix and known issues in :ref:`neuronx-cc-rn` and :ref:`transformers-neuronx-rn` 
   



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
     - * ``Llama-2-13B`` model training support ( `tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`_ )
       * ZeRO-1 Optimizer support  that works with tensor parallelism and pipeline parallelism
       * See more at :ref:`neuronx-nemo-rn` and `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - Trn1/Trn1n
   
   * - Neuron Distributed (neuronx-distributed) for Training
     - * ``pad_model`` API to pad attention heads that do not divide by the number of NeuronCores, this will allow users to use any supported tensor-parallel degree. See  :ref:`api_guide`
       * ``Llama-2-7B`` model training support  (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/tp_zero1_llama2_7b_hf_pretrain>`_) (:ref:`tutorial <llama2_7b_tp_zero1_tutorial>`)
       * See more at :ref:`neuronx-distributed-rn` and  :ref:`api_guide`
     - Trn1/Trn1n

   * - Neuron Distributed (neuronx-distributed) for Inference
     - * ``T5-3B`` model inference support (:pytorch-neuron-src:`tutorial <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`)
       * ``pad_model`` API to pad attention heads that do not divide by the number of NeuronCores, this will allow users to use any supported tensor-parallel degree. See  :ref:`api_guide` 
       * See more at :ref:`neuronx-distributed-rn` and  :ref:`api_guide`
     - Inf2,Trn1/Trn1n

   * - Transformers Neuron (transformers-neuronx) for Inference
     - * Introducing ``--model-type=transformer`` compiler flag that deprecates ``--model-type=transformer-inference`` compiler flag. 
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n

   * - PyTorch Neuron (torch-neuronx)
     - * Performance optimizations in ``torch_neuronx.analyze`` API. See :ref:`torch_neuronx_analyze_api`
       * ``Stable Diffusion XL(Refiner and Base)`` model inference support  ( `sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sdxl_base_and_refiner_1024_inference.ipynb>`_)
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * New  ``--optlevel``(or ``-O``) compiler option that enables different optimizations with tradeoff between faster model compile time and faster model execution. See more at :ref:`neuron-compiler-cli-reference-guide`
       * See more at :ref:`neuronx-cc-rn`
     - Inf2/Trn1/Trn1n

   * - Neuron Tools
     - * Neuron SysFS support for showing connected devices on ``trn1.32xl``, ``inf2.24xl`` and ``inf2.48xl`` instances. See :ref:`neuron-sysfs-ug`
       * See more at :ref:`neuron-tools-rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Documentation Updates
     - * Neuron Calculator now supports multiple model configurations for Tensor Parallel Degree computation. See :ref:`neuron_calculator`
       * Announcement to deprecate ``--model-type=transformer-inference`` flag. See :ref:`announce-deprecation-transformer-flag`
       * See more at :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1
   
   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

To learn about the model architectures currently supported on Inf1, Inf2, Trn1 and Trn1n instances, please see :ref:`model_architecture_fit`.


.. _neuron-2.13.0-whatsnew:

Neuron 2.13.2 (09/01/2023)
---------------------------

This is a patch release that fixes issues in Kubernetes (K8) deployments related to Neuron Device Plugin crashes and other pod scheduling issues. This release also adds support for zero-based Neuron Device indexing in K8 deployments, see the :ref:`Neuron K8 release notes <neuron-k8-rn>` for more details on the specific bug fixes.

Updating to latest Neuron Kubernetes components and Neuron Driver is highly encouraged for customers using Kubernetes.

Please :ref:`follow these instructions in setup guide <setup-guide-index>` to upgrade to latest Neuron release.


Neuron 2.13.1 (08/29/2023)
--------------------------
This release adds support for ``Llama 2`` model training (`tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`_) using `neuronx-nemo-megatron <https://github.com/aws-neuron/neuronx-nemo-megatron>`_ library, and adds support for ``Llama 2`` model inference using ``transformers-neuronx`` library (`tutorial <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb>`_) . 

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
       * ``GPT-3`` model training support ( `tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-gpt-job.md>`_ )
       * See more at `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - Trn1/Trn1n

   * - Transformers Neuron (transformers-neuronx) for Inference
     - * Latency optimizations for  ``Llama`` and ``GPT-2`` models inference.
       * Neuron Persistent Cache support (:ref:`developer guide <transformers_neuronx_developer_guide>`)
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n
   
   * - Neuron Distributed (neuronx-distributed) for Training
     - * Now Stable, removed beta support
       * ZeRO-1 Optimizer support with tensor parallel. (:ref:`tutorial <gpt_neox_tp_zero1_tutorial>`)
       * Sequence Parallel support. (:ref:`api guide <api_guide>`)
       * GPT-NeoX model training support. (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training>`_) (:ref:`tutorial <gpt_neox_tp_zero1_tutorial>`)
       * See more at :ref:`neuronx-distributed-rn` and  :ref:`api_guide`
     - Trn1/Trn1n

   * - Neuron Distributed (neuronx-distributed) for Inference
     - * KV Cache Support for LLM Inference (:ref:`release notes <neuronx-distributed-rn>`)
     - Inf2,Trn1/Trn1n


   * - PyTorch Neuron (torch-neuronx)
     - * Seedable dropout enabled by default for training
       * KV Cache inference support ( :pytorch-neuron-src:`tutorial <torch-neuronx/t5-inference-tutorial.ipynb>` )
       * ``camembert-base`` training script. (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_text_classification/CamembertBase.ipynb>`_)
       * New models inference support that include `Stable Diffusion XL <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sdxl_1024_inference.ipynb>`_ , CLIP (`clip-vit-base-patch32 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_clip_base_inference_on_inf2.ipynb>`_ , `clip-vit-large-patch14 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_clip_large_inference_on_inf2.ipynb>`_ ) , `Vision Perceiver <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_perceiver_vision_inference.ipynb>`_ , `Language Perceiver <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_perceiver_language_inference.ipynb>`_ and :pytorch-neuron-src:`T5 <torch-neuronx/t5-inference-tutorial.ipynb>`
     - Trn1/Trn1n,Inf2


   * - Neuron Tools
     - * New data types support for Neuron Collective Communication Test Utility (NCCOM-TEST)  --check option: fp16, bf16, (u)int8, (u)int16, and (u)int32 
       * Neuron SysFS support for FLOP count(flop_count) and connected Neuron Device ids (connected_devices).  See :ref:`neuron-sysfs-ug`
       * See more at :ref:`neuron-tools-rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Neuron Runtime 
     - * Runtime version and Capture Time support to NTFF
       * Async DMA copies support to improve Neuron Device copy times for all instance types
       * Logging and error messages improvements for Collectives timeouts and when loading NEFFs.
       * See more at :ref:`neuron-runtime-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - End of Support Announcements and Documentation Updates 
     - * Announcing End of support for ``AWS Neuron reference for Megatron-LM`` starting Neuron 2.13. See more at :ref:`announce-eol-megatronlm`
       * Announcing end of support for ``torch-neuron`` version 1.9 starting Neuron 2.14. See more at :ref:`announce-eol-pytorch19`
       * Added TensorFlow 2.x (``tensorflow-neuronx``) analyze_model API section. See more at :ref:`tensorflow-ref-neuron-analyze_model-api`
       * Upgraded ``numpy`` version to ``1.21.6`` in various training scripts for `Text Classification <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training>`_
       * Updated ``bert-japanese`` training Script to use ``multilingual-sentiments`` dataset. See `hf-bert-jp <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_bert_jp> `_
       * See more at :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1
   
   * - Known Issues and Limitations
     - * See :ref:`neuron-2.13.0-known-issues`
     - Trn1/Trn1n , Inf2, Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

To learn about the model architectures currently supported on Inf1, Inf2, Trn1 and Trn1n instances, please see :ref:`model_architecture_fit`.

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
       * Removed constraints on ``tp_degree`` in tensor-parallel configurations for ``GPT2``, ``OPT``, and ``BLOOM`` . See more at :ref:`transformers-neuronx-rn`
       * Added multi-query / multi-group attention support for ``GPT2``.
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n
   
   * - Support for Inf2 and Trn1 instances on Triton Inference Server
     - * Support for Model Inference serving on Triton for Inf2 and Trn1 instances. See more at `Triton Server Python Backend <https://github.com/triton-inference-server/python_backend/tree/main/inferentia#using-triton-with-inferentia-2-or-trn1>`_
       * See tutorial at `Triton on SageMaker - Deploying on Inf2 <https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-triton/inferentia2>`_
     - Inf2, Trn1

   * - Support for new computer vision models 
     - * Performance optimizations in Stable Diffusion 2.1 model script and added [beta] support for Stable Diffusion 1.5 models.
       * [Beta] Script for training CLIP model for Image Classification.
       * [Beta] Script for inference of Multimodal perceiver model
       * Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_
     - Inf2, Trn1/Trn1n

   * - New Features in ``neuronx-distributed`` for training
     - * Added parallel cross entropy loss function.
       * See more at :ref:`tp_api_guide`
     - Trn1/Trn1n

   * - ``lazy_load`` and ``async_load`` API for model loading in inference and performance enhancements in ``torch-neuronx`` 
     - * Added ``lazy_load`` and ``async_load`` API to accelerate model loading for Inference. See more at :ref:`torch_neuronx_lazy_async_load_api`
       * Optimize DataParallel API to load onto multiple cores simultaneously when device IDs specified are consecutive.
       * See more at :ref:`torch-neuronx-rn`
     - Inf2, Trn1/Trn1n
  
   * - [Beta] Asynchronous Execution support and Enhancements in Neuron Runtime 
     - * Added beta asynchronous execution feature which can reduce latency by roughly 12% for training workloads. See more at :ref:`nrt-configuration`
       * AllReduce with All-to-all communication pattern enabled for 16 ranks on TRN1/TRN1N within the instance (intranode)
       * See more at :ref:`neuron-runtime-rn`
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
       * See more at :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1
   
   * - Known Issues and Limitations
     - * See :ref:`neuron-2.12.0-known-issues`
     - Trn1/Trn1n , Inf2, Inf1
  
   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

To learn about the model architectures currently supported on Inf1, Inf2, Trn1 and Trn1n instances, please see :ref:`model_architecture_fit`.

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
-------------------------

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
       * See more at :ref:`transformers-neuronx-rn` 
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
       * See more at :ref:`neuronx-distributed-index`
     - Inf2, Trn1/Trn1n

   * - Neuron Calculator and Documentation Updates  
     - * New :ref:`neuron_calculator` Documentation section to help determine number of Neuron Cores needed for LLM Inference.
       * Added App Note :ref:`neuron_llm_inference`
       * See more at :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n

   * - Enhancements to Neuron SysFS
     - * Support for detailed breakdown of memory usage across the NeuronCores
       * See more at :ref:`neuron-sysfs-ug`
     - Inf1, Inf2, Trn1/Trn1n

   * - Support for Ubuntu 22
     - * See more at :ref:`setup-guide-index` for setup instructions on Ubuntu22
     - Inf1, Inf2, Trn1/Trn1n

   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

To learn about the model architectures currently supported on Inf1, Inf2, Trn1 and Trn1n instances, please see :ref:`model_architecture_fit`.


.. _neuron-2.10.0-whatsnew:

Neuron 2.10.0 (05/01/2023)
-------------------------

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
       * Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_
     - Inf2, Trn1/Trn1n

   * - Profiling support in PyTorch Neuron(``torch-neuronx``) for Inference with TensorBoard
     - * See more at :ref:`torch-neuronx-profiling-with-tb`
     - Inf2, Trn1/Trn1n
  
   * - New Features and Performance Enhancements in transformers-neuronx
     - * Support for the HuggingFace generate function. 
       * Model Serialization support for GPT2 models. (including model saving, loading, and weight swapping)
       * Improved prompt context encoding performance.
       * See :ref:`transformers_neuronx_readme` for examples and usage
       * See more at :ref:`transformers-neuronx-rn` 
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
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

To learn about the model architectures currently supported on Inf1, Inf2, Trn1 and Trn1n instances, please see :ref:`model_architecture_fit`.



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

   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1, Inf2, Inf1

   * - Release included packages
     - * see :ref:`neuron-release-content`
     - Trn1, Inf2, Inf1

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

To learn about the model architectures currently supported on Inf1, Inf2, Trn1 and Trn1n instances, please see :ref:`model_architecture_fit`.




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
     - * New sample scripts for deploying LLM models with ``transformer-neuronx`` under       `aws-neuron-samples <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference>`_  GitHub repository.
      
       * New sample scripts for deploying models with ``torch-neuronx`` under `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_  GitHub repository.

   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`

   * - Release included packages
     - * see :ref:`neuron-release-content`

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.


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
     - Support of PyTorch DistributedDataParallel (DDP) API in PyTorch Neuron (``torch-neuronx``). For resources how to use PyTorch DDP API with Neuron, please check :ref:`neuronx-ddp-tutorial`.

   * - Inference support in ``torch-neuronx``
     - For more details please visit :ref:`pytorch-neuronx-main`` page. You can also try Neuron Inference samples `<https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_ in the ``aws-neuron-samples`` GitHub repo.     

   * - Neuron Custom C++ Operators[Beta]
     - Initial support for Neuron Custom C++ Operators [Beta] , with Neuron Custom C++ Operators (“CustomOps”) you can now write CustomOps that run on NeuronCore-v2 chips. For more resources please check :ref:`neuron_c++customops` section.


   * - ``transformers-neuronx`` [Beta] 
     - ``transformers-neuronx``  is a new library enabling LLM model inference. It contains models that are checkpoint-compatible with HuggingFace Transformers, and currently supports Transformer Decoder models like GPT2, GPT-J and OPT. Please check `aws-neuron-samples repository <https://github.com/aws-neuron/transformers-neuronx>`_  


   * - Neuron sysfs filesystem
     - Neuron sysfs filesystem exposes Neuron Devices under ``/sys/devices/virtual/neuron_device`` providing visibility to Neuron Driver and Runtime at the system level. By performing several simple CLIs such as reading or writing to a sysfs file, you can get information such as Neuron Runtime status, memory usage, Driver info etc. For resources about Neuron sysfs filesystem visit :ref:`neuron-sysfs-ug`.


   * - TFLOPS support in Neuron System Tools
     - Neuron System Tools now also report model actual TFLOPs rate in both ``neuron-monitor`` and ``neuron-top``. More details can be found in the :ref:`Neuron Tools documentation <neuron-tools>`.

   * - New sample scripts for training
     - This release adds multiple new sample scripts for training models with ``torch-neuronx``, Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_

   * - New sample scripts for inference
     - This release adds multiple new sample scripts for deploying models with ``torch-neuronx``, Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_

   * - Neuron GitHub samples repository for Amazon EKS
     - A new AWS Neuron GitHub samples repository for Amazon EKS, Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-eks-samples>`_


For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.



.. _neuron-2.6.0-whatsnew:

Neuron 2.6.0 (12/12/2022)
-------------------------

This release introduces the support of PyTorch 1.12 version, and introduces PyTorch Neuron (``torch-neuronx``) profiling through Neuron Plugin for TensorBoard. Pytorch Neuron (``torch-neuronx``) users can now profile their models through the following TensorBoard views:

* Operator Framework View
* Operator HLO View
* Operator Trace View

This release introduces the support of LAMB optimizer for FP32 mode, and adds support for :ref:`capturing snapshots <torch-neuronx-snapshotting>` of inputs, outputs and graph HLO for debugging.

In addition, this release introduces the support of new operators and resolves issues that improve stability for Trn1 customers.

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

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

       * Support for new :ref:`API for core placement <torch_neuron_core_placement_api>`
      
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

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.


.. _neuron-2.4.0-whatsnew:

Neuron 2.4.0 (10/27/2022)
-------------------------

This release introduces new features and resolves issues that improve stability. The release introduces "memory utilization breakdown" feature in both :ref:`Neuron Monitor <neuron-monitor-ug>` and :ref:`Neuron Top <neuron-top-ug>` system tools. The release introduces support for "NeuronCore Based Sheduling" capability to the Neuron Kubernetes Scheduler and introduces new operators support in :ref:`Neuron Compiler <neuronx-cc>` and :ref:`PyTorch Neuron <torch-neuronx-rn>`. This release introduces also additional eight (8) samples of models' fine tuning using PyTorch Neuron. The new samples can be found in the `AWS Neuron Samples GitHub <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_ repository.


.. _neuron-2.3.0-whatsnew:

Neuron 2.3.0 (10/10/2022)
-------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

Overview
~~~~~~~~

This Neuron 2.3.0 release extends Neuron 1.x and adds support for the new AWS Trainium powered Amazon EC2 Trn1 instances. With this release, you can now run deep learning training workloads on Trn1 instances to save training costs by up to 50% over equivalent GPU-based EC2 instances, while getting the highest training performance in AWS cloud for popular NLP models.


.. list-table::
   :widths: auto
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - * :ref:`rn2.3.0_new`
       * :ref:`neuron-packages-changes`
       * :ref:`announce-aws-neuron-github-org`
       * :ref:`announce-neuron-rtd-eol`

   * - Tested workloads and known issues
     - * :ref:`rn2.3.0_tested`
       * :ref:`rn2.3.0-known-issues` 

.. _rn2.3.0_new:

New features and capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: /release-notes/templates/n2.x-trn1ga-whats-new.txt

.. _rn2.3.0_tested:

Tested Workloads
~~~~~~~~~~~~~~~~

The following workloads were tested in this release:

* Distributed data-parallel pre-training of Hugging Face BERT model on single Trn1.32xl instance (32 NeuronCores).
* Distributed data-parallel pre-training of Hugging Face BERT model on multiple Trn1.32xl instances.
* HuggingFace BERT MRPC task finetuning on single NeuronCore or multiple NeuronCores (data-parallel).
* Megatron-LM GPT3 (6.7B parameters) pre-training on single Trn1.32xl instance. 
* Megatron-LM GPT3 (6.7B parameters) pre-training on multi Trn1.32xl instances. 
* Multi-Layer Perceptron (ML) model training on single NeuronCore or multiple NeuronCores (data-parallel).

.. _rn2.3.0-known-issues:

Known Issues
~~~~~~~~~~~~

* For maximum training performance, please set environment variables ``XLA_USE_BF16=1`` to enable full BF16 and Stochastic Rounding.
