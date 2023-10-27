.. _prev-rn:

Previous Releases Notes (Neuron 2.x)
====================================

.. contents:: Table of contents
   :local:
   :depth: 1


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
     - * Now Stable, removed Experimental support
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

This release introduces  ZeRO-1 optimizer for model training in ``torch-neuronx`` , introduces experimental support for ``GPT-NeoX``, ``BLOOM`` , ``Llama`` and ``Llama 2(coming soon)`` models in ``transformers-neuronx``. This release also adds support for model inference serving on Triton Inference Server for Inf2 & Trn1 instances, ``lazy_load`` API and ``async_load`` API for model loading in ``torch-neuronx``, as well as other new features,
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
     - * [Experimental] Support for inference of ``GPT-NeoX``, ``BLOOM`` and ``Llama`` models. 
       * [Experimental] Support for ``Llama 2`` coming soon. Please monitor the `transformers-neuronx repository <https://github.com/aws-neuron/transformers-neuronx/tree/main/src/transformers_neuronx>`_ for updates.
       * Removed constraints on ``tp_degree`` in tensor-parallel configurations for ``GPT2``, ``OPT``, and ``BLOOM`` . See more at :ref:`transformers-neuronx-rn`
       * Added multi-query / multi-group attention support for ``GPT2``.
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n
   
   * - Support for Inf2 and Trn1 instances on Triton Inference Server
     - * Support for Model Inference serving on Triton for Inf2 and Trn1 instances. See more at `Triton Server Python Backend <https://github.com/triton-inference-server/python_backend/tree/main/inferentia#using-triton-with-inferentia-2-or-trn1>`_
       * See tutorial at `Triton on SageMaker - Deploying on Inf2 <https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-triton/inferentia2>`_
     - Inf2, Trn1

   * - Support for new computer vision models 
     - * Performance optimizations in Stable Diffusion 2.1 model script and added [experimental] support for Stable Diffusion 1.5 models.
       * [Experimental] Script for training CLIP model for Image Classification.
       * [Experimental] Script for inference of Multimodal perceiver model
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
  
   * - [Experimental]Asynchronous Execution support and Enhancements in Neuron Runtime 
     - * Added experimental asynchronous execution feature which can reduce latency by roughly 12% for training workloads. See more at :ref:`nrt-configuration`
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

   * - Neuron Distributed Library [Experimental]
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
  
   * - Performance Enhancements in PyTorch C++ Custom Operators (Experimental)
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

   * - Neuron Custom C++ Operators[Experimental]
     - Initial support for Neuron Custom C++ Operators [Experimental] , with Neuron Custom C++ Operators (“CustomOps”) you can now write CustomOps that run on NeuronCore-v2 chips. For more resources please check :ref:`neuron_c++customops` section.


   * - ``transformers-neuronx`` [Experimental] 
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
      
       * Experimental support for tracing models larger than 2GB using ``extract-weights`` flag (TF2.x only), see :ref:`tensorflow-ref-neuron-tracing-api`

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
