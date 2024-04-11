.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.18.0-whatsnew:


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


.. _latest-neuron-release-artifacts:

Release Artifacts
-------------------

.. contents:: Table of contents
   :local:
   :depth: 1

Trn1 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.18.1

Inf2 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.18.1

Inf1 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.18.1

Supported Python Versions for Inf1 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.18.1

Supported Python Versions for Inf2/Trn1 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.18.1

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
