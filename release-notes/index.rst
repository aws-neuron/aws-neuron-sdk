.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.19.0-whatsnew:

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


.. _latest-neuron-release-artifacts:

Release Artifacts
-------------------

.. contents:: Table of contents
   :local:
   :depth: 1

Trn1 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.19.0

Inf2 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.19.0

Inf1 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.19.0

Supported Python Versions for Inf1 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.19.0

Supported Python Versions for Inf2/Trn1 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.19.0

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