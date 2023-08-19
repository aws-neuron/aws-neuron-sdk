.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
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



   * - Apache MXNet (Incubating)
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
^^^^^^^^^^^^^^^^^

Trn1 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.12.2

Inf2 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.12.2

Inf1 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.12.2


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`

