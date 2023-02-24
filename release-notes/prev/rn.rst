.. _prev-rn:

Previous Releases Notes (Neuron 2.x)
====================================

.. contents:: Table of contents
   :local:
   :depth: 1


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
