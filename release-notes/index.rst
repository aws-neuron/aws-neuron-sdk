.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:

.. _neuron2x-trn1ga:

Neuron 2.3.0 (10/10/2022)
----------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

Overview
~~~~~~~~

This Neuron 2.x release extends Neuron 1.x and adds support for the new AWS Trainium powered Amazon EC2 Trn1 instances. With this release, you can now run deep learning training workloads on Trn1 instances to save training costs by up to 50% over equivalent GPU-based EC2 instances, while getting the highest training performance in AWS cloud for popular NLP models.


.. include:: /release-notes/templates/n2.x-trn1-ga-quick.txt


.. _rn2.3.0_new:

New features and capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: /release-notes/templates/n2.x-trn1ga-whats-new.txt

.. _main-rn-tested:

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




.. _components-rn:

Neuron Components Release Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inf1 and Trn1 common packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size


   * - Component
     - Instance/s
     - Package/s
     - Details

   * - Neuron Runtime Driver
     - Trn1, Inf1
     - * ``aws-neuronx-dkms``  (.deb, .rpm)
       
     - * :ref:`neuron-driver-release-notes`


   * - Containers
     - Trn1, Inf1
     - * ``aws-neuronx-k8-plugin`` (.deb, .rpm)

       * ``aws-neuronx-k8-scheduler`` (.deb, .rpm)
       
       * ``aws-neuronx-oci-hooks`` (.deb, .rpm)

     - * :ref:`neuron-k8-rn`

       * :ref:`neuron-containers-release-notes`


Trn1 only packages
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


   * - Neuron Runtime
     - Trn1
     - * ``aws-neuronx-runtime-lib`` (.deb, .rpm)
       
     - * :ref:`neuron-runtime-rn`
     

   * - PyTorch Neuron
     - Trn1
     - * ``torch-neuronx`` (.whl)
     - * :ref:`torch-neuronx-rn`

       * :ref:`pytorch-neuron-supported-operators`
       

   * - Neuron Compiler (Trn1 only)
     - Trn1
     - * ``neuronx-cc`` (.whl)
     - * :ref:`neuronx-cc-rn`

   * - Collective Communication library
     - Trn1
       
     - * ``aws-neuronx-collective`` (.deb, .rpm)

     - * :ref:`neuron-collectives-rn`

   * - Neuron System Tools
     - Trn1
     - * ``aws-neuronx-tools``  (.deb, .rpm)
     - * :ref:`neuron-tools-rn`


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


   * - TensorFlow Model Server Neuron
     - Inf1
     - * ``tensorflow-model-server-neuron`` (.deb, .rpm)
     - * :ref:`tensorflow-modelserver-rn`


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

   * - Neuron System Tools
     - Inf1
     - * ``aws-neuron-tools``  (.deb, .rpm)
     - * :ref:`neuron-tools-rn`


   * - NeuronPerf
     - Inf1
     - * ``neuronperf`` (.whl)
     - * :ref:`neuronperf_rn`





Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`prev-n1-rn`

