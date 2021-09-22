.. _neuron-whatsnew:

What's New
==========

.. contents::
   :local:
   :depth: 2

Neuron 1.15.2 (09/22/2021)
--------------------------

Neuron 1.15.2 includes bug fixes for the tensorflow-model-server-neuron 2.5.1.1.6.8.0 package and several other bug fixes for tensorflow-neuron/tensorflow-model-server-neuron packages.


Neuron 1.15.1 (08/30/2021)
--------------------------

Neuron 1.15.1 includes bug fixes for the aws-neuron-dkms package and several other bug fixes for related packages.

Neuron 1.15.0 (08/12/2021)
--------------------------

Neuron 1.15.0 is the first release to support TensorFlow 2. In this release TensorFlow 2 supports language transformer base models like BERT. The TensorFlow 2 support will be enhanced in future releases to support additional models.

* **TensorFlow 2.x** - To get started with TensorFlow 2.x:

  *  Run the TensorFlow 2  :ref:`HuggingFace distilBERT Tutorial </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>`.
  *  Read :ref:`tf2_faq`
  *  See newly introduced :ref:`TensorFlow-Neuron 2.x Tracing API <tensorflow-ref-neuron-tracing-api>`.
  *  See :ref:`tensorflow-ref-neuron-accelerated-ops`.


* **Documentation**

  *  **New** :ref:`models-inferentia` application note added in this release. This application note describes what types of deep learning model architectures perform well out of the box and provides guidance on techniques you can use to optimize your deep learning models for Inferentia.
  *  **New** :ref:`Neuron inference performance page <appnote-performance-benchmark>` provides performance information for popular models and links to test these models in your own environment. The data includes throughout and latency numbers, cost per inference, for both realtime and offline applications.
  *  **New** :ref:`TensorFlow 2 HuggingFace distilBERT Tutorial </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>`.
  *  **New** :ref:`Bring your own HuggingFace pretrained BERT container to Sagemaker Tutorial </src/examples/pytorch/byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb>`.



* **More information**

  *  :ref:`tensorflow-neuron-rn`
  *  :ref:`neuron-cc-rn`
  *  :ref:`tensorflow-modelserver-rn`



   

Detailed release notes
----------------------

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   
   * - Software 
     - Details
      
   * - General     
      
     - * :ref:`neuron-release-content`
      
       * :ref:`software-deprecation`
      
   * - PyTorch   
     - * :ref:`pytorch-neuron-rn`
      
       * :ref:`neuron-cc-ops-pytorch` 
      
      
      
   * - TensorFlow
     - * :ref:`tensorflow-neuron-rn`
      
       * :ref:`tensorflow-ref-neuron-accelerated-ops`
      
       * :ref:`neuron-cc-ops-tensorflow`
      
       * :ref:`tensorflow-modelserver-rn`
      
      
      
   * - Apache MXNet (Incubating)
      
     - * :ref:`mxnet-neuron-rn`
      
       * :ref:`neuron-cc-ops-mxnet`
      
      

   * - Compiler              
     - * :ref:`neuron-cc-rn`
      
   * - Runtime
     - * :ref:`neuron-runtime-release-notes`
      
   * - Tools
      
     - * :ref:`neuron-tools-rn`
      
       * :ref:`neuron-tensorboard-rn`
      
   * - DLAMI and Conda Packages
      
     - * :ref:`dlami-neuron-rn`
      
       * :ref:`DLAMI and Neuron versions Matrix <dlami-neuron-matrix>`

   * - Software Deprecation
   
     - * :ref:`software-deprecation`
      

Previous Releases
-----------------

.. toctree::
   :maxdepth: 1

   README
   
