.. _neuron-whatsnew:

What's New
==========

.. contents::
   :local:
   :depth: 2
   
Main release notes
------------------

This release (Neuron 1.13.0) introduces higher performance , updated frameworks support, additional tutorials, and adding models and tools:

* Additional compiler improvements boost performance up to 20% higher throughput compared to previous release across model types.
* Improving usability for NLP models, with out-of-the-box 12x higher-throughput at 70% lower cost for Hugging Face Transformers pre-trained BERT Base models, see :ref:`pytorch-tutorials-neuroncore-pipeline-pytorch`.
* Upgrade Apache MXNet (Incubating) to 1.8, where Neuron is now a plugin, see :ref:`mxnet-neuron-rn`.
* PyTorch ResNext models now functional with new operator support, see :ref:`pytorch-neuron-rn`.
* PyTorch Yolov5 support, see :ref:`pytorch-neuron-rn`.
* MXNet (Incubating): Gluon API and Neuron support for NLP BERT models, see :ref:`mxnet-neuron-rn`.
* PyTorch Convolution operator support has been extended to include most Conv1d and Conv3d variants, please see :ref:`neuron-cc-ops-pytorch`  for the complete list of operators.
* First release of Neuron plugin for TensorBoard, see :ref:`neuron-tensorboard-rn`.


**Software Deprecation**

* :ref:`eol-conda-packages`
* :ref:`eol-ubuntu16`
* :ref:`eol-classic-tensorboard`


Detailed release notes
----------------------

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   
   * - Software 
     - Details
      
   * - Main     
      
     - * :ref:`main-rn`
      
       * :ref:`neuron-release-content`
      
       * :ref:`software-deprecation`
      
   * - PyTorch   
     - * :ref:`pytorch-neuron-rn`
      
       * :ref:`neuron-cc-ops-pytorch` 
      
      
      
   * - TensorFlow
     - * :ref:`tensorflow-neuron-rn`
      
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
      
       * :ref:`conda-pytorch-release-notes`
      
       * :ref:`conda-tensorflow-release-notes`
      
       * :ref:`conda-mxnet-release-notes`
   * - Software Deprecation
   
     - * :ref:`software-deprecation`
      

Previous Releases
-----------------

.. toctree::
   :maxdepth: 1

   neuronrelease/previous-rn
   
