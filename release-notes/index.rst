.. _neuron-whatsnew:

What's New
==========

.. contents::
   :local:
   :depth: 2
   
Neuron 1.14.0 (05/28/2021)
--------------------------

This release (Neuron 1.14.0) introduces first release of Neuron PyTorch 1.8.1, tutorials update, performance enhancements and memory optimizations for Neuron PyTorch, Neuron TensorFlow and Neuron MXNet.


* Neuron PyTorch - First release of Neuron PyTorch 1.8.1.
* Neuron PyTorch - Convolution operator support has been extended to include ConvTranspose2d variants.
* Neuron PyTorch - Updated  tutorials to use Hugging Face Transformers 4.6.0.
* Neuron PyTorch - Additional performance enhancements, memory optimizations, and bug fixes. see :ref:`pytorch-neuron-rn`.
* Neuron Compiler - New feature  -  Uncompressed NEFF format for faster loading models prior inference. Enable it by –enable-fast-loading-neuron-binaries. Some cases of large models may be detrimentally  impacted as it will not be compressed but many cases will benefit.
* Neuron Compiler - Additional performance enhancements, memory optimizations, and bug fixes, see :ref:`neuron-cc-rn`.
* Neuron TensorFlow - Performance enhancements, memory optimizations, and bug fixes. see :ref:`tensorflow-neuron-rn`. 
* Neuron MXNet - Enhancements and minor bug fixes (MXNet 1.8), see :ref:`mxnet-neuron-rn`.
* Neuron Runtime - Performance enhancements, memory optimizations, and bug fixes. :ref:`neuron-runtime-release-notes`.
* Neuron Tools - Minor bug fixes and enhancements.
* Software Deprecation

    * End of support for Neuron Conda packages in Deep Learning AMI, users should use pip upgrade commands to upgrade to latest Neuron version in DLAMI, see `blog <https://aws.amazon.com/blogs/developer/neuron-conda-packages-eol/>`_.
    * End of support for Ubuntu 16, see  `documentation <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/deprecation.html>`_.


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
   
