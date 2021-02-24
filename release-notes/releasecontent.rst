.. _neuron-release-content:

Release Content
===============

.. contents:: Table of Contents
   :local:
   :depth: 2


Current Release - Neuron v1.12.1 (Feb 24th 2021)
------------------------------------------------



Machine Learning Frameworks Supported Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Software
     - Supported
     - Maintenance
     - End Of Support
   * - PyTorch
     - * PyTorch 1.5
      
       * PyTorch 1.7
     - 
     - * PyTorch 1.3 (07/16/2020)
   * - TensorFlow
     - * TensorFlow 1.15
     - 
     - 
   * - TensorFlow Model Server
     - * TensorFlow Model Server 1.15
     - 
     -
   * - MXNet
     - * MXNet 1.5
     - 
     -
   * - TensorBoard
     - * TensorBoard 1.15
     - 
     -
   * - ONNX
     - * No
     - 
     - * Yes (11/17/2020)

Dependency Software Supported Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Software
     - Supported
     - Maintenance
     - End Of Support
   * - Python
     - * Python 3.6
       * Python 3.7
     - 
     - * Python 3.5 (2/24/2021)
   * - Neuron Conda Packages
     - * torch-neuron 1.7.1.1.2.15.0 
     
       * tensorflow-neuron 1.15.5.1.2.8.0

       * mxnet-neuron 1.5.1.1.3.7.0
       
       * See https://conda.repos.neuron.amazonaws.com/
     - 
     - 


Neuron Packages
^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Software
     - Included Versions
   * - Neuron Compiler
     - neuron-cc-1.2.7
   * - Neuron Runtime
     - neuron-runtime-1.4.9
   * - Neuron Tools
     - neuron-tools-1.4.8
   * - Neuron Kernel module
     - neuron-dkms-1.4.5
   * - Neuron Kubernetes Plugin
     - aws-neuron-k8-plugin-1.4.5
   * - Neuron Kubernetes Scheduler
     - aws-neuron-k8-scheduler-1.4.5


See :ref:`neuron-maintenance-policy` for more information.



Deep Learning AMI (DLAMI) Versions Matrix
-----------------------------------------

The DLAMI version can be determined by examining the AMI name of the EC2 instance from EC2 console or examining the file README in the default home directory.

.. include:: /neuron-intro/dlami/dlami-neuron-matrix-table.rst

See :ref:`dlami-section` for more information.


Previous Releases Content
-------------------------

.. toctree::
   :maxdepth: 1

   neuronrelease/previous-content

