.. _neuron-release-content:

Release Details
===============

.. contents::
   :local:
   :depth: 2


Current Release - Neuron v1.15.1 (August 30 2021)
------------------------------------------------



ML Frameworks supported versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Software
     - Supported
   * - PyTorch
     - * PyTorch 1.5
       * PyTorch 1.7
       * PyTorch 1.8
   * - TensorFlow
     - * TensorFlow 1.15
       * TensorFlow 2.1.4
       * TensorFlow 2.2.3
       * TensorFlow 2.3.3
       * TensorFlow 2.4.2
       * TensorFlow 2.5.0
   * - TensorFlow Model Server
     - * TensorFlow Model Server 1.15
       * TensorFlow Model Server 2.1.4
       * TensorFlow Model Server 2.2.3
       * TensorFlow Model Server 2.3.3
       * TensorFlow Model Server 2.4.2
       * TensorFlow Model Server 2.5.0
   * - Apache MXNet (Incubating)
     - * MXNet 1.5
       * MXNet 1.8
   * - TensorBoard
     - * TensorBoard 1.15
       * Neuron Plugin for TensorBoard 2.1.0.0

Dependency Software Supported Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Software
     - Supported
   * - Python
     - * Python 3.6
       * Python 3.7
       * Python 3.8 [Experimental]
       


Neuron Packages
^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Software
     - Included Versions
   * - Neuron Compiler
     - neuron_cc-1.6.13.0
   * - Neuron Runtime
     - aws-neuron-runtime-1.6.24.0
   * - Neuron Tools
     - aws-neuron-tools-1.7.25.0
   * - Neuron Kernel module
     - aws-neuron-dkms-2.1.5.0
   * - Neuron Kubernetes Plugin
     - aws-neuron-k8-plugin-1.6.22.0
   * - Neuron Kubernetes Scheduler
     - aws-neuron-k8-scheduler-1.6.22.0


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

