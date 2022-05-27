.. _neuron-whatsnew:

What's New
==========

.. contents::
   :local:
   :depth: 2

.. _latest-neuron-release:

Neuron 1.19.1 (05/27/2022)
--------------------------

**Neuron 1.19.1** is a patch release. This release fixes a bug in Neuron Driver (``aws-neuron-dkms``). Neuron driver version 2.3.11 included in this release fixes a bug that causes kernel panic when a large memory allocation on Neuron device fails.  Neuron Driver 2.3.11 also introduces a new functionality required by the upcoming Neuron 1.20.0 release.  Because the new functionality is mandatory for Neuron 1.20.0 support, Neuron Driver 2.3.11 adds a compatibility check that will prevents Neuron 1.20.0 from running with older versions of the driver.   An attempt to run Neuron 1.20.0 with an older version of the driver will result in the application terminating with an error message.

In addition, this release updates ``tensorflow-neuron`` installation instructions to pin ``protobuf`` version to avoid `compatibility issues <https://github.com/protocolbuffers/protobuf/issues/10051>`__ with older versions of TensorFlow.

.. important ::

   For successful installation or update to next releases (Neuron 1.20.0 and newer):
      * Uninstall ``aws-neuron-dkms`` by running: ``sudo apt remove aws-neuron-dkms`` or ``sudo yum remove aws-neuron-dkms``
      * Install or upgrade to latest Neuron driver (``aws-neuron-dkms``) by following the ":ref:`neuron-install-guide`" instructions.


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

       * :ref:`software-maintenance`

   * - PyTorch
     - * :ref:`pytorch-neuron-rn`

       * :ref:`neuron-cc-ops-pytorch`


   * - TensorFlow 2.x
     - * :ref:`tensorflow-neuron-rn-v2`

       * :ref:`tensorflow-ref-neuron-accelerated-ops`

       * :ref:`tensorflow-modelserver-rn-v2`



   * - TensorFlow 1.x
     - * :ref:`tensorflow-neuron-rn`

       * :ref:`neuron-cc-ops-tensorflow`

       * :ref:`tensorflow-modelserver-rn`


   * - Apache MXNet (Incubating)

     - * :ref:`mxnet-neuron-rn`

       * :ref:`neuron-cc-ops-mxnet`



   * - Compiler
     - * :ref:`neuron-cc-rn`

       * :ref:`neuron-supported-operators`

   * - Runtime
     - * :ref:`neuron-runtime-release-notes`

       * :ref:`neuron-driver-release-notes`

   * - Containers
     - * :ref:`neuron-k8-rn`

       * :ref:`neuron-containers-release-notes`

   * - DLAMI
     - * `AWS Deep Learning AMI (Amazon Linux 2) <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-amazon-linux-2/>`_

       * `AWS Deep Learning AMI (Ubuntu 18.04) <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-ubuntu-18-04/>`_

       * `AWS Deep Learning Base AMI (Amazon Linux 2) <https://aws.amazon.com/releasenotes/aws-deep-learning-base-ami-amazon-linux-2/>`_

       * `AWS Deep Learning Base AMI (Ubuntu 18.04) <https://aws.amazon.com/releasenotes/aws-deep-learning-base-ami-ubuntu-18-04/>`_


   * - Tools

     - * :ref:`neuron-tools-rn`

       * :ref:`neuron-tensorboard-rn`

   * - NeuronPerf

     - * :ref:`neuronperf_rn`
      
   * - Software Deprecation

     - * :ref:`software-maintenance`

   * - Software Maintenance

     - * :ref:`software-maintenance`


Previous Releases
-----------------

.. toctree::
   :maxdepth: 1

   README

