.. _neuron-whatsnew:

What's New
==========

.. contents::
   :local:
   :depth: 2

.. _latest-neuron-release:

Neuron 1.19.2 (08/02/2022)
--------------------------

**Neuron 1.19.2** This is a patch release. The release include a :ref:`security update <ndriver_2_3_26_0>` for Neuron Driver (``aws-neuron-dkms``) and includes compiler bug fix that ignore MXNet dropout for 'training' while performing inference. 
Please update the Neuron Driver to the latest (version 2.3.26 or newer) so that you can benefit from operational and security updates included in this release.

.. important ::

   You must update to the latest Neuron Driver (aws-neuron-dkms version 2.3.26 or newer) before installing or upgrading to latest Neuron release.
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

