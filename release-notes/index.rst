.. _neuron-whatsnew:

What's New
==========

.. contents::
   :local:
   :depth: 2

.. _latest-neuron-release:

Neuron 1.19.0 (04/29/2022)
--------------------------

**Neuron 1.19.0** release adds support for PyTorch version 1.11, updates torch-neuron 1.10 to 1.10.2, and adds support for TensorFlow version 2.8, as well as minor enhancements and bug fixes.

Please note that starting with this release (*Neuron 1.19.0*), installing ``aws-neuron-runtime-base`` and ``oci-add-hooks`` are no longer required for Neuron Kubernetes device driver plugin. In addition starting with this release, *torch-neuron 1.5* :ref:`will no longer be supported <eol-pt-15>`.

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

