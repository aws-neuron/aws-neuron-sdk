.. _neuron-whatsnew:

What's New
==========

.. contents::
   :local:
   :depth: 2

.. _latest-neuron-release:

Neuron 1.18.0 (03/25/2022)
--------------------------

**Neuron 1.18.0** release introduces the beta release of :ref:`NeuronPerf <neuronperf>`, NeuronPerf is a Python library with a simple API that enables fast measurements of performance when running models with Neuron. This release adds new 5 models to the :ref:`appnote-performance-benchmark` together with  NeuronPerf scripts used to compile these models and run the benchmarks.


This release also introduces additional ``torch-neuron`` packages that support C++11 ABI, updates TensorFlow-Neuron 2.5 to 2.5.3, adds support for TensorFlow-Neuron 2.6 and 2.7, and introduces Runtime NEURON_RT_NUM_CORES :ref:`environment variable <nrt-configuration>`. In addition this release include minor enhancements and bug fixes in Compiler, Neuron Framework Extensions, Runtime 2.x library and tools. See below detailed release notes.

Starting with this release, *TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4* will :ref:`no longer be supported <eol-tf-21-24>` . We will also :ref:`stop supporting PyTorch Neuron version 1.5 <announce-eol-pt-1-5>` starting with Neuron 1.19.0 release, and :ref:`will stop supporting <eol-ncgs-env_2>`  ``NEURONCORE_GROUP_SIZES`` environment variable starting with Neuron 1.20.0 release.

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

