.. _neuron-whatsnew:

What's New
==========

.. contents::
   :local:
   :depth: 2

.. _latest-neuron-release:

Neuron 1.17.2 (02/18/2022)
--------------------------

**Neuron 1.17.2** is a patch release. This release fixes a bug in TensorFlow Neuron versions 2.1, 2.2. 2.3 and 2.4. The fixed bug was causing a memory leak of 128B for each inference. Starting this release, TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4 are :ref:`entering maintenance mode <maintenance_tf21_tf24>`. Future releases of TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4 will address security issues only.

Neuron 1.17.1 (02/16/2022)
--------------------------

**Neuron 1.17.1** is a patch release. This release fixes a bug in TensorFlow Neuron that caused a memory leak. The memory leak was approximately 128b for each inference and 
exists in all versions of TensorFlow Neuron versions part of Neuron 1.16.0 to Neuron 1.17.0 releases. see :ref:`pre-release-content` for exact versions included in each release.  This release only fixes the memory leak for TensorFlow versions 1.15 and 2.5 from Neuron.  The other versions of TensorFlow Neuron will be fixed in a shortly upcoming release.


Neuron 1.17.0 (01/20/2022)
--------------------------

**Neuron 1.17.0** release introduces the support of PyTorch 1.10,  Tensorflow 2.5 update to version 2.5.2, new operators support in PyTorch 
and TensorFlow 1.15, in addition to enhancements and bug fixes in PyTorch, TensorFlow, MxNet, Compiler, Runtime and Tools.

- **PyTorch**
   * First PyTorch 1.10 support.
   * Added new operators support.
   * See :ref:`pytorch-neuron-rn` and :ref:`neuron-cc-ops-pytorch` for more details.
- **TensorFlow 2.x**   
   * Updated Tensorflow 2.5 to version 2.5.2.
   * Updated tensorflow-model-server 2.5 to version 2.5.3.
   * See :ref:`tensorflow-neuron-rn-v2` and :ref:`tensorflow-modelserver-rn-v2` for more details.
- **TensorFlow 1.15**   
   * Added new operators support.
   * See :ref:`tensorflow-neuron-rn` and :ref:`neuron-cc-ops-tensorflow` for more details.
- **MXNet**   
   * Added support for ``mx_neuron.__version__`` to get the build version of MXNet Neuron plugin.
   * See :ref:`mxnet-neuron-rn` for more details.
- **Tools 2.x**
   * ``neuron-top`` - Added “all” tab that aggregates all running Neuron processes into a single view.  
   * ``neuron-top`` - Improved startup time by approximately 1.5 seconds in most cases.
   * See :ref:`neuron-tools-rn` for more details.
- **Compiler**
   * Enhancements and minor bug fixes.
   * See :ref:`neuron-cc-rn` for more details.
- **Runtime 2.x**
   * Enhancements and minor bug fixes.
   * See :ref:`neuron-runtime-release-notes` for more details.


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


   * - Tools

     - * :ref:`neuron-tools-rn`

       * :ref:`neuron-tensorboard-rn`

   * - Software Deprecation

     - * :ref:`software-deprecation`

   * - Software Maintenance

     - * :ref:`software-maintenance`


Previous Releases
-----------------

.. toctree::
   :maxdepth: 1

   README

