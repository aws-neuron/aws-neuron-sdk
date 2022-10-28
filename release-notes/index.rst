.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:

Neuron 2.4.0 (10/27/2022)
-------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

Overview
~~~~~~~~
This release introduces new features and resolves issues that improve stability. The release introduces "memory utilization breakdown" feature in both :ref:`Neuron Monitor <neuron-monitor-ug>` and :ref:`Neuron Top <neuron-top-ug>` system tools. The release introduces support for "NeuronCore Based Sheduling" capability to the Neuron Kubernetes Scheduler and introduces new operators support in :ref:`Neuron Compiler <neuronx-cc>` and :ref:`PyTorch Neuron <torch-neuronx-rn>`. This release introduces also additional eight (8) samples of models' fine tuning using PyTorch Neuron. The new samples can be found in the `AWS Neuron Samples GitHub <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_ repository.

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

.. list-table::
   :widths: auto
   :align: left
   :class: table-smaller-font-size

   * - Get started with Neuron
     - * :ref:`torch_quick_start`
       * :ref:`docs-quick-links`

   * - Frequently Asked Questions (FAQ)
     - * :ref:`neuron2-intro-faq`
       * :ref:`neuron-training-faq`


   * - Troubleshooting
     - * :ref:`PyTorch Neuron Troubleshooting on Trn1 <pytorch-neuron-traning-troubleshooting>`
       * :ref:`Neuron Runtime Troubleshooting on Trn1  <trouble-shoot-trn1>`   

   * - Neuron architecture and features
     - * :ref:`neuron-architecture-index`
       * :ref:`neuron-features-index`


   * - Neuron Components release notes
     - * :ref:`components-rn`




.. _components-rn:

Neuron Components Release Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inf1 and Trn1 common packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size


   * - Component
     - Instance/s
     - Package/s
     - Details

   * - Neuron Runtime Driver
     - Trn1, Inf1
     - * ``aws-neuronx-dkms``  (.deb, .rpm)
       
     - * :ref:`neuron-driver-release-notes`


   * - Containers
     - Trn1, Inf1
     - * ``aws-neuronx-k8-plugin`` (.deb, .rpm)

       * ``aws-neuronx-k8-scheduler`` (.deb, .rpm)
       
       * ``aws-neuronx-oci-hooks`` (.deb, .rpm)

     - * :ref:`neuron-k8-rn`

       * :ref:`neuron-containers-release-notes`


Trn1 only packages
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   

   * - Component
     - Instance/s
     - Package/s
     - Details


   * - Neuron Runtime
     - Trn1
     - * ``aws-neuronx-runtime-lib`` (.deb, .rpm)
       
     - * :ref:`neuron-runtime-rn`
     

   * - PyTorch Neuron
     - Trn1
     - * ``torch-neuronx`` (.whl)
     - * :ref:`torch-neuronx-rn`

       * :ref:`pytorch-neuron-supported-operators`
       

   * - Neuron Compiler (Trn1 only)
     - Trn1
     - * ``neuronx-cc`` (.whl)
     - * :ref:`neuronx-cc-rn`

   * - Collective Communication library
     - Trn1
       
     - * ``aws-neuronx-collective`` (.deb, .rpm)

     - * :ref:`neuron-collectives-rn`

   * - Neuron System Tools
     - Trn1
     - * ``aws-neuronx-tools``  (.deb, .rpm)
     - * :ref:`neuron-tools-rn`


.. note::

   In next releases ``aws-neuronx-tools`` and ``aws-neuronx-runtime-lib`` will add support for Inf1.


Inf1 only packages
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   

   * - Component
     - Instance/s
     - Package/s
     - Details


   * - PyTorch Neuron
     - Inf1
     - * ``torch-neuron`` (.whl)
     - * :ref:`pytorch-neuron-rn`

       * :ref:`neuron-cc-ops-pytorch`


   * - TensorFlow Neuron
     - Inf1
     - * ``tensorflow-neuron`` (.whl)
     - * :ref:`tensorflow-neuron-rn`

       * :ref:`neuron-cc-ops-tensorflow`


   * - TensorFlow Model Server Neuron
     - Inf1
     - * ``tensorflow-model-server-neuron`` (.deb, .rpm)
     - * :ref:`tensorflow-modelserver-rn`


   * - Apache MXNet (Incubating)
     - Inf1
     - * ``mx_neuron`` (.whl)
     - * :ref:`mxnet-neuron-rn`

       * :ref:`neuron-cc-ops-mxnet`


   * - Neuron Compiler (Inf1 only)
     - Inf1
     - * ``neuron-cc`` (.whl)
     - * :ref:`neuron-cc-rn`

       * :ref:`neuron-supported-operators`

   * - Neuron System Tools
     - Inf1
     - * ``aws-neuron-tools``  (.deb, .rpm)
     - * :ref:`neuron-tools-rn`


   * - NeuronPerf
     - Inf1
     - * ``neuronperf`` (.whl)
     - * :ref:`neuronperf_rn`


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`prev-n1-rn`

