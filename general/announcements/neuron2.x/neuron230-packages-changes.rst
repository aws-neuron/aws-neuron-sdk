.. post:: Oct 10, 2022 03:00
    :language: en
    :tags: neuron2.x

.. _neuron-packages-changes:

Introducing Packaging and installation changes
----------------------------------------------

Starting with :ref:`Neuron release 2.3 <neuron2x-trn1ga>`, Neuron introduces changes in Neuron packages and installation instructions.

.. contents::  Table of contents
   :local:
   :depth: 2

.. _neuron-new-packages:

New Neuron packages
^^^^^^^^^^^^^^^^^^^

Starting with :ref:`Neuron release 2.3 <neuron2x-trn1ga>`, Neuron introduces the following new packages:

.. list-table:: New Neuron packages
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - New Package
     - Package Type
     - Description
     - Supported Instances 
     
       (At the time of releasing :ref:`Neuron release 2.3 <neuron2x-trn1ga>`)

   * - ``torch-neuronx``
     - .whl (pip)
     - PyTorch Neuron package using `PyTorch XLA <https://pytorch.org/xla>`_ 
     - Trn1

   * - ``neuronx-cc``
     - .whl (pip)
     - Neuron Compiler with XLA front-end
     - Trn1

   * - ``aws-neuronx-runtime-lib``
     - .deb (apt), .rpm (yum)
     - Neuron Runtime library
     - Trn1

   * - ``aws-neuronx-collective``
     - .deb (apt), .rpm (yum)
     - Collective Communication library          
     - Trn1

   * - ``aws-neuronx-tools``
     - .deb (apt), .rpm (yum)
     - Neuron System Tools
     - Trn1

.. note::

   In next releases ``aws-neuronx-tools`` and ``aws-neuronx-runtime-lib`` will add support for Inf1.

Why are we introducing new Neuron packages?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add Neuron support for training neural-networks, Neuron 2.x introduces new capabilities and major architectural updates. For example, Neuron adds support for Collective Communication Operations, in :ref:`new packages <neuron-new-packages>` such as ``aws-neuron-collective``. 

In addition, some of those updates and new capabilities are not backward compatible, for example the Pytorch Neuron package that adds support for training neural-networks uses `PyTorch XLA <https://pytorch.org/xla>`_ as a backend. To reduce the possibility of customers using features that are not backward compatible, the new capabilities are introduced in new Neuron packages. For example, PyTorch Neuron and Neuron Compiler will support  different packages for Inf1 and for Trn1: ``torch-neuron`` and ``neuron-cc`` will support Inf1 instances, and ``torch-neuronx`` and ``neuronx-cc`` will support Trn1 instances.

.. _neuron-packages-renaming:

Renamed Neuron Packages
^^^^^^^^^^^^^^^^^^^^^^^

Starting with :ref:`Neuron release 2.3 <neuron2x-trn1ga>`, the following  Neuron packages will change names: 


.. list-table:: Neuron package with changed names
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size   

   * - New name
     - Old name (deprecated package)
     - Package Type
     - Description
     - Supported Instances 

   * - ``aws-neuronx-oci-hooks``
     - ``aws-neuron-runtime-base``
     - .deb (apt), .rpm (yum)
     - OCI Hooks support
     - Trn1, Inf1

   * - ``aws-neuronx-dkms``
     - ``aws-neuron-dkms``
     - .deb (apt), .rpm (yum)
     - Neuron Driver
     - Trn1, Inf1     




   * - ``aws-neuronx-k8-plugin``
     - ``aws-neuron-k8-plugin``
     - .deb (apt), .rpm (yum)
     - Neuron Kubernetes plugin
     - Trn1, Inf1

   * - ``aws-neuronx-k8-scheduler``
     - ``aws-neuron-k8-scheduler``
     - .deb (apt), .rpm (yum)
     - Neuron Scheduler plugin
     - Trn1, Inf1

Why are we changing package names?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To avoid situations where customers may accidentally install Neuron packages with features that are not backward compatible, we have introduced additional packages with different names for the same Neuron component. 

.. _neuron-installation-instruction-change:

Updated installation and update instructions 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting with :ref:`Neuron release 2.3 <neuron2x-trn1ga>`, Neuron installation and update instructions will include pinning of the major version of the Neuron package. For example, to install latest Neuron tools package, call ``sudo apt-get install aws-neuronx-tools=2.*`` and to install latest PyTorch Neuron package for Trn1, call ``pip install torch-neuronx==1.11.0.1.*``. 


Why are we changing installation and update instructions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Neuron installation and update instructions now guide customers to pin the major version of the different Neuron packages as mentioned in :ref:`neuron-installation-instruction-change`. This is done to future-proof instructions for new, backwards-incompatible major version releases.

.. note:: The change of the installation and update instructions will not include instruction to install or update ``torch-neuron`` and ``neuron-cc``.

What do I need to do?
~~~~~~~~~~~~~~~~~~~~~

Please follow the :ref:`Neuron setup guide <setup-guide-index>` to update to latest Neuron releases.

