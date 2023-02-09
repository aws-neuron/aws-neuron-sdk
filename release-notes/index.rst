.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.7.0-whatsnew:

Neuron 2.7.0 (02/08/2023)
-------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release introduces new capabilities and libraries, as well as features and tools that improves usability. This release introduces the following:

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details

   * - PyTorch 1.13
     - Support of PyTorch 1.13 version for PyTorch Neuron (``torch-neuronx``). For resources see :ref:`pytorch-neuronx-main`

   * - PyTorch DistributedDataParallel (DDP) API
     - Support of PyTorch DistributedDataParallel (DDP) API in PyTorch Neuron (``torch-neuronx``). For resources how to use PyTorch DDP API with Neuron, please check :ref:`neuronx-ddp-tutorial`.

   * - Inference support in ``torch-neuronx``
     - For more details please visit :ref:`pytorch-neuronx-main`` page. You can also try Neuron Inference samples `<https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_ in the ``aws-neuron-samples`` GitHub repo.     

   * - Neuron Custom C++ Operators[Experimental]
     - Initial support for Neuron Custom C++ Operators [Experimental] , with Neuron Custom C++ Operators (“CustomOps”) you can now write CustomOps that run on NeuronCore-v2 chips. For more resources please check :ref:`neuron_c++customops` section.


   * - ``transformers-neuronx`` [Experimental] 
     - ``transformers-neuronx``  is a new library enabling LLM model inference. It contains models that are checkpoint-compatible with HuggingFace Transformers, and currently supports Transformer Decoder models like GPT2, GPT-J and OPT. Please check `aws-neuron-samples repository <https://github.com/aws-neuron/transformers-neuronx>`_  


   * - Neuron sysfs filesystem
     - Neuron sysfs filesystem exposes Neuron Devices under ``/sys/devices/virtual/neuron_device`` providing visibility to Neuron Driver and Runtime at the system level. By performing several simple CLIs such as reading or writing to a sysfs file, you can get information such as Neuron Runtime status, memory usage, Driver info etc. For resources about Neuron sysfs filesystem visit :ref:`neuron-sysfs-ug`.


   * - TFLOPS support in Neuron System Tools
     - Neuron System Tools now also report model actual TFLOPs rate in both ``neuron-monitor`` and ``neuron-top``. More details can be found in the :ref:`Neuron Tools documentation <neuron-tools>`.

   * - New sample scripts for training
     - This release adds multiple new sample scripts for training models with ``torch-neuronx``, Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_

   * - New sample scripts for inference
     - This release adds multiple new sample scripts for deploying models with ``torch-neuronx``, Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_

   * - Neuron GitHub samples repository for Amazon EKS
     - A new AWS Neuron GitHub samples repository for Amazon EKS, Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-eks-samples>`_


For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.


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


   * - Neuron Runtime
     - Trn1 , Inf1
     - * Trn1: ``aws-neuronx-runtime-lib`` (.deb, .rpm)

       * Inf1: Runtime is linked into the ML frameworks packages
       
     - * :ref:`neuron-runtime-rn`

   * - Neuron Runtime Driver
     - Trn1, Inf1
     - * ``aws-neuronx-dkms``  (.deb, .rpm)
       
     - * :ref:`neuron-driver-release-notes`

   * - Neuron System Tools
     - Trn1, Inf1
     - * ``aws-neuronx-tools``  (.deb, .rpm)
     - * :ref:`neuron-tools-rn`



   * - Containers
     - Trn1, Inf1
     - * ``aws-neuronx-k8-plugin`` (.deb, .rpm)

       * ``aws-neuronx-k8-scheduler`` (.deb, .rpm)
       
       * ``aws-neuronx-oci-hooks`` (.deb, .rpm)

     - * :ref:`neuron-k8-rn`

       * :ref:`neuron-containers-release-notes`

   * - NeuronPerf (Inference only)
     - Trn1, Inf1 
     - * ``neuronperf`` (.whl)
     - * :ref:`neuronperf_rn`


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


   * - Neuron Custom C++ Operators
     - Trn1
  
     - * ``aws-neuronx-gpsimd-customop`` (.deb, .rpm)
  
       * ``aws-neuronx-gpsimd-tools`` (.deb, .rpm)
  
     - * :ref:`gpsimd-customop-lib-rn`

       * :ref:`gpsimd-customop-tools-rn`


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
     - * ``tensorflow-model-server-neuronx`` (.deb, .rpm)
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


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`prev-n1-rn`

