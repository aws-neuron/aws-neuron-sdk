.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.9.0-whatsnew:


Neuron 2.9.0 (03/27/2023)
-------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release adds support for EC2 Trn1n instances, introduces new features, performance optimizations, minor enhancements and bug fixes. This release introduces the following:

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - Support for EC2 Trn1n instances
     - * Updated Neuron Runtime for Trn1n instances     
      
       * Overall documentation update to include Trn1n instances
     - Trn1n

   * - New Analyze API in PyTorch Neuron (``torch-neuronx``)  
     - * A new API that return list of supported and unsupported PyTorch operators for a model. See :ref:`torch_neuronx_analyze_api`
     - Trn1, Inf2
  
   * - Support models that are larger than 2GB in PyTorch Neuron (``torch-neuron``) on Inf1
     - * See ``separate_weights`` flag to :func:`torch_neuron.trace` to support models that are larger than 2GB
     - Inf1


   * - Performance Improvements
     - * Up to 10% higher throughput when training GPT3 6.7B model on multi-node
     - Trn1


   * - Dynamic Batching support in TensorFlow 2.x Neuron (``tensorflow-neuronx``)
     - * See :ref:`tensorflow-neuronx-special-flags` for details.
     - Trn1, Inf2



   * - NeuronPerf support for Trn1/Inf2 instances
     - * Added Trn1/Inf2 support for PyTorch Neuron (``torch-neuronx``) and TensorFlow 2.x Neuron (``tensorflow-neuronx``)
     - Trn1, Inf2

   * - Hierarchical All-Reduce and Reduce-Scatter collective communication
     - * Added support for hierarchical All-Reduce and Reduce-Scatter in Neuron Runtime to enable better scalability of distributed workloads .
     - Trn1, Inf2
  
   * - New Tutorials added
     - * :ref:`Added tutorial to fine-tune T5 model <torch-hf-t5-finetune>`
      
       * Added tutorial to demonstrate use of Libtorch with PyTorch Neuron (``torch-neuronx``) for inference :ref:`[html] <pytorch-tutorials-libtorch>`
     - Trn1, Inf2

   * - New sample scripts for 3D Parallelism training in Megatron-LM reference for Neuron
     - * See `GPT3-65b <https://github.com/aws-neuron/aws-neuron-reference-for-megatron-lm/blob/master/examples/pretrain_gpt3_65B_3D.sh>`_ and `GPT3-175b <https://github.com/aws-neuron/aws-neuron-reference-for-megatron-lm/blob/master/examples/pretrain_gpt3_175B.sh>`_ examples. 
     - Trn1, Inf2

   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1, Inf2, Inf1

   * - Release included packages
     - * see :ref:`neuron-release-content`
     - Trn1, Inf2, Inf1

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

To learn about the model architectures currently supported on Inf1, Inf2, Trn1 and Trn1n instances, please see :ref:`model_architecture_fit`.

.. _components-rn:

Neuron Components Release Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inf1, Trn1/Trn1n and Inf2 common packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
     - Trn1/Trn1n, Inf1, Inf2
     - * Trn1/Trn1n: ``aws-neuronx-runtime-lib`` (.deb, .rpm)

       * Inf1: Runtime is linked into the ML frameworks packages
       
     - * :ref:`neuron-runtime-rn`

   * - Neuron Runtime Driver
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-dkms``  (.deb, .rpm)
       
     - * :ref:`neuron-driver-release-notes`

   * - Neuron System Tools
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-tools``  (.deb, .rpm)
     - * :ref:`neuron-tools-rn`



   * - Containers
     - Trn1/Trn1n, Inf1, Inf2
     - * ``aws-neuronx-k8-plugin`` (.deb, .rpm)

       * ``aws-neuronx-k8-scheduler`` (.deb, .rpm)
       
       * ``aws-neuronx-oci-hooks`` (.deb, .rpm)

     - * :ref:`neuron-k8-rn`

       * :ref:`neuron-containers-release-notes`

   * - NeuronPerf (Inference only)
     - Trn1/Trn1n, Inf1, Inf2
     - * ``neuronperf`` (.whl)
     - * :ref:`neuronperf_rn`


   * - TensorFlow Model Server Neuron
     - Trn1/Trn1n, Inf1, Inf2
     - * ``tensorflow-model-server-neuronx`` (.deb, .rpm)
     - * :ref:`tensorflow-modeslserver-neuronx-rn`


Trn1/Trn1n and Inf2 only packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
     - Trn1/Trn1n, Inf2
     - * ``torch-neuronx`` (.whl)
     - * :ref:`torch-neuronx-rn`

       * :ref:`pytorch-neuron-supported-operators`
       

   * - TensorFlow Neuron
     - Trn1/Trn1n, Inf2
     - * ``tensorflow-neuronx`` (.whl)
     - * :ref:`tensorflow-neuronx-release-notes`


   * - Neuron Compiler (Trn1/Trn1n, Inf2 only)
     - Trn1/Trn1n, Inf2
     - * ``neuronx-cc`` (.whl)
     - * :ref:`neuronx-cc-rn`

   * - Collective Communication library
     - Trn1/Trn1n, Inf2
       
     - * ``aws-neuronx-collective`` (.deb, .rpm)

     - * :ref:`neuron-collectives-rn`


   * - Neuron Custom C++ Operators
     - Trn1/Trn1n, Inf2
  
     - * ``aws-neuronx-gpsimd-customop`` (.deb, .rpm)
  
       * ``aws-neuronx-gpsimd-tools`` (.deb, .rpm)
  
     - * :ref:`gpsimd-customop-lib-rn`

       * :ref:`gpsimd-customop-tools-rn`


   * - ``transformers-neuronx``
     - Trn1/Trn1n, Inf2
       
     - * GitHub repository `(link) <https://github.com/aws-neuron/transformers-neuronx>`_

     - * `Release Notes <https://github.com/aws-neuron/transformers-neuronx/blob/master/releasenotes.md>`_


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
       
       * :ref:`tensorflow-neuron-rn-v2` 



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


Release Content
~~~~~~~~~~~~~~~

* :ref:`neuron-release-content`


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`prev-n1-rn`

