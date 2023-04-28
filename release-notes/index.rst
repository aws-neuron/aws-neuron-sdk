.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.10.0-whatsnew:


Neuron 2.10.0 (04/28/2023)
-------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release adds support for FSDP, introduces new features, performance optimizations, minor enhancements and bug fixes. This release introduces the following:

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - Added initial FSDP support for Training (Experimental) in PyTorch Neuron (``torch-neuronx``)
     - * Iniitial support includes MiCS (Minimize Communication Scale) in FSDP. See more at :ref:`torch-neuronx-rn`    
     - Trn1/Trn1n

   * - New sample scripts for inference
     - * This release adds new sample scripts for deploying Stable Diffusion 2.1 and other models with ``torch-neuronx``, Please check `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_
     - Trn1, Inf2

   * - Added Profiling support in PyTorch Neuron for Inference (``torch-neuronx``)  
     - * See more at :ref:`torch-neuronx-profiling-with-tb`
     - Inf2, Trn1/Trn1n
  
   * -  Added support for the HuggingFace generate in transfomers-neuronx 
     - * See more at :ref:`transformers-neuronx-rn`
     - Inf2, Trn1/Trn1n
   
   * -  Added support for Model Serialization in transfomers-neuronx 
     - * Support for model saving, loading, and weight swapping. See more at :ref:`transformers-neuronx-rn`
     - Inf2, Trn1/Trn1n
   
   * -  Performance improvements in transfomers-neuronx 
     - * Improved performance by removing unnecessary KV-cache tensor resetting. 
       * Improved prompt context encoding performance (``OPT`_ , `GPT2`_``) 
       * See more at :ref:`transformers-neuronx-rn`
     - Inf2, Trn1/Trn1n

   * - Support models larger than 2GB in TensorFlow 2.x Neuron 
     - * See :ref:`tensorflow-neuronx-special-flags` for details. (``tensorflow-neuronx``) 
       * See :ref:`Special Flags <tensorflow-ref-neuron-tracing-api>` for details. (``tensorflow-neuron``)
     - Trn1/Trn1n, Inf2, Inf1
  
   * - Added new operators and options in Compiler. (``neuronx-cc``)
     - * Added support for the HLO operator ``BitcastConvertType`` and ``TopK`` operator.
       * Added two new ``model-type`` option arguments for model specific optimizations.
       * See more at :ref:`neuronx-cc-rn`
     - Inf2, Trn1/Trn1n

   * - Support for using multiple GPSIMD Cores in Neuron Custom C++ Operators
     - * See :ref:`custom-ops-api-ref-guide`
       * Updated package name to ``aws-neuronx-gpsimd-customop-lib``
     - Trn1/Trn1n
   
   * - Runtime Support for Weight Deduplication across multiple instances of same model. 
     - * Support for Sharing weights when loading multiple instance versions of the same model on different NeuronCores.
       * See more at :ref:`nrt-configuration`
     - Trn1, Inf2

   * - Added new Neuron Collectives benchmarking tool, ``nccom-test``  
     - * Supports enabling benchmarking sweeps on various Neuron Collective Communication operations. See :ref:`nccom-test` for more details.
     - Trn1, Inf2

   * - Expanded support for Neuron profiling to include runtime setup/teardown times
     - * Expanded support for Neuron profiling to include runtime setup/teardown times and collapsed execution of NeuronCore engines and DMA.  See :ref:`neuron-tensorboard-rn` for more details.
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

