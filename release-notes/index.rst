.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.11.0-whatsnew:


Neuron 2.11.0 (06/14/2023)
-------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release introduces Neuron Distributed, a new python library to simplify training and inference of large models, improving usability with features like S3 model caching, standalone profiler tool, support for Ubuntu22, as well as other new features,
performance optimizations, minor enhancements and bug fixes. This release introduces the following:


.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

  
   * - New Features and Performance Enhancements in ``transformers-neuronx``
     - * Support for ``int8`` inference. See example at :ref:`int8_weight_storage_support`
       * Improved prompt context encoding performance. See more at :ref:`transformers_neuronx_developer_guide`
       * Improved collective communications performance for Tensor Parallel inference on Inf2 and Trn1.
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n

   * - Neuron Profiler Tool 
     - * Support for as a stand alone tool to profile and get visualized insights on execution of models on Trainium and Inferentia devices.
       * See more at :ref:`neuron-profile-ug`
     - Inf1, Inf2, Trn1/Trn1n

   * - Neuron Compilation Cache through S3
     - * Support for sharing compiled models across Inf2 and Trn1 nodes through S3
       * See more at :ref:`pytorch-neuronx-parallel-compile-cli`
     - Inf2, Trn1/Trn1n

   * - New script to scan a model for supported/unsupported operators
     - * Script to scan a model for supported/unsupported operators before training, scan output includes supported and unsupported operators at both XLA operators and PyTorch operators level.
       * See a sample tutorial at :ref:`torch-analyze-for-training-tutorial`
     - Inf2, Trn1/Trn1n

   * - Neuron Distributed Library [Experimental]
     - * New Python Library based on PyTorch enabling distributed training and inference of large models.
       * Initial support for tensor-parallelism.
       * See more at :ref:`neuronx-distributed-index`
     - Inf2, Trn1/Trn1n

   * - Neuron Calculator and Documentation Updates  
     - * New :ref:`neuron_calculator` Documentation section to help determine number of Neuron Cores needed for LLM Inference.
       * Added App Note :ref:`neuron_llm_inference`
       * See more at :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n

   * - Enhancements to Neuron SysFS
     - * Support for detailed breakdown of memory usage across the NeuronCores
       * See more at :ref:`neuron-sysfs-ug`
     - Inf1, Inf2, Trn1/Trn1n

   * - Support for Ubuntu 22
     - * See more at :ref:`setup-guide-index` for setup instructions on Ubuntu22
     - Inf1, Inf2, Trn1/Trn1n

   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

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


   * - Neuron Documentation
     - Trn1/Trn1n, Inf1, Inf2
     - * 
     - * :ref:`neuron-documentation-rn`


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


   * - Transformers Neuron
     - Trn1/Trn1n, Inf2
     - * ``transformers-neuronx`` (.whl)
     - * :ref:`transformers-neuronx-rn`


   * - Neuron Distributed
     - Trn1/Trn1n, Inf2
  
     - * ``neuronx-distributed`` (.whl)
  
     - * :ref:`neuronx-distributed-rn`




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


.. _latest-neuron-release-artifacts:

Release Artifacts
^^^^^^^^^^^^^^^^^

Trn1 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.11.0

Inf2 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.11.0

Inf1 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.11.0


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`

