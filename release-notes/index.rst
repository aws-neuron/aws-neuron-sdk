.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.14.0-whatsnew:


Neuron 2.14.1 (09/26/2023)
--------------------------

This is a patch release that fixes compiler issues in certain configurations of ``Llama`` and ``Llama-2`` model inference using ``transformers-neuronx``.

.. note::

   There is still a known compiler issue for inference of some configurations of ``Llama`` and ``Llama-2`` models that will be addressed in future Neuron release.
   Customers are advised to use ``--optlevel 1 (or -O1)`` compiler flag to mitigate this known compiler issue.  
    
   See :ref:`neuron-compiler-cli-reference-guide` on the usage of ``--optlevel 1`` compiler flag. Please see more on the compiler fix and known issues in :ref:`neuronx-cc-rn` and :ref:`transformers-neuronx-rn` 
   



Neuron 2.14.0 (09/15/2023)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release introduces support for ``Llama-2-7B`` model training and ``T5-3B`` model inference using ``neuronx-distributed``. It also adds support for  ``Llama-2-13B`` model training using ``neuronx-nemo-megatron``. Neuron 2.14 also adds support for ``Stable Diffusion XL(Refiner and Base)`` model inference using ``torch-neuronx`` . This release also introduces other new features, performance optimizations, minor enhancements and bug fixes.
This release introduces the following:

.. note::
   This release deprecates ``--model-type=transformer-inference`` compiler flag. Users are highly encouraged to migrate to the ``--model-type=transformer`` compiler flag.


.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - AWS Neuron Reference for Nemo Megatron library (``neuronx-nemo-megatron``)
     - * ``Llama-2-13B`` model training support ( `tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`_ )
       * ZeRO-1 Optimizer support  that works with tensor parallelism and pipeline parallelism
       * See more at :ref:`neuronx-nemo-rn` and `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - Trn1/Trn1n
   
   * - Neuron Distributed (neuronx-distributed) for Training
     - * ``pad_model`` API to pad attention heads that do not divide by the number of NeuronCores, this will allow users to use any supported tensor-parallel degree. See  :ref:`api_guide`
       * ``Llama-2-7B`` model training support  (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/tp_zero1_llama2_7b_hf_pretrain>`_) (:ref:`tutorial <llama2_7b_tp_zero1_tutorial>`)
       * See more at :ref:`neuronx-distributed-rn` and  :ref:`api_guide`
     - Trn1/Trn1n

   * - Neuron Distributed (neuronx-distributed) for Inference
     - * ``T5-3B`` model inference support (:pytorch-neuron-src:`tutorial <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`)
       * ``pad_model`` API to pad attention heads that do not divide by the number of NeuronCores, this will allow users to use any supported tensor-parallel degree. See  :ref:`api_guide` 
       * See more at :ref:`neuronx-distributed-rn` and  :ref:`api_guide`
     - Inf2,Trn1/Trn1n

   * - Transformers Neuron (transformers-neuronx) for Inference
     - * Introducing ``--model-type=transformer`` compiler flag that deprecates ``--model-type=transformer-inference`` compiler flag. 
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n

   * - PyTorch Neuron (torch-neuronx)
     - * Performance optimizations in ``torch_neuronx.analyze`` API. See :ref:`torch_neuronx_analyze_api`
       * ``Stable Diffusion XL(Refiner and Base)`` model inference support  ( `sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sdxl_base_and_refiner_1024_inference.ipynb>`_)
     - Trn1/Trn1n,Inf2

   * - Neuron Compiler (neuronx-cc)
     - * New  ``--optlevel``(or ``-O``) compiler option that enables different optimizations with tradeoff between faster model compile time and faster model execution. See more at :ref:`neuron-compiler-cli-reference-guide`
       * See more at :ref:`neuronx-cc-rn`
     - Inf2/Trn1/Trn1n

   * - Neuron Tools
     - * Neuron SysFS support for showing connected devices on ``trn1.32xl``, ``inf2.24xl`` and ``inf2.48xl`` instances. See :ref:`neuron-sysfs-ug`
       * See more at :ref:`neuron-tools-rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Documentation Updates
     - * Neuron Calculator now supports multiple model configurations for Tensor Parallel Degree computation. See :ref:`neuron_calculator`
       * Announcement to deprecate ``--model-type=transformer-inference`` flag. See :ref:`announce-deprecation-transformer-flag`
       * See more at :ref:`neuron-documentation-rn`
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

   * - AWS Neuron Reference for NeMo Megatron
     - Trn1/Trn1n
     - * `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - * :ref:`neuronx-nemo-rn`



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

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.14.1

Inf2 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.14.1

Inf1 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.14.1


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`

