.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.15.0-whatsnew:



Neuron 2.15.0 (10/26/2023)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release adds support for PyTorch 2.0 (Beta), increases performance for both training and inference workloads, adding ability to train models like ``Llama-2-70B`` using ``neuronx-distributed``. With this release, we are also adding pipeline parallelism support for ``neuronx-distributed`` enabling full 3D parallelism support to easily scale training to large model sizes.
Neuron 2.15 also introduces support for training ``resnet50``, ``milesial/Pytorch-UNet`` and ``deepmind/vision-perceiver-conv`` models using ``torch-neuronx``, as well as new sample code for ``flan-t5-xl`` model inference using ``neuronx-distributed``, in addition to other performance optimizations, minor enhancements and bug fixes.

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - Neuron Distributed (neuronx-distributed) for Training
     - * Pipeline parallelism support. See :ref:`api_guide` , :ref:`pp_developer_guide` and :ref:`pipeline_parallelism_overview`
       * ``Llama-2-70B`` model training script  (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/llama2/tp_pp_llama2_70b_hf_pretrain>`_) (:ref:`tutorial <llama2_70b_tp_pp_tutorial>`)
       * Mixed precision support. See :ref:`pp_developer_guide`
       * Support serialized checkpoint saving and loading using ``save_xser`` and ``load_xser`` parameters. See :ref:`api_guide` 
       * See more at :ref:`neuronx-distributed-rn` 
     - Trn1/Trn1n

   * - Neuron Distributed (neuronx-distributed) for Inference
     - * ``flan-t5-xl`` model inference script (:pytorch-neuron-src:`tutorial <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`)
       * See more at :ref:`neuronx-distributed-rn` and  :ref:`api_guide`
     - Inf2,Trn1/Trn1n

   * - Transformers Neuron (transformers-neuronx) for Inference
     - * Serialization support for ``Llama``, ``Llama-2``, ``GPT2`` and ``BLOOM`` models . See :ref:`developer guide <transformers_neuronx_developer_guide>` and `tutorial <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb>`_
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n

   * - PyTorch Neuron (torch-neuronx)
     - * Introducing ``PyTorch 2.0`` Beta support. See :ref:`introduce-pytorch-2-0` . See  :ref:`llama-2-7b training <llama2_7b_tp_zero1_tutorial>` , `bert training <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/dp_bert_hf_pretrain>`_ and  `t5-3b inference <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/neuronx_distributed/t5-inference/t5-inference-tutorial.html>`_ samples.
       * Scripts for training `resnet50[Beta] <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/resnet50>`_ ,
         `milesial/Pytorch-UNet[Beta] <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/unet_image_segmentation>`_ and `deepmind/vision-perceiver-conv[Beta] <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_image_classification/VisionPerceiverConv.ipynb>`_ models.
     - Trn1/Trn1n,Inf2

   * - AWS Neuron Reference for Nemo Megatron library (``neuronx-nemo-megatron``)
     - * ``Llama-2-70B`` model training sample using pipeline parallelism and tensor parallelism ( `tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`_ )
       * ``GPT-NeoX-20B`` model training using pipeline parallelism and tensor parallelism 
       * See more at :ref:`neuronx-nemo-rn` and `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - Trn1/Trn1n

   * - Neuron Compiler (neuronx-cc)
     - * New ``llm-training`` option argument to ``--distribution_strategy`` compiler option for optimizations related to distributed training. See more at :ref:`neuron-compiler-cli-reference-guide`
       * See more at :ref:`neuronx-cc-rn`
     - Inf2/Trn1/Trn1n

   * - Neuron Tools
     - * ``alltoall`` Collective Communication operation, previously released in Neuron Collectives v2.15.13, was added as a testable operation in ``nccom-test``. See :ref:`nccom-test`
       * See more at :ref:`neuron-tools-rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Documentation Updates
     - * New :ref:`App Note <activation_memory_reduction>` and :ref:`Developer Guide <activation_memory_reduction_developer_guide>` about Activation memory reduction using ``sequence parallelism`` and ``activation recomputation`` in ``neuronx-distributed``
       * Added a new Model Samples and Tutorials summary page. See :ref:`model_samples_tutorials`
       * Added Neuron SDK Classification guide. See :ref:`sdk-classification`
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

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.15.0

Inf2 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.15.0

Inf1 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.15.0


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`

