.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.13.0-whatsnew:



Neuron 2.13.0 (08/28/2023)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^


This release introduces support for ``GPT-NeoX`` 20B model training in ``neuronx-distributed`` including Zero-1 optimizer capability. It also adds support for ``Stable Diffusion XL`` and ``CLIP`` models inference in ``torch-neuronx``. Neuron 2.13 also introduces `AWS Neuron Reference for Nemo Megatron <https://github.com/aws-neuron/neuronx-nemo-megatron>`_ library supporting distributed training of LLMs like ``GPT-3 175B``. This release also introduces other new features, performance optimizations, minor enhancements and bug fixes.
This release introduces the following:



.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - AWS Neuron Reference for Nemo Megatron library
     - * Modified versions of the open-source packages `NeMo <https://github.com/NVIDIA/NeMo>`_ and `Apex <https://github.com/NVIDIA/apex>`_ that have been adapted for use with AWS Neuron and AWS EC2 Trn1 instances.
       * ``GPT-3`` model training support ( `tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-gpt-job.md>`_ )
       * See more at `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_
     - Trn1/Trn1n

   * - Transformers Neuron (transformers-neuronx) for Inference
     - * Latency optimizations for  ``Llama`` and ``GPT-2`` models inference.
       * Neuron Persistent Cache support (:ref:`developer guide <transformers_neuronx_developer_guide>`)
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n
   
   * - Neuron Distributed (neuronx-distributed) for Training
     - * Now Stable, removed Experimental support
       * ZeRO-1 Optimizer support with tensor parallel. (:ref:`tutorial <gpt_neox_tp_zero1_tutorial>`)
       * Sequence Parallel support. (:ref:`api guide <api_guide>`)
       * GPT-NeoX model training support. (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training>`_) (:ref:`tutorial <gpt_neox_tp_zero1_tutorial>`)
       * See more at :ref:`neuronx-distributed-rn` and  :ref:`api_guide`
     - Trn1/Trn1n

   * - Neuron Distributed (neuronx-distributed) for Inference
     - * KV Cache Support for LLM Inference (:ref:`release notes <neuronx-distributed-rn>`)
     - Inf2,Trn1/Trn1n


   * - PyTorch Neuron (torch-neuronx)
     - * Seedable dropout enabled by default for training
       * KV Cache inference support ( :pytorch-neuron-src:`tutorial <torch-neuronx/t5-inference-tutorial.ipynb>` )
       * ``camembert-base`` training script. (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_text_classification/CamembertBase.ipynb>`_)
       * New models inference support that include `Stable Diffusion XL <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sdxl_1024_inference.ipynb>`_ , CLIP (`clip-vit-base-patch32 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_clip_base_inference_on_inf2.ipynb>`_ , `clip-vit-large-patch14 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_clip_large_inference_on_inf2.ipynb>`_ ) , `Vision Perceiver <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_perceiver_vision_inference.ipynb>`_ , `Language Perceiver <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_perceiver_language_inference.ipynb>`_ and :pytorch-neuron-src:`T5 <torch-neuronx/t5-inference-tutorial.ipynb>`
     - Trn1/Trn1n,Inf2


   * - Neuron Tools
     - * New data types support for Neuron Collective Communication Test Utility (NCCOM-TEST)  --check option: fp16, bf16, (u)int8, (u)int16, and (u)int32 
       * Neuron SysFS support for FLOP count(flop_count) and connected Neuron Device ids (connected_devices).  See :ref:`neuron-sysfs-ug`
       * See more at :ref:`neuron-tools-rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Neuron Runtime 
     - * Runtime version and Capture Time support to NTFF
       * Async DMA copies support to improve Neuron Device copy times for all instance types
       * Logging and error messages improvements for Collectives timeouts and when loading NEFFs.
       * See more at :ref:`neuron-runtime-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - End of Support Announcements and Documentation Updates 
     - * Announcing End of support for ``AWS Neuron reference for Megatron-LM`` starting Neuron 2.13. See more at :ref:`announce-eol-megatronlm`
       * Announcing end of support for ``torch-neuron`` version 1.9 starting Neuron 2.14. See more at :ref:`announce-eol-pytorch19`
       * Added TensorFlow 2.x (``tensorflow-neuronx``) analyze_model API section. See more at :ref:`tensorflow-ref-neuron-analyze_model-api`
       * Upgraded ``numpy`` version to ``1.21.6`` in various training scripts for `Text Classification <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training>`_
       * Updated ``bert-japanese`` training Script to use ``multilingual-sentiments`` dataset. See `hf-bert-jp <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_bert_jp> `_
       * See more at :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1
   
   * - Known Issues and Limitations
     - * See :ref:`neuron-2.13.0-known-issues`
     - Trn1/Trn1n , Inf2, Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.

To learn about the model architectures currently supported on Inf1, Inf2, Trn1 and Trn1n instances, please see :ref:`model_architecture_fit`.

.. _neuron-2.13.0-known-issues:

2.13.0 Known Issues and Limitations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Currently we see a NaN generated when the model implementation uses torch.dtype(float32.min) or torch.dtype(float32.max) along with XLA_USE_BF16/XLA_DOWNCAST_BF16. This is because, float32.min or float32.max gets downcasted to Inf in bf16 thereby producing a NaN. Short term fix is that we can use a small/large fp32 number instead of using float32.min/float32.max. Example, for mask creation, we can use -/+1e4 instead of min/max values. The issue will be addressed in future Neuron releases.   



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

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.13.0

Inf2 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.13.0

Inf1 packages

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.13.0


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`

