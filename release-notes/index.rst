.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.16.0-whatsnew:



Neuron 2.16.0 (12/21/2023)
--------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

Neuron 2.16 adds support for Llama-2-70B training and inference, upgrades to PyTorch 2.1 (beta) and adds new support for PyTorch Lightning Trainer (beta) as well as performance improvements and adding Amazon Linux 2023 support.

**Training highlights**: NeuronX Distributed library LLM models training performance is improved by up to 15%. LLM model training user experience is improved by introducing support of PyTorch Lightning Trainer (beta), and a new model optimizer wrapper which will minimize the amount of changes needed to partition models using NeuronX Distributed primitives.  

**Inference highlights**: PyTorch inference now allows to dynamically swap different fine-tuned weights for an already loaded model, as well as overall improvements of LLM inference throughput and latency with Transformers NeuronX. Two new reference model samples for LLama-2-70b and Mistral-7b model inference.

**User experience**: This release introduces two new capabilities: A new tool, Neuron Distributed Event Tracing (NDET) which improves debuggability, and the support of profiling collective communication operators in the Neuron Profiler tool.

More release content can be found in the table below and each component release notes.



.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances


   * - Transformers NeuronX (transformers-neuronx) for Inference
     - * [Beta] Support for Grouped Query Attention(GQA). See :ref:`developer guide <transformers_neuronx_developer_guide>` 
       * [Beta] Support for ``Llama-2-70b`` model inference using ``Grouped Query Attention``. See `tutorial <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-70b-sampling.ipynb>`_ 
       * [Beta] Support for ``Mistral-7B-Instruct-v0.1`` model inference. See :ref:`sample code <mistral_gqa_code_sample>`
       * See more at :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Training
     - * [Beta] Support for ``PyTorch Lightning``  to train models using ``tensor parallelism`` and ``data parallelism`` . See :ref:`api guide <api_guide>` , :ref:`developer guide <ptl_developer_guide>` and :ref:`tutorial <llama2_7b_tp_zero1_ptl_tutorial>`
       * Support for Model and Optimizer Wrapper training API that handles the parallelization. See :ref:`api guide <api_guide>` and :ref:`model_optimizer_wrapper_developer_guide`
       * New ``save_checkpoint``  and ``load_checkpoint`` APIs to save/load checkpoints during distributed training. See :ref:`save_load_developer_guide`
       * Support for a new ``Query-Key-Value(QKV)`` module that provides the ability to replicate the Key Value heads and adds flexibility to use higher Tensor parallel degree during Training. See :ref:`api guide <api_guide>` and :ref:`tutorial <llama2_tp_pp_tutorial>`
       * See more at :ref:`neuronx-distributed-rn` 
     - Trn1/Trn1n

   * - NeuronX Distributed (neuronx-distributed) for Inference
     - * Support weight-deduplication amongst TP shards by giving ability to save weights separately than in NEFF files.  See :ref:`developer guide<nxd_inference_developer_guide>`
       * ``Llama-2-7B`` model inference script (:ref:`[html] </src/examples/pytorch/neuronx_distributed/llama/llama2_inference.ipynb>` :pytorch-neuron-src:`[notebook] <neuronx_distributed/llama/llama2_inference.ipynb>`)
       * See more at :ref:`neuronx-distributed-rn` and  :ref:`api_guide`
     - Inf2,Trn1/Trn1n

   * - PyTorch NeuronX (torch-neuronx)
     - * [Beta]Support for] ``PyTorch 2.1``. See :ref:`introduce-pytorch-2-1` . See  `llama-2-13b inference <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb>`_ sample.
       * Support to separate out model weights from NEFF files and new ``replace_weights`` API to replace the separated weights. See :ref:`torch_neuronx_replace_weights_api` and :ref:`torch_neuronx_trace_api`
       * [Beta] Script for training ``stabilityai/stable-diffusion-2-1-base`` and  ``runwayml/stable-diffusion-v1-5`` models . See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/stable_diffusion/>`_ 
       * [Beta] Script for training ``facebook/bart-large`` model. See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_summarization/BartLarge.ipynb>`_ 
       * [Beta] Script for ``stabilityai/stable-diffusion-2-inpainting`` model inference.  See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sd2_inpainting_936_624_inference.ipynb>`_ 
     - Trn1/Trn1n,Inf2

   * - Neuron Tools
     - * New ``Neuron Distributed Event Tracing (NDET) tool`` to help visualize execution trace logs and diagnose errors in multi-node workloads. See :ref:`neuron-det-ug` 
       * Support for multi-worker jobs in ``neuron-profile`` . See :ref:`neuron-profile-ug`
       * See more at :ref:`neuron-tools-rn`
     - Inf1/Inf2/Trn1/Trn1n
  
   * - Documentation Updates
     - * Added setup guide instructions for ``AL2023`` OS. See :ref:`setup-guide-index`
       * Added announcement for name change of Neuron Components. See :ref:`announce-component-name-change`
       * Added announcement for End of Support for ``PyTorch 1.10`` . See :ref:`announce-eos_pytorch110`
       * Added announcement for End of Support for ``PyTorch 2.0`` Beta. See :ref:`announce-eos_pytorch2`
       * See more at :ref:`neuron-documentation-rn`
     - Inf1, Inf2, Trn1/Trn1n
  
   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`
     - Trn1/Trn1n , Inf2, Inf1
   
   * - Known Issues and Limitations
     - * See :ref:`neuron-2.16.0-known-issues`
     - Trn1/Trn1n , Inf2, Inf1

   * - Release Artifacts
     - * see :ref:`latest-neuron-release-artifacts`
     - Trn1/Trn1n , Inf2, Inf1


.. _neuron-2.16.0-known-issues:

2.16.0 Known Issues and Limitations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* We recommend running multi-node training jobs on AL2023 using Amazon EKS. Parallel Cluster currently does not support AL2023.
* There are known compiler issues impacting inference accuracy of certain model configurations of ``Llama-2-13b`` when ``amp = fp16`` is used. If this issue is observed, ``amp=fp32`` should be used as a work around.  This issue will be addressed in future Neuron releases.
* Execution time reported in ``neuron-profile`` tool is sometimes in-accurate due to a bug in how the time is captured.  The bug will be addressed in upcoming Neuron releases.
* See component release notes below for any additional known issues.


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



   * - Apache MXNet
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
-------------------

.. contents:: Table of contents
   :local:
   :depth: 1

Trn1 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.16.0

Inf2 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.16.0

Inf1 packages
^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.16.0

Supported Python Versions for Inf1 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.16.0

Supported Python Versions for Inf2/Trn1 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.16.0

Supported Numpy Versions
^^^^^^^^^^^^^^^^^^^^^^^^
Neuron supports versions >= 1.21.6 and <= 1.22.2

Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`


