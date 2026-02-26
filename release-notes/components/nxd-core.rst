.. meta::
    :description: Complete release notes for the NxD Core component across all AWS Neuron SDK versions.
    :keywords: nxd core, neuronx-distributed, release notes, aws neuron sdk
    :date-modified: 09/18/2025

.. _nxd-core_rn:

Component Release Notes for NxD Core
====================================

The release notes for the NxD Core (``neuronx-distributed``) Neuron component. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _nxd-core-2-26-0-rn:

NxD Core [0.15.22259] (Neuron 2.26.0 Release)
-----------------------------------------------

Date of Release: 09/18/2025

Improvements
~~~~~~~~~~~~~~~

**NxD Core inference improvements**

* Non-distributed inference in parallel layers: Updated parallel layers to support non-distributed inference when parallel state isn't initialized. In non-parallel environments, RowParallelLinear and ColumnParallelLinear now function as ``nn.Linear``, and ``ParallelEmbedding`` now functions as ``nn.Embedding``. This change enables you to simplify model code that works on device and on CPU by enabling you to use the parallel layer in both cases.
* Added a ``compiler_flag_hook`` argument to ModelBuilder, which you can use to override compiler flags for different submodels and buckets.

Bug Fixes
~~~~~~~~~

* Added additional instance types to the ``hardware`` enum. For example, ``inf2`` now maps to ``trn1``.
* Other minor bug fixes and improvements.

Known Issues
~~~~~~~~~~~~

* At high batch size (>=32), we have observed performance degradation with ``shard-on-load`` for some models such as Llama3.1-8B. Our current recommendation is to disable this feature by enabling ``save_sharded_checkpoint`` in ``NeuronConfig`` when you trace and compile the model.
* ``spmd_mode = True`` does not work when provided to the ``parallel_model_trace`` API. ``parallel_model_trace`` will be deprecated in the next Neuron SDK release.


----

.. _nxd-core-2-25-0-rn:

NxD Core [0.14.18461] (Neuron 2.25.0 Release)
-----------------------------------------------

Date of Release: 07/31/2025

Improvements
~~~~~~~~~~~~~~~

**Inference**

* ModelBuilder V2: ModelBuilder V2 provides a simplified version of the ModelBuilder API that is more flexible and extensible. This API includes basic building blocks that you can use to trace, compile, and load modules to Neuron. For more information, see :ref:`nxd-core-model-builder-v2` and the updated `Llama-3.2-1B reference inference sample <https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/llama>`__.

**Training**

* Support for Shared Experts: Shared Experts allow multiple model components to utilize the same expert neural networks. This release adds full support for Shared Experts in training workloads.

Date of Release: 06/24/2025

Improvements
~~~~~~~~~~~~~~~

**Inference:**

* Add ``--auto-cast=none`` compiler arg by default in ModelBuilder to ensure model dtypes are preserved during compilation.
* Update ModelBuilder to cast model weights based on dtypes defined in module parameters.
* Add support for PyTorch 2.7. This release includes support for PyTorch 2.5, 2.6, and 2.7.
* Other minor fixes and improvements.

**Training:**

* Added support for transformers 4.48.0

Bug Fixes
~~~~~~~~~

* Other minor fixes and improvements.


----

.. _nxd-core-2-23-0-rn:

NxD Core [0.12.12111] (Neuron 2.23.0 Release)
-----------------------------------------------

Date of Release: 05/20/2025

Improvements
~~~~~~~~~~~~~~~

**Inference:**

* Improve the Model Builder API. Note: The Model Builder API is in beta.
  
  * Add Neuron Persistent Cache support to Model Builder. Now, Model Builder caches compiled model artifacts to reduce compilation time.
  * Improve the performance of weight sharding in Model Builder to support shard-on-load in NxD Inference.
  * Improve the performance of Model Builder trace when HLO ``debug`` mode is enabled.

* Add a Llama-3.2-1B reference inference sample using NxD Core.
* Remove the unsupported NxD inference examples. You can use the NxD Inference library to run inference with on Neuron using NxD.
* Other minor fixes and improvements.

**Training:**

* Context parallel support for sequence lengths up to 32k on TRN1 (beta feature)

**General:**

* Update the package version to include additional information.

Bug Fixes
~~~~~~~~~

* Other minor fixes and improvements.


----

.. _nxd-core-2-22-0-rn:

NxD Core [0.11.0] (Neuron 2.22.0 Release)
------------------------------------------

Date of Release: 04/03/2025

Improvements
~~~~~~~~~~~~~~~

**Inference:**

* Improve the performance of weight sharding by up to 60-70%, depending on the model.
* You can now configure modules to skip during quantization with the ``modules_to_not_convert`` argument.
* Other minor fixes and improvements.

**Training:**

* Fixed issue with wikicorpus dataset download
* Updated model load for LoRA checkpoints

Bug Fixes
~~~~~~~~~

* Fixed issue with wikicorpus dataset download
* Updated model load for LoRA checkpoints
* Other minor fixes and improvements.

Known Issues
~~~~~~~~~~~~

* With PT2.5, some of the key workloads like Llama3-8B training may show reduced performance when using ``--llm-training`` compiler flag as compared to PT2.1. In such a case, try removing ``--llm-training`` flag from ``NEURON_CC_FLAGS`` in the run.sh only if using Neuron Kernel Interface.


----

.. _nxd-core-2-21-1-rn:

NxD Core [0.10.1] (Neuron 2.21.1 Release)
------------------------------------------

Date of Release: 01/14/2025

Improvements
~~~~~~~~~~~~~~~

**Inference:**

* Fix an issue with sequence parallel support for quantized models.

Bug Fixes
~~~~~~~~~

* Fix an issue with sequence parallel support for quantized models.


----

.. _nxd-core-2-21-0-rn:

NxD Core [0.10.0] (Neuron 2.21.0 Release)
------------------------------------------

Date of Release: 12/20/2024

Improvements
~~~~~~~~~~~~~~~

**Training:**

* Added support for HuggingFace Llama3 70B with Trn2 instances
* Added support for PyTorch 2.5
* Added DPO support for post-training model alignment
* Added fused QKV optimization in GQA models
* Support for Mixture-of-Experts with Tensor, Sequence, and Pipeline parallelism

Known Issues
~~~~~~~~~~~~

* With PT2.5, some of the key workloads like Llama3-8B training may show reduced performance when using ``--llm-training`` compiler flag as compared to PT2.1. In such a case, try removing ``--llm-training`` flag from ``NEURON_CC_FLAGS`` in the run.sh


----

.. _nxd-core-2-20-0-rn:

NxD Core [0.9.0] (Neuron 2.20.0 Release)
-----------------------------------------

Date of Release: 09/16/2024

Improvements
~~~~~~~~~~~~~~~

**Training:**

* Added LoRA adaptor support
* Added support for GPU compatible precision support using ZeRO-1

**Inference:**

* Added inference example for DBRX, and Mixtral models
* Improved inference performance with sequence length autobucketing
* Improved trace time for inference examples
* Reduced memory usage by sharing weights across prefill and decode traced models


----

.. _nxd-core-2-19-0-rn:

NxD Core [0.8.0] (Neuron 2.19.0 Release)
-----------------------------------------

Date of Release: 07/03/2024

Improvements
~~~~~~~~~~~~~~~

* Added support for Interleave pipeline parallel. At large cluster sizes, interleave pipeline schedule should help to reduce the pipeline bubble, thereby increasing training throughput.
* Added integration with flash attention kernel for longer sequence length training. See :ref:`Llama3 8K sequence-length training sample <llama3_tp_zero1_tutorial>`.
* Added support for naive speculative decoding, enabling assistance during the token generation process by predicting tokens with a draft model and verifying the predicted tokens with the original target model. Refer to the Neuronx Distributed inference developer guide for an example.
* Added integration with flash attention kernel for longer sequence length inference. See an end to end example of CodeLlama-13b model with 16K sequence length.
* Added support for scaled inference to run for Llama-2 70b or similar sized models

Known Issues
~~~~~~~~~~~~

* Model checkpointing saves sharded checkpoints. Users will have to write a script to combine the shards
* Validation/Evaluation with interleaved pipeline feature is not supported.
* Due to weights not being able to be shared across context encoding and token generation trace, inference scale is tested for models up to size Llama-2-70b. For model configurations above this, there is a risk of OOM errors.
* Tracing Llama-2-70b sized models for inference and loading them to device can take close to two hours. This is due to duplicate sharding of weights for both context encoding and token generation traces.


----

.. _nxd-core-2-18-0-rn:

NxD Core [0.7.0] (Neuron 2.18.0 Release)
-----------------------------------------

Date of Release: 04/01/2024

Improvements
~~~~~~~~~~~~~~~

* Added support for Pipeline-parallelism training using PyTorch-lightning
* Added support for fine-tuning a model and running evaluation on the fine-tuned model using optimum-neuron
* Added support for auto-partitioning the pipeline parallel stages for training large models
* Added support for async checkpointing, optimizing the checkpoint saving time.
* Added support for auto-resume from a checkpoint, in case training job crashes.
* Added support for sequence length autobucketing in inference
* Added support for inference with bfloat16
* Improved performance for Llama-2-7b inference example.

Known Issues
~~~~~~~~~~~~

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.


----

.. _nxd-core-2-16-0-rn:

NxD Core [0.6.0] (Neuron 2.16.0 Release)
-----------------------------------------

Date of Release: 12/21/2023

Improvements
~~~~~~~~~~~~~~~

* Added support for Model/Optimizer wrapper that handles the parallelization in both model and optimizer.
* Added support for PyTorch-lightning. This allows users to train models using Tensor-parallelism and Data-parallelism.
* Added new checkpoint save/load APIs that handles the parallelization and dumps/loads the checkpoint.
* Added a new QKV module which has the ability to replicate the KV heads and produce the query, key and value states.
* Reduced the model initialization time when pipeline-parallel distributed strategy is used.
* Added support for limiting max parallel compilations in parallel_model_trace. This resolves many out of memory errors by reducing the host memory usage.
* Added example for Llama-2-7b inference. This is still early in development and is not well-optimized. The current recommendation is to use ``transformers-neuronx`` for optimal performance of llama inference.

Known Issues
~~~~~~~~~~~~

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.
* Pipeline-parallelism is not supported as part of PyTorch-lightning integration.


----

.. _nxd-core-2-15-0-rn:

NxD Core [0.5.0] (Neuron 2.15.0 Release)
-----------------------------------------

Date of Release: 10/26/2023

Improvements
~~~~~~~~~~~~~~~

* Added support for pipeline-parallelism for distributed training.
* Added support for serialized checkpoint saving/loading, resulting in better checkpoint saving/loading time.
* Added support for mixed precision training using ``torch.autocast``.
* Fixed an issue with Zero1 checkpoint saving/loading.

Bug Fixes
~~~~~~~~~

* Fixed an issue with Zero1 checkpoint saving/loading.

Known Issues
~~~~~~~~~~~~

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.


----

.. _nxd-core-2-14-0-rn:

NxD Core [0.4.0] (Neuron 2.14.0 Release)
-----------------------------------------

Date of Release: 09/15/2023

Improvements
~~~~~~~~~~~~~~~

* Added API for padding attention heads when they are not divisible by tensor-parallel degree
* Added a constant threadpool for distributed inference
* Fixed a bug with padding_idx in ParallelEmbedding layer
* Fixed an issue with checkpoint loading to take into account the stride parameter in tensor parallel layers

Bug Fixes
~~~~~~~~~

* Fixed a bug with padding_idx in ParallelEmbedding layer
* Fixed an issue with checkpoint loading to take into account the stride parameter in tensor parallel layers

Known Issues
~~~~~~~~~~~~

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.


----

.. _nxd-core-2-13-0-rn:

NxD Core [0.3.0] (Neuron 2.13.0 Release)
-----------------------------------------

Date of Release: 08/28/2023

Improvements
~~~~~~~~~~~~~~~

* Added Zero1 Optimizer support that works with tensor-parallelism
* Added support for sequence-parallel that works with tensor-parallelism
* Added IO aliasing feature in parallel_trace api, which can allow marking certains tensors as state tensors
* Fixed hangs when tracing models using parallel_trace for higher TP degree

Bug Fixes
~~~~~~~~~

* Fixed hangs when tracing models using parallel_trace for higher TP degree

Known Issues
~~~~~~~~~~~~

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.


----

.. _nxd-core-2-12-0-rn:

NxD Core [0.2.0] (Neuron 2.12.0 Release)
-----------------------------------------

Date of Release: 07/19/2023

Improvements
~~~~~~~~~~~~~~~

* Added parallel cross entropy loss function.

Known Issues
~~~~~~~~~~~~

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.


----

.. _nxd-core-2-11-0-rn:

NxD Core [0.1.0] (Neuron 2.11.0 Release)
-----------------------------------------

Date of Release: 06/14/2023

Improvements
~~~~~~~~~~~~~~~

* Releasing the Neuron Distributed (``neuronx-distributed``) library for enabling large language model training/inference.
* Added support for tensor-parallelism training/inference.

Known Issues
~~~~~~~~~~~~

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.