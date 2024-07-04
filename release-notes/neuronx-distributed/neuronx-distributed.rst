.. _neuronx-distributed-rn:


Neuron Distributed Release Notes (``neuronx-distributed``)
==========================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for Neuronx-Distributed library.

Neuron Distributed [0.8.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 07/03/2024

New in this release
-------------------

* Added support for Interleave pipeline parallel. At large cluster sizes, interleave pipeline schedule should help to reduce the pipeline bubble, thereyby increasing training throughput.
* Added integration with flash attention kernel for longer sequence length training. See :ref:`Llama3 8K sequence-length training sample<llama2_7b_tp_zero1_tutorial>`.
* Added support for naive speculative decoding, enabling assistance during the token generation process by predicting tokens with a draft model and verifying the predicted tokens with the original target model. Refer to the Neuronx Distributed inference developer guide for an example. 
* Added integration with flash attention kernel for longer sequence length inference. See an end to end example of CodeLlama-13b model with 16K sequence length.
* Added support for scaled inference to run for Llama-2 70b or similar sized models

Known Issues and Limitations
----------------------------

* Model checkpointing saves sharded checkpoints. Users will have to write a script to combine the shards
* Validation/Evaluation with interleaved pipeline feature is not supported.
* Due to weights not being able to be shared across context encoding and token generation trace, inference scale is tested for models up to size Llama-2-70b. For model configurations above this, there is a risk of OOM errors.
* Tracing Llama-2-70b sized models for inference and loading them to device can take close to two hours. This is due to duplicate sharding of weights for both context encoding and token generation traces.

Neuron Distributed [0.7.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 04/01/2024

New in this release
-------------------

* Added support for Pipeline-parallelism training using PyTorch-lightning
* Added support for fine-tuning a model and running evaluation on the fine-tuned model using optimum-neuron
* Added support for auto-partitioning the pipeline parallel stages for training large models
* Added support for async checkpointing, optimizing the checkpoint saving time.
* Added support for auto-resume from a checkpoint, in case training job crashes.
* Added support for sequence length autobucketing in inference
* Added support for inference with bfloat16
* Improved performance for Llama-2-7b inference example.

Known Issues and Limitations
----------------------------

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.

Neuron Distributed [0.6.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 12/21/2023

New in this release
-------------------

* Added support for Model/Optimizer wrapper that handles the parallelization in both model and optimizer.
* Added support for PyTorch-lightning. This allows users to train models using Tensor-parallelism and Data-parallelism.
* Added new checkpoint save/load APIs that handles the parallelization and dumps/loads the checkpoint.
* Added a new QKV module which has the ability to replicate the KV heads and produce the query, key and value states.
* Reduced the model initialization time when pipeline-parallel distributed strategy is used.
* Added support for limiting max parallel compilations in parallel_model_trace. This resolves many out of memory errors by reducing the host memory usage.
* Added example for Llama-2-7b inference. This is still early in development and is not well-optimized. The current recommendation is to use `transformers-neuronx` for optimal performance of llama inference.

Known Issues and Limitations
----------------------------

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.
* Pipeline-parallelism is not supported as part of PyTorch-lightning integration.

Neuron Distributed [0.5.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 10/26/2023

New in this release
-------------------

* Added support for pipeline-parallelism for distributed training.
* Added support for serialized checkpoint saving/loading, resulting in better checkpoint saving/loading time.
* Added support for mixed precision training using `torch.autocast`.
* Fixed an issue with Zero1 checkpoint saving/loading.


Known Issues and Limitations
----------------------------

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.

Neuron Distributed [0.4.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 9/15/2023

New in this release
-------------------

* Added API for padding attention heads when they are not divisible by tensor-parallel degree
* Added a constant threadpool for distributed inference
* Fixed a bug with padding_idx in ParallelEmbedding layer
* Fixed an issue with checkpoint loading to take into account the stride parameter in tensor parallel layers

Known Issues and Limitations
----------------------------

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.

Neuron Distributed [0.3.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 8/28/2023

New in this release
-------------------

* Added Zero1 Optimizer support that works with tensor-parallelism
* Added support for sequence-parallel that works with tensor-parallelism
* Added IO aliasing feature in parallel_trace api, which can allow marking certains tensors as state tensors
* Fixed hangs when tracing models using parallel_trace for higher TP degree

Known Issues and Limitations
----------------------------

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.

Neuron Distributed [0.2.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 7/19/2023

New in this release
-------------------

* Added parallel cross entropy loss function.

Known Issues and Limitations
----------------------------

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.

Date: 6/14/2023

New in this release
-------------------

* Releasing the Neuron Distributed (``neuronx-distributed``) library for enabling large language model training/inference.
* Added support for tensor-parallelism training/inference.

Known Issues and Limitations
----------------------------

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.
