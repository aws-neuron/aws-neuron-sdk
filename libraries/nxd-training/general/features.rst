.. _nxdt_features:

Neuronx Distributed Training Library Features
=============================================

The library is meant to provide an end-to-end framework for training on Trainium instances. The NxD Training is a
collection of open-source libraries, tools, and resources that empowers customers to run end-to-end training workflows
on Neuron. Its an extension to Neuronx-Distributed (NxD) library. NxD Training incorporates the distributed strategies
primitives from NxD (i.e., NxD Parallel Primitives),while maintaining a design that is ready to integrate partitioning
technologies from native PyTorch or from OpenXLA such as GSPMD. NxD Training also supports  PyTorch Lightning (PTL)
Trainer and extends NxD to include data engineering features from NeMo, such as data loaders, datasets, and tokenizers,
as well as ML engineering capabilities from NeMo like monitoring, logging, and experiment management. Furthermore,
the NxD Training framework introduces support for training techniques such as pre-training and fine-tuning, along with
a model hub featuring end-to-end examples for state of the art models like LLama, GPT, and Mixtral MoE implemented using
both HuggingFace and Megatron-LM model classes.

The framework uses the distributed training technology from NxD. This allows the framework to support all the
sharding techniques and Modules already supported by NxD.

.. contents:: Table of contents
   :local:
   :depth: 2

Distributed Techniques
-----------------------

1. Data-parallelism
2. `Tensor-parallelism <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tensor_parallelism_overview.html#tensor-parallelism-overview>`_
3. `Sequence-Parallelism <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/activation_memory_reduction.html#sequence-parallelism>`_
4. `Pipeline-parallelism <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/pipeline_parallelism_overview.html>`_
    1. 1F1B pipeline schedule
    2. Interleave pipeline schedule (or virtual pipeline parallel)
5. `Zero1 <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/zero1_gpt2.html#what-is-zero-1>`_
6. Expert-parallelism

Modules
--------

1. `Grouped Query Attention layer <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#gqa-qkv-linear-module>`_
2. Mixture of Experts (MoE)

Model/Optimizer Precision
-------------------------

To cater to different types of precision that can affect the overall training, the library provides an option to
configure the following:

1. Zero1 with Master weights in FP32
2. BF16 + Stochastic Rounding
3. FP32

Checkpoint Saving/Loading
-------------------------
When we are working with large models and running training for a long time, checkpointing becomes an important
part of training models. The framework supports the following features for checkpointing:

1. Save/Load sharded checkpoints
2. Asynchronous checkpoint saving/loading
3. Ability to keep only the last K checkpoints
4. Auto-resume training jobs from previous checkpoints
5. Ability to dump a checkpoint to S3

To optimize the checkpointing time, we have enabled dumping of checkpoints from all ranks to distribute the workload
and parallelize the checkpoint saving. Similarly when loading checkpoints, the API would load only on 1 data-parallel
rank and broadcast it to all ranks. This improves the checkpoint loading time as it avoids contention on the file
system.

Training Recipes
----------------

The library supports the following training recipes:

1. Pre-training: The library shows examples of pretraining models like LLama2/3-8B/70B , GPT, Mistral, and Mixtral MoE
2. Supervised Fine-tuning: Showcase fine-tuning of llama-3 model with a chat dataset.
