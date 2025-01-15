.. _neuronx-distributed-training-rn:


NxD Training Release Notes (``neuronx-distributed-training``)
=============================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for Neuronx Distributed Training library.

.. _neuronx-distributed-training-rn-1-1-0:

Neuronx Distributed Training [1.1.1]

Date: 1/14/2025

Features in this release
------------------------

* Added a flag in Llama3/3.1 70B config to control the dtype of reduce-scatter operations in Column/Row Parallel linear layers.


Neuronx Distributed Training [1.1.0]

Date: 12/20/2024

Features in this release
------------------------

* Added support for HuggingFace Llama3/3.1 70B with trn2 instances
* Added support for custom pipeline parallel cuts in HuggingFace Llama3
* Added support for PyTorch 2.5
* Added support for DPO post-training model alignment
* Added support for Mixtral 8x7B Megatron and HuggingFace models
* Added option in checkpoint converter to download and convert checkpoints using HuggingFace model identifier
* Fix the validation loss to properly compute the average loss across the validation epoch
* Minor bug fixes for error logging and imports

Known Issues and Limitations
----------------------------

* Autocast option may not properly cast all inputs to bf16, recommended to use mixed precision option (currently is default) in configs for best results
* With PT2.5, some of the key workloads like Llama3-8B training may show a reduced performance when using `--llm-training` compiler flag as compared to PT2.1.
In such a case, try removing `--llm-training` flag from `compiler_flags` in the config.yaml

.. _neuronx-distributed-training-rn-1-0-1:

Neuronx Distributed Training [1.0.1]

Date: 11/20/2024

Features in this release
------------------------

* Added support for transformers 4.36.0

.. _neuronx-distributed-training-rn-1-0-0:

Neuronx Distributed Training [1.0.0]

Date: 09/16/2024

Features in this release
------------------------

This is the first release of NxD Training (NxDT), NxDT is a PyTorch-based library that adds support for user-friendly distributed training experience through a YAML configuration file compatible with NeMo,, allowing users to easily set up their training workflows. At the same time, NxDT maintains flexibility, enabling users to choose between using the YAML configuration file, PyTorch Lightning Trainer, or writing their own custom training script using the NxD Core.
The library supports PyTorch model classes including Hugging Face and Megatron-LM. Additionally, it leverages NeMo's data engineering and data science modules enabling end-to-end training workflows on NxDT, and providing a compatability with NeMo through minimal changes to the YAML configuration file for models that are already supported in NxDT. Furthermore, the functionality of the Neuron NeMo Megatron (NNM) library is now part of NxDT, ensuring a smooth migration path from NNM to NxDT.

This release of NxDT includes:

* Installation through `neuronx-distributed-training` package.
* Open Source Github repository: https://github.com/aws-neuron/neuronx-distributed-training 
* Support for YAML based interface allowing users to configure training from a config file.
* Support for 3D-parallelism, sequence-parallelism and zero1.
* Support for megatron-model and hugging-face based Llama model.
* Support flash attention kernel.
* Support for async checkpointing and s3 checkpointing.
* Examples to pretrain and fine-tune Llama model

Known Issues and Limitations
----------------------------

* Model checkpointing saves sharded checkpoints. Users will have to write a script to combine the shards
* Validation/Evaluation with interleaved pipeline feature is not supported.
* NxDT shows slightly higher memory utilization as compared to NxD based examples.
