.. meta::
    :description: Complete release notes for the NxD Training component across all AWS Neuron SDK versions.
    :keywords: nxd training, neuronx-distributed-training, release notes, aws neuron sdk
    :date-modified: 07/31/2025

.. _nxd-training_rn:

Component Release Notes for NxD Training
========================================

The release notes for the NxD Training (``neuronx-distributed-training``) Neuron component. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _nxd-training-2-25-0-rn:

NxD Training [1.5.0] (Neuron 2.25.0 Release)
---------------------------------------------

Date of Release: 07/31/2025

Improvements
~~~~~~~~~~~~~~~

* None

Bug Fixes
~~~~~~~~~

* Disable ``expert_index`` in Mixture of Experts (MoE) forwarding to limit the output to just hidden states and router logits (as expected).


----

.. _nxd-training-2-24-0-rn:

NxD Training [1.4.0] (Neuron 2.24.0 Release)
---------------------------------------------

Date of Release: 06/26/2025

Improvements
~~~~~~~~~~~~~~~

* Added support for PyTorch 2.7


----

.. _nxd-training-2-23-0-rn:

NxD Training [1.3.0] (Neuron 2.23.0 Release)
---------------------------------------------

Date of Release: 05/16/2025

Improvements
~~~~~~~~~~~~~~~

* (Beta release) Added autocast for HF based Llama3 8B and Llama3 70B models
* (Beta release) Added support for context parallel sequence lengths up to 32k on TRN1
* Added support for ORPO
* Added support for nemo-toolkit 2.1
* Added support for Transformers 4.48.0
* Added support for PyTorch-Lightning 2.5.0
* Added support for PyTorch 2.6


----

.. _nxd-training-2-22-0-rn:

NxD Training [1.2.0] (Neuron 2.22.0 Release)
---------------------------------------------

Date of Release: 04/03/2025

Improvements
~~~~~~~~~~~~~~~

* Added support for LoRA supervised fine-tuning.
* Added option to configure collectives data types.
* Minor fixes to reduce the amount of logs during training.
* Removes ``--llm-training`` flag by default in all configs, except llama2. Note: this flag should not be enabled when using the Neuron Kernel Interface.

Bug Fixes
~~~~~~~~~

* Minor fixes to reduce the amount of logs during training.


----

.. _nxd-training-2-21-1-rn:

NxD Training [1.1.1] (Neuron 2.21.1 Release)
---------------------------------------------

Date of Release: 01/14/2025

Improvements
~~~~~~~~~~~~~~~

* Added a flag in Llama3/3.1 70B config to control the dtype of reduce-scatter operations in Column/Row Parallel linear layers.


----

.. _nxd-training-2-21-0-rn:

NxD Training [1.1.0] (Neuron 2.21.0 Release)
---------------------------------------------

Date of Release: 12/20/2024

Improvements
~~~~~~~~~~~~~~~

* Added support for HuggingFace Llama3/3.1 70B with trn2 instances
* Added support for custom pipeline parallel cuts in HuggingFace Llama3
* Added support for PyTorch 2.5
* Added support for DPO post-training model alignment
* Added support for Mixtral 8x7B Megatron and HuggingFace models
* Added option in checkpoint converter to download and convert checkpoints using HuggingFace model identifier
* Fix the validation loss to properly compute the average loss across the validation epoch
* Minor bug fixes for error logging and imports

Bug Fixes
~~~~~~~~~

* Fix the validation loss to properly compute the average loss across the validation epoch
* Minor bug fixes for error logging and imports

Known Issues
~~~~~~~~~~~~

* Autocast option may not properly cast all inputs to bf16, recommended to use mixed precision option (currently is default) in configs for best results
* With PT2.5, some of the key workloads like Llama3-8B training may show a reduced performance when using ``--llm-training`` compiler flag as compared to PT2.1. In such a case, try removing ``--llm-training`` flag from ``compiler_flags`` in the config.yaml


----

.. _nxd-training-2-20-1-rn:

NxD Training [1.0.1] (Neuron 2.20.1 Release)
---------------------------------------------

Date of Release: 11/20/2024

Improvements
~~~~~~~~~~~~~~~

* Added support for transformers 4.36.0


----

.. _nxd-training-2-20-0-rn:

NxD Training [1.0.0] (Neuron 2.20.0 Release)
---------------------------------------------

Date of Release: 09/16/2024

Improvements
~~~~~~~~~~~~~~~

* This is the first release of NxD Training (NxDT), NxDT is a PyTorch-based library that adds support for user-friendly distributed training experience through a YAML configuration file compatible with NeMo, allowing users to easily set up their training workflows. At the same time, NxDT maintains flexibility, enabling users to choose between using the YAML configuration file, PyTorch Lightning Trainer, or writing their own custom training script using the NxD Core.
* The library supports PyTorch model classes including Hugging Face and Megatron-LM. Additionally, it leverages NeMo's data engineering and data science modules enabling end-to-end training workflows on NxDT, and providing a compatability with NeMo through minimal changes to the YAML configuration file for models that are already supported in NxDT. Furthermore, the functionality of the Neuron NeMo Megatron (NNM) library is now part of NxDT, ensuring a smooth migration path from NNM to NxDT.

**This release of NxDT includes:**

* Installation through ``neuronx-distributed-training`` package.
* Open Source Github repository: https://github.com/aws-neuron/neuronx-distributed-training
* Support for YAML based interface allowing users to configure training from a config file.
* Support for 3D-parallelism, sequence-parallelism and zero1.
* Support for megatron-model and hugging-face based Llama model.
* Support flash attention kernel.
* Support for async checkpointing and s3 checkpointing.
* Examples to pretrain and fine-tune Llama model

Known Issues
~~~~~~~~~~~~

* Model checkpointing saves sharded checkpoints. Users will have to write a script to combine the shards
* Validation/Evaluation with interleaved pipeline feature is not supported.
* NxDT shows slightly higher memory utilization as compared to NxD based examples.