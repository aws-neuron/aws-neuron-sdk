.. _neuronx-nemo-rn:


AWS Neuron Reference for Nemo Megatron(``neuronx-nemo-megatron``) Release Notes
===============================================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for ``neuronx-nemo-megatron`` library.

``neuronx-nemo-megatron`` [0.8.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 12/20/2024

New in this release
-------------------

* Added support for HuggingFace to NeMo checkpoint conversion when virtual pipeline parallel is enabled.
* Added support for Python 3.11
* Added support for PyTorch 2.5
* Added collective compute coalescing for ZeRO-1 optimizer
* Bug fix for flash attention to ensure proper mixed precision data type handling

Known Issues and Limitations
----------------------------

None at this time.


``neuronx-nemo-megatron`` [0.7.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 09/16/2024

New in this release
-------------------

* Fixed issue with linear warmup with cosine annealing
* Fixed indexing issues with MPI job checkpoint conversion.
* Fixed pipeline parallel bug for NeMo to HF checkpoint conversion.

Known Issues and Limitations
----------------------------

None at this time.


``neuronx-nemo-megatron`` [0.6.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 07/03/2024

New in this release
-------------------

* Added support for fp32 gradient accumulation.
* Added support for flash attention kernel.
* Added option for zero1 with master weights.
* Checkpoint conversion script improvements.
* S3 checkpointing improvements.
* Zero1 checkpointing improvements
* Various bug fixes and improvements.


Known Issues and Limitations
----------------------------

None at this time.


``neuronx-nemo-megatron`` [0.5.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 04/01/2024

New in this release
-------------------

* Added support for LoRA fine tuning.
* Added support for Mistral 7B and sliding window attention
* Added support for Zero1 Automatic Mixed Precision.
* Improved throughput at scale of hundreds of nodes.
* Improved support for FP32 optimizer states.
* Merges up and gate projection in Llama for improved throughput.
* Various bug fixes and improvements.
* Fixes for checkpoint restoration accuracy issues.
* Fixes Zero1 checkpointing issues.


Known Issues and Limitations
----------------------------

None at this time.


``neuronx-nemo-megatron`` [0.4.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 10/15/2023

New in this release
-------------------

* Added Llama 70B model pre-training and finetuning support that works with tensor-parallelism and pipeline parallelism using Group Query Attention (GQA)
* Added GPT-NeoX 20B using  tensor parallelism and pipeline parallelism.
* Added Checkpoint conversion scripts from Nemo to HuggingFace models for LLama 7B, 13B, 70B, GPT-NeoX FineTuning
* Stability fixes for hangs observed for long running jobs checkpointing at regular time intervals.
* Enabled python 3.10 support with Nemo.

Known Issues and Limitations
----------------------------

* We are seeing few extra graph compilations than before. These are not limiting functionality or performance.
* Llama2-70B : Tested and validated on 8 nodes. Scaling beyond might see memory issues.

``neuronx-nemo-megatron`` [0.3.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 9/15/2023

New in this release
-------------------

* Added Llama 13B model support that works with tensor-parallelism and pipeline parallelism
* Zero1 Optimizer support that works with tensor-parallelism and pipeline parallelism
* Fixes for loading/saving checkpoint OOM issues while loading large models
* Added Docker support
* Feature to save only the last checkpoint and delete previous ones to conserve disk space
* Added FP32 OptimizerState option for mixed precision
* Added Validation loop support

Known Issues and Limitations
----------------------------

* Tested validation logic with smaller global batch sizes (32). Not tested larger global batch sizes.

