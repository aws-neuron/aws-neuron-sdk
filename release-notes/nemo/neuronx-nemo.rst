.. _neuronx-nemo-rn:


AWS Neuron Reference for Nemo Megatron(``neuronx-nemo-megatron``) Release Notes 
===============================================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for ``neuronx-nemo-megatron`` library.

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

