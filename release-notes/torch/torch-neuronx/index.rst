.. _torch-neuronx-rn:

PyTorch Neuron (``torch-neuronx``) release notes
================================================

.. contents:: Table of Contents
   :local:
   :depth: 1

PyTorch Neuron for Trainium is a software package that enables PyTorch
users to train their models on Trainium.

Release [1.12.0.1.4.0]
----------------------
Date: 12/12/2022

Summary
~~~~~~~

What’s new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added support for PyTorch 1.12.
- Setting XLA_DOWNCAST_BF16=1 now also enables stochastic rounding by default (as done with XLA_USE_BF16=1).
- Added support for :ref:`capturing snapshots <torch-neuronx-snapshotting>` of inputs, outputs and graph HLO for debug.
- Fixed issue with parallel compile error when both train and evaluation are enabled in HuggingFace fine-tuning tutorial.
- Added support for LAMB optimizer in FP32 mode.

Resolved Issues
~~~~~~~~~~~~~~~

NaNs seen with transformers version >= 4.21.0 when running HF BERT fine-tuning or pretraining with XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running HuggingFace BERT (any size) fine-tuning tutorial or pretraining tutorial with transformers version >= 4.21.0 and using XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1, you will see NaNs in the loss immediately at the first step. More details on the issue can be found at `pytorch/xla#4152 <https://github.com/pytorch/xla/issues/4152>`_. The workaround is to use 4.20.0 or earlier (the tutorials currently recommend version 4.15.0) or add the line ``transformers.modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16`` to your Python training script (as now done in latest tutorials). `A permanent fix <https://github.com/huggingface/transformers/pull/20562>`_ will become part of an upcoming HuggingFace transformers release.

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

Number of data parallel training workers on one Trn1 instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of workers used in single-instance data parallel
training can be one of the following values: 1 or 2 for trn1.2xlarge and 1, 2, 8 or 32 for trn1.32xlarge.

Release [1.11.0.1.2.0]
----------------------
Date: 10/27/2022

Summary
~~~~~~~

What’s new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added support for argmax.
- Clarified error messages for runtime errors ``NRT_UNINITIALIZED`` and ``NRT_CLOSED``.
- When multi-worker training is launched using torchrun on one instance, framework now handles runtime state cleanup at end of training.

Resolved Issues
~~~~~~~~~~~~~~~

Drop-out rate ignored in dropout operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A known issue in the compiler's implementation of dropout caused drop-rate to be ignored in the last release. It is fixed in the current release.

Runtime error "invalid offset in Coalesced\_memloc\_..." followed by "Failed to process dma block: 1703"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Previously, when running MRPC fine-tuning tutorial with ``bert-base-*`` model, you would encounter runtime error "invalid offset in Coalesced\_memloc\_..." followed by "Failed to process dma block: 1703". This is fixed in the current release.

Compilation error: "TongaSBTensor[0x7fb2a46e0830]:TongaSB partitions[0] uint8 %138392[128, 512]"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Previously, when compiling MRPC fine-tuning tutorial with ``bert-large-*`` and FP32 (no XLA_USE_BF16=1) for two workers or more, you would encounter compiler error that looks like ``Error message:  TongaSBTensor[0x7fb2a46e0830]:TongaSB partitions[0] uint8 %138392[128, 512]`` followed by ``Error class:    KeyError``. Single worker fine-tuning is not affected. This is fixed in the current release.

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

Number of data parallel training workers on one Trn1 instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of workers used in single-instance data parallel
training can be one of the following values: 1 or 2 for trn1.2xlarge and 1, 2, 8 or 32 for trn1.32xlarge.


Release [1.11.0.1.1.1]
----------------------
Date: 10/10/2022


Summary
~~~~~~~

This is the initial release of PyTorch Neuron that supports Trainium for
users to train their models on the new EC2 Trn1 instances.


What’s new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Announcing the first PyTorch Neuron release for training.

- XLA device support for Trainium
- PyTorch 1.11 with XLA backend support in torch.distributed
- torch-xla distributed support
- Single-instance and multi-instance distributed training using torchrun
- Support for ParallelCluster and SLURM with node-level scheduling granularity
- Persistent cache for compiled graph
- :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`
  utility to help speed up compilation
- Optimizer support: SGD, AdamW
- Loss functions supported: NLLLoss
- Python versions supported: 3.7, 3.8
- Multi-instance training support with EFA
- Support PyTorch’s BF16 automatic mixed precision

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

Number of data parallel training workers on one Trn1 instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of workers used in single-instance data parallel
training can be one of the following values: 1 or 2 for trn1.2xlarge and 1, 2, 8 or 32 for trn1.32xlarge.

Drop-out rate ignored in dropout operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A known issue in the compiler's implementation of dropout caused drop-rate to be ignored. Will be fixed in a follow-on release.

Runtime error "invalid offset in Coalesced\_memloc\_..." followed by "Failed to process dma block: 1703"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently, when running MRPC fine-tuning tutorial with ``bert-base-*`` model, you will encounter runtime error "invalid offset in Coalesced\_memloc\_..." followed by "Failed to process dma block: 1703".
This issue will be fixed in an upcoming release.

Compilation error: "TongaSBTensor[0x7fb2a46e0830]:TongaSB partitions[0] uint8 %138392[128, 512]"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When compiling MRPC fine-tuning tutorial with ``bert-large-*`` and FP32 (no XLA_USE_BF16=1) for two workers or more, you will encounter compiler error that looks like ``Error message:  TongaSBTensor[0x7fb2a46e0830]:TongaSB partitions[0] uint8 %138392[128, 512]`` followed by ``Error class:    KeyError``. Single worker fine-tuning is not affected. This issue will be fixed in an upcoming release.
