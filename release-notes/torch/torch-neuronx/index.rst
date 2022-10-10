.. _torch-neuronx-rn:

PyTorch Neuron (``torch-neuronx``) release notes
================================================

.. contents:: Table of Contents
   :local:
   :depth: 1

PyTorch Neuron for Trainium is a software package that enables PyTorch
users to train their models on Trainium.

Release [1.11.0.1.1.1]
----------------------
Date: 10/1/2022


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
- `neuron_parallel_compile <../../frameworks/torch/api-reference-guide/training/pytorch-neuron-parallel-compile.html>`__
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
