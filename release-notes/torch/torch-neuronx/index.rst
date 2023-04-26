.. |FSDP| replace:: :class:`~torch.distributed.fsdp.FullyShardedDataParallel`
.. |Trn1| replace:: :ref:`Trn1 <aws-trn1-arch>`
.. |Inf2| replace:: :ref:`Inf2 <aws-inf2-arch>`

.. _torch-neuronx-rn:

PyTorch Neuron (``torch-neuronx``) release notes
================================================

.. contents:: Table of Contents
   :local:
   :depth: 1

PyTorch Neuron for |Trn1|/|Inf2| is a software package that enables PyTorch
users to train, evaluate, and perform inference on second-generation Neuron
hardware (See: :ref:`NeuronCore-v2 <neuroncores-v2-arch>`).


Release [1.13.0.1.7.0]
----------------------
Date: 04/xx/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- Added an improved Neuron-optimized AdamW optimizer implementation.
- Added an improved Neuron-optimized :class:`torch.nn.Dropout` implementation.

  - Added an assertion when the :class:`torch.nn.Dropout` argument
    ``inplace=True`` during training. This is currently not supported on Neuron.

- Added XLA lowering for ``aten::count_nonzero``
- Added initial |FSDP| (FSDP) support.
- Added MiCS (Minimize Communication Scale) support in FSDP. Supported
  partition sizes are 2,8,32 and multiples of 32.

Inference support:

- Added profiling support for models compiled with :func:`torch_neuronx.trace`
- Added torch_neuronx.DataParallel for models compiled with :func:`torch_neuronx.trace`


Resolved Issues (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unexpected behavior with :class:`torch.autocast`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed an issue where :class:`torch.autocast` did not correctly autocast
when using ``torch.bfloat16``


Resolved Issues (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Error when using the original model after ``torch_neuronx.trace``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed an issue where model parameters would be moved to the Neuron ``'xla'``
device during :func:`torch_neuronx.trace` and would no longer be available to
execute on the original device. This made it more difficult to compare Neuron
models against CPU since previously this would require manually moving
parameters back to CPU.

Error when using the ``xm.xla_device()`` object followed by using ``torch_neuronx.trace``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed an issue where XLA device execution and :func:`torch_neuronx.trace` could
not be performed in the same python process.

Error when executing ``torch_neuronx.trace`` with ``torch.bfloat16`` input/output tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed an issue where :func:`torch_neuronx.trace` could not compile models which
consumed or produced ``torch.bfloat16`` values.

Known Issues and Limitations (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|FSDP| does not support all features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |FSDP| checkpoint consolidation not yet supported.
- Restoration of random states from Activation checkpoints only works with the
  new Neuron dropout operator implementation.

Memory leaking in ``glibc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``glibc`` malloc memory leaks affect Neuron and may be temporarily limited by
setting ``MALLOC_ARENA_MAX``.

Slower BERT bf16 Phase 1 Single Node Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As of the Neuron 2.9.0 release, :ref:`BERT phase 1 pretraining <hf-bert-pretraining-tutorial>`
performance has regressed by approximately 8-9% when executed on a *single
node* only (i.e. just one ``trn1.32xlarge`` instance).

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convolution is not supported during training.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the
scripts that don't use DDP. We also see a throughput drop with DDP. This is a
known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its
own graph. This causes an error in the runtime, and you may see errors that
look like this:

::

    bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Lower throughput for BERT-large training on AL2 instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We see a performance drop of roughly 5-10% for BERT model training on AL2
instances. This is because of the increase in time required for tracing the
model.

Known Issues and Limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmin` produces incorrect results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.argmin` produces incorrect results.

No automatic partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when Neuron encounters an operation that it does not support during
:func:`torch_neuronx.trace`, this will cause an error. The intended behavior
when tracing is to automatically partition the model into separate subgraphs
that run on NeuronCores and subgraphs that run on CPU. See
:ref:`pytorch-neuron-supported-operators` for a list of supported operators.

Torchscript serialization error with compiled artifacts larger than 4GB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using :func:`torch_neuronx.trace`, compiled artifacts which exceed 4GB
cannot be serialized. Serializing the torchscript artifact will trigger a
segfault. This issue is resolved in torch but is not yet
released: https://github.com/pytorch/pytorch/pull/99104


Release [1.13.0.1.6.1]
----------------------
Date: 04/19/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- No changes

Inference support:

- Enable deserialized TorchScript modules to be compiled with :func:`torch_neuronx.trace`


Release [1.13.0.1.6.0]
----------------------
Date: 03/28/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- Added pipeline parallelism support in AWS Samples for Megatron-LM

Inference support:

- Added model analysis API: torch_neuronx.analyze
- Added HLO opcode support for:

  - kAtan2
  - kAfterAll
  - kMap

- Added XLA lowering support for:

  - aten::glu
  - aten::scatter_reduce

- Updated torch.nn.MSELoss to promote input data types to a compatible type

Resolved Issues (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~

GRPC timeout errors when running Megatron-LM GPT 6.7B tutorial on multiple instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running AWS Samples for Megatron-LM GPT 6.7B tutorial over multiple instances, you may encounter GRPC timeout errors like below:

::

    E0302 01:10:20.511231294  138645 chttp2_transport.cc:1098]   Received a GOAWAY with error code ENHANCE_YOUR_CALM and debug data equal to "too_many_pings"
    2023-03-02 01:10:20.511500: W tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc:157] RPC failed with status = "UNAVAILABLE: Too many pings" and grpc_error_string = "{"created":"@1677719420.511317309","description":"Error received from peer ipv4:10.1.35.105:54729","file":"external/com_github_grpc_grpc/src/core/lib/surface/call.cc","file_line":1056,"grpc_message":"Too many pings","grpc_status":14}", maybe retrying the RPC


or:

::

    2023-03-08 21:18:27.040863: F tensorflow/compiler/xla/xla_client/xrt_computation_client.cc:476] Non-OK-status: session->session()->Run(session_work->feed_inputs, session_work->outputs_handles, &outputs) status: UNKNOWN: Stream removed


This is due to excessive DNS lookups during execution, and is fixed in this release.


NaNs seen with transformers version >= 4.21.0 when running HF GPT fine-tuning or pretraining with XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using Hugging Face transformers version >= 4.21.0 can produce NaN outputs for GPT models when using full BF16 (XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1) plus stochastic rounding. This issue occurs due to large negative constants used to implement attention masking (https://github.com/huggingface/transformers/pull/17306). To workaround this issue, please use transformers version <= 4.20.0.


Resolved Issues (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmax` now supports single argument call variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously only the 3 argument variant of :func:`torch.argmax` was supported. Now the single argument call variant is supported.

Known Issues and Limitations (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Slower BERT bf16 Phase 1 Single Node Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Neuron 2.9.0 release, :ref:`BERT phase 1 pretraining <hf-bert-pretraining-tutorial>`
performance has regressed by approximately 8-9% when executed on a *single
node* only (i.e. just one ``trn1.32xlarge`` instance).

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the scripts that don't use DDP. We also see a throughput drop 
with DDP. This is a known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its own graph. This causes an error in the runtime, and
you may see errors that look like this: ``bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Lower throughput for BERT-large training on AL2 instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We see a performance drop of roughly 5-10% for BERT model training on AL2 instances. This is because of the increase in time required for tracing the model.

Known Issues and Limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmin` produces incorrect results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.argmin` now supports both the single
argument call variant and the 3 argument variant.
However, :func:`torch.argmin` currently produces
incorrect results.

Error when using the ``xm.xla_device()`` object followed by using ``torch_neuronx.trace``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Executing a model using the ``xm.xla_device()`` object followed by using ``torch_neuronx.trace`` in the same process can produce errors in specific situations due to torch-xla caching behavior. It is recommended that only one type of execution is used per process.

Error when executing ``torch_neuronx.trace`` with ``torch.bfloat16`` input/output tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Executing ``torch_neuronx.trace`` with ``torch.bfloat16`` input/output tensors can cause an error. It is currently recommended to use an alternative torch data type in combination with compiler casting flags instead.


No automatic partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, there's no automatic partitioning of a model into subgraphs that run on NeuronCores and subgraphs that run on CPU
Operations in the model that are not supported by Neuron would result in compilation error. Please see :ref:`pytorch-neuron-supported-operators` for a list of supported operators.


Release [1.13.0.1.5.0]
----------------------
Date: 02/24/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- Added SPMD flag for XLA backend to generate global collective-compute replica groups

Inference support:

- Expanded inference support to inf2
- Added Dynamic Batching

Resolved Issues
~~~~~~~~~~~~~~~

Known Issues and Limitations (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the scripts that don't use DDP. We also see a throughput drop
with DDP. This is a known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its own graph. This causes an error in the runtime, and
you may see errors that look like this: ``bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Lower throughput for BERT-large training on AL2 instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We see a performance drop of roughly 5-10% for BERT model training on AL2 instances. This is because of the increase in time required for tracing the model.

Known Issues and Limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmax` and :func:`torch.argmin` do not support the single argument call variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.argmax` and :func:`torch.argmin` do not support the single
argument call variant. Only the 3 argument variant of these functions is
supported. The ``dim`` argument *must be* specified or this function will
fail at the call-site. Secondly, :func:`torch.argmin` may produce
incorrect results.

No automatic partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, there's no automatic partitioning of a model into subgraphs that run on NeuronCores and subgraphs that run on CPU
Operations in the model that are not supported by Neuron would result in compilation error. Please see :ref:`pytorch-neuron-supported-operators` for a list of supported operators.

Release [1.13.0.1.4.0]
----------------------
Date: 02/08/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- Added support for PyTorch 1.13
- Added support for Python version 3.9
- Added support for torch.nn.parallel.DistributedDataParallel (DDP) along with a :ref:`tutorial <neuronx-ddp-tutorial>`
- Added optimized lowering for Softmax activation
- Added support for LAMB optimizer in BF16 mode

Added initial support for inference on Trn1, including the following features:

- Trace API (torch_neuronx.trace)
- Core placement API (experimental)
- Python 3.7, 3.8 and 3.9 support
- Support for tracing models larger than 2 GB

The following inference features are not included in this release:

- Automatic partitioning of a model into subgraphs that run on NeuronCores and subgraphs that run on CPU
- cxx11 ABI wheels

Resolved Issues
~~~~~~~~~~~~~~~

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the scripts that don't use DDP. We also see a throughput drop
with DDP. This is a known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its own graph. This causes an error in the runtime, and
you may see errors that look like this: ``bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Lower throughput for BERT-large training on AL2 instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We see a performance drop of roughly 5-10% for BERT model training on AL2 instances. This is because of the increase in time required for tracing the model.


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
