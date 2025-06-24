.. _introduce-pytorch-2-7:

Introducing PyTorch 2.7 Support
===============================

.. contents:: Table of contents
   :local:
   :depth: 2


What are we introducing?
------------------------

Starting with the :ref:`Neuron 2.24 <neuron-2.24.0-whatsnew>` release, customers will be able to upgrade to ``PyTorch NeuronX(torch-neuronx)`` supporting ``PyTorch 2.7``.

:ref:`setup-torch-neuronx` is updated to include installation instructions for PyTorch NeuronX 2.7 for Amazon Linux 2023 and Ubuntu 22. Note that PyTorch NeuronX 2.7 is supported on Python 3.9, 3.10, and 3.11.

Please review :ref:`migration guide <migrate_to_pytorch_2.7>` for possible changes to training scripts. No code changes are required for inference scripts.


.. _how-pytorch-2.7-different:

How is PyTorch NeuronX 2.7 different compared to PyTorch NeuronX 2.5?
---------------------------------------------------------------------

PyTorch NeuronX 2.7 uses Torch-XLA v2.7 and PyTorch v2.7 which have C++11 ABI enabled by default. 

Additionally, Torch-XLA v2.7 includes a fix for training performance issue https://github.com/pytorch/xla/issues/9037 .

See `Torch-XLA 2.7 release <https://github.com/pytorch/xla/releases/tag/v2.7.0>`__ for a full list.

See :ref:`migrate_to_pytorch_2.7` for changes needed to use PyTorch NeuronX 2.7.

.. note::

   GSPMD and Torch Dynamo (torch.compile) support in Neuron will be available in a future release.

.. _install_pytorch_neuron_2.7:

How can I install PyTorch NeuronX 2.7?
--------------------------------------------

To install PyTorch NeuronX 2.7 please follow the :ref:`setup-torch-neuronx` guides for Amazon Linux 2023 and Ubuntu 22 AMI. Please also refer to the Neuron multi-framework DLAMI :ref:`setup guide <setup-ubuntu22-multi-framework-dlami>` for Ubuntu 22 with a pre-installed virtual environment for PyTorch NeuronX 2.7 that you can use to get started. PyTorch NeuronX 2.7 can be installed using the following:

.. code::

    python -m pip install --upgrade neuronx-cc==2.* torch-neuronx==2.7.* torchvision

.. note::

   PyTorch NeuronX 2.7 is currently available for Python 3.9, 3.10, 3.11.

.. _migrate_to_pytorch_2.7:

Migrate your application to PyTorch 2.7
---------------------------------------

Please make sure you have first installed the PyTorch NeuronX 2.7 as described above in :ref:`installation guide <install_pytorch_neuron_2.7>`


Migrating training scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^

To migrate the training scripts from PyTorch NeuronX 2.5/2.6 to PyTorch NeuronX 2.7, implement the following changes: 

.. note::

    ``xm`` below refers to ``torch_xla.core.xla_model``, ``xr`` refers to ``torch_xla.runtime``, and ``xmp`` refers to ``torch_xla.distributed.xla_multiprocessing``

* The environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used) and will be removed in an upcoming release. Please switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to convert model to BF16 format. (see :ref:`migration_from_xla_downcast_bf16`)
* The functions ``xm.xrt_world_size()``, ``xm.xla_model.get_ordinal()``, and ``xm.xla_model.get_local_ordinal()`` are deprecated and removed so there's error when used. Please switch to ``xr.world_size()``, ``xr.global_ordinal()``, and ``xr.local_ordinal()`` respectively as replacements.
* The default behavior of ``torch.load`` parameter ``weights_only`` is changed from ``False`` to ``True``. Leaving ``weights_only`` as ``True`` can cause issues with pickling.
* If using ``xmp.spawn``, the ``nprocs`` argument limited to 1 or None since v2.1. Previously, passing a value > 1 would result in a warning. In torch-xla 2.6+, passing a value > 1 would result in an error with an actionable message to use ``NEURON_NUM_DEVICES`` to set the number of NeuronCores to use.

See :ref:`v2.6 migration guide <migrate_to_pytorch_2.6>` for additional changes needed if you are migrating from PyTorch NeuronX 2.5.
See :ref:`v2.5 migration guide <migrate_to_pytorch_2.x>` for additional changes needed if you are migrating from PyTorch NeuronX 2.1.

Migrating inference scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are no code changes required in the inference scripts.


Troubleshooting and Known Issues
--------------------------------

TypeError: AdamW.__init__() got an unexpected keyword argument 'decoupled_weight_decay'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AdamW now has an additional argument “decoupled_weight_decay” which is default to False. If you get “TypeError: AdamW.__init__() got an unexpected keyword argument 'decoupled_weight_decay'” with NeuronX Distributed, please update to the latest version.


Tensor split on second dimension of 2D array not working
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when using tensor split operation on a 2D array in the second dimension, the resulting tensors don't have the expected data (https://github.com/pytorch/xla/issues/8640). The work-around is to set ``XLA_DISABLE_FUNCTIONALIZATION=0``. Another work-around is to use ``torch.tensor_split``.

Lower BERT pretraining performance when switch to using ``model.to(torch.bfloat16)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, BERT pretraining performance is ~11% lower when switching to using ``model.to(torch.bfloat16)`` as part of migration away from the deprecated environment variable ``XLA_DOWNCAST_BF16`` due to https://github.com/pytorch/xla/issues/8545. As a work-around to recover the performance, you can set ``XLA_DOWNCAST_BF16=1`` which would still work in torch-neuronx 2.5 and 2.7 although there will be deprecation warnings (as noted below).


Warning "XLA_DOWNCAST_BF16 will be deprecated after the 2.6 release, please downcast your model directly"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used). Please switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to cast model to BF16. (see :ref:`migration_from_xla_downcast_bf16`)


AttributeError: <module 'torch_xla.core.xla_model' ... does not have the attribute 'xrt_world_size'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an error that ``torch_xla.core.xla_model.xrt_world_size()`` is removed in torch-xla version 2.7. Please switch to using ``torch_xla.runtime.world_size()`` instead. If using Hugging Face transformers/accelerate libraries, please use transformers==4.53.* and accelerate==1.7.*.

AttributeError: <module 'torch_xla.core.xla_model' ... does not have the attribute 'get_ordinal'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an error that ``torch_xla.core.xla_model.xla_model.get_ordinal()`` is removed in torch-xla version 2.7. Please switch to using ``torch_xla.runtime.global_ordinal()`` instead. If using Hugging Face transformers/accelerate libraries, please use transformers==4.53.* and accelerate==1.7.*.

AttributeError: <module 'torch_xla.core.xla_model' ... does not have the attribute 'get_local_ordinal'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an error that ``torch_xla.core.xla_model.xla_model.get_local_ordinal()`` is removed in torch-xla version 2.7. Please switch to using ``torch_xla.runtime.local_ordinal()`` instead. If using Hugging Face transformers/accelerate libraries, please use transformers==4.53.* and accelerate==1.7.*.


Socket Error: Socket failed to bind
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In PyTorch 2.7, there needs to be a socket available for both torchrun and the ``init_process_group`` to bind. Both of these, by default,
will be set to unused sockets. If you plan to use a ``MASTER_PORT`` environment variable then this error may occur, if the port you set it to
is already in use.

.. code:: 

    [W socket.cpp:426] [c10d] The server socket has failed to bind to [::]:2.700 (errno: 98 - Address already in use).
    [W socket.cpp:426] [c10d] The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
    [E socket.cpp:462] [c10d] The server socket has failed to listen on any local network address.
    RuntimeError: The server socket has failed to listen on any local network address. 
    The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).

To resolve the issue, please ensure if you are setting ``MASTER_PORT`` that the port you're setting it to is not used anywhere else in your scripts. Otherwise,
you can leave ``MASTER_PORT`` unset, and torchrun will set the default port for you.


``AttributeError: module 'torch' has no attribute 'xla'`` Failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In PyTorch 2.7, training scripts might fail during activation checkpointing with the error shown below.

.. code::

    AttributeError: module 'torch' has no attribute 'xla'


The solution is to use ``torch_xla.utils.checkpoint.checkpoint`` instead of ``torch.utils.checkpoint.checkpoint`` as the checkpoint function while wrapping pytorch modules for activation checkpointing.
Refer to the pytorch/xla discussion regarding this `issue <https://github.com/pytorch/xla/issues/5766>`_.
Also set ``use_reentrant=True`` while calling the torch_xla checkpoint function. Failure to do so will lead to ``XLA currently does not support use_reentrant==False`` error.
For more details on checkpointing, refer the `documentation <https://pytorch.org/docs/stable/checkpoint.html>`_.


Error ``Attempted to access the data pointer on an invalid python storage`` when using HF Trainer API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While using HuggingFace Transformers Trainer API to train (i.e. :ref:`HuggingFace Trainer API fine-tuning tutorial<torch-hf-bert-finetune>`), you may see the error "Attempted to access the data pointer on an invalid python storage". This is a known `issue <https://github.com/huggingface/transformers/issues/2.778>`_ and has been fixed in the version ``4.37.3`` of HuggingFace Transformers.


``ImportError: libcrypt.so.1: cannot open shared object file: No such file or directory`` on Amazon Linux 2023
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

torch-xla version 2.5+ now requires ``libcrypt.so.1`` shared library. Currently, Amazon Linux 2023 includes ``libcrypt.so.2`` shared library by default so you may see `ImportError: libcrypt.so.1: cannot open shared object file: No such file or directory`` when using torch-neuronx 2.1+ on Amazon Linux 2023. To install ``libcrypt.so.1`` on Amazon Linux 2023, please run the following installation command (see also https://github.com/amazonlinux/amazon-linux-2023/issues/182 for more context):

.. code::

   sudo yum install libxcrypt-compat


``FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'`` Failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In PyTorch 2.7, users might face the error shown below due to incompatible ``libneuronxla`` and ``torch-neuronx`` versions being installed.

.. code::

    FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'

Check that the version of ``libneuronxla`` that support PyTorch NeuronX 2.7 is ``2.2.*``. If not, then uninstall ``libneuronxla`` using ``pip uninstall libneuronxla`` and then reinstall the packages following the installation guide :ref:`installation guide <install_pytorch_neuron_2.7>`


``Input dimension should be either 1 or equal to the output dimension it is broadcasting into`` or ``IndexError: index out of range`` error during Neuron Parallel Compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running Neuron Parallel Compile with HF Trainer API, you may see the error ``Status: INVALID_ARGUMENT: Input dimension should be either 1 or equal to the output dimension it is broadcasting into`` or ``IndexError: index out of range`` in Accelerator's ``pad_across_processes`` function. This is due to data-dependent operation in evaluation metrics computation. Data-dependent operations would result in undefined behavior with Neuron Parallel Compile trial execution (execute empty graphs with zero outputs). To work-around this error, please disable compute_metrics when NEURON_EXTRACT_GRAPHS_ONLY is set to 1:

.. code:: python

   compute_metrics=None if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY") else compute_metrics

Compiler assertion error when running Stable Diffusion training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with PyTorch 2.7 (torch-neuronx), we are seeing the following compiler assertion error with Stable Diffusion training when gradient accumulation is enabled. This will be fixed in an upcoming release. For now, if you would like to run Stable Diffusion training with Neuron SDK release 2.23, please disable gradient accumulation in torch-neuronx 2.7.

.. code:: bash

    ERROR 222163 [NeuronAssert]: Assertion failure in usr/lib/python3.9/concurrent/futures/process.py at line 239 with exception:
    too many partition dims! {{0,+,960}[10],+,10560}[10]


Frequently Asked Questions (FAQ)
--------------------------------

Do I need to recompile my models with PyTorch 2.7?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes.

Do I need to update my scripts for PyTorch 2.7?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Please see the :ref:`migration guide <migrate_to_pytorch_2.7>`

What environment variables will be changed with PyTorch NeuronX 2.7 ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used). Please switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to cast model to BF16. (see :ref:`migration_from_xla_downcast_bf16`)

What features will be missing with PyTorch NeuronX 2.7?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch NeuronX 2.7 has all of the supported features in PyTorch NeuronX 2.6, with known issues listed above, and unsupported features as listed in :ref:`torch-neuronx-rn`.

Can I use Neuron Distributed and Transformers Neuron libraries with PyTorch NeuronX 2.7?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, NeuronX Distributed and Transformers NeuronX are supported by PyTorch NeuronX 2.7.  AWS Neuron Reference for NeMo Megatron has reached end-of-support in release 2.23.

Can I still use PyTorch 2.6 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch 2.6 is supported since release 2.23.

Can I still use PyTorch 2.5 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch 2.5 is supported for releases 2.21 to 2.24 and will reach end-of-life in a future release. Additionally, the CVE `CVE-2025-32434 <https://github.com/advisories/GHSA-53q9-r3pm-6pq6>`_ affects PyTorch version 2.5. We recommend upgrading to the new version of Torch-NeuronX by following :ref:`setup-torch-neuronx`.

Can I still use PyTorch 2.1 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch 2.1 is supported for release 2.21 and has reached end-of-life in release 2.22. Additionally, the CVEs `CVE-2024-31583 <https://github.com/advisories/GHSA-pg7h-5qx3-wjr3>`_ and `CVE-2024-31580 <https://github.com/advisories/GHSA-5pcm-hx3q-hm94>`_ affect PyTorch versions 2.1 and earlier.  We recommend upgrading to the new version of Torch-NeuronX by following :ref:`setup-torch-neuronx`.
