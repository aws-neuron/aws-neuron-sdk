.. _introduce-pytorch-2-5:

Introducing PyTorch 2.5 Support
===============================

.. contents:: Table of contents
   :local:
   :depth: 2


What are we introducing?
------------------------

Starting with the :ref:`Neuron 2.21 <neuron-2.21.0-whatsnew>` release, customers will be able to upgrade to ``PyTorch NeuronX(torch-neuronx)`` supporting ``PyTorch 2.5``.

:ref:`setup-torch-neuronx` is updated to include installation instructions for PyTorch NeuronX 2.5 for Amazon Linux 2023 and Ubuntu 22. Note that PyTorch NeuronX 2.5 does not support Python 3.8 which is default in Ubuntu 20. To use Ubuntu 20, customers will need to install Python 3.9+.

Please review :ref:`migration guide <migrate_to_pytorch_2_5>` for possible changes to training scripts. No code changes are required for inference scripts.


.. _how-pytorch-2-5-different:

How is PyTorch NeuronX 2.5 different compared to PyTorch NeuronX 2.1?
---------------------------------------------------------------------

PyTorch NeuronX 2.5 uses Torch-XLA 2.5 which has improved support for eager debug mode, Automatic Mixed Precission, PJRT device auto-detection, FP8, and others. See `Torch-XLA 2.5 release <https://github.com/pytorch/xla/releases/tag/v2.5.0>`__ for a full list.

See :ref:`migrate_to_pytorch_2_5` for changes needed to use PyTorch NeuronX 2.5.

.. note::

   GSPMD and Torch Dynamo (torch.compile) support in Neuron will be available in a future release.

.. _install_pytorch_neuron_2_5:

How can I install PyTorch NeuronX 2.5?
--------------------------------------------

To install PyTorch NeuronX 2.5 please follow the :ref:`setup-torch-neuronx` guides for Amazon Linux 2023 and Ubuntu 22 AMI. Please also refer to the Neuron multi-framework DLAMI :ref:`setup guide <setup-ubuntu22-multi-framework-dlami>` for Ubuntu 22 with a pre-installed virtual environment for PyTorch NeuronX 2.5 that you can use to get started. PyTorch NeuronX 2.5 can be installed using the following:

.. code::

    python -m pip install --upgrade neuronx-cc==2.* torch-neuronx==2.5.* torchvision

.. note::

   PyTorch NeuronX 2.5 is currently available for Python 3.9, 3.10, 3.11.

.. _migrate_to_pytorch_2_5:

Migrate your application to PyTorch 2.5
---------------------------------------

Please make sure you have first installed the PyTorch NeuronX 2.5 as described above in :ref:`installation guide <install_pytorch_neuron_2_5>`


Migrating training scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^

To migrate the training scripts from PyTorch NeuronX 2.1 to PyTorch NeuronX 2.5, implement the following changes: 

.. note::

    ``xm`` below refers to ``torch_xla.core.xla_model`` and ``xr`` refers to ``torch_xla.runtime``

* The environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used). Please switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to convert model to BF16 format. (see :ref:`migration_from_xla_downcast_bf16`)
* The ``torch_xla.experimental.pjrt`` module which was replaced by ``torch_xla.runtime`` in Torch-XLA 2.1, has been removed in Torch-XLA 2.5. Users should now utilize the ``torch_xla.runtime`` module as a replacement.
* ``torch_xla.runtime.using_pjrt`` is removed because PJRT is the sole Torch-XLA runtime.
* ``xm.all_reduce`` no longer operates in-place for single tensors. To fix this, please convert the single tensor to an array (e.g.. ``[single_tensor]``) or assign the output of ``xm.all_reduce`` to a variable.
* The functions ``xm.xrt_world_size()``, ``xm.xla_model.get_ordinal()``, and ``xm.xla_model.get_local_ordinal()`` are deprecated (warning when used). Please switch to ``xr.world_size``, ``xr.global_ordinal``, and ``xr.local_ordinal`` respectively as replacements.
* ``torch_xla.experimental.xla_sharding`` is now replaced by ``torch_xla.distributed.spmd.xla_sharding``.
* Class ``ZeroRedundancyOptimizer`` now has two new arguments that replaces the optional boolean argument ``coalesce_cc``:
    * ``bucket_cap_mb_all_gather`` (int, Optional): Number of MegaBytes of the tensor bucket to fill before doing all-gather. Default: 0 (disable  all gather coalescing).
    * ``bucket_cap_mb_reduce_scatter`` (int, Optional): Number of MegaBytes of the tensor bucket to fill before doing reduce-scatter. Default: 0 (disable reduce scatter coalescing).

Migrating inference scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are no code changes required in the inference scripts.


Troubleshooting and Known Issues
--------------------------------

Neuronx-Distributed Training Llama 3.1 70B 8-node tutorial failed with OSError when the Neuron Cache is placed on FSx mount
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently, the Neuronx-Distributed Training Llama 3.1 70B 8-node tutorial failed with OSError (Errno 61) when the Neuron Cache is placed on FSx mount:

.. code:: bash

    [rank197]: RuntimeError: Bad StatusOr access: INVALID_ARGUMENT: RunNeuronCCImpl: error condition !(error != 400): <class 'OSError'>: [Errno 61] No data available: '/fsxl/neuron_cache/neuronxcc-2.16.372.0+4a9b2326/MODULE_3540044791706521849+4eb52b03/model.neff' -> '/tmp/tmpx7bvfpmm/model.neff'

We found that the error is due to FSx failing during file copy when there are multiple readers (13 workers fail to copy out of 256). This issue doesn’t affect simpler models like BERT.

To work-around the issue, please use the shared NFS mount (/home directory on a Parallel Cluster) instead of FSx to store Neuron Cache. This will be fixed in an upcoming release.

Running in-place update operations (e.g. all_reduce) on 0-dimensional tensors result in buffer aliasing errors in torch 2.5 and earlier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Torch's lazy tensor core has a feature where 0-dimensional tensors are stored in a device cache, so scalar constant values can be transferred once and then reused. The values in the device cache are supposed to be marked read-only and never participate in parameter aliasing. However, due to a bug in torch-xla 2.5 (`#8499 <https://github.com/pytorch/xla/issues/8499>`_), sometimes the read-only flag can be dropped, allowing these tensors to be donated, resulting in aliasing errors later when the cached value is used again.

A work-around is to avoid using 0-dimensional tensors by changing them to be 1d tensor of length 1 (`example <https://github.com/aws-neuron/neuronx-nemo-megatron/pull/36/commits/0b2354666508ac75cb6150083211fa6823864ebe>`_).
If modifying library code is not possible, disable XLA parameter aliasing by setting environment variable XLA_ENABLE_PARAM_ALIASING=0

Tensor split on second dimension of 2D array not working
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when using tensor split operation on a 2D array in the second dimension, the resulting tensors don't have the expected data (https://github.com/pytorch/xla/issues/8640). The work-around is to set ``XLA_DISABLE_FUNCTIONALIZATION=0``. Another work-around is to use ``torch.tensor_split``.

Import torch_xla crashed with ``TypeError: must be called with a dataclass type or instance`` with torch-xla 2.5 and torch 2.5.1+cpu (CPU flavor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using torch 2.5.1+cpu (CPU flavor) on python 3.10, importing torch_xla crashed with ``TypeError: must be called with a dataclass type or instance`` due to installed triton version 3.2.0 (https://github.com/pytorch/xla/issues/8560). To work-around, please remove the installed triton package or downgrade to triton==3.1.0 or use the regular torch 2.5.1 (GPU flavor).

Certain sequence of operations with ``xm.save()`` could corrupt tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using the ``xm.save`` function to save tensors, please use ``xm.mark_step()`` before ``xm.save`` to avoid the error described in https://github.com/pytorch/xla/issues/8422 where parameter aliasing could corrupt other tensor values. This issue will be fixed in a future release.

(Here ``xm`` is ``torch_xla.core.xla_model`` following PyTorch/XLA convention)

Lower BERT pretraining performance when switch to using ``model.to(torch.bfloat16)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, BERT pretraining performance is ~11% lower when switching to using ``model.to(torch.bfloat16)`` as part of migration away from the deprecated environment variable ``XLA_DOWNCAST_BF16`` due to https://github.com/pytorch/xla/issues/8545. As a work-around to recover the performance, you can set ``XLA_DOWNCAST_BF16=1`` which would still work in torch-neuronx 2.5 and 2.6 although there will be end-of-support warnings (as noted below).

Warning "XLA_DOWNCAST_BF16 will be deprecated after the 2.5 release, please downcast your model directly"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used). Please switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to cast model to BF16. (see :ref:`migration_from_xla_downcast_bf16`)


WARNING:root:torch_xla.core.xla_model.xrt_world_size() will be removed in release 2.7. is deprecated. Use torch_xla.runtime.world_size instead.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a warning that ``torch_xla.core.xla_model.xrt_world_size()`` will be removed in a future release. Please switch to using ``torch_xla.runtime.world_size`` instead.


WARNING:torch_xla.core.xla_model.xla_model.get_ordinal() will be removed in release 2.7. is deprecated. Use torch_xla.runtime.global_ordinal instead.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a warning that ``torch_xla.core.xla_model.xla_model.get_ordinal()`` will be removed in a future release. Please switch to using ``torch_xla.runtime.global_ordinal`` instead.


AttributeError: module 'torch_xla.runtime' has no attribute 'using_pjrt'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Torch-XLA 2.5, ``torch_xla.runtime.using_pjrt`` is removed because PJRT is the sole Torch-XLA runtime.
See `commit PR <https://github.com/pytorch/xla/commit/d6fb5391d09578c8804b1331a5e7a4f72bf981db>`__.


Socket Error: Socket failed to bind
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In PyTorch 2.5, there needs to be a socket available for both torchrun and the ``init_process_group`` to bind. Both of these, by default,
will be set to unused sockets. If you plan to use a ``MASTER_PORT`` environment variable then this error may occur, if the port you set it to
is already in use.

.. code:: 

    [W socket.cpp:426] [c10d] The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use).
    [W socket.cpp:426] [c10d] The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
    [E socket.cpp:462] [c10d] The server socket has failed to listen on any local network address.
    RuntimeError: The server socket has failed to listen on any local network address. 
    The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).

To resolve the issue, please ensure if you are setting ``MASTER_PORT`` that the port you're setting it to is not used anywhere else in your scripts. Otherwise,
you can leave ``MASTER_PORT`` unset, and torchrun will set the default port for you.


``AttributeError: module 'torch' has no attribute 'xla'`` Failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In PyTorch 2.5, training scripts might fail during activation checkpointing with the error shown below.

.. code::

    AttributeError: module 'torch' has no attribute 'xla'


The solution is to use ``torch_xla.utils.checkpoint.checkpoint`` instead of ``torch.utils.checkpoint.checkpoint`` as the checkpoint function while wrapping pytorch modules for activation checkpointing.
Refer to the pytorch/xla discussion regarding this `issue <https://github.com/pytorch/xla/issues/5766>`_.
Also set ``use_reentrant=True`` while calling the torch_xla checkpoint function. Failure to do so will lead to ``XLA currently does not support use_reentrant==False`` error.
For more details on checkpointing, refer the `documentation <https://pytorch.org/docs/stable/checkpoint.html>`_.


Error ``Attempted to access the data pointer on an invalid python storage`` when using HF Trainer API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While using HuggingFace Transformers Trainer API to train (i.e. :ref:`HuggingFace Trainer API fine-tuning tutorial<torch-hf-bert-finetune>`), you may see the error "Attempted to access the data pointer on an invalid python storage". This is a known `issue <https://github.com/huggingface/transformers/issues/27578>`_ and has been fixed in the version ``4.37.3`` of HuggingFace Transformers.

``ImportError: libcrypt.so.1: cannot open shared object file: No such file or directory`` on Amazon Linux 2023
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

torch-xla version 2.5+ now requires ``libcrypt.so.1`` shared library. Currently, Amazon Linux 2023 includes ``libcrypt.so.2`` shared library by default so you may see `ImportError: libcrypt.so.1: cannot open shared object file: No such file or directory`` when using torch-neuronx 2.1+ on Amazon Linux 2023. To install ``libcrypt.so.1`` on Amazon Linux 2023, please run the following installation command (see also https://github.com/amazonlinux/amazon-linux-2023/issues/182 for more context):

.. code::

   sudo dnf install libxcrypt-compat


``FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'`` Failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In PyTorch 2.5, users might face the error shown below due to incompatible ``libneuronxla`` and ``torch-neuronx`` versions being installed.

.. code::

    FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'

Check that the version of ``libneuronxla`` that support PyTorch NeuronX 2.5 is ``2.1.*``. If not, then uninstall ``libneuronxla`` using ``pip uninstall libneuronxla`` and then reinstall the packages following the installation guide :ref:`installation guide <install_pytorch_neuron_2_5>`


GlibC error on Amazon Linux 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If using Torch-NeuronX 2.5 on Amazon Linux 2, you will see a GlibC error below. Please switch to a newer supported OS such as Ubuntu 22 or Amazon Linux 2023.

.. code:: bash

   ImportError: /lib64/libc.so.6: version `GLIBC_2.27' not found (required by /tmp/debug/_XLAC.cpython-38-x86_64-linux-gnu.so)

``Input dimension should be either 1 or equal to the output dimension it is broadcasting into`` or ``IndexError: index out of range`` error during Neuron Parallel Compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running Neuron Parallel Compile with HF Trainer API, you may see the error ``Status: INVALID_ARGUMENT: Input dimension should be either 1 or equal to the output dimension it is broadcasting into`` or ``IndexError: index out of range`` in Accelerator's ``pad_across_processes`` function. This is due to data-dependent operation in evaluation metrics computation. Data-dependent operations would result in undefined behavior with Neuron Parallel Compile trial execution (execute empty graphs with zero outputs). To work-around this error, please disable compute_metrics when NEURON_EXTRACT_GRAPHS_ONLY is set to 1:

.. code:: python

   compute_metrics=None if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY") else compute_metrics

Compiler assertion error when running Stable Diffusion training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with PyTorch 2.5 (torch-neuronx), we are seeing the following compiler assertion error with Stable Diffusion training when gradient accumulation is enabled. This will be fixed in an upcoming release. For now, if you would like to run Stable Diffusion training with Neuron SDK release 2.21/2.22, please disable gradient accumulation in torch-neuronx 2.5.

.. code:: bash

    ERROR 222163 [NeuronAssert]: Assertion failure in usr/lib/python3.9/concurrent/futures/process.py at line 239 with exception:
    too many partition dims! {{0,+,960}[10],+,10560}[10]


Frequently Asked Questions (FAQ)
--------------------------------

Do I need to recompile my models with PyTorch 2.5?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes.

Do I need to update my scripts for PyTorch 2.5?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Please see the :ref:`migration guide <migrate_to_pytorch_2_5>`

What environment variables will be changed with PyTorch NeuronX 2.5 ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used). Please switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to cast model to BF16. (see :ref:`migration_from_xla_downcast_bf16`)

What features will be missing with PyTorch NeuronX 2.5?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch NeuronX 2.5 now has most of the supported features in PyTorch NeuronX 2.1, with known issues listed above, and unsupported features as listed in :ref:`torch-neuronx-rn`.

Can I use Neuron Distributed and Transformers Neuron libraries with PyTorch NeuronX 2.5?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, NeuronX Distributed, and Transformers NeuronX, and AWS Neuron Reference for NeMo Megatron libraries will work with PyTorch NeuronX 2.5.

Can I still use PyTorch 2.1 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch 2.1 is supported for release 2.21 and will reach end-of-life in a future release. Additionally, the CVEs `CVE-2024-31583 <https://github.com/advisories/GHSA-pg7h-5qx3-wjr3>`_ and `CVE-2024-31580 <https://github.com/advisories/GHSA-5pcm-hx3q-hm94>`_ affect PyTorch versions 2.1 and earlier.  We recommend upgrading to the new version of Torch-NeuronX by following :ref:`setup-torch-neuronx`.
