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

To install PyTorch NeuronX 2.5 please follow the :ref:`setup-torch-neuronx` guides for Amazon Linux 2023 and Ubuntu 22 AMI. Please also refer to the Neuron multi-framework DLAMI :ref:`setup guide <setup-ubuntu22-multi-framework-dlami>` for Ubuntu 22 with a pre-installed virtual environment for PyTorch NeuronX 2.5 that you can use to easily get started. PyTorch NeuronX 2.5 can be installed using the following:

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

* The environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used). Please switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to convert model to BF16 format. (see :ref:`<migration_from_xla_downcast_bf16>`)
* The ``torch_xla.experimental.pjrt`` module which was replaced by ``torch_xla.runtime`` in Torch-XLA 2.1, has been removed in Torch-XLA 2.5. Users should now utilize the ``torch_xla.runtime`` module as a replacement.
* ``torch_xla.runtime.using_pjrt`` is removed because PJRT is the sole Torch-XLA runtime.
* ``xm.all_reduce`` no longer operates in-place for single tensors. To fix this, please convert the single tensor to an array (e.g.. ``[single_tensor]``) or assign the output of ``xm.all_reduce`` to a variable.
* The functions ``xm.xrt_world_size()`` and ``xm.xla_model.get_ordinal()`` are deprecated (warning when used). Please switch to ``xr.world_size`` and ``xr.global_ordinal`` respectively as replacements.
* ``torch_xla.experimental.xla_sharding`` is now replaced by ``torch_xla.distributed.spmd.xla_sharding``.
* Class ``ZeroRedundancyOptimizer`` now has two new arguments that replaces the optional boolean argument ``coalesce_cc``:
    * ``bucket_cap_mb_all_gather`` (int, Optional): Number of MegaBytes of the tensor bucket to fill before doing all-gather. Default: 0 (disable  all gather coalescing).
    * ``bucket_cap_mb_reduce_scatter`` (int, Optional): Number of MegaBytes of the tensor bucket to fill before doing reduce-scatter. Default: 0 (disable reduce scatter coalescing).

Migrating inference scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are no code changes required in the inference scripts.


Troubleshooting and Known Issues
--------------------------------

Certain sequence of operations with ``xm.save()`` could corrupt tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using the ``xm.save`` function to save tensors, please use ``xm.mark_step()`` before ``xm.save`` to avoid the error described in https://github.com/pytorch/xla/issues/8422 where parameter aliasing could corrupt other tensor values. This issue will be fixed in a future release.

(Here ``xm`` is ``torch_xla.core.xla_model`` following PyTorch/XLA convention)

Lower BERT pretraining performance with torch-neuronx 2.5 compared to torch-neuronx 2.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, BERT pretraining performance is ~11% lower with torch-neuronx 2.5 compared to torch-neuronx 2.1. This is due to the switch to using ``model.to(torch.bfloat16)`` as part of migration away from the deprecated environment variable ``XLA_DOWNCAST_BF16``. As a work-around to recover the performance, you can set ``XLA_DOWNCAST_BF16=1`` which would still work in torch-neuronx 2.5 although there will be deprecation warnings (as noted below).

Warning "XLA_DOWNCAST_BF16 will be deprecated after the 2.5 release, please downcast your model directly"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used). Please switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to cast model to BF16. (see :ref:`migration_from_xla_downcast_bf16`)


WARNING:root:torch_xla.core.xla_model.xrt_world_size() will be removed in release 2.7. is deprecated. Use torch_xla.runtime.world_size instead.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a warning that ``torch_xla.core.xla_model.xrt_world_size()`` will be removed in a future release. Please switch to using ``torch_xla.runtime.world_size`` instead.


WARNING:torch_xla.core.xla_model.xla_model.get_ordinal() will be removed in release 2.7. is deprecated. Use torch_xla.runtime.global_ordinal instead.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a warning that ``torch_xla.core.xla_model.xla_model.get_ordinal() `` will be removed in a future release. Please switch to using ``torch_xla.runtime.global_ordinal`` instead.


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


Incorrect device assignment when using ellipsis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Usage of ellipsis (``...``) with PyTorch/XLA 2.5 can lead to incorrect device assignment of the tensors as 'lazy' instead of 'xla'.
Refer to the example shown

.. code:: python

    import torch
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()

    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
    print(f"x.device : {x.device}")
    y = x[:3, ...]
    print(f"y.device : {y.device}")
    print(x + y)


leads to

.. code::

    x.device : xla:0
    y.device : lazy:0
    RuntimeError: torch_xla/csrc/tensor.cpp:57 : Check failed: tensor.device().type() == at::kCPU (lazy vs. cpu)


This only happens for scenarios where ellipsis is used to extract a subset of a tensor with the same size as that of the original tensor. An issue is created with pytorch/xla to fix this behavior (`Ref <https://github.com/pytorch/xla/issues/6398>`_).
Potential workaround is to avoid using ellipsis and instead replace it with ``:`` for each corresponding dimensions in the buffer.

For the faulty code shown above, replace it with

.. code:: python

    import torch
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()

    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
    print(f"x.device : {x.device}")
    # Replaced '...' with ':'
    y = x[:3, :]
    print(f"y.device : {y.device}")
    print(x + y)

Error ``Attempted to access the data pointer on an invalid python storage`` when using HF Trainer API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While using HuggingFace Transformers Trainer API to train (i.e. :ref:`HuggingFace Trainer API fine-tuning tutorial<torch-hf-bert-finetune>`), you may see the error "Attempted to access the data pointer on an invalid python storage". This is a known `issue <https://github.com/huggingface/transformers/issues/27578>`_ and has been fixed in the version ``4.37.3`` of HuggingFace Transformers.

``ImportError: libcrypt.so.1: cannot open shared object file: No such file or directory`` on Amazon Linux 2023
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

torch-xla version 2.5+ now requires ``libcrypt.so.1`` shared library. Currently, Amazon Linux 2023 includes ``libcrypt.so.2`` shared library by default so you may see `ImportError: libcrypt.so.1: cannot open shared object file: No such file or directory`` when using torch-neuronx 2.1+ on Amazon Linux 2023. To install ``libcrypt.so.1`` on Amazon Linux 2023, please run the following installation command (see also https://github.com/amazonlinux/amazon-linux-2023/issues/182 for more context):

.. code::

   sudo yum install libxcrypt-compat


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

``"EOFError: Ran out of input"`` or ``"_pickle.UnpicklingError: invalid load key, '!'"`` errors during Neuron Parallel Compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With PyTorch 2.5 (torch-neuronx), HF Trainer API's use of XLA function ``.mesh_reduce`` causes ``"EOFError: Ran out of input"`` or ``"_pickle.UnpicklingError: invalid load key, '!'"`` errors during Neuron Parallel Compile. To work-around this issue, you can add the following code snippet (after python imports) to replace ``xm.mesh_reduce`` with a form that uses ``xm.all_gather`` instead of ``xm.rendezvous()`` with payload. This will add additional small on-device graphs (as opposed to the original ``xm.mesh_reduce`` which runs on CPU).

.. code:: python

    import copy
    import torch_xla.core.xla_model as xm
    def mesh_reduce(tag, data, reduce_fn):
        xm.rendezvous(tag)
        xdatain = copy.deepcopy(data)
        xdatain = xdatain.to("xla")
        xdata = xm.all_gather(xdatain, pin_layout=False)
        cpu_xdata = xdata.detach().to("cpu")
        cpu_xdata_split = torch.split(cpu_xdata, xdatain.shape[0])
        xldata = [x for x in cpu_xdata_split]
        return reduce_fn(xldata)
    xm.mesh_reduce = mesh_reduce


``Check failed: tensor_data`` error during when using ``torch.utils.data.DataLoader`` with ``shuffle=True``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With PyTorch 2.5 (torch-neuronx), using ``torch.utils.data.DataLoader`` with ``shuffle=True`` would cause the following error in ``synchronize_rng_states`` (i.e. :ref:`ZeRO1 tutorial<zero1-gpt2-pretraining-tutorial>`):

.. code:: bash

    RuntimeError: torch_xla/csrc/xla_graph_executor.cpp:562 : Check failed: tensor_data 

This is due to ``synchronize_rng_states`` using ``xm.mesh_reduce`` to synchronize RNG states. ``xm.mesh_reduce`` in turn uses  ``xm.rendezvous()`` with payload, which as noted in 2.x migration guide, would result in extra graphs that could lead to lower performance due to change in ``xm.rendezvous()`` in torch-xla 2.x. In the case of :ref:`ZeRO1 tutorial<zero1-gpt2-pretraining-tutorial>`, using ``xm.rendezvous()`` with payload also lead to the error above. This limitation will be fixed in an upcoming release. For now, to work around the issue, please disable shuffle in DataLoader when ``NEURON_EXTRACT_GRAPHS_ONLY`` environment is set automatically by Neuron Parallel Compile:

.. code:: python

    train_dataloader = DataLoader(
        train_dataset, shuffle=(os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None) == None), collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )

Additionally, as in the previous section, you can add the following code snippet (after python imports) to replace ``xm.mesh_reduce`` with a form that uses ``xm.all_gather`` instead of ``xm.rendezvous()`` with payload. This will add additional small on-device graphs (as opposed to the original ``xm.mesh_reduce`` which runs on CPU).

.. code:: python

    import copy
    import torch_xla.core.xla_model as xm
    def mesh_reduce(tag, data, reduce_fn):
	xm.rendezvous(tag)
	xdatain = copy.deepcopy(data)
	xdatain = xdatain.to("xla")
	xdata = xm.all_gather(xdatain, pin_layout=False)
	cpu_xdata = xdata.detach().to("cpu")
	cpu_xdata_split = torch.split(cpu_xdata, xdatain.shape[0])
	xldata = [x for x in cpu_xdata_split]
	return reduce_fn(xldata)
    xm.mesh_reduce = mesh_reduce

Compiler assertion error when running Stable Diffusion training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with PyTorch 2.5 (torch-neuronx), we are seeing the following compiler assertion error with Stable Diffusion training when gradient accumulation is enabled. This will be fixed in an upcoming release. For now, if you would like to run Stable Diffusion training with Neuron SDK release 2.21, please disable gradient accumulation in torch-neuronx 2.5.

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
