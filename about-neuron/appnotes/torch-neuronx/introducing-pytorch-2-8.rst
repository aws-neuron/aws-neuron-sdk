.. _introduce-pytorch-2-8:

Introducing PyTorch 2.8 Support
===============================

.. contents:: Table of contents
   :local:
   :depth: 2


What are we introducing?
------------------------

Starting with the :ref:`Neuron 2.26 <neuron-2.26.0-whatsnew>` release, customers can now upgrade to PyTorch NeuronX (``torch-neuronx``) with specific support for PyTorch version 2.8.

:ref:`setup-torch-neuronx` is updated to include installation instructions for PyTorch NeuronX 2.8 for Ubuntu 22.04. Note that PyTorch NeuronX 2.8 is supported on Python 3.10 and 3.11, with 3.12+ support coming in a future release.

Review :ref:`migration guide <migrate_to_pytorch_2.8>` for possible changes to training scripts. No code changes are required for inference scripts.


.. _how-pytorch-2.8-different:

How is PyTorch NeuronX 2.8 different compared to PyTorch NeuronX 2.7?
---------------------------------------------------------------------

See `Torch-XLA 2.8 release <https://github.com/pytorch/xla/releases/tag/v2.8.0>`__ for a full list of changes.

See :ref:`migrate_to_pytorch_2.8` for changes needed to use PyTorch NeuronX 2.8.

.. note::

   GSPMD and Torch Dynamo (torch.compile) support in Neuron will be available in a future release.

.. _install_pytorch_neuron_2.8:

How can I install PyTorch NeuronX 2.8?
--------------------------------------------

To install PyTorch NeuronX 2.8, follow the :ref:`setup-torch-neuronx` guides for Ubuntu 22.04 AMI. Refer to the Neuron Multi-Framework DLAMI :ref:`setup guide <setup-ubuntu22-multi-framework-dlami>` for Ubuntu 22.04 with a pre-installed virtual environment for PyTorch NeuronX 2.8 that you can use to get started. PyTorch NeuronX 2.8 can be installed using the following:

.. code::

    python -m pip install --upgrade neuronx-cc==2.* torch-neuronx==2.8.* torchvision

.. note::

   PyTorch NeuronX 2.8 is currently available for Python 3.10 and 3.11, with 3.12+ support coming in a future release.

.. note::

   To use Amazon Linux 2023, you will need to install Python 3.10 or 3.11 to use PyTorch NeuronX 2.8.

.. _migrate_to_pytorch_2.8:

Migrate your application to PyTorch 2.8
---------------------------------------

First, install the PyTorch NeuronX 2.8 as described above in :ref:`installation guide <install_pytorch_neuron_2.8>`


Migrating training scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^

There are no code changes required in the training scripts to move from PyTorch NeuronX 2.7 to PyTorch NeuronX 2.8.

See :ref:`v2.7 migration guide <migrate_to_pytorch_2.7>` for additional changes needed if you are migrating from PyTorch NeuronX 2.6.
See :ref:`v2.6 migration guide <migrate_to_pytorch_2.6>` for additional changes needed if you are migrating from PyTorch NeuronX 2.5.

Migrating inference scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are no code changes required in the inference scripts.


Troubleshooting and Known Issues
--------------------------------

[v2.8] Lower BERT/LLaMA performance with torch-xla 2.8.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the publicly released version of torch-xla 2.8.0 from public PyPI repositories would result in lower performance for models like BERT and LLaMA (https://github.com/pytorch/xla/issues/9605). To fix this, switch to using the updated torch-xla version 2.8.1 from public PyPI repositories.

Using the latest torch-xla 2.7/2.8 may result in increase in host memory usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using torch-xla 2.7/2.8 may result in an increase in host memory usage compared to torch-xla 2.6. In one example, LLama2 pretraining with ZeRO1 and sequence length 16k could see an increase of 1.6% in host memory usage.

TypeError: AdamW.__init__() got an unexpected keyword argument 'decoupled_weight_decay'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AdamW now has an additional argument ``decoupled_weight_decay`` which defaults to False. If you get ``TypeError: AdamW.__init__() got an unexpected keyword argument 'decoupled_weight_decay'`` with NeuronX Distributed, update to the latest version.


Tensor split on second dimension of 2D array not working
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when using the tensor split operation on a 2D array in the second dimension, the resulting tensors do not contain the expected data (https://github.com/pytorch/xla/issues/8640). The workaround is to set ``XLA_DISABLE_FUNCTIONALIZATION=0``. Another workaround is to use ``torch.tensor_split``.

Lower BERT pretraining performance when switch to using ``model.to(torch.bfloat16)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, BERT pretraining performance is approximately 11% lower when switching to using ``model.to(torch.bfloat16)`` as part of migration away from the deprecated environment variable ``XLA_DOWNCAST_BF16`` due to https://github.com/pytorch/xla/issues/8545. As a workaround to recover the performance, you can set ``XLA_DOWNCAST_BF16=1``, which will still work in torch-neuronx 2.5 to 2.8 although there will be end-of-support warnings (as noted below).


DeprecationWarning: Use torch_xla.device instead
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a end-of-support warning when using ``torch_xla.core.xla_model.xla_device()``. Switch to ``torch_xla.device()`` instead.

DeprecationWarning: Use torch_xla.sync instead
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a end-of-support warning when using ``torch_xla.core.xla_model.mark_step()``. Switch to ``torch_xla.sync()`` instead.

Warning "XLA_DOWNCAST_BF16 will be deprecated after the 2.6 release, please downcast your model directly"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warnings are shown when used). Switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to cast model to BF16. (see :ref:`migration_from_xla_downcast_bf16`)


AttributeError: <module 'torch_xla.core.xla_model' ... does not have the attribute 'xrt_world_size'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an error that ``torch_xla.core.xla_model.xrt_world_size()`` was removed since torch-xla version 2.7+. Switch to using ``torch_xla.runtime.world_size()`` instead. If using Hugging Face transformers/accelerate libraries, use transformers==4.53.* and accelerate==1.7.* or newer.

AttributeError: <module 'torch_xla.core.xla_model' ... does not have the attribute 'get_ordinal'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an error that ``torch_xla.core.xla_model.get_ordinal()`` was removed since torch-xla version 2.7+. Switch to using ``torch_xla.runtime.global_ordinal()`` instead. If using Hugging Face transformers/accelerate libraries, use transformers==4.53.* and accelerate==1.7.* or newer.

AttributeError: <module 'torch_xla.core.xla_model' ... does not have the attribute 'get_local_ordinal'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an error that ``torch_xla.core.xla_model.get_local_ordinal()`` was removed since torch-xla version 2.7+. Switch to using ``torch_xla.runtime.local_ordinal()`` instead. If using Hugging Face transformers/accelerate libraries, use transformers==4.53.* and accelerate==1.7.* or newer.


Socket Error: Socket failed to bind
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In PyTorch 2.1+ including 2.8, there must be a socket available for both torchrun and the ``init_process_group`` to bind. By default, both 
will be set to use unused sockets. If you plan to use a ``MASTER_PORT`` environment variable then this error may occur if the port you set it to
is already in use.

.. code:: 

    [W socket.cpp:426] [c10d] The server socket has failed to bind to [::]:2.700 (errno: 98 - Address already in use).
    [W socket.cpp:426] [c10d] The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
    [E socket.cpp:462] [c10d] The server socket has failed to listen on any local network address.
    RuntimeError: The server socket has failed to listen on any local network address. 
    The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).

To resolve the issue, if you are setting ``MASTER_PORT``, ensure that the port you're setting it to is not used anywhere else in your scripts. Otherwise,
you can leave ``MASTER_PORT`` unset and torchrun will set the default port for you.


``AttributeError: module 'torch' has no attribute 'xla'`` Failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In PyTorch 2.8, training scripts might fail during activation checkpointing with the error shown below.

.. code::

    AttributeError: module 'torch' has no attribute 'xla'


The solution is to use ``torch_xla.utils.checkpoint.checkpoint`` instead of ``torch.utils.checkpoint.checkpoint`` as the checkpoint function while wrapping pytorch modules for activation checkpointing.
Refer to the pytorch/xla discussion regarding this `issue <https://github.com/pytorch/xla/issues/5766>`_.
Also set ``use_reentrant=True`` while calling the torch_xla checkpoint function. Failure to do so will lead to ``XLA currently does not support use_reentrant==False`` error.
For more details on checkpointing, refer the `documentation <https://pytorch.org/docs/stable/checkpoint.html>`_.


Error ``Attempted to access the data pointer on an invalid python storage`` when using HF Trainer API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While using HuggingFace Transformers Trainer API to train (i.e. :ref:`HuggingFace Trainer API fine-tuning tutorial<torch-hf-bert-finetune>`), you may see the error "Attempted to access the data pointer on an invalid python storage". This is a known `issue <https://github.com/huggingface/transformers/issues/27778>`_ and has been fixed in the version ``4.37.3`` of HuggingFace Transformers.

``Input dimension should be either 1 or equal to the output dimension it is broadcasting into`` or ``IndexError: index out of range`` error during Neuron Parallel Compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running Neuron Parallel Compile with HF Trainer API, you may see the errors ``Status: INVALID_ARGUMENT: Input dimension should be either 1 or equal to the output dimension it is broadcasting into`` or ``IndexError: index out of range`` in Accelerator's ``pad_across_processes`` function. This is due to data-dependent operations in evaluation metrics computation. Data-dependent operations would result in undefined behavior with Neuron Parallel Compile trial execution (execute empty graphs with zero outputs). To work around this error, disable compute_metrics when NEURON_EXTRACT_GRAPHS_ONLY is set to 1:

.. code:: python

   compute_metrics=None if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY") else compute_metrics

Compiler assertion error when running Stable Diffusion training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With PyTorch 2.8 (torch-neuronx), you may encounter the following compiler assertion error with Stable Diffusion training when gradient accumulation is enabled. This will be fixed in an upcoming release. For now, if you want to run Stable Diffusion training, disable gradient accumulation in torch-neuronx 2.8 by keeping the `default gradient accumulation steps of 1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/stable_diffusion/run.py#L20>`__.

.. code:: bash

    ERROR 222163 [NeuronAssert]: Assertion failure in usr/lib/python3.9/concurrent/futures/process.py at line 239 with exception:
    too many partition dims! {{0,+,960}[10],+,10560}[10]


Frequently Asked Questions (FAQ)
--------------------------------

Do I need to recompile my models with PyTorch 2.8?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes.

Do I need to update my scripts for PyTorch 2.8?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See the :ref:`migration guide <migrate_to_pytorch_2.8>`

What environment variables will be changed with PyTorch NeuronX 2.8 ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warnings are shown when used). Switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to cast model to BF16. (see :ref:`migration_from_xla_downcast_bf16`)

What features will be missing with PyTorch NeuronX 2.8?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch NeuronX 2.8 has all of the supported features in PyTorch NeuronX 2.7, with known issues listed above, and unsupported features as listed in :ref:`torch-neuronx-rn`.

Can I use Neuron Distributed libraries with PyTorch NeuronX 2.8?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, NeuronX Distributed libraries are supported by PyTorch NeuronX 2.8. Transformers NeuronX has reached end-of-support in release 2.26. AWS Neuron Reference for NeMo Megatron has reached end-of-support in release 2.23.

Can I still use PyTorch 2.7 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch 2.7 is supported since release 2.24.

Can I still use PyTorch 2.6 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch 2.6 is supported since release 2.23.

Can I still use PyTorch 2.5 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch 2.5 reached end-of-support in release 2.25.

Can I still use Amazon Linux 2023?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes. You will need to install Python 3.10 or 3.11 to use PyTorch NeuronX 2.8.
