.. _introduce-pytorch-2-6:

Introducing PyTorch 2.6 Support
===============================

.. contents:: Table of contents
   :local:
   :depth: 2


What are we introducing?
------------------------

Starting with the :ref:`Neuron 2.23 <neuron-2.23.0-whatsnew>` release, customers can now upgrade to PyTorch NeuronX (``torch-neuronx``) with specific support for PyTorch version 2.6.

:ref:`setup-torch-neuronx` is updated to include installation instructions for PyTorch NeuronX 2.6 for Amazon Linux 2023 and Ubuntu 22.04. Note that PyTorch NeuronX 2.6 is supported on Python 3.9, 3.10, and 3.11.

Review :ref:`migration guide <migrate_to_pytorch_2.6>` for possible changes to training scripts. No code changes are required for inference scripts.


.. _how-pytorch-2.6-different:

How is PyTorch NeuronX 2.6 different compared to PyTorch NeuronX 2.5?
---------------------------------------------------------------------

PyTorch NeuronX 2.6 uses Torch-XLA 2.6 which has improved support for Automatic Mixed Precision and buffer aliasing. Additionally:

* Reintroduced ``XLA_USE_32BIT_LONG`` to give customers the flexibility to use INT32 for their workloads. This flag was removed in v2.5.
* Added xm.xla_device_kind() to return the XLA device kind string ('NC_v2' for Trainium1, 'NC_v3' and 'NC_v3d' for Trainium2). See :ref:`logical-neuroncore-config` for more info.

See `Torch-XLA 2.6 release <https://github.com/pytorch/xla/releases/tag/v2.6.0>`__ for a full list.

See :ref:`migrate_to_pytorch_2.6` for changes needed to use PyTorch NeuronX 2.6.

.. note::

   GSPMD and Torch Dynamo (torch.compile) support in Neuron will be available in a future release.

.. _install_pytorch_neuron_2.6:

How can I install PyTorch NeuronX 2.6?
--------------------------------------------

To install PyTorch NeuronX 2.6, follow the :ref:`setup-torch-neuronx` guides for Amazon Linux 2023 and Ubuntu 22.04 AMI. Refer to the Neuron Multi-Framework DLAMI :ref:`setup guide <setup-ubuntu22-multi-framework-dlami>` for Ubuntu 22.04 with a pre-installed virtual environment for PyTorch NeuronX 2.6 that you can use to get started. PyTorch NeuronX 2.6 can be installed using the following:

.. code::

    python -m pip install --upgrade neuronx-cc==2.* torch-neuronx==2.6.* torchvision

.. note::

   PyTorch NeuronX 2.6 is currently available for Python 3.9, 3.10, 3.11.

.. _migrate_to_pytorch_2.6:

Migrate your application to PyTorch 2.6
---------------------------------------

First, install the PyTorch NeuronX 2.6 as described above in :ref:`installation guide <install_pytorch_neuron_2.6>`


Migrating training scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^

To migrate the training scripts from PyTorch NeuronX 2.5 to PyTorch NeuronX 2.6, implement the following changes: 

.. note::

    ``xm`` below refers to ``torch_xla.core.xla_model``, ``xr`` refers to ``torch_xla.runtime``, and ``xmp`` refers to ``torch_xla.distributed.xla_multiprocessing``

* The environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used) and will be removed in an upcoming release. Switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to convert model to BF16 format. (see :ref:`migration_from_xla_downcast_bf16`)
* The functions ``xm.xrt_world_size()``, ``xm.get_ordinal()``, and ``xm.get_local_ordinal()`` are deprecated (warnings are shown when used). Switch to ``xr.world_size()``, ``xr.global_ordinal()``, and ``xr.local_ordinal()`` respectively as replacements.
* The default behavior of ``torch.load`` parameter ``weights_only`` is changed from ``False`` to ``True``. Setting ``weights_only`` to ``True`` may cause issues with pickling custom objects.
* If using ``xmp.spawn``, the ``nprocs`` argument is limited to 1 or None since v2.1. Previously, passing a value > 1 would result in a warning. In torch-xla 2.6, passing a value > 1 will result in an error with an actionable message to use ``NEURON_NUM_DEVICES`` to set the number of NeuronCores to use.

See :ref:`v2.5 migration guide <migrate_to_pytorch_2.5>` for additional changes needed if you are migrating from PyTorch NeuronX 2.1.

Migrating inference scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are no code changes required in the inference scripts.


Troubleshooting and Known Issues
--------------------------------

Tensor split on second dimension of 2D array not working
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when using the tensor split operation on a 2D array in the second dimension, the resulting tensors do not contain the expected data (https://github.com/pytorch/xla/issues/8640). The workaround is to set ``XLA_DISABLE_FUNCTIONALIZATION=0``. Another workaround is to use ``torch.tensor_split``.


Lower BERT pretraining performance with torch-neuronx 2.6 compared to torch-neuronx 2.5
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, BERT pretraining performance is ~10% lower with torch-neuronx 2.6 compared to torch-neuronx 2.5. This is due to a known regression in the torch-xla library https://github.com/pytorch/xla/issues/9037 and may affect other models with high graph tracing overhead. To work around this issue, build the ``r2.6_aws_neuron`` branch of torch-xla as follows (see :ref:`pytorch-neuronx-install-cxx11` for C++11 ABI version):

.. code:: bash

   # Setup build env (make sure you are in a python virtual env). Replace "apt" with "yum" on AL2023.
   sudo apt install cmake
   pip install yapf==0.30.0
   wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
   sudo cp bazelisk-linux-amd64 /usr/local/bin/bazel
   # Clone repos
   git clone --recursive https://github.com/pytorch/pytorch --branch v2.6.0
   cd pytorch/
   git clone --recursive https://github.com/pytorch/xla.git --branch r2.6_aws_neuron
   _GLIBCXX_USE_CXX11_ABI=0 python setup.py bdist_wheel
   # The pip wheel will be present in ./dist
   cd xla/
   CXX_ABI=0 python setup.py bdist_wheel
   # The pip wheel will be present in ./dist and can be installed instead of the torch-xla released in pypi.org

Lower BERT pretraining performance when switch to using ``model.to(torch.bfloat16)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, BERT pretraining performance is approximately 11% lower when switching to using ``model.to(torch.bfloat16)`` as part of migration away from the deprecated environment variable ``XLA_DOWNCAST_BF16`` due to https://github.com/pytorch/xla/issues/8545. As a workaround to recover the performance, you can set ``XLA_DOWNCAST_BF16=1``, which will still work in torch-neuronx 2.5 and 2.6 although there will be end-of-support warnings (as noted below).


Warning "XLA_DOWNCAST_BF16 will be deprecated after the 2.6 release, please downcast your model directly"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used). Switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to cast model to BF16. (see :ref:`migration_from_xla_downcast_bf16`)


WARNING:root:torch_xla.core.xla_model.xrt_world_size() will be removed in release 2.7. is deprecated. Use torch_xla.runtime.world_size instead.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a warning that ``torch_xla.core.xla_model.xrt_world_size()`` will be removed in a future release. Switch to using ``torch_xla.runtime.world_size`` instead.


WARNING:torch_xla.core.xla_model.get_ordinal() will be removed in release 2.7. is deprecated. Use torch_xla.runtime.global_ordinal instead.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a warning that ``torch_xla.core.xla_model.get_ordinal()`` will be removed in a future release. Switch to using ``torch_xla.runtime.global_ordinal`` instead.

WARNING:torch_xla.core.xla_model.get_local_ordinal() will be removed in release 2.7. is deprecated. Use torch_xla.runtime.local_ordinal instead.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. warning::
    ``torch_xla.core.xla_model.get_local_ordinal()`` will be removed in a future release. Use ``torch_xla.runtime.local_ordinal`` instead.
    


Socket Error: Socket failed to bind
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In PyTorch 2.6, there must be a socket available for both torchrun and the ``init_process_group`` to bind. By default, both 
will be set to use unused sockets. If you plan to use a ``MASTER_PORT`` environment variable then this error may occur if the port you set it to
is already in use.

.. code:: 

    [W socket.cpp:426] [c10d] The server socket has failed to bind to [::]:2.600 (errno: 98 - Address already in use).
    [W socket.cpp:426] [c10d] The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
    [E socket.cpp:462] [c10d] The server socket has failed to listen on any local network address.
    RuntimeError: The server socket has failed to listen on any local network address. 
    The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).

To resolve the issue, ensure you are setting ``MASTER_PORT`` to a port value that is not used anywhere else in your scripts. Otherwise,
you can leave ``MASTER_PORT`` unset and torchrun will set the default port for you.


``AttributeError: module 'torch' has no attribute 'xla'`` Failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In PyTorch 2.6, training scripts might fail during activation checkpointing with the error shown below.

.. code::

    AttributeError: module 'torch' has no attribute 'xla'


The solution is to use ``torch_xla.utils.checkpoint.checkpoint`` instead of ``torch.utils.checkpoint.checkpoint`` as the checkpoint function while wrapping pytorch modules for activation checkpointing.
Refer to the pytorch/xla discussion regarding this `issue <https://github.com/pytorch/xla/issues/5766>`_.
Also set ``use_reentrant=True`` while calling the torch_xla checkpoint function. Failure to do so will lead to ``XLA currently does not support use_reentrant==False`` error.
For more details on checkpointing, refer the `documentation <https://pytorch.org/docs/stable/checkpoint.html>`_.


Error ``Attempted to access the data pointer on an invalid python storage`` when using HF Trainer API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While using HuggingFace Transformers Trainer API to train (i.e. :ref:`HuggingFace Trainer API fine-tuning tutorial<torch-hf-bert-finetune>`), you may see the error "Attempted to access the data pointer on an invalid python storage". This is a known `issue <https://github.com/huggingface/transformers/issues/2.678>`_ and has been fixed in the version ``4.37.3`` of HuggingFace Transformers.


``ImportError: libcrypt.so.1: cannot open shared object file: No such file or directory`` on Amazon Linux 2023
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

torch-xla version 2.6+ now requires ``libcrypt.so.1`` shared library. Currently, Amazon Linux 2023 includes ``libcrypt.so.2`` shared library by default so you may see ``ImportError: libcrypt.so.1: cannot open shared object file: No such file or directory`` when using torch-neuronx 2.1+ on Amazon Linux 2023. To install ``libcrypt.so.1`` on Amazon Linux 2023, run the following installation command (see also https://github.com/amazonlinux/amazon-linux-2023/issues/182 for more context):

.. code::

   sudo dnf install libxcrypt-compat


``FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'`` Failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In PyTorch 2.6, users might face the error shown below due to incompatible ``libneuronxla`` and ``torch-neuronx`` versions being installed.

.. code::

    FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'

Check that the version of ``libneuronxla`` that support PyTorch NeuronX 2.6 is ``2.2.*``. If not, then uninstall ``libneuronxla`` using ``pip uninstall libneuronxla`` and then reinstall the packages following the installation guide :ref:`installation guide <install_pytorch_neuron_2.6>`


``Input dimension should be either 1 or equal to the output dimension it is broadcasting into`` or ``IndexError: index out of range`` error during Neuron Parallel Compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running Neuron Parallel Compile with HF Trainer API, you may see the errors ``Status: INVALID_ARGUMENT: Input dimension should be either 1 or equal to the output dimension it is broadcasting into`` or ``IndexError: index out of range`` in Accelerator's ``pad_across_processes`` function. This is due to data-dependent operation in evaluation metrics computation. Data-dependent operations would result in undefined behavior with Neuron Parallel Compile trial execution (execute empty graphs with zero outputs). To work-around this error, disable compute_metrics when NEURON_EXTRACT_GRAPHS_ONLY is set to 1:

.. code:: python

   compute_metrics=None if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY") else compute_metrics

Compiler assertion error when running Stable Diffusion training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With PyTorch 2.6 (torch-neuronx), you may encounter the following compiler assertion error with Stable Diffusion training when gradient accumulation is enabled. This will be fixed in an upcoming release. For now, if you want to run Stable Diffusion training, disable gradient accumulation in torch-neuronx 2.6 by keeping the `default gradient accumulation steps of 1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/stable_diffusion/run.py#L20>`__.

.. code:: bash

    ERROR 222163 [NeuronAssert]: Assertion failure in usr/lib/python3.9/concurrent/futures/process.py at line 239 with exception:
    too many partition dims! {{0,+,960}[10],+,10560}[10]


Frequently Asked Questions (FAQ)
--------------------------------

Do I need to recompile my models with PyTorch 2.6?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes.

Do I need to update my scripts for PyTorch 2.6?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See the :ref:`migration guide <migrate_to_pytorch_2.6>`

What environment variables will be changed with PyTorch NeuronX 2.6 ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used). Switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to cast model to BF16. (see :ref:`migration_from_xla_downcast_bf16`)

What features will be missing with PyTorch NeuronX 2.6?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch NeuronX 2.6 has all of the supported features in PyTorch NeuronX 2.5, with known issues listed above, and unsupported features as listed in :ref:`pytorch_rn`.

Can I use Neuron Distributed and Transformers Neuron libraries with PyTorch NeuronX 2.6?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, NeuronX Distributed, and Transformers NeuronX, and AWS Neuron Reference for NeMo Megatron libraries will work with PyTorch NeuronX 2.6.

Can I still use PyTorch 2.5 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch 2.5 is supported for releases 2.21/2.22/2.23 and will reach end-of-life in a future release. Additionally, the CVE `CVE-2025-32434 <https://github.com/advisories/GHSA-53q9-r3pm-6pq6>`_ affects PyTorch version 2.5. We recommend upgrading to the new version of Torch-NeuronX by following :ref:`setup-torch-neuronx`.

Can I still use PyTorch 2.1 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch 2.1 is supported for release 2.21 and has reached end-of-life in release 2.22. Additionally, the CVEs `CVE-2024-31583 <https://github.com/advisories/GHSA-pg7h-5qx3-wjr3>`_ and `CVE-2024-31580 <https://github.com/advisories/GHSA-5pcm-hx3q-hm94>`_ affect PyTorch versions 2.1 and earlier.  We recommend upgrading to the new version of Torch-NeuronX by following :ref:`setup-torch-neuronx`.
