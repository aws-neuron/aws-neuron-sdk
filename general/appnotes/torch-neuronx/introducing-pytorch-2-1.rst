.. _introduce-pytorch-2-1:

Introducing PyTorch 2.1 Support  
===============================

.. contents:: Table of contents
   :local:
   :depth: 2


What are we introducing?
------------------------

Starting with the :ref:`Neuron 2.18 <neuron-2.18.0-whatsnew>` release, customers will be able to upgrade to ``PyTorch NeuronX(torch-neuronx)`` supporting ``PyTorch 2.1``. 
PyTorch NeuronX 2.x now uses the PyTorch-XLA PJRT instead of XRT to provide better scalability and simpler Neuron integration.

We have updated :ref:`setup-torch-neuronx` to include installation instructions for PyTorch NeuronX 2.1 for AL2023, Ubuntu 20 and Ubuntu 22. Users will also have to make possible training and inference script changes which
are shown below in :ref:`migration guide <migrate_to_pytorch_2_1>`.


.. _how-pytorch-2-1-different:

How is PyTorch NeuronX 2.1 different than PyTorch NeuronX 1.13?
-------------------------------------------------------------

By upgrading to ``PyTorch NeuronX 2.1``, we will be removing the previous ``XRT`` runtime and ``XRT`` server that manages your program, applications will now be managed by individual ``PJRT`` clients instead. 
For more details on the changes between ``XRT`` and ``PJRT`` with ``PyTorch/XLA`` see this `documentation <https://github.com/pytorch/xla/blob/r2.1/docs/pjrt.md>`_.

In addition, the behavior of ``xm.rendezvous()`` APIs have been updated in PyTorch 2.1. There's no code change needed to switch from PyTorch NeuronX 1.13 to PyTorch NeuronX 2.1, except for snapshotting
which is discussed in the below :ref:`migration guide <migrate_to_pytorch_2_1>`

HLO snapshot dumping is available in PyTorch Neuron 2.1 via the ``XLA_FLAGS`` environment variable, using a combination of the ``--xla_dump_to`` and ``--xla_dump_hlo_snapshots`` command-line arguments.
For example:

.. code::

    XLA_FLAGS="--xla_dump_hlo_snapshots --xla_dump_to=./dump" python foo.py


will have ``foo.py``'s PJRT runtime execution snapshots dumped into ``./dump`` directory. See :ref:`torch-neuronx-snapshotting` section for more information.

.. note::

    Snapshot dumping triggered by a runtime error such as NaN is not yet available in PyTorch NeuronX 2.1. It will be available in a future release.


Starting with ``PyTorch/XLA 2.1``, functionalization changes result in new graphs leading to lower performance while training. Refer similar discussions `here <https://github.com/pytorch/xla/issues/6294>`_. We set ``XLA_DISABLE_FUNCTIONALIZATION=1`` as default to help with better performance. More on functionalization in Pytorch can be found `here <https://dev-discuss.pytorch.org/t/functionalization-in-pytorch-everything-you-wanted-to-know/965>`_.

.. note::

    In ``PyTorch/XLA 2.1``, the HLOModuleProto files dumped in the neuron cache ``/var/tmp/neuron-compile-cache`` (default path) is suffixed as ``.hlo_module.pb`` which was earlier dumped out as ``.hlo.pb`` in ``PyTorch/XLA 1.13``


.. _install_pytorch_neuron_2_1:

How can I install PyTorch NeuronX 2.1?
--------------------------------------------

To install PyTorch NeuronX 2.1 please follow the :ref:`setup-torch-neuronx` guides for AL2023, Ubuntu 20 AMI and Ubuntu 22 AMI. Please also refer to the Neuron multi framework DLAMI :ref:`setup guide <setup-ubuntu22-multi-framework-dlami>` for Ubuntu 22 with a pre-installed virtual environment for PyTorch NeuronX 2.1 that you can use to easily get started. PyTorch NeuronX 2.1 can be installed using the following:

.. code::

    python -m pip install --upgrade neuronx-cc==2.* torch-neuronx==2.1.* torchvision


.. note::
 PyTorch NeuronX DLAMIs for Ubuntu 20 does not yet have a pre-installed PyTorch 2.1. Please use Ubuntu 20 AMI and Ubuntu 22 AMI setup guide instructions.

.. _migrate_to_pytorch_2_1:

Migrate your application to PyTorch 2.1 and PJRT
------------------------------------------------

Please make sure you have first installed the PyTorch NeuronX 2.1 as described above in :ref:`installation guide <install_pytorch_neuron_2_1>`


Migrating Training scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^

Following changes need to be made to migrate the training scripts from PyTorch NeuronX 1.13 to PyTorch NeuronX 2.1.


.. dropdown::  Activation Checkpointing changes
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
    :open:


    Starting with PyTorch Neuron 2.1, users will have to use ``torch_xla.utils.checkpoint.checkpoint`` instead of ``torch.utils.checkpoint.checkpoint`` as the checkpointing function while wrapping pytorch modules for activation checkpointing. Refer to the pytorch/xla discussion regarding this `issue <https://github.com/pytorch/xla/issues/5766>`_. 
    Also set ``use_reentrant=True`` while calling the torch_xla checkpoint function. Failure to do so will lead to ``XLA currently does not support use_reentrant==False`` error. For more details on checkpointing, refer the `documentation <https://pytorch.org/docs/stable/checkpoint.html>`_.


.. dropdown::  Changes to ``xm.rendezvous()`` behavior
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
    :open:

    
    As ``xm.rendezvous()`` behavior has changed in PyTorch/XLA 2.x, PyTorch NeuronX 2.1 has implemented synchronization API to be compatible with the change. There are no code changes users have to do related to ``xm.rendezvous()``. Users can however see possible performance drops and memory issues when calling ``xm.rendezvous()`` with a payload on large XLA graphs.


Migrating Inference scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are no code changes required in the inference scripts.


Troubleshooting
---------------

Socket Error: Socket failed to bind
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description
~~~~~~~~~~~

In PyTorch 2.1, there needs to be a socket available for both torchrun and the ``init_process_group`` to bind. Both of these, by default,
will be set to unused sockets. If you plan to use a ``MASTER_PORT`` environment variable then this error may occur, if the port you set it to
is already in use.

.. code:: 

    [W socket.cpp:426] [c10d] The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use).
    [W socket.cpp:426] [c10d] The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
    [E socket.cpp:462] [c10d] The server socket has failed to listen on any local network address.
    RuntimeError: The server socket has failed to listen on any local network address. 
    The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).


Solution
~~~~~~~~

Please ensure if you are setting ``MASTER_PORT`` that the port you're setting it to is not used anywhere else in your scripts. Otherwise,
you can leave ``MASTER_PORT`` unset, and torchrun will set the default port for you.


``AttributeError: module 'torch' has no attribute 'xla'`` Failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In PyTorch 2.1, training scripts might fail during activation checkpointing with the error shown below.

.. code::

    AttributeError: module 'torch' has no attribute 'xla'


The solution is to use ``torch_xla.utils.checkpoint.checkpoint`` instead of ``torch.utils.checkpoint.checkpoint`` as the checkpoint function while wrapping pytorch modules for activation checkpointing.
Refer to the pytorch/xla discussion regarding this `issue <https://github.com/pytorch/xla/issues/5766>`_.
Also set ``use_reentrant=True`` while calling the torch_xla checkpoint function. Failure to do so will lead to ``XLA currently does not support use_reentrant==False`` error.
For more details on checkpointing, refer the `documentation <https://pytorch.org/docs/stable/checkpoint.html>`_.


Incorrect device assignment when using ellipsis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Usage of ellipsis (``...``) with PyTorch/XLA 2.1 can lead to incorrect device assignment of the tensors as 'lazy' instead of 'xla'.
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


This only happens for scenarios where ellipsis is used to extract a subset of a tensor with the same size as that of the original tensor. An issue is created with pytorch/xla to fix this behavior `Ref <https://github.com/pytorch/xla/issues/6398>`_.
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


Lower performance for BERT-Large
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently we see 8% less performance when running BERT-Large pretraining tutorial with Torch-Neuronx 2.1.


Divergence (non-convergence) of loss for BERT/LLaMA when using release 2.16 compiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently, when using release 2.16 compiler version ``2.12.54.0+f631c2365``, you may see divergence (non-convergence) of loss curve. To workaround this issue, please use release 2.15 compiler version ``2.11.0.35+4f5279863``.


Error "Attempted to access the data pointer on an invalid python storage" when using HF Trainer API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While using HuggingFace Transformers Trainer API to train (i.e. :ref:`HuggingFace Trainer API fine-tuning tutorial<torch-hf-bert-finetune>`), you may see the error "Attempted to access the data pointer on an invalid python storage". This is a known `issue <https://github.com/huggingface/transformers/issues/27578>`_ and has been fixed in the version ``4.37.3`` of HuggingFace Transformers.

``ImportError: libcrypt.so.1: cannot open shared object file: No such file or directory`` on Amazon Linux 2023
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

torch-xla version 2.1+ now requires ``libcrypt.so.1`` shared library. Currently, Amazon Linux 2023 includes ``libcrypt.so.2`` shared library by default so you may see `ImportError: libcrypt.so.1: cannot open shared object file: No such file or directory`` when using torch-neuronx 2.1+ on Amazon Linux 2023. To install ``libcrypt.so.1`` on Amazon Linux 2023, please run the following installation command (see also https://github.com/amazonlinux/amazon-linux-2023/issues/182 for more context):

.. code::

   sudo yum install libxcrypt-compat


``FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'`` Failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In PyTorch 2.1, users might face the error shown below due to incompatible ``libneuronxla`` and ``torch-neuronx`` versions being installed.

.. code::

    FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'

Check that the version of ``libneuronxla`` is ``2.0.*``. If not, then uninstall ``libneuronxla`` using ``pip uninstall libneuronxla`` and then reinstall the packages following the installation guide :ref:`installation guide <install_pytorch_neuron_2_1>`


Frequently Asked Questions (FAQ)
--------------------------------

What is the difference between PJRT and Neuron Runtime?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PJRT is the framework-level interface that enables frameworks such as PyTorch and JAX to compile HLO graphs using Neuron Compiler and
execute compiled graphs using Neuron Runtime. Neuron Runtime is device-specific runtime that enables compiled graphs to run on the Neuron devices.
Both runtimes will be used by Neuron SDK to support PyTorch NeuronX 2.x.

Do I need to recompile my models with PyTorch 2.1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes.

Do I need to update my scripts for PyTorch 2.1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
No changes are required for PyTorch 2.1 if users are migrating from PyTorch 1.13. If migrating from PyTorch 2.0, users can optionally get rid of the ``torch_xla.experimental.pjrt*`` imports
for ``init_process_group`` call. Please see the :ref:`migration guide <migrate_to_pytorch_2_1>`

What environment variables will be changed with PJRT?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Any of the previous XRT or libtpu.so environment variables that start with ``XRT`` or ``TPU`` (ex:- TPU_NUM_DEVICES) can be removed from scripts.
``PJRT_DEVICE`` is the new environment variable to control your compute device, by default it will be set to ``NEURON``.
Also ``NEURON_DUMP_HLO_SNAPSHOT`` and ``NEURON_NC0_ONLY_SNAPSHOT`` are no longer support in 2.1. Please see snapshotting guide for updated 2.1 instructions.

What features will be missing with PyTorch NeuronX 2.1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch NeuronX 2.1 now have most of the supported features in PyTorch NeuronX 1.13, with known issues listed above, and unsupported features as listed in release notes.

Can I use Neuron Distributed and Transformers Neuron libraries with PyTorch NeuronX 2.1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, Neuron Distributed and Transformers Neuron libraries will work with PyTorch NeuronX 2.1.

Can I still use PyTorch 1.13 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, PyTorch 1.13 will continue to be supported.
