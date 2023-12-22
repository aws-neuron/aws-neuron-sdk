.. _introduce-pytorch-2-1:

Introducing PyTorch 2.1 Support (Beta)  
======================================

.. contents:: Table of contents
   :local:
   :depth: 2


What are we introducing?
------------------------

Starting with the :ref:`Neuron 2.16 <neuron-2.16.0-whatsnew>` release, customers will be able to upgrade to Beta version of ``PyTorch Neuron(torch-neuronx)`` supporting ``PyTorch 2.1``. 
PyTorch/XLA 2.x uses a new default runtime PJRT, which will also be used by ``PyTorch Neuron 2.1 Beta``. Neuron plans to support ``torch.compile`` (``TorchDynamo``) feature in future release of the Neuron SDK.

We have updated :ref:`setup-torch-neuronx` to include installation instructions for PyTorch Neuron 2.1 Beta for AL2023, Ubuntu 20 and Ubuntu 22. Users will also have to make possible training and inference script changes which
are shown below in :ref:`migration guide <migrate_to_pytorch_2_1>`.


.. _how-pytorch-2-1-different:

How is PyTorch Neuron 2.1 different than PyTorch Neuron 1.13?
-------------------------------------------------------------

By upgrading to ``PyTorch Neuron 2.1``, we will be removing the previous ``XRT`` runtime and ``XRT`` server that manages your program, applications will now be managed by individual ``PJRT`` clients instead. 
For more details on the changes between ``XRT`` and ``PJRT`` with ``PyTorch/XLA`` see this `documentation <https://github.com/pytorch/xla/blob/r2.1/docs/pjrt.md>`_.

In addition, the behavior of ``xm.rendezvous()`` APIs have been updated in PyTorch 2.1. Users might need to make possible code changes in the training/inference
scripts which is discussed in the below :ref:`migration guide <migrate_to_pytorch_2_1>`


.. _how-pytorch-2-1-different:

How is PyTorch Neuron 2.1(Beta) different than PyTorch Neuron 2.0(Beta)?
------------------------------------------------------------------------

The experience with ``init_process_group()`` API is still the same between the two versions. PyTorch Neuron 2.1 overrides ``init_method='pjrt://'`` with  ``init_method='xla://'``, so users can skip this update.
Import of ``torch_xla.experimental.pjrt*`` is also no longer required.

HLO snapshot dumping is now available in PyTorch Neuron 2.1 which was missing before in PyTorch Neuron 2.0 via the ``XLA_FLAGS`` environment variable, using a combination of the ``--xla_dump_to`` and ``--xla_dump_hlo_snapshots`` command-line arguments.
For example:

.. code::

    XLA_FLAGS="--xla_dump_hlo_snapshots --xla_dump_to=./dump" python foo.py


will have ``foo.py``'s PJRT runtime execution snapshots dumped into ``./dump`` directory.

.. _install_pytorch_neuron_2_1:

How can I install PyTorch Neuron 2.1 (Beta)?
--------------------------------------------

To install PyTorch Neuron 2.1 Beta please follow the :ref:`setup-torch-neuronx` guides for AL2023, Ubuntu 20 AMI and Ubuntu 22 AMI. PyTorch Neuron 2.1 Beta can be installed using the following:

.. code::

    python -m pip install --upgrade neuronx-cc==2.* --pre torch-neuronx==2.1.* torchvision


.. note::
 PyTorch Neuron DLAMIs for Ubuntu 20 does not yet have a pre-installed PyTorch 2.1 Beta. Please use Ubuntu 20 AMI and Ubuntu 22 AMI setup guide instructions.

.. _migrate_to_pytorch_2_1:

Migrate your application to PyTorch 2.1 and PJRT
------------------------------------------------

Please make sure you have first installed the PyTorch Neuron 2.1 Beta as described above in :ref:`installation guide <install_pytorch_neuron_2_1>`


Migrating Training scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^

Following changes need to be made to migrate the training scripts.

.. dropdown::  Changes to ``xm.rendezvous()`` behavior
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
    :open:

    
    As ``xm.rendezvous()`` behavior has changed in PyTorch/XLA 2.x, PyTorch Neuron 2.1 has implemented synchronization API to be compatible with the change. There are no code changes users have to do related to ``xm.rendezvous()``. Users can however see possible performance drops and memory issues when calling ``xm.rendezvous()`` with a payload on large XLA graphs.
    These performance drops and memory issues will be addressed in future Neuron release.


Migrating Inference scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^
There should not be any code changes required in the inference scripts.


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
Also set ``use_reentrant=True`` while calling the torch_xla checkpoint function. Failure to do so will lead to ``XLA currently does not support use_reentrant==False`` error.
For more details on checkpointing, refer the `documentation <https://pytorch.org/docs/stable/checkpoint.html>`_.


Lower performance for BERT-Large
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently we see 8% less performance when running BERT-Large pretraining tutorial with Torch-Neuronx 2.1.


Divergence (non-convergence) of loss for BERT/LLaMA when using release 2.16 compiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently, when using release 2.16 compiler version ``2.12.54.0+f631c2365``, you may see divergence (non-convergence) of loss curve. To workaround this issue, please use release 2.15 compiler version ``2.11.0.35+4f5279863``.


Error "Attempted to access the data pointer on an invalid python storage" when using HF Trainer API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently, if using HuggingFace Transformers Trainer API to train (i.e. :ref:`HuggingFace Trainer API fine-tuning tutorial<torch-hf-bert-finetune>`), you may see the error "Attempted to access the data pointer on an invalid python storage". This is a known `issue <https://github.com/huggingface/transformers/issues/27578>`_ and will be fixed in a future release.


Frequently Asked Questions (FAQ)
--------------------------------

What is the difference between PJRT and Neuron Runtime?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PJRT is a separate runtime than Neuron Runtime. Both runtimes will be used by Neuron SDK to support PyTorch Neuron 2.x Beta.

Do I need to recompile my models with PyTorch 2.1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes.

Do I need to update my scripts for PyTorch 2.1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, script changes might be needed in Beta support. Please see the :ref:`migration guide <migrate_to_pytorch_2_x>`

What environment variables will be changed with PJRT?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Any of the previous XRT or libtpu.so environment variables that start with ``XRT`` or ``TPU`` (ex:- TPU_NUM_DEVICES) can be removed from scripts.
``PJRT_DEVICE`` is the new environment variable to control your compute device, by default it will be set to ``NEURON``.

What features will be missing with PyTorch Neuron 2.1 Beta?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Because Neuron support for PyTorch 2.1 is still in beta, we have some missing features from PyTorch Neuron 1.13 that we expect to have available in future Neuron release. 
The following features are not currently available in PyTorch Neuron 2.1 Beta :

* NEURON_FRAMEWORK_DEBUG: :ref:`torch-neuronx-snapshotting`
* Neuron Profiler in torch_neuronx: :ref:`pytorch-neuronx-debug`
* Analyze command with neuron_parallel_compile: :ref:`pytorch-neuronx-parallel-compile-cli`

Can I use Neuron Distributed and Transformers Neuron libraries with PyTorch Neuron 2.1 Beta?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, please note that they will be considered Beta if using them with PyTorch Neuron 2.1 Beta.

Can I still use PyTorch 1.13 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, PyTorch 1.13 will continue to be supported.
