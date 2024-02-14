.. _introduce-pytorch-2-0:

Introducing PyTorch 2.0 Support (End of Support)  
=================================================

.. contents:: Table of contents
   :local:
   :depth: 2

.. note::
 Neuron no longer supports PyTorch 2.0. Please migrate to PyTorch 2.1 via :ref:`migration guide <migrate_to_pytorch_2_1>`.

What are we introducing?
------------------------

Starting with the :ref:`Neuron 2.15 <neuron-2.15.0-whatsnew>` release, customers will be able to upgrade to Beta version of ``PyTorch Neuron(torch-neuronx)`` supporting ``PyTorch 2.0``. 
PyTorch/XLA 2.0 uses a new default runtime PJRT, which will also be used by ``PyTorch Neuron 2.0 Beta``. Neuron plans to support ``torch.compile`` (``TorchDynamo``) feature in future release of the Neuron SDK.

We have updated :ref:`setup-torch-neuronx` to include installation instructions for PyTorch Neuron 2.0 Beta for Ubuntu 20 and Ubuntu 22. Users will also have to make possible training and inference script changes which
are shown below in :ref:`migration guide <migrate_to_pytorch_2_0>`.


.. _how-pytorch-2-0-different:

How is PyTorch Neuron 2.0 different than PyTorch Neuron 1.13?
-------------------------------------------------------------

By upgrading to ``PyTorch Neuron 2.0``, we will be removing the previous ``XRT`` runtime and ``XRT`` server that manages your program, applications will now be managed by individual ``PJRT`` clients instead. 
For more details on the changes between ``XRT`` and ``PJRT`` with ``PyTorch/XLA`` see this `documentation <https://github.com/pytorch/xla/blob/r2.0/docs/pjrt.md>`_.

In addition, the behavior of ``init_process_group()`` and ``xm.rendezvous()`` APIs have been updated in PyTorch 2.0. Users might need to make possible code changes in the training/inference
scripts which is discussed in the below :ref:`migration guide <migrate_to_pytorch_2_0>`



.. _install_pytorch_neuron_2_0:

How can I install PyTorch Neuron 2.0 (Beta)?
--------------------------------------------

To install PyTorch Neuron 2.0 Beta please follow the :ref:`setup-torch-neuronx` guides for Ubuntu 20 AMI and Ubuntu 22 AMI. PyTorch Neuron 2.0 Beta can be installed using the following:

.. code::

    python -m pip install --upgrade neuronx-cc==2.* --pre torch-neuronx==2.0.* torchvision


.. note::
 PyTorch Neuron DLAMIs for Ubuntu 20 does not yet have a pre-installed PyTorch 2.0 Beta. Please use Ubuntu 20 AMI and Ubuntu 22 AMI setup guide instructions.

.. _migrate_to_pytorch_2_0:

Migrate your application to PyTorch 2.0 and PJRT
------------------------------------------------

Please make sure you have first installed the PyTorch Neuron 2.0 Beta as described above in :ref:`installation guide <install_pytorch_neuron_2_0>`


Migrating Training scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^

Following changes need to be made to migrate the training scripts.

.. _changes_for_init_process_group:

.. dropdown::  Changes to ``init_process_group()``
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
    :open:
    
    As PJRT backend is invoked along with the ``PyTorch/XLA`` backend, we need to initialize our backend for PJRT. Following code changes need to be made where ``init_process_group`` is called. 


    Old:

    .. code:: python 

        torch.distributed.init_process_group('xla')

    New:

    .. code:: python

        # Now we have to import pjrt_backend to use pjrt:// for the init_process_group
        import torch_xla.experimental.pjrt_backend
        # Also, to use pjrt functions after you call init_process_group
        import torch_xla.experimental.pjrt as pjrt
        # Call init_process_group with new pjrt:// init_method
        torch.distributed.init_process_group('xla', init_method='pjrt://')

.. dropdown::  Changes to ``xm.rendezvous()`` behavior
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
    :open:

    
    As ``xm.rendezvous()`` behavior has changed in PyTorch/XLA 2.0, PyTorch Neuron 2.0 has implemented synchronization API to be compatible with the change. There are no code changes users have to do related to ``xm.rendezvous()``. Users can however see possible performance drops and memory issues when calling ``xm.rendezvous()`` with a payload on large XLA graphs.
    These performance drops and memory issues will be addressed in future Neuron release.



Please see this :ref:`BERT tutorial <hf-bert-pretraining-tutorial>` for an example of changes within a training script to migrate it to PyTorch Neuron 2.0 



Migrating Inference scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In most cases, there should not be any code changes needed in inference scripts unless  ``init_process_group()`` is being called.  If ``init_process_group`` is being called, users need
to update the code as outlined in :ref:`Changes to init_process_group() <changes_for_init_process_group>` section above.



Troubleshooting
---------------

``init_process_group()`` Failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description
~~~~~~~~~~~
Because PyTorch/XLA has changed the ``init_method`` for ``init_process_group()``, make sure you are using the correct parameters for this.
A common error would be:

.. code::

    RuntimeError: No rendezvous handler for pjrt://


This error means you have not properly imported the rendezvous handler from ``torch_xla.experimental.pjrt_backend``

Solution
~~~~~~~~

Make sure you are calling ``init_process_group`` and not forgetting the import statement like so:

.. code:: python 

    import torch_xla.experimental.pjrt_backend
    torch.distributed.init_process_group('xla', init_method='pjrt://')


Socket Error: Socket failed to bind
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description
~~~~~~~~~~~

In PyTorch 2.0, there needs to be a socket available for both torchrun and the ``init_process_group`` to bind. Both of these, by default,
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


Frequently Asked Questions (FAQ)
--------------------------------

What is the difference between PJRT and Neuron Runtime?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PJRT is a separate runtime than Neuron Runtime. Both runtimes will be used by Neuron SDK to support PyTorch Neuron 2.0 Beta.

Do I need to recompile my models with PyTorch 2.0?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes.

Do I need to update my scripts for PyTorch 2.0?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, script changes might be needed in Beta support. Please see the :ref:`migration guide <migrate_to_pytorch_2_0>`

What environment variables will be changed with PJRT?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Any of the previous XRT or libtpu.so environment variables that start with ``XRT`` or ``TPU`` (ex:- TPU_NUM_DEVICES) can be removed from scripts.
``PJRT_DEVICE`` is the new environment variable to control your compute device, by default it will be set to ``NEURON``.

What features will be missing with PyTorch Neuron 2.0 Beta?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Because Neuron support for PyTorch 2.0 is still in beta, we have some missing features from PyTorch Neuron 1.13 that we expect to have available in future Neuron release. 
The following features are not currently available in PyTorch Neuron 2.0 Beta :

* NEURON_FRAMEWORK_DEBUG: :ref:`torch-neuronx-snapshotting`
* HLO Snapshotting: :ref:`torch-neuronx-snapshotting`
* Neuron Profiler in torch_neuronx: :ref:`pytorch-neuronx-debug`
* Analyze command with neuron_parallel_compile: :ref:`pytorch-neuronx-parallel-compile-cli`

Can I use Neuron Distributed and Transformers Neuron libraries with PyTorch Neuron 2.0 Beta?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, please note that they will be considered Beta if using them with PyTorch Neuron 2.0 Beta.

Can I still use PyTorch 1.13 version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, PyTorch 1.13 will continue to be supported.
