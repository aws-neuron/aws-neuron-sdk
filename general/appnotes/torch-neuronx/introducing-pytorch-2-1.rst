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

In PyTorch 2.1, there needs to be a socket available for both torchrun and the ``init_process_group`` to bind. Both of these, by default,
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

Error ``Attempted to access the data pointer on an invalid python storage`` when using HF Trainer API
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


Error ``cannot import name 'builder' from 'google.protobuf.internal'`` after installing compiler from earlier releases (2.19 or earlier)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using release 2.20 packages and you choose to install compiler from an earlier release (2.19 or earlier), you may see the error ``cannot import name 'builder' from 'google.protobuf.internal`` with the earlier compiler's dependency on protobuf version 3.19. To work-around this issue, please install protobuf 3.20.3 (``pip install protobuf==3.20.3``, ignoring the pip dependency check error due to earlier compiler's dependency on protobuf version 3.19).

.. code:: bash

    File "/home/ubuntu/aws_neuron_venv_pytorch/lib/python3.11/site-packages/torch_neuronx/proto/metaneff_pb2.py", line 5, in <module> from google.protobuf.internal import builder as _builder
    ImportError: cannot import name 'builder' from 'google.protobuf.internal' (/home/ubuntu/aws_neuron_venv_pytorch/lib/python3.11/site-packages/google/protobuf/internal/__init__.py)

Lower accuracy when fine-tuning Roberta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently with release 2.20, we see lower accuracy (68% vs expected 89%) when fine-tuning Roberta-large and MRPC. This will be fixed in a future release. To work-around, you may use the compiler from release 2.19, noting the ``protobuf`` issue above:

.. code:: bash

    python3 -m pip install neuronx-cc==2.14.227.0+2d4f85be protobuf==3.20.3


Slower loss convergence for NxD LLaMA-3 70B pretraining using ZeRO1 tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with Torch-NeuronX 2.1, we see slower loss convergence in the :ref:`LLaMA-3 70B tutorial for neuronx-distributed<llama3_tp_pp_tutorial>` when using the recommended flags (``NEURON_CC_FLAGS="--distribution-strategy llm-training --model-type transformer"``). To work-around this issue, please only use ``--model-type transformer`` flag (``NEURON_CC_FLAGS="--model-type transformer"``).

Gradient accumulation is not yet supported for Stable Diffusion due to a compiler error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with torch-neuronx 2.1, we are seeing a compiler assertion error with Stable Diffusion training when gradient accumulation is enabled. To train Stable Diffusion with gradient accumulation, please use torch-neuronx 1.13 instead of 2.1.

Enable functionalization to resolve slower loss convergence for NxD LLaMA-2 70B pretraining using ZeRO1 tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously with Torch-NeuronX 2.1, we see slower loss convergence in the :ref:`LLaMA-2 70B tutorial for neuronx-distributed<llama2_tp_pp_tutorial>`. This issue is now resolved. Customer can now run the tutorial with the recommended flags (``NEURON_CC_FLAGS="--distribution-strategy llm-training --model-type transformer"``) and turning on functionalization (``XLA_DISABLE_FUNCTIONALIZATION=0``). Turning on functionalization results in slightly higher device memory usage and ~11% lower in performance due to a known issue with torch-xla 2.1 (https://github.com/pytorch/xla/issues/7174). The higher device memory usage also limits LLaMA-2 70B tutorial to run on 16 trn1.32xlarge nodes at the minimum, and running on 8 nodes would result in out-of-memory error. See the :ref:`list of environment variables<>` for more information about ``XLA_DISABLE_FUNCTIONALIZATION``.

Enabling functionalization (``XLA_DISABLE_FUNCTIONALIZATION=0``) results in 15% lower performance and non-convergence for the BERT pretraining tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with torch-neuronx 2.1, enabling functionalization (``XLA_DISABLE_FUNCTIONALIZATION=0``) would result in 15% lower performance and non-convergence for the BERT pretraining tutorial. The lower performance is due to missing aliasing for gradient accumulation and is a known issue with torch-xla 2.1 (https://github.com/pytorch/xla/issues/7174). The non-convergence is due to an issue in marking weights as static (buffer address not changing), which can be worked around by setting ``NEURON_TRANSFER_WITH_STATIC_RING_OPS`` to empty string (``NEURON_TRANSFER_WITH_STATIC_RING_OPS=""``. See the :ref:`list of environment variables<>` for more information about ``XLA_DISABLE_FUNCTIONALIZATION``. and ``NEURON_TRANSFER_WITH_STATIC_RING_OPS``.

.. code:: bash

   export NEURON_TRANSFER_WITH_STATIC_RING_OPS=""


GlibC error on Amazon Linux 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If using Torch-NeuronX 2.1 on Amazon Linux 2, you will see a GlibC error below. Please switch to a newer supported OS such as Ubuntu 20, Ubuntu 22, or Amazon Linux 2023.

.. code:: bash

   ImportError: /lib64/libc.so.6: version `GLIBC_2.27' not found (required by /tmp/debug/_XLAC.cpython-38-x86_64-linux-gnu.so)


``"EOFError: Ran out of input"`` or ``"_pickle.UnpicklingError: invalid load key, '!'"`` errors during Neuron Parallel Compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With torch-neuronx 2.1, HF Trainer API's use of XLA function ``.mesh_reduce`` causes ``"EOFError: Ran out of input"`` or ``"_pickle.UnpicklingError: invalid load key, '!'"`` errors during Neuron Parallel Compile. To work-around this issue, you can add the following code snippet (after python imports) to replace ``xm.mesh_reduce`` with a form that uses ``xm.all_gather`` instead of ``xm.rendezvous()`` with payload. This will add additional small on-device graphs (as opposed to the original ``xm.mesh_reduce`` which runs on CPU).

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

With torch-neuronx 2.1, using ``torch.utils.data.DataLoader`` with ``shuffle=True`` would cause the following error in ``synchronize_rng_states`` (i.e. :ref:`ZeRO1 tutorial<zero1-gpt2-pretraining-tutorial>`):

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


Compiler error when ``torch_neuronx.xla_impl.ops.set_unload_prior_neuron_models_mode(True)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently with torch-neuronx 2.1, using the ``torch_neuronx.xla_impl.ops.set_unload_prior_neuron_models_mode(True)`` (as previously done in the :ref:`ZeRO1 tutorial<zero1-gpt2-pretraining-tutorial>`) to unload graphs during execution would cause a compilation error ``Expecting value: line 1 column 1 (char 0)``. You can remove this line as it is not recommended for use. Please see the updated :ref:`ZeRO1 tutorial<zero1-gpt2-pretraining-tutorial>` in release 2.18.

Compiler assertion error when running Stable Diffusion training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with torch-neuronx 2.1, we are seeing the following compiler assertion error with Stable Diffusion training when gradient accumulation is enabled. This will be fixed in an upcoming release. For now, if you would like to run Stable Diffusion training with Neuron SDK release 2.18, please use ``torch-neuronx==1.13.*`` or disable gradient accumulation in torch-neuronx 2.1.

.. code:: bash

    ERROR 222163 [NeuronAssert]: Assertion failure in usr/lib/python3.8/concurrent/futures/process.py at line 239 with exception:
    too many partition dims! {{0,+,960}[10],+,10560}[10]


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
