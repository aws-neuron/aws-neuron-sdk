.. _neuron-2-25-0-pytorch:

.. meta::
   :description: The official release notes for the AWS Neuron SDK PyTorch support component, version 2.25.0. Release date: 7/31/2025.

AWS Neuron SDK 2.25.0: PyTorch support release notes
====================================================

**Date of release**: July 31, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.25.0 release notes home <neuron-2-25-0-whatsnew>`

Released versions
-----------------

- ``2.7.0.2.9.*``
- ``2.6.0.2.9.*``

Improvements
------------

- The `Core Placement API <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/inference/api-torch-neuronx-core-placement.html>`_ is no longer beta/experimental and instructions to use it has been updated.
To migrate, replace any function scope ``torch_neuron.experimental.`` with ``torch_neuron.``. The change will have no effect on behavior or performance. For example, replace ``torch_neuronx.experimental.set_neuron_cores`` with ``torch_neuronx.set_neuron_cores``. If you use ``torch_neuron.experimental.*`` scope it will work as before but now will also emit a warning “In a future version torch_neuronx.experimental.<func> will be removed.  Call torch_neuronx.<func> instead."

Known issues
------------

Please see the :ref:`Introducing PyTorch 2.7 Support<introduce-pytorch-2-7>` for a full list of known issues with v2.7.
Please see the :ref:`Introducing PyTorch 2.6 Support<introduce-pytorch-2-6>` for a full list of known issues with v2.6.

[v2.7] Using the latest torch-xla v2.7 may result in increase in host memory usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the latest ``torch-xla`` v2.7 may result in increase in host memory usage compared to ``torch-xla`` 2.6. In on example, LLama2 pretraining with ZeRO1 and sequence length 16k could see an increase of 1.6% in host memory usage.

Updating Ubuntu OS kernel version from 5.15 to 6.8 may result in lower performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when switching Ubuntu OS kernel version from 5.15 to 6.8, you may see performance differences due to the new kernel scheduler (CFS vs EEVDF). For example, BERT pretraining performance could be lower by up to 10%. You may try using an older OS kernel (such as Amazon Linux 2023) or experiment with the kernel real-time scheduler by running ``sudo chrt --fifo 99`` before your command (such as ``sudo chrt --fifo 99 {script-here}``) to improve the performance. Note that adjusting the real-time scheduler can also result in lower performance. See https://www.kernel.org/doc/html/latest/scheduler/sched-eevdf.html for more information.

Tensor split on second dimension of 2D array not working
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when using tensor split operation on a 2D array in the second dimension, the resulting tensors don't have the expected data (https://github.com/pytorch/xla/issues/8640). The work-around is to set ``XLA_DISABLE_FUNCTIONALIZATION=0``. Another work-around is to use ``torch.tensor_split``.


[v2.6] Lower BERT pretraining performance with torch-neuronx 2.6 compared to torch-neuronx 2.5
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BERT pretraining performance is ~10% lower with ``torch-neuronx`` 2.6 compared to ``torch-neuronx`` 2.5. This is due to a known regression in torch-xla https://github.com/pytorch/xla/issues/9037 and can affect other models with high graph tracing overhead. This is fixed in torch-xla v2.7.

To work-around this issue in ``torch-xla`` v2.6, please build the ``r2.6_aws_neuron`` branch of torch-xla as follows (see :ref:`pytorch-neuronx-install-cxx11` for C++11 ABI version):

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
   # pip wheel will be present in ./dist
   cd xla/
   CXX_ABI=0 python setup.py bdist_wheel
   # pip wheel will be present in ./dist and can be installed instead of the torch-xla released in pypi.org


Lower BERT pretraining performance when switch to using ``model.to(torch.bfloat16)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, BERT pretraining performance is ~11% lower when switching to using ``model.to(torch.bfloat16)`` as part of migration away from the deprecated environment variable ``XLA_DOWNCAST_BF16`` due to https://github.com/pytorch/xla/issues/8545. As a work-around to recover the performance, you can set ``XLA_DOWNCAST_BF16=1`` which would still work in torch-neuronx 2.5 and 2.6 although there will be deprecation warnings (as noted below).

Warning "XLA_DOWNCAST_BF16 will be deprecated after the 2.5 release, please downcast your model directly"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (warning when used). Please switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to cast model to BF16. (see :ref:`migration_from_xla_downcast_bf16`)


[v2.6] AttributeError: <module 'torch_xla.core.xla_model' ... does not have the attribute 'xrt_world_size'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an error that ``torch_xla.core.xla_model.xrt_world_size()`` is removed in torch-xla version 2.7. Please switch to using ``torch_xla.runtime.world_size()`` instead.

[v2.6] AttributeError: <module 'torch_xla.core.xla_model' ... does not have the attribute 'get_ordinal'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an error that ``torch_xla.core.xla_model.xla_model.get_ordinal()`` is removed in torch-xla version 2.7. Please switch to using ``torch_xla.runtime.global_ordinal()`` instead.

AttributeError: module 'torch_xla.runtime' has no attribute 'using_pjrt'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Torch-XLA 2.5+, ``torch_xla.runtime.using_pjrt`` is removed because PJRT is the sole Torch-XLA runtime.
See `commit PR <https://github.com/pytorch/xla/commit/d6fb5391d09578c8804b1331a5e7a4f72bf981db>`_.

