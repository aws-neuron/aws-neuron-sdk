.. _neuron-2-26-0-pytorch:

.. meta::
   :description: The official release notes for the AWS Neuron SDK PyTorch support component, version 2.26.0. Release date: 9/18/2025.

AWS Neuron SDK 2.26.0: PyTorch support release notes
====================================================

**Date of release**:  September 18, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.26.0 release notes home <neuron-2-26-0-whatsnew>`

Released versions
-----------------

- ``2.8.0.2.10.*``
- ``2.7.0.2.10.*``
- ``2.6.0.2.10.*``

Improvements
------------

- Added support for PyTorch 2.8 (see :ref:`Introducing PyTorch 2.8 Support<introduce-pytorch-2-8>`)

Known issues
------------

.. note::
   * See :ref:`Introducing PyTorch 2.8 Support<introduce-pytorch-2-8>` for a full list of known issues with v2.8.
   * See :ref:`Introducing PyTorch 2.7 Support<introduce-pytorch-2-7>` for a full list of known issues with v2.7.
   * See :ref:`Introducing PyTorch 2.6 Support<introduce-pytorch-2-6>` for a full list of known issues with v2.6.

* [PyTorch v2.8] Using the publicly released version of torch-xla 2.8.0 from public PyPI repositories would result in lower performance for models like BERT and LLaMA (https://github.com/pytorch/xla/issues/9605). To fix this, switch to using the updated torch-xla version 2.8.1 from public PyPI repositories.

* [PyTorch v2.7] Using the latest torch-xla v2.7 may result in an increase in host memory usage compared to torch-xla v2.6. In one example, LLama2 pretraining with ZeRO1 and sequence length 16k could see an increase of 1.6% in host memory usage.

* Currently, when switching Ubuntu OS kernel version from 5.15 to 6.8, you may see performance differences due to the new kernel scheduler (CFS vs EEVDF). For example, BERT pretraining performance could be lower by up to 10%. You may try using an older OS kernel (i.e. Amazon Linux 2023) or experiment with the kernel real-time scheduler by running ``sudo chrt --fifo 99`` before your command (i.e. ``sudo chrt --fifo 99 <script>``) to improve the performance. Note that adjusting the real-time scheduler can also result in lower performance. See https://www.kernel.org/doc/html/latest/scheduler/sched-eevdf.html for more information.

* Currently, when using the tensor split operation on a 2D array in the second dimension, the resulting tensors do not contain the expected data (https://github.com/pytorch/xla/issues/8640). The workaround is to set ``XLA_DISABLE_FUNCTIONALIZATION=0``. Another workaround is to use ``torch.tensor_split``.

* [PyTorch v2.6]  BERT pretraining performance is approximately 10% lower with torch-neuronx 2.6 compared to torch-neuronx 2.5. This is due to a known regression in torch-xla https://github.com/pytorch/xla/issues/9037 and may affect other models with high graph tracing overhead. This is fixed in torch-xla 2.7 and 2.8. To work around this issue in torch-xla 2.6, build the ``r2.6_aws_neuron`` branch of torch-xla as follows (see :ref:`pytorch-neuronx-install-cxx11` for C++11 ABI version):

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

* Currently, BERT pretraining performance is approximately 11% lower when switching to using ``model.to(torch.bfloat16)`` as part of migration away from the deprecated environment variable ``XLA_DOWNCAST_BF16`` due to https://github.com/pytorch/xla/issues/8545. As a workaround to recover the performance, you can set ``XLA_DOWNCAST_BF16=1``, which will still work in torch-neuronx 2.5 through 2.8 although there will be end-of-support warnings (as noted below).

* Environment variables ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` are deprecated (see the warning raised below). Switch to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to cast model to BF16. (see :ref:`migration_from_xla_downcast_bf16`).

.. code:: bash

   Warning: ``XLA_DOWNCAST_BF16`` will be deprecated after the 2.5 release, please downcast your model directly

* [PyTorch v2.8+] ``DeprecationWarning: Use torch_xla.device instead``. This is a warning that ``torch_xla.core.xla_model.xla_device()`` is deprecated. Switch to using ``torch_xla.device()`` instead.

* [PyTorch v2.8+] ``DeprecationWarning: Use torch_xla.sync instead``. This is a warning that ``torch_xla.core.xla_model.mark_step()`` is deprecated. Switch to using ``torch_xla.sync()`` instead.

* [PyTorch v2.7+] ``AttributeError: module 'torch_xla.core.xla_model' ... does not have the attribute 'xrt_world_size'``. This is an error that notes that ``torch_xla.core.xla_model.xrt_world_size()`` is removed in torch-xla version 2.7+. Switch to using ``torch_xla.runtime.world_size()`` instead.

* [PyTorch v2.7+] ``AttributeError: module 'torch_xla.core.xla_model' ... does not have the attribute 'get_ordinal'``. This is an error that notes that ``torch_xla.core.xla_model.get_ordinal()`` is removed in torch-xla version 2.7+. Switch to using ``torch_xla.runtime.global_ordinal()`` instead.

* [PyTorch v2.5+] ``AttributeError: module 'torch_xla.runtime' has no attribute 'using_pjrt'``. In Torch-XLA 2.5+, ``torch_xla.runtime.using_pjrt`` is removed because PJRT is the sole Torch-XLA runtime. See this `PyTorch commit PR on GitHub <https://github.com/pytorch/xla/commit/d6fb5391d09578c8804b1331a5e7a4f72bf981db>`_.

Previous release notes
----------------------

* :ref:`neuron-2-25-0-pytorch`
* :ref:`pytorch-neuron-rn`
