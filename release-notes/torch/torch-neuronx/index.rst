.. |Trn1| replace:: :ref:`Trn1 <aws-trn1-arch>`
.. |Inf2| replace:: :ref:`Inf2 <aws-inf2-arch>`

.. _torch-neuronx-rn:

PyTorch Neuron (``torch-neuronx``) release notes
================================================

.. contents:: Table of Contents
   :local:
   :depth: 1

PyTorch Neuron for |Trn1|/|Inf2| is a software package that enables PyTorch
users to train, evaluate, and perform inference on second-generation Neuron
hardware (See: :ref:`NeuronCore-v2 <neuroncores-v2-arch>`).


Release [2.1.2.2.2.0]
---------------------
Date: 07/03/2024

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Improvements in ZeRO1 to have FP32 master weights support and BF16 all-gather
* Added custom SILU enabled via ``NEURON_CUSTOM_SILU`` environment variable
* Neuron Parallel Compile now handle non utf-8 characters in trial-run log and reports compilation time results when enabled with ``NEURON_PARALLEL_COMPILE_DUMP_RESULTS``
* Support for using DummyStore during PJRT process group initialization by setting ``TORCH_DIST_INIT_BARRIER=0`` and ``XLA_USE_DUMMY_STORE=1``

Known limitations
~~~~~~~~~~~~~~~~~
The following features are not yet supported in this version of Torch-Neuronx 2.1:
* (Training) GSPMD
* (Training/Inference) TorchDynamo (torch.compile)
* (Training) DDP/FSDP

Resolved Issues
~~~~~~~~~~~~~~~


Resolved an issue with slower loss convergence for GPT-2 pretraining using ZeRO1 tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously with Torch-NeuronX 2.1, we see slower loss convergence in the :ref:`ZeRO1 tutorial<zero1-gpt2-pretraining-tutorial>`. This issue is now resolved. Customer can now run the tutorial with the recommended flags (``NEURON_CC_FLAGS="--distribution-strategy llm-training --model-type transformer"``).

Resolved an issue with slower loss convergence for NxD LLaMA-2 70B pretraining using ZeRO1 tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously with Torch-NeuronX 2.1, we see slower loss convergence in the :ref:`LLaMA-2 70B tutorial for neuronx-distributed<llama2_tp_pp_tutorial>`. This issue is now resolved. Customer can now run the tutorial with the recommended flags (``NEURON_CC_FLAGS="--distribution-strategy llm-training --model-type transformer"``) and turning on functionalization (``XLA_DISABLE_FUNCTIONALIZATION=0``). Turning on functionalization results in slightly higher device memory usage and ~11% lower in performance due to a known issue with torch-xla 2.1 (https://github.com/pytorch/xla/issues/7174). The higher device memory usage also limits LLaMA-2 70B tutorial to run on 16 trn1.32xlarge nodes at the minimum, and running on 8 nodes would result in out-of-memory error. See the :ref:`list of environment variables<>` for more information about ``XLA_DISABLE_FUNCTIONALIZATION``.

Resolved an issue where upon a compiler error during XLA JIT execution, the framework process exits with a stack dump followed by a core dump
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously, when there's a compiler error during XLA JIT execution, the framework process exits with a stack dump following by a core dump:

.. code:: bash

    2024-06-10 04:31:49.733004: F ./torch_xla/csrc/runtime/debug_macros.h:20] Non-OK-status: status.status() status: INTERNAL: RunNeuronCCImpl: error condition error != 0: <class 'subprocess.CalledProcessError'>: Command '' died with <Signals.SIGHUP: 1>.
    *** Begin stack trace ***
            tsl::CurrentStackTrace()
            std::unique_ptr<xla::PjRtLoadedExecutable, std::default_delete<xla::PjRtLoadedExecutable> > ConsumeValue<std::unique_ptr<xla::PjRtLoadedExecutable, std::default_delete<xla::PjRtLoadedExecutable> > >(absl::lts_20230125::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable, std::default_delete<xla::PjRtLoadedExecutable> > >&&)
            torch_xla::runtime::PjRtComputationClient::Compile(std::vector<torch_xla::runtime::ComputationClient::CompileInstance, std::allocator<torch_xla::runtime::ComputationClient::CompileInstance> >)
            ...
            Py_RunMain
            Py_BytesMain
            _start
    *** End stack trace ***
    Aborted (core dumped)

This is now fixed so that the above error is more succinct:

.. code:: bash

    RuntimeError: Bad StatusOr access: INTERNAL: RunNeuronCCImpl: error condition error != 0: <class 'subprocess.CalledProcessError'>: Command '' died with <Signals.SIGHUP: 1>.

Resolved an issue where S3 caching during distributed training can lead to S3 throttling error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using S3 location as Neuron Cache path (specified via NEURON_COMPILE_CACHE_URL or --cache_dir option in NEURON_CC_FLAGS), you may get the error ``An error occurred (SlowDown) when calling the PutObject operation`` as in:

.. code:: bash

    2024-04-18 01:51:38.231524: F ./torch_xla/csrc/runtime/debug_macros.h:20] Non-OK-status: status.status() status: INVALID_ARGUMENT: RunNeuronCCImpl: error condition !(error != 400): <class 'boto3.exceptions.S3UploadFailedError'>: Failed to upload /tmp/tmp4d8d4r2d/model.hlo to bucket/llama-compile-cache/neuronxcc-2.13.68.0+6dfecc895/MODULE_9048582265414220701+5d2d81ce/model.hlo_module.pb: An error occurred (SlowDown) when calling the PutObject operation (reached max retries: 4): Please reduce your request rate.

This issue is now resolved in release 2.19.

Resolved error "ImportError: cannot import name 'packaging' from 'pkg_resources'" when using latest setuptools version 70
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As reported in https://github.com/aws-neuron/aws-neuron-sdk/issues/893, When running examples in environment where the latest setuptools version 70 is installed, you may get the following error:

.. code:: bash

    ImportError: cannot import name 'packaging' from 'pkg_resources' (/home/ubuntu/aws_neuron_venv_pytorch/lib/python3.8/site-packages/pkg_resources/__init__.py)

In release 2.19 torch-neuronx now depends on setuptools version <= 69.5.1.

Resolved compiler assertion error when training using Hugging Face ``deepmind/language-perceiver`` model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The follow assertion error when training with Hugging Face ``deepmind/language-perceiver`` model is now resolved in release 2.19 compiler:

.. code:: bash

    ERROR 176659 [NeuronAssert]: Assertion failure in usr/lib/python3.8/multiprocessing/process.py at line 108 with exception:
    Unsupported batch-norm-training op: tensor_op_name: _batch-norm-training.852 | hlo_id: 852| file_name:  | Line: 0 | Column: 0 | .

Resolved lower accuracy for BERT-base finetuning using HF Trainer API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With release 2.19 compiler, the MRPC dataset accuracy for BERT-base finetuning after 5 epochs is now 87% as expected.


Resolved the issue with increased in Neuron Parallel Compile time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Torch-NeuronX 2.1, the time to run Neuron Parallel Compile for some model configuration has decreased.

Known issues
~~~~~~~~~~~~

Please see the :ref:`Introducing PyTorch 2.1 Support<introduce-pytorch-2-1>` for a full list of known issues.

Slower loss convergence for NxD LLaMA-3 70B pretraining using ZeRO1 tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with Torch-NeuronX 2.1, we see slower loss convergence in the :ref:`LLaMA-3 70B tutorial for neuronx-distributed<llama3_tp_pp_tutorial>` when using the recommended flags (``NEURON_CC_FLAGS="--distribution-strategy llm-training --model-type transformer"``). To work-around this issue, please only use ``--model-type transformer`` flag (``NEURON_CC_FLAGS="--model-type transformer"``).

Gradient accumulation is not yet supported for Stable Diffusion due to a compiler error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with torch-neuronx 2.1, we are seeing a compiler assertion error with Stable Diffusion training when gradient accumulation is enabled. To train Stable Diffusion with gradient accumulation, please use torch-neuronx 1.13 instead of 2.1.

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

This is due to ``synchronize_rng_states`` using ``xm.mesh_reduce`` to synchronize RNG states. ``xm.mesh_reduce`` in turn uses ``xm.rendezvous()`` with payload, which as noted in 2.x migration guide, would result in extra graphs that could lead to lower performance due to change in ``xm.rendezvous()`` in torch-xla 2.x. In the case of :ref:`ZeRO1 tutorial<zero1-gpt2-pretraining-tutorial>`, using ``xm.rendezvous()`` with payload also lead to the error above. This limitation will be fixed in an upcoming release. For now, to work around the issue, please disable shuffle in DataLoader when ``NEURON_EXTRACT_GRAPHS_ONLY`` environment is set automatically by Neuron Parallel Compile:

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



Lower performance for BERT-Large
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see 8% less performance when running the BERT-Large pre-training tutorial with Torch-NeuronX 2.1 as compared to Torch-NeuronX 1.13.


Release [1.13.1.2.10.12.0]
-----------------------
Date: 07/03/2024


Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~
Improvements in ZeRO1 to have FP32 master weights support and BF16 all-gather
Added custom SILU enabled via ``NEURON_CUSTOM_SILU`` environment variable
Neuron Parallel Compile now handle non utf-8 characters in trial-run log and reports compilation time results when enabled with ``NEURON_PARALLEL_COMPILE_DUMP_RESULTS``

Resolved Issues
~~~~~~~~~~~~~~~

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory leaking in ``glibc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``glibc`` malloc memory leaks affect Neuron and may be temporarily limited by
setting ``MALLOC_ARENA_MAX`` or using ``jemalloc`` library (see https://github.com/aws-neuron/aws-neuron-sdk/issues/728).

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the
scripts that don't use DDP. We also see a throughput drop with DDP. This is a
known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its
own graph. This causes an error in the runtime, and you may see errors that
look like this:

::

    bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Known Issues and Limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Torchscript serialization error with compiled artifacts larger than 4GB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using :func:`torch_neuronx.trace`, compiled artifacts which exceed 4GB
cannot be serialized. Serializing the torchscript artifact will trigger a
segfault. This issue is resolved in torch but is not yet
released: https://github.com/pytorch/pytorch/pull/99104


Release [2.1.2.2.1.0]
---------------------

Date: 04/01/2024

Summary
~~~~~~~

This release of 2.1 includes support for Neuron Profiler, multi-instance distributed training, Nemo Megatron, and HuggingFace Trainer API.

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to previously supported features (Transformers-NeuronX, Torch-NeuronX Trace API, Torch-NeuronX training, NeuronX-Distributed training), torch-neuronx 2.1 now includes support for:

* (Inference) NeuronX-Distributed inference
* (Training/Inference) Neuron Profiler
* (Training) Multi-instance distributed training
* (Training) Nemo Megatron
* (Training) `analyze` feature in `neuron_parallel_compile`
* (Training) HuggingFace Trainer API

Additionally, auto-bucketing is a new feature for torch-neuronx and Neuronx-Distributed allowing users to define bucket models that can be serialized into a single model for multi-shape inference.

Known limitations
~~~~~~~~~~~~~~~~~

The following features are not yet supported in this version of Torch-NeuronX 2.1:

* (Training) GSPMD
* (Training) TorchDynamo (torch.compile)
* (Training) DDP/FSDP
* (Training) S3 caching during distributed training can lead to throttling issues


Resolved issues
~~~~~~~~~~~~~~~

"Attempted to access the data pointer on an invalid python storage"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When using Hugging Face Trainer API with transformers version >= 4.35 and < 4.37.3, user would see the error ``"Attempted to access the data pointer on an invalid python storage"`` during model checkpoint saving. This issue is fixed in transformers version >= 4.37.3. See https://github.com/huggingface/transformers/issues/27578 for more information.

Too many graph compilations when using HF Trainer API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using Hugging Face transformers version >= 4.35 and < 4.37.3, user would see many graph compilations (see https://github.com/aws-neuron/aws-neuron-sdk/issues/813 for more information). To work around this issue, in transformers version >= 4.37.3, user can add the option ``--save_safetensors False`` to Trainer API function call and modify the installed  ``trainer.py`` as follows (don't move model to CPU before saving checkpoint):

.. code:: bash

   # Workaround https://github.com/aws-neuron/aws-neuron-sdk/issues/813
   sed -i "s/model\.to(\"cpu\")//" `python -c "import site; print(site.getsitepackages()[0])"`/trainer.py


Divergence (non-convergence) of loss for BERT/LLaMA when using release 2.16 compiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With release 2.18, the divergence (non-convergence) of BERT/LLaMA loss is resolved. No compiler flag change is required.

Known Issues
~~~~~~~~~~~~

Please see the :ref:`Introducing PyTorch 2.1 Support<introduce-pytorch-2-1>` for a full list of known issues.


GlibC error on Amazon Linux 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If using Torch-NeuronX 2.1 on Amazon Linux 2, you will see a GlibC error below. Please switch to a newer supported OS such as Ubuntu 20, Ubuntu 22, or Amazon Linux 2023.

.. code:: bash

    ImportError: /lib64/libc.so.6: version `GLIBC_2.27' not found (required by /tmp/debug/_XLAC.cpython-38-x86_64-linux-gnu.so)


``"EOFError: Ran out of input"`` or ``"_pickle.UnpicklingError: invalid load key, '!'"`` errors during Neuron Parallel Compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With torch-neuronx 2.1, HF Trainer API's use of XLA function ``.mesh_reduce`` causes ``"EOFError: Ran out of input"`` or ``"_pickle.UnpicklingError: invalid load key, '!'"`` errors during Neuron Parallel Compile. This is an issue with the trial execution of empty NEFFs and should not affect the normal execution of the training script.

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

Additionally, you can add the following code snippet (after python imports) to replace ``xm.mesh_reduce`` with a form that uses ``xm.all_gather`` instead of ``xm.rendezvous()`` with payload. This will add additional small on-device graphs (as opposed to the original ``xm.mesh_reduce`` which runs on CPU).

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

Currently, with torch-neuronx 2.1, we are seeing the following compiler assertion error with Stable Diffusion training. This will be fixed in an upcoming release. For now, if you would like to run Stable Diffusion training with Neuron SDK release 2.18, please use ``torch-neuronx==1.13.*``.

.. code:: bash

    ERROR 222163 [NeuronAssert]: Assertion failure in usr/lib/python3.8/concurrent/futures/process.py at line 239 with exception:
    too many partition dims! {{0,+,960}[10],+,10560}[10]

Compiler assertion error when training using Hugging Face ``deepmind/language-perceiver`` model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with torch-neuronx 2.1, we are seeing the following compiler assertion error when training with Hugging Face ``deepmind/language-perceiver`` model. This will be fixed in an upcoming release. For now, if you would like to train Hugging Face ``deepmind/language-perceiver`` model with Neuron SDK release 2.18, please use ``torch-neuronx==1.13.*``.

.. code:: bash

    ERROR 176659 [NeuronAssert]: Assertion failure in usr/lib/python3.8/multiprocessing/process.py at line 108 with exception:
    Unsupported batch-norm-training op: tensor_op_name: _batch-norm-training.852 | hlo_id: 852| file_name:  | Line: 0 | Column: 0 | .

Lower performance for BERT-Large
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see 8% less performance when running the BERT-Large pre-training tutorial with Torch-NeuronX 2.1 as compared to Torch-NeuronX 1.13.

Slower loss convergence for GPT-2 pretraining using ZeRO1 tutorial when using recommended compiler flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently with Torch-NeuronX 2.1, we see slower loss convergence in the :ref:`ZeRO1 tutorial<zero1-gpt2-pretraining-tutorial>` when using recommended compiler flags. To work-around this issue and restore faster convergence, please replace the ``NEURON_CC_FLAGS`` as below:

.. code:: python

   # export NEURON_CC_FLAGS="--retry_failed_compilation --distribution-strategy llm-training --model-type transformer"
   export NEURON_CC_FLAGS="--retry_failed_compilation -O1"

Slower loss convergence for NxD LLaMA 70B pretraining using ZeRO1 tutorial when using recommended compiler flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently with Torch-NeuronX 2.1, we see slower loss convergence in the :ref:`LLaMA-2 70B tutorial for neuronx-distributed<llama2_tp_pp_tutorial>` when using recommended compiler flags. To work-around this issue and restore faster convergence, please replace the ``NEURON_CC_FLAGS`` as below:

.. code:: python

   # export NEURON_CC_FLAGS="--retry_failed_compilation --distribution-strategy llm-training --model-type transformer"
   export NEURON_CC_FLAGS="--retry_failed_compilation"


Lower accuracy for BERT-base finetuning using HF Trainer API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with Torch-NeuronX 2.1, MRPC dataset accuracy for BERT-base finetuning after 5 epochs is 83% instead of 87%. A work-around is to remove the option ``--model-type=transformer`` from ``NEURON_CC_FLAGS``. This will be fixed in an upcoming release.

Increased in Neuron Parallel Compile time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, with Torch-NeuronX 2.1, the time to run Neuron Parallel Compile for some model configuration is increased. In one example, the Neuron Parallel Compile time for NeuronX Nemo-Megatron LLaMA 13B is 2x compared to when using Torch-NeuronX 1.13. This will be fixed in an upcoming release.


Release [1.13.1.1.14.0]
-----------------------

Date: 04/01/2024

Summary
~~~~~~~

Auto-bucketing is a new feature for torch-neuronx and Neuronx-Distributed allowing users to define bucket models that can be serialized into a single model for multi-shape inference.

Resolved issues
~~~~~~~~~~~~~~~

* (Inference) Fixed an issue where transformers-neuronx inference errors could crash the application and cause it to hang. Inference errors should now correctly throw a runtime exception.
* (Inference/Training) Fixed an issue where :func:`torch.argmin` produced incorrect results.
* (Training) ``neuron_parallel_compile`` tool now use ``traceback.print_exc`` instead of ``format`` to support Python 3.10.
* (Training) Fixed an issue in ZeRO1 when sharded params are initialized with torch.double.

Known issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory leaking in ``glibc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``glibc`` malloc memory leaks affect Neuron and may be temporarily limited by
setting ``MALLOC_ARENA_MAX`` or using ``jemalloc`` library (see https://github.com/aws-neuron/aws-neuron-sdk/issues/728).

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the
scripts that don't use DDP. We also see a throughput drop with DDP. This is a
known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its
own graph. This causes an error in the runtime, and you may see errors that
look like this:

::

    bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Known issues and limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Torchscript serialization error with compiled artifacts larger than 4GB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using :func:`torch_neuronx.trace`, compiled artifacts that exceed 4GB
cannot be serialized. Serializing the TorchScript artifact triggers a
segmentation fault. This issue is resolved in PyTorch but is not yet
released: https://github.com/pytorch/pytorch/pull/99104


Release [2.1.1.2.0.0b0] (Beta)
------------------------------

Date: 12/21/2023

Summary
~~~~~~~

Introducing the beta release of Torch-NeuronX with PyTorch 2.1 support.

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

This version of Torch-NeuronX 2.1 supports:

* (Inference) Transformers-NeuronX
* (Inference) Torch-NeuronX Trace API
* (Training) NeuronX-Distributed training
* (Training) Torch-NeuronX training
* (Training) New snapshotting capability enabled via the XLA_FLAGS environment variable (see :ref:`debug guide <pytorch-neuronx-debug>`)

Known limitations
~~~~~~~~~~~~~~~~~

The following features are not yet supported in this version of Torch-NeuronX 2.1:

* (Training/Inference) Neuron Profiler
* (Inference) NeuronX-Distributed inference
* (Training) Nemo Megatron
* (Training) GSPMD
* (Training) TorchDynamo (torch.compile)
* (Training) `analyze` feature in `neuron_parallel_compile`
* (Training) HuggingFace Trainer API (see `Known Issues` below)

Additional limitations are noted in the `Known Issues` section below.

Known Issues
~~~~~~~~~~~~

Please see the :ref:`Introducing PyTorch 2.1 Support (Beta)<introduce-pytorch-2-1>` for a full list of known issues.

Lower performance for BERT-Large
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see 8% less performance when running the BERT-Large pre-training tutorial with Torch-NeuronX 2.1 as compared to Torch-NeuronX 1.13.

Divergence (non-convergence) of loss for BERT/LLaMA when using release 2.16 compiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when using release 2.16 compiler version 2.12.54.0+f631c2365, you may see divergence (non-convergence) of loss curve. To workaround this issue, please use release 2.15 compiler version 2.11.0.35+4f5279863.

Error "Attempted to access the data pointer on an invalid python storage" when using HF Trainer API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if using HuggingFace Transformers Trainer API to train (i.e. :ref:`HuggingFace Trainer API fine-tuning tutorial<torch-hf-bert-finetune>`), you may see the error "Attempted to access the data pointer on an invalid python storage". This is a known issue https://github.com/huggingface/transformers/issues/27578 and will be fixed in a future release.


Release [1.13.1.1.13.0]
-----------------------
Date: 12/21/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added :ref:`Weight Replacement API For Inference<_torch_neuronx_replace_weights_api>`)

Resolved issues
~~~~~~~~~~~~~~~

- Add bucketting logic to control the size of tensors for all-gather and reduce-scatter
- Fixed ZeRO-1 bug for inferring local ranks in 2-D configuration (https://github.com/pytorch/xla/pull/5936)

Known issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory leaking in ``glibc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``glibc`` malloc memory leaks affect Neuron and may be temporarily limited by
setting ``MALLOC_ARENA_MAX`` or using ``jemalloc`` library (see https://github.com/aws-neuron/aws-neuron-sdk/issues/728).

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the
scripts that don't use DDP. We also see a throughput drop with DDP. This is a
known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its
own graph. This causes an error in the runtime, and you may see errors that
look like this:

::

    bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Known issues and limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmin` produces incorrect results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.argmin` produces incorrect results.


Torchscript serialization error with compiled artifacts larger than 4GB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using :func:`torch_neuronx.trace`, compiled artifacts that exceed 4GB
cannot be serialized. Serializing the TorchScript artifact triggers a
segmentation fault. This issue is resolved in PyTorch but is not yet
released: https://github.com/pytorch/pytorch/pull/99104


Release [2.0.0.2.0.0b0] (Beta)
------------------------------

Date: 10/26/2023

Summary
~~~~~~~

Introducing the beta release of Torch-NeuronX with PyTorch 2.0 and PJRT support.

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Updating from XRT to PJRT runtime. For more info see: <link to intro pjrt doc>
- (Inference) Added the ability to partition unsupported ops to CPU during traced inference (See ``torch_neuronx.trace`` API guide)

Known issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Snapshotting is not supported
- NEURON_FRAMEWORK_DEBUG=1 is not supported
- Analyze in neuron_parallel_compile is not supported
- Neuron Profiler is not supported
- VGG11 with input sizes 300x300 may show accuracy issues
- Possible issues with NeMo Megatron checkpointing
- S3 caching with neuron_parallel_compile may show compilation errors
- Compiling without neuron_parallel_compile on multiple nodes may show compilation errors
- GPT2 inference may show errors with torch_neuronx.trace

Release [1.13.1.1.12.0]
-----------------------
Date: 10/26/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- (Training) Added coalescing of all-gather and reduce-scatter inside ZeRO1, which should help in improving performance at high cluster sizes.
- (Inference) Added the ability to partition unsupported ops to CPU during traced inference. (See ``torch_neuronx.trace`` API guide)
- (Inference) Previously undocumented arguments trace API args ``state`` and ``options`` are now unsupported (have no effect) and will result in a deprecation warning if used.

Resolved issues
~~~~~~~~~~~~~~~

- Fixed an issue where torch.topk would fail on specific dimensions
- (Inference) Fixed an issue where NaNs could be produced when using torch_neuronx.dynamic_batch
- (Inference) Updated torch_neuronx.dynamic_batch to better support Modules (traced, scripted, and normal modules) with multiple Neuron subgraphs
- (Inference) Isolate frontend calls to the Neuron compiler to working directories, so concurrent compilations do not conflict by being run from the same directory.

Known issues and limitations (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory leaking in ``glibc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``glibc`` malloc memory leaks affect Neuron and may be temporarily limited by
setting ``MALLOC_ARENA_MAX``.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the
scripts that don't use DDP. We also see a throughput drop with DDP. This is a
known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its
own graph. This causes an error in the runtime, and you may see errors that
look like this:

::

    bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Known issues and limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmin` produces incorrect results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.argmin` produces incorrect results.


Torchscript serialization error with compiled artifacts larger than 4GB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using :func:`torch_neuronx.trace`, compiled artifacts that exceed 4GB
cannot be serialized. Serializing the TorchScript artifact triggers a
segmentation fault. This issue is resolved in PyTorch but is not yet
released: https://github.com/pytorch/pytorch/pull/99104


Release [1.13.1.1.11.0]
----------------------
Date: 9/15/2023

Summary
~~~~~~~

Resolved issues
~~~~~~~~~~~~~~~

- Fixed an issue in :func:`torch_neuronx.analyze` which could cause failures with scalar inputs.
- Improved performance of :func:`torch_neuronx.analyze`.


Release [1.13.1.1.10.1]
----------------------
Date: 9/01/2023

Summary
~~~~~~~

Minor bug fixes and enhancements.


Release [1.13.1.1.10.0]
----------------------
Date: 8/28/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Removed support for Python 3.7
- (Training) Added a neuron_parallel_compile command to clear file locks left behind when a neuron_parallel_compile execution was interrupted (neuron_parallel_compile --command clear-locks)
- (Training) Seedable dropout now enabled by default

Resolved issues
~~~~~~~~~~~~~~~

- (Training) Convolution is now supported
- Fixed segmentation fault when using torch-neuronx to compile models on U22
- Fixed XLA tensor stride information in torch-xla package, which blocked lowering of log_softmax and similar functions and showed errors like:
::

      File "/home/ubuntu/waldronn/asr/test_env/lib/python3.7/site-packages/torch/nn/functional.py", line 1930, in log_softmax
            ret = input.log_softmax(dim)
        RuntimeError: dimensionality of sizes (3) must match dimensionality of strides (1)


Known issues and limitations (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory leaking in ``glibc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``glibc`` malloc memory leaks affect Neuron and may be temporarily limited by
setting ``MALLOC_ARENA_MAX``.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the
scripts that don't use DDP. We also see a throughput drop with DDP. This is a
known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its
own graph. This causes an error in the runtime, and you may see errors that
look like this:

::

    bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Known issues and limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmin` produces incorrect results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.argmin` produces incorrect results.

No automatic partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when Neuron encounters an operation that it does not support during
:func:`torch_neuronx.trace`, it may exit with the following compiler error: "Import of the HLO graph into the Neuron Compiler has failed.
This may be caused by unsupported operators or an internal compiler error."
The intended behavior
when tracing is to automatically partition the model into separate subgraphs
that run on NeuronCores and subgraphs that run on CPU. This will be supported in a future release. See
:ref:`pytorch-neuron-supported-operators` for a list of supported operators.

Torchscript serialization error with compiled artifacts larger than 4GB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using :func:`torch_neuronx.trace`, compiled artifacts that exceed 4GB
cannot be serialized. Serializing the TorchScript artifact triggers a
segmentation fault. This issue is resolved in PyTorch but is not yet
released: https://github.com/pytorch/pytorch/pull/99104


Release [1.13.1.1.9.0]
----------------------
Date: 7/19/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- Uses jemalloc as the primary malloc lib to avoid memory leak at checkpointing
- Added support for ZeRO-1 along with :ref:`tutorial <zero1-gpt2-pretraining-tutorial>`

Inference support:

- Add async load and lazy model load options to accelerate model loading
- Optimize DataParallel API to load onto multiple cores simultaneously when device IDs specified in device_ids are consecutive

Resolved issues (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Remove extra graph creation in torch_neuronx.optim.adamw when the beta/lr parameters values become 0 or 1.
- Stability improvements and faster failure on hitting a fault in XRT server used by XLA.

Known issues and limitations (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory leaking in ``glibc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``glibc`` malloc memory leaks affect Neuron and may be temporarily limited by
setting ``MALLOC_ARENA_MAX``.

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convolution is not supported during training.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the
scripts that don't use DDP. We also see a throughput drop with DDP. This is a
known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its
own graph. This causes an error in the runtime, and you may see errors that
look like this:

::

    bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Known issues and limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmin` produces incorrect results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.argmin` produces incorrect results.

No automatic partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when Neuron encounters an operation that it does not support during
:func:`torch_neuronx.trace`, it may exit with the following compiler error: "Import of the HLO graph into the Neuron Compiler has failed.
This may be caused by unsupported operators or an internal compiler error."
The intended behavior
when tracing is to automatically partition the model into separate subgraphs
that run on NeuronCores and subgraphs that run on CPU. This will be supported in a future release. See
:ref:`pytorch-neuron-supported-operators` for a list of supported operators.

Torchscript serialization error with compiled artifacts larger than 4GB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using :func:`torch_neuronx.trace`, compiled artifacts that exceed 4GB
cannot be serialized. Serializing the TorchScript artifact triggers a
segmentation fault. This issue is resolved in PyTorch but is not yet
released: https://github.com/pytorch/pytorch/pull/99104


Release [1.13.1.1.8.0]
----------------------
Date: 6/14/2023

Summary
~~~~~~~

- Added s3 caching to NeuronCache.
- Added extract/compile/analyze phases to neuron_parallel_compile.

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- Added S3 caching support to NeuronCache. Removed NeuronCache options --cache_size/cache_ttl (please delete cache directories as needed).
- Added separate extract and compile phases Neuron Parallel Compile.
- Added model analyze API to Neuron Parallel Compile.

Known issues and limitations (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory leaking in ``glibc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``glibc`` malloc memory leaks affect Neuron and may be temporarily limited by
setting ``MALLOC_ARENA_MAX``.

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convolution is not supported during training.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the
scripts that don't use DDP. We also see a throughput drop with DDP. This is a
known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its
own graph. This causes an error in the runtime, and you may see errors that
look like this:

::

    bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Known issues and limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmin` produces incorrect results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.argmin` produces incorrect results.

No automatic partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when Neuron encounters an operation that it does not support during
:func:`torch_neuronx.trace`, this will cause an error. The intended behavior
when tracing is to automatically partition the model into separate subgraphs
that run on NeuronCores and subgraphs that run on CPU. See
:ref:`pytorch-neuron-supported-operators` for a list of supported operators.

Torchscript serialization error with compiled artifacts larger than 4GB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using :func:`torch_neuronx.trace`, compiled artifacts that exceed 4GB
cannot be serialized. Serializing the TorchScript artifact triggers a
segmentation fault. This issue is resolved in PyTorch but is not yet
released: https://github.com/pytorch/pytorch/pull/99104


Release [1.13.1.1.7.0]
----------------------
Date: 05/01/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- Added an improved Neuron-optimized AdamW optimizer implementation.
- Added an improved Neuron-optimized :class:`torch.nn.Dropout` implementation.
- Added an assertion when the :class:`torch.nn.Dropout` argument
  ``inplace=True`` during training. This is currently not supported on Neuron.
- Added XLA lowering for ``aten::count_nonzero``

Inference support:

- Added profiling support for models compiled with :func:`torch_neuronx.trace`
- Added torch_neuronx.DataParallel for models compiled with :func:`torch_neuronx.trace`


Resolved issues (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unexpected behavior with :class:`torch.autocast`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed an issue where :class:`torch.autocast` did not correctly autocast
when using ``torch.bfloat16``

Resolved slower BERT bf16 Phase 1 Single Node Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As of the Neuron 2.9.0 release, :ref:`BERT phase 1 pretraining <hf-bert-pretraining-tutorial>`
performance has regressed by approximately 8-9% when executed on a *single
node* only (i.e. just one ``trn1.32xlarge`` instance). This is resolved in 2.10 release.

Resolved lower throughput for BERT-large training on AL2 instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting in release 2.7, we see a performance drop of roughly 5-10% for BERT model training on AL2
instances. This is resolved in release 2.10.

Resolved issues (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Error when using the original model after ``torch_neuronx.trace``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed an issue where model parameters would be moved to the Neuron ``'xla'``
device during :func:`torch_neuronx.trace` and would no longer be available to
execute on the original device. This made it more difficult to compare Neuron
models against CPU since previously this would require manually moving
parameters back to CPU.

Error when using the ``xm.xla_device()`` object followed by using ``torch_neuronx.trace``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed an issue where XLA device execution and :func:`torch_neuronx.trace` could
not be performed in the same python process.

Error when executing ``torch_neuronx.trace`` with ``torch.bfloat16`` input/output tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed an issue where :func:`torch_neuronx.trace` could not compile models which
consumed or produced ``torch.bfloat16`` values.

Known issues and limitations (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory leaking in ``glibc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``glibc`` malloc memory leaks affect Neuron and may be temporarily limited by
setting ``MALLOC_ARENA_MAX``.

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convolution is not supported during training.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the
scripts that don't use DDP. We also see a throughput drop with DDP. This is a
known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its
own graph. This causes an error in the runtime, and you may see errors that
look like this:

::

    bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.


Known issues and limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmin` produces incorrect results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.argmin` produces incorrect results.

No automatic partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, when Neuron encounters an operation that it does not support during
:func:`torch_neuronx.trace`, this will cause an error. The intended behavior
when tracing is to automatically partition the model into separate subgraphs
that run on NeuronCores and subgraphs that run on CPU. See
:ref:`pytorch-neuron-supported-operators` for a list of supported operators.

Torchscript serialization error with compiled artifacts larger than 4GB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using :func:`torch_neuronx.trace`, compiled artifacts that exceed 4GB
cannot be serialized. Serializing the TorchScript artifact triggers a
segmentation fault. This issue is resolved in PyTorch but is not yet
released: https://github.com/pytorch/pytorch/pull/99104


Release [1.13.0.1.6.1]
----------------------
Date: 04/19/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- No changes

Inference support:

- Enable deserialized TorchScript modules to be compiled with :func:`torch_neuronx.trace`



Release [1.13.0.1.6.1]
----------------------
Date: 04/19/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- No changes

Inference support:

- Enable deserialized TorchScript modules to be compiled with :func:`torch_neuronx.trace`


Release [1.13.0.1.6.0]
----------------------
Date: 03/28/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- Added pipeline parallelism support in AWS Samples for Megatron-LM

Inference support:

- Added model analysis API: torch_neuronx.analyze
- Added HLO opcode support for:

  - kAtan2
  - kAfterAll
  - kMap

- Added XLA lowering support for:

  - aten::glu
  - aten::scatter_reduce

- Updated torch.nn.MSELoss to promote input data types to a compatible type

Resolved issues (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~

GRPC timeout errors when running Megatron-LM GPT 6.7B tutorial on multiple instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running AWS Samples for Megatron-LM GPT 6.7B tutorial over multiple instances, you may encounter GRPC timeout errors like below:

::

    E0302 01:10:20.511231294  138645 chttp2_transport.cc:1098]   Received a GOAWAY with error code ENHANCE_YOUR_CALM and debug data equal to "too_many_pings"
    2023-03-02 01:10:20.511500: W tensorflow/core/distributed_runtime/rpc/grpc_remote_master.cc:157] RPC failed with status = "UNAVAILABLE: Too many pings" and grpc_error_string = "{"created":"@1677719420.511317309","description":"Error received from peer ipv4:10.1.35.105:54729","file":"external/com_github_grpc_grpc/src/core/lib/surface/call.cc","file_line":1056,"grpc_message":"Too many pings","grpc_status":14}", maybe retrying the RPC


or:

::

    2023-03-08 21:18:27.040863: F tensorflow/compiler/xla/xla_client/xrt_computation_client.cc:476] Non-OK-status: session->session()->Run(session_work->feed_inputs, session_work->outputs_handles, &outputs) status: UNKNOWN: Stream removed


This is due to excessive DNS lookups during execution, and is fixed in this release.


NaNs seen with transformers version >= 4.21.0 when running HF GPT fine-tuning or pretraining with XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using Hugging Face transformers version >= 4.21.0 can produce NaN outputs for GPT models when using full BF16 (XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1) plus stochastic rounding. This issue occurs due to large negative constants used to implement attention masking (https://github.com/huggingface/transformers/pull/17306). To workaround this issue, please use transformers version <= 4.20.0.


Resolved issues (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmax` now supports single argument call variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously only the 3 argument variant of :func:`torch.argmax` was supported. Now the single argument call variant is supported.

Known issues and limitations (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Slower BERT bf16 Phase 1 Single Node Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Neuron 2.9.0 release, :ref:`BERT phase 1 pretraining <hf-bert-pretraining-tutorial>`
performance has regressed by approximately 8-9% when executed on a *single
node* only (i.e. just one ``trn1.32xlarge`` instance).

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the scripts that don't use DDP. We also see a throughput drop
with DDP. This is a known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its own graph. This causes an error in the runtime, and
you may see errors that look like this: ``bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Lower throughput for BERT-large training on AL2 instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We see a performance drop of roughly 5-10% for BERT model training on AL2 instances. This is because of the increase in time required for tracing the model.

Known issues and limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmin` produces incorrect results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.argmin` now supports both the single
argument call variant and the 3 argument variant.
However, :func:`torch.argmin` currently produces
incorrect results.

Error when using the ``xm.xla_device()`` object followed by using ``torch_neuronx.trace``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Executing a model using the ``xm.xla_device()`` object followed by using ``torch_neuronx.trace`` in the same process can produce errors in specific situations due to torch-xla caching behavior. It is recommended that only one type of execution is used per process.

Error when executing ``torch_neuronx.trace`` with ``torch.bfloat16`` input/output tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Executing ``torch_neuronx.trace`` with ``torch.bfloat16`` input/output tensors can cause an error. It is currently recommended to use an alternative torch data type in combination with compiler casting flags instead.


No automatic partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, there's no automatic partitioning of a model into subgraphs that run on NeuronCores and subgraphs that run on CPU
Operations in the model that are not supported by Neuron would result in compilation error. Please see :ref:`pytorch-neuron-supported-operators` for a list of supported operators.


Release [1.13.0.1.5.0]
----------------------
Date: 02/24/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- Added SPMD flag for XLA backend to generate global collective-compute replica groups

Inference support:

- Expanded inference support to inf2
- Added Dynamic Batching

Resolved issues
~~~~~~~~~~~~~~~

Known issues and limitations (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the scripts that don't use DDP. We also see a throughput drop
with DDP. This is a known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its own graph. This causes an error in the runtime, and
you may see errors that look like this: ``bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Lower throughput for BERT-large training on AL2 instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We see a performance drop of roughly 5-10% for BERT model training on AL2 instances. This is because of the increase in time required for tracing the model.

Known issues and limitations (Inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.argmax` and :func:`torch.argmin` do not support the single argument call variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.argmax` and :func:`torch.argmin` do not support the single
argument call variant. Only the 3 argument variant of these functions is
supported. The ``dim`` argument *must be* specified or this function will
fail at the call-site. Secondly, :func:`torch.argmin` may produce
incorrect results.

No automatic partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, there's no automatic partitioning of a model into subgraphs that run on NeuronCores and subgraphs that run on CPU
Operations in the model that are not supported by Neuron would result in compilation error. Please see :ref:`pytorch-neuron-supported-operators` for a list of supported operators.

Release [1.13.0.1.4.0]
----------------------
Date: 02/08/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training support:

- Added support for PyTorch 1.13
- Added support for Python version 3.9
- Added support for torch.nn.parallel.DistributedDataParallel (DDP) along with a :ref:`tutorial <neuronx-ddp-tutorial>`
- Added optimized lowering for Softmax activation
- Added support for LAMB optimizer in BF16 mode

Added initial support for inference on Trn1, including the following features:

- Trace API (torch_neuronx.trace)
- Core placement API (Beta)
- Python 3.7, 3.8 and 3.9 support
- Support for tracing models larger than 2 GB

The following inference features are not included in this release:

- Automatic partitioning of a model into subgraphs that run on NeuronCores and subgraphs that run on CPU
- cxx11 ABI wheels

Resolved issues
~~~~~~~~~~~~~~~

Known issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

DDP shows slow convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently we see that the models converge slowly with DDP when compared to the scripts that don't use DDP. We also see a throughput drop
with DDP. This is a known issue with torch-xla: https://pytorch.org/xla/release/1.13/index.html#mnist-with-real-data

Runtime crash when we use too many workers per node with DDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, if we use 32 workers with DDP, we see that each worker generates its own graph. This causes an error in the runtime, and
you may see errors that look like this: ``bootstrap.cc:86 CCOM WARN Call to accept failed : Too many open files``.

Hence, it is recommended to use fewer workers per node with DDP.

Lower throughput for BERT-large training on AL2 instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We see a performance drop of roughly 5-10% for BERT model training on AL2 instances. This is because of the increase in time required for tracing the model.


Release [1.12.0.1.4.0]
----------------------
Date: 12/12/2022

Summary
~~~~~~~

What’s new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added support for PyTorch 1.12.
- Setting XLA_DOWNCAST_BF16=1 now also enables stochastic rounding by default (as done with XLA_USE_BF16=1).
- Added support for :ref:`capturing snapshots <torch-neuronx-snapshotting>` of inputs, outputs and graph HLO for debug.
- Fixed issue with parallel compile error when both train and evaluation are enabled in HuggingFace fine-tuning tutorial.
- Added support for LAMB optimizer in FP32 mode.

Resolved issues
~~~~~~~~~~~~~~~

NaNs seen with transformers version >= 4.21.0 when running HF BERT fine-tuning or pretraining with XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running HuggingFace BERT (any size) fine-tuning tutorial or pretraining tutorial with transformers version >= 4.21.0 and using XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1, you will see NaNs in the loss immediately at the first step. More details on the issue can be found at `pytorch/xla#4152 <https://github.com/pytorch/xla/issues/4152>`_. The workaround is to use 4.20.0 or earlier (the tutorials currently recommend version 4.15.0) or add the line ``transformers.modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16`` to your Python training script (as now done in latest tutorials). `A permanent fix <https://github.com/huggingface/transformers/pull/20562>`_ will become part of an upcoming HuggingFace transformers release.

Known issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

Number of data parallel training workers on one Trn1 instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of workers used in single-instance data parallel
training can be one of the following values: 1 or 2 for trn1.2xlarge and 1, 2, 8 or 32 for trn1.32xlarge.

Release [1.11.0.1.2.0]
----------------------
Date: 10/27/2022

Summary
~~~~~~~

What’s new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added support for argmax.
- Clarified error messages for runtime errors ``NRT_UNINITIALIZED`` and ``NRT_CLOSED``.
- When multi-worker training is launched using ``torchrun`` on one instance, framework now handles runtime state cleanup at end of training.

Resolved issues
~~~~~~~~~~~~~~~

Drop-out rate ignored in dropout operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A known issue in the compiler's implementation of dropout caused drop-rate to be ignored in the last release. It is fixed in the current release.

Runtime error "invalid offset in Coalesced\_memloc\_..." followed by "Failed to process dma block: 1703"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Previously, when running MRPC fine-tuning tutorial with ``bert-base-*`` model, you would encounter runtime error "invalid offset in Coalesced\_memloc\_..." followed by "Failed to process dma block: 1703". This is fixed in the current release.

Compilation error: "TongaSBTensor[0x7fb2a46e0830]:TongaSB partitions[0] uint8 %138392[128, 512]"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Previously, when compiling MRPC fine-tuning tutorial with ``bert-large-*`` and FP32 (no XLA_USE_BF16=1) for two workers or more, you would encounter compiler error that looks like ``Error message:  TongaSBTensor[0x7fb2a46e0830]:TongaSB partitions[0] uint8 %138392[128, 512]`` followed by ``Error class:    KeyError``. Single worker fine-tuning is not affected. This is fixed in the current release.

Known issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

Number of data parallel training workers on one Trn1 instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of workers used in single-instance data parallel
training can be one of the following values: 1 or 2 for trn1.2xlarge and 1, 2, 8 or 32 for trn1.32xlarge.


Release [1.11.0.1.1.1]
----------------------
Date: 10/10/2022


Summary
~~~~~~~

This is the initial release of PyTorch Neuron that supports Trainium for
users to train their models on the new EC2 Trn1 instances.


What’s new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Announcing the first PyTorch Neuron release for training.

- XLA device support for Trainium
- PyTorch 1.11 with XLA backend support in ``torch.distributed``
- torch-xla distributed support
- Single-instance and multi-instance distributed training using ``torchrun``
- Support for ParallelCluster and SLURM with node-level scheduling granularity
- Persistent cache for compiled graph
- :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`
  utility to help speed up compilation
- Optimizer support: SGD, AdamW
- Loss functions supported: NLLLoss
- Python versions supported: 3.7, 3.8
- Multi-instance training support with EFA
- Support PyTorch’s BF16 automatic mixed precision

Known issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolution is not supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this release, convolution is not supported.

Number of data parallel training workers on one Trn1 instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of workers used in single-instance data parallel
training can be one of the following values: 1 or 2 for trn1.2xlarge and 1, 2, 8 or 32 for trn1.32xlarge.

Drop-out rate ignored in dropout operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A known issue in the compiler's implementation of dropout caused drop-rate to be ignored. Will be fixed in a follow-on release.

Runtime error "invalid offset in Coalesced\_memloc\_..." followed by "Failed to process dma block: 1703"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently, when running MRPC fine-tuning tutorial with ``bert-base-*`` model, you will encounter runtime error "invalid offset in Coalesced\_memloc\_..." followed by "Failed to process dma block: 1703".
This issue will be fixed in an upcoming release.

Compilation error: "TongaSBTensor[0x7fb2a46e0830]:TongaSB partitions[0] uint8 %138392[128, 512]"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When compiling MRPC fine-tuning tutorial with ``bert-large-*`` and FP32 (no XLA_USE_BF16=1) for two workers or more, you will encounter compiler error that looks like ``Error message:  TongaSBTensor[0x7fb2a46e0830]:TongaSB partitions[0] uint8 %138392[128, 512]`` followed by ``Error class:    KeyError``. Single worker fine-tuning is not affected. This issue will be fixed in an upcoming release.
