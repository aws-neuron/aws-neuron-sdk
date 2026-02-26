.. meta::
    :description: Complete release notes for the Neuron PyTorch framework component across all AWS Neuron SDK versions.
    :keywords: pytorch, torch-neuronx, torch-neuron, transformers-neuronx, release notes, aws neuron sdk
    :date-modified: 02/26/2026

.. _pytorch_rn:

Component Release Notes for Neuron PyTorch Framework
=====================================================

The release notes for the Neuron PyTorch framework component. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _pytorch-2-28-0-rn:   

PyTorch Framework [2.28.0] (Neuron 2.28.0 Release)
--------------------------------------------------------

**Date of Release**: 02/26/2026

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* No new improvements in this release.

Breaking Changes
^^^^^^^^^^^^^^^^

* **PyTorch/XLA replaced by TorchNeuron in PyTorch 2.10**: Starting with PyTorch 2.10 support (planned for a future Neuron release), AWS Neuron will use native PyTorch support via TorchNeuron instead of PyTorch/XLA. PyTorch 2.9 is the last version using PyTorch/XLA. Users will need to update their scripts when upgrading to PyTorch 2.10 or later. See :ref:`native-pytorch-trainium` for complete details.

Bug Fixes
^^^^^^^^^

* No new bug fixes in this release.

Known Issues
^^^^^^^^^^^^

* **Segmentation faults with certain vision models**: Vision models including ``yolos``, ``wav2vec2``, and ``convbert`` crash with segmentation faults during model tracing.

  **How to check if affected**: If your model tracing fails with a segmentation fault, you are likely affected by this issue.

  **Workaround**: Downgrade to torch-neuronx 2.8, which does not exhibit this issue.

  See `GitHub issue #1265 <https://github.com/aws-neuron/aws-neuron-sdk/issues/1265>`_ for updates.

* **Performance degradation with public PyPI torch-xla 2.8.0**: Using the publicly released version of torch-xla 2.8.0 from public PyPI repositories results in 10-15% performance degradation for BERT and LLaMA models (`pytorch/xla#9605 <https://github.com/pytorch/xla/issues/9605>`_).

  **Workaround**: Upgrade to torch-xla version 2.8.1 from public PyPI repositories, which resolve this performance issue.

  See :doc:`/setup/torch-neuronx` for detailed installation instructions.

* **PyTorch NeuronX 2.7 does not support Python 3.12**: torch-neuronx 2.7 supports Python 3.10 and 3.11 only. Python 3.12 is not supported in torch-neuronx 2.7.

  **Impact**: Attempting to install or run torch-neuronx 2.7 with Python 3.12 will fail with dependency errors.

  **Workaround**: Use Python 3.10 or 3.11 with torch-neuronx 2.7, or upgrade to torch-neuronx 2.9 which supports Python 3.12.

  See :ref:`setup-guide-index` for complete system requirements and compatibility information.


.. _pytorch-2-27-0-rn:

PyTorch Framework [2.27.0] (Neuron 2.27.0 Release)
--------------------------------------------------------

**Date of Release**: 12/19/2025

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Added support for PyTorch 2.9
* Improved model tracing performance for large models by up to 90% through trace API optimizations that avoid copying weights and state buffers to the device and guarantee state restoration after tracing.
* Fixed GitHub issue #1240 impacting torch-neuronx 2.7 to 2.9
* Fixed GitHub issue #834 impacting torch-neuronx 2.7 to 2.9
* Fixed issue in PyTorch 2.8 where PJRT_Client_Destroy was not being called, which prevented NRT:nrt_close from being invoked.

Breaking Changes
^^^^^^^^^^^^^^^^

* PyTorch 2.6 has reached end-of-support since release 2.27.
* Transitioning to PyTorch Native Support: In the next Neuron release that will support PyTorch 2.10, AWS Neuron will transition from PyTorch/XLA to native PyTorch support via TorchNeuron. PyTorch 2.9 will be the last version based on PyTorch/XLA.

Bug Fixes
^^^^^^^^^

* Fixed resource leaks and "nrtucode: internal error: 832 object(s) leaked, improper teardown" errors by ensuring proper cleanup of Neuron Runtime resources on program exit.

Known Issues
^^^^^^^^^^^^

* Using the publicly released version of torch-xla 2.8.0 from public PyPI repositories would result in lower performance for models like BERT and LLaMA.
* Using the latest torch-xla v2.7 may result in an increase in host memory usage compared to torch-xla v2.6.
* PyTorch NeuronX 2.7 supports Python 3.10, and 3.11 only. Python 3.12 is not supported.

----

.. _pytorch-2-26-1-rn:

PyTorch Framework [2.26.1] (Neuron 2.26.1 Release)
--------------------------------------------------------

Date of Release: 10/29/2025

torch-neuronx
~~~~~~~~~~~~~

Bug Fixes
^^^^^^^^^

* Fixed an issue with out-of-memory errors by enabling the use of the :doc:`Neuron Runtime API </neuron-runtime/api/index>` to apply direct memory allocation.


----

.. _pytorch-2-26-0-rn:

PyTorch Framework [2.26.0] (Neuron 2.26.0 Release)
--------------------------------------------------------

Date of Release: 09/18/2025

Released Versions: ``2.8.0.2.10.*``, ``2.7.0.2.10.*``, ``2.6.0.2.10.*``

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Added support for PyTorch 2.8 (see :ref:`Introducing PyTorch 2.8 Support<introduce-pytorch-2-8>`)

Known Issues
^^^^^^^^^^^^

.. note::
   * See :ref:`Introducing PyTorch 2.8 Support<introduce-pytorch-2-8>` for a full list of known issues with v2.8.
   * See :ref:`Introducing PyTorch 2.7 Support<introduce-pytorch-2-7>` for a full list of known issues with v2.7.
   * See :ref:`Introducing PyTorch 2.6 Support<introduce-pytorch-2-6>` for a full list of known issues with v2.6.

* [PyTorch v2.8] Using the publicly released version of torch-xla 2.8.0 from public PyPI repositories would result in lower performance for models like BERT and LLaMA (https://github.com/pytorch/xla/issues/9605). To fix this, switch to using the updated torch-xla version 2.8.1 from public PyPI repositories.

* [PyTorch v2.7] Using the latest torch-xla v2.7 may result in an increase in host memory usage compared to torch-xla v2.6. In one example, LLama2 pretraining with ZeRO1 and sequence length 16k could see an increase of 1.6% in host memory usage.

* Currently, when switching Ubuntu OS kernel version from 5.15 to 6.8, you may see performance differences due to the new kernel scheduler (CFS vs EEVDF). For example, BERT pretraining performance could be lower by up to 10%. You may try using an older OS kernel (i.e. Amazon Linux 2023) or experiment with the kernel real-time scheduler by running ``sudo chrt --fifo 99`` before your command (i.e. ``sudo chrt --fifo 99 <script>``) to improve the performance. Note that adjusting the real-time scheduler can also result in lower performance. See https://www.kernel.org/doc/html/latest/scheduler/sched-eevdf.html for more information.

* Currently, when using the tensor split operation on a 2D array in the second dimension, the resulting tensors do not contain the expected data (https://github.com/pytorch/xla/issues/8640). The workaround is to set ``XLA_DISABLE_FUNCTIONALIZATION=0``. Another workaround is to use ``torch.tensor_split``.

* [PyTorch v2.6] BERT pretraining performance is approximately 10% lower with torch-neuronx 2.6 compared to torch-neuronx 2.5. This is due to a known regression in torch-xla https://github.com/pytorch/xla/issues/9037 and may affect other models with high graph tracing overhead. This is fixed in torch-xla 2.7 and 2.8. To work around this issue in torch-xla 2.6, build the ``r2.6_aws_neuron`` branch of torch-xla as follows (see :ref:`pytorch-neuronx-install-cxx11` for C++11 ABI version):

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


----

.. _pytorch-2-25-0-rn:

PyTorch Framework [2.25.0] (Neuron 2.25.0 Release)
--------------------------------------------------------

Date of Release: 07/31/2025

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* The Core Placement API is no longer beta/experimental and the instructions on how to use it have been updated.

Breaking Changes
^^^^^^^^^^^^^^^^

* To migrate, replace any function scope ``torch_neuron.experimental.`` with ``torch_neuron.``. The change will have no effect on behavior or performance.

Known Issues
^^^^^^^^^^^^

* Using the latest torch-xla v2.7 may result in increase in host memory usage compared torch-xla v2.6.
* When switching Ubuntu OS kernel version from 5.15 to 6.8, you may see performance differences due to the new kernel scheduler (CFS vs EEVDF).
* When using tensor split operation on a 2D array in the second dimension, the resulting tensors don't have the expected data.
* BERT pretraining performance is ~10% lower with torch-neuronx 2.6 compared to torch-neuronx 2.5.


----

.. _pytorch-2-21-1-rn:

PyTorch Framework [2.21.1] (Neuron 2.21.1 Release)
--------------------------------------------------------

Date of Release: 01/14/2025

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* The transformers dependency has been pinned to ``transformers<4.48``


----

.. _pytorch-2-21-0-rn:

PyTorch Framework [2.21.0] (Neuron 2.21.0 Release)
--------------------------------------------------------

Date of Release: 12/20/2024

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Flash decoding support for speculative decoding
* Enabled on-device generation support in speculative decoding flows
* Added support for EAGLE speculative decoding support with greedy and lossless sampling
* Support for CPU compilation and sharded model saving
* Performance optimized MLP and QKV kernels added for llama models with support for sequence parallel norm
* Added support to control concurrent compilation workers
* Added option to skip AllGather using duplicate Q weights during shard over sequence

Bug Fixes
^^^^^^^^^

* Fixed padding issues when requested batch size is smaller than neff compiled size
* Fixed sequence parallel norm issue when executor is used with speculative decoding flows

Known Issues
^^^^^^^^^^^^

* GPT-NeoX is sensitive to ``fp16`` and customers are advised to use only ``amp="f32"`` for GPT-NeoX.
* Using ``cache_layout=constants.LAYOUT_BSH`` in NeuronConfig has known limitations with compilation. Customers are advised to use ``constants.LAYOUT_SBH`` instead.


----

.. _pytorch-2-20-0-rn:

PyTorch Framework [2.20.0] (Neuron 2.20.0 Release)
--------------------------------------------------------

Date of Release: 09/16/2024

torch-neuron
~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Minor updates.

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* This release adds support for Neuron Kernel Interface (NKI), Python 3.11, and protobuf versions 3.20+, as well as improved BERT performance.
* Added support for Neuron Kernel Interface (NKI).
* Added support for Python 3.11.
* Added support for protobuf versions 3.20+.
* (Training) Increased performance for BERT-Large pretraining by changing ``NEURON_TRANSFER_WITH_STATIC_RING_OPS`` default.
* (Training) Improved Neuron Cache locking mechanism for better Neuron Cache performance during multi-node training
* (Inference) Added support for weight separated models for DataParallel class.

Known Issues
^^^^^^^^^^^^

* Error ``cannot import name 'builder' from 'google.protobuf.internal'`` after installing compiler from earlier releases (2.19 or earlier)
* Lower accuracy when fine-tuning Roberta
* Slower loss convergence for NxD LLaMA-3 70B pretraining using ZeRO1 tutorial

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Support for model serialization (save and load) of all models except the ``GPTJForSampling`` and ``GPTNeoXForSampling`` model classes, which reduces future model load time by saving a transformed and sharded set of weights as a new safetensors checkpoint.
* Support for on device sampling (Top P) with Continuous batching
* Support for Scaled RoPE for LLAMA 3.1 models
* Support for multi-node inference for LLAMA 3.1 405B model for specific sequence lengths
* Support for FlashDecoding (using ``shard_over_sequence``) for supporting long context lengths upto 128k

Bug Fixes
^^^^^^^^^

* Fixes to handle ``seq_ids`` consistently across vLLM versions
* Fixes for KV head full replication logic errors

----

.. _pytorch-2-19-0-rn:

PyTorch Framework [2.19.0] (Neuron 2.19.0 Release)
--------------------------------------------------------

Date of Release: 07/03/2024

torch-neuron
~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Minor updates.

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Improvements in ZeRO1 to have FP32 master weights support and BF16 all-gather
* Added custom SILU enabled via ``NEURON_CUSTOM_SILU`` environment variable
* Neuron Parallel Compile now handle non utf-8 characters in trial-run log and reports compilation time results when enabled with ``NEURON_PARALLEL_COMPILE_DUMP_RESULTS``
* Support for using DummyStore during PJRT process group initialization by setting ``TORCH_DIST_INIT_BARRIER=0`` and ``XLA_USE_DUMMY_STORE=1``

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Support for compiler optimized flash attention kernel to support context lengths of 16k/32k for Llama models
* Streamer support enabled for BLOOM, GPTJ, GPT2, GPT-NeoX and LLAMA models
* Support for on device generation for TopK in Mixtral models
* Continuous batching support for Mistral v0.2
* Minor API improvements with type annotations for NeuronConfig, end-of-support warnings for old arguments, and exposing top-level configurations
* Performance improvements such as an optimized logit ordering for continuous batching in Llama models, optimized QKV padding for certain GQA models, faster implementation of cumsum operation to improve TopP performance

Bug Fixes
^^^^^^^^^

* Removed ``start_ids=None`` from ``generate()``
* Mistral decoding issue that occurs during multiple sampling runs
* Mistralv0.1 sliding window error
* Off-by-one error in window context encoding
* Better error messaging

Known Issues
^^^^^^^^^^^^

* ``on_device_generation=GenerationConfig(do_sample=True)`` has some known failures for Llama models. Customers are advised not to use ``on_device_generation`` in such cases.
* GPT-NeoX is sensitive to ``fp16`` and customers are advised to use only ``amp="f32"`` for GPT-NeoX.
* Using ``cache_layout=constants.LAYOUT_BSH`` in NeuronConfig has known limitations with compilation. Customers are advised to use ``constants.LAYOUT_SBH`` instead.

----

.. _pytorch-2-18-0-rn:

PyTorch Framework [2.18.0] (Neuron 2.18.0 Release)
--------------------------------------------------------

Date of Release: 04/10/2024

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* [Beta] Added support for continuous batching and a reference integration with vLLM (Llama models only)

Known Issues
^^^^^^^^^^^^

* There is a known compiler issue for inference of some configurations of Llama-2 70B that can cause accuracy degredation. Customers are advised to use the ``--enable-mixed-precision-accumulation`` compiler flag if Llama-2 70B accuracy issues occur.
* There is a known compiler issue for inference of some configurations of Llama-2 13B that can cause accuracy degredation. Customers are advised to use the ``--enable-saturate-infinity --enable-mixed-precision-accumulation`` compiler flags if Llama-2 13B accuracy issues occur.
* There is a known compiler issue for inference of some configurations of GPT-2 that can cause accuracy degredation. Customers are advised to use the ``--enable-saturate-infinity --enable-mixed-precision-accumulation`` compiler flags if GPT-2 accuracy issues occur.
* GPT-NeoX is sensitive to ``fp16`` and customers are advised to use only ``amp="f32"`` for GPT-NeoX.
* Using ``cache_layout=constants.LAYOUT_BSH`` in NeuronConfig has known limitations with compilation. Customers are advised to use ``constants.LAYOUT_SBH`` instead.

----

.. _pytorch-2-17-0-rn:

PyTorch Framework [2.17.0] (Neuron 2.17.0 Release)
--------------------------------------------------------

Date of Release: 04/01/2024

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Added support for on device log-softmax and on device sampling for TopK
* Added support for on device embedding for all models
* Added support for Speculative Decoding
* [Beta] Added support for Mixtral-8x7b MoE
* [Beta] Added support for mistralai/Mistral-7B-Instruct-v0.2 with no sliding window
* Added faster checkpoint loading support for both sharded and whole checkpoints
* Added the ability to download checkpoints directly from huggingface hub repositories
* Added NeuronAutoModelForCausalLM class which automatically loads architecture-specific classes
* Added a warmup to all kernels to avoid unexpected initialization latency spikes

Bug Fixes
^^^^^^^^^

* Users no longer need a copy of the original checkpoint and can use safetensor checkpoints for optimal speed.

Known Issues
^^^^^^^^^^^^

* There is a known compiler issue for inference of some configurations of Llama-2 70B that can cause accuracy degredation. Customers are advised to use the ``--enable-mixed-precision-accumulation`` compiler flag if Llama-2 70B accuracy issues occur.
* There is a known compiler issue for inference of some configurations of Llama-2 13B that can cause accuracy degredation. Customers are advised to use the ``--enable-saturate-infinity --enable-mixed-precision-accumulation`` compiler flags if Llama-2 13B accuracy issues occur.
* There is a known compiler issue for inference of some configurations of GPT-2 that can cause accuracy degredation. Customers are advised to use the ``--enable-saturate-infinity --enable-mixed-precision-accumulation`` compiler flags if GPT-2 accuracy issues occur.
* GPT-NeoX is sensitive to ``fp16`` and customers are advised to use only ``amp="f32"`` for GPT-NeoX.


----

.. _pytorch-2-16-0-rn:

PyTorch Framework [2.16.0] (Neuron 2.16.0 Release)
--------------------------------------------------------

Date of Release: 12/21/2023

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* [Beta] Added support for Llama-2 70B
* [Beta] Added support for Mistral 7B
* [Beta] Added support for PyTorch 2.1
* [Beta] Added support for Grouped Query Attention (GQA)
* [Beta] Added support for ``safetensors`` serialization
* [Beta] Added support for early stopping in the ``sample_llama`` function
* [Beta] Added sparse attention support for GPT2
* Added support for ``BatchNorm``
* Use the ``--auto-cast=none`` compiler flag by default for all models. This flag improves accuracy for ``float32`` operations

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* [Beta] Added support for Llama-2 70B
* [Beta] Added support for Mistral 7B
* [Beta] Added support for Grouped Query Attention (GQA)
* [Beta] Added support for ``safetensors`` serialization
* [Beta] Added support for early stopping in the ``sample_llama`` function
* [Beta] Added sparse attention support for GPT2

Bug Fixes
^^^^^^^^^

* Resolved an issue in ``top_p`` in the ``sample_llama`` function so that it now selects the same number of tokens that the Hugging Face ``top_p`` implementation selects.

Known Issues
^^^^^^^^^^^^

* There is a known compiler issue for inference of some configurations of Llama-2 70B that can cause accuracy degredation. Customers are advised to use the ``--enable-mixed-precision-accumulation`` compiler flag if Llama-2 70B accuracy issues occur.
* There are known compiler issues impacting inference accuracy of certain model configurations of ``Llama-2-13b`` when ``amp = fp16`` is used. If this issue is observed, ``amp=fp32`` should be used as a work around. This issue will be addressed in future Neuron releases.

----

.. _pytorch-2-15-0-rn:

PyTorch Framework [2.15.0] (Neuron 2.15.0 Release)
--------------------------------------------------------

Date of Release: 10/26/2023

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* [Beta] Added support for ``int8`` quantization for Llama
* [Beta] Added multi bucket context encoding support for BLOOM
* [Beta] Added model Serialization for all supported models (except GPT-J and GPT-NeoX)
* [Beta] Added the ability to return output logit scores during sampling
* Added support for ``SOLU`` activation and ``GroupNorm``

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* [Beta] Added support for ``int8`` quantization for Llama
* [Beta] Added multi bucket context encoding support for BLOOM
* [Beta] Added model Serialization for all supported models (except GPT-J and GPT-NeoX)
* [Beta] Added the ability to return output logit scores during sampling

Bug Fixes
^^^^^^^^^

* [GPT2] Fixed an issue in ``GPT2ForSamplingWithContextBroadcasting`` where the input prompt would get truncated if it was longer than the ``context_length_estimate``.

----

.. _pytorch-2-14-0-rn:

PyTorch Framework [2.14.0] (Neuron 2.14.0 Release)
--------------------------------------------------------

Date of Release: 09/15/2023

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Use the ``--model-type=transformer`` compiler flag by default for all models. This flag improves performance and compilation time for all models. This flag replaces the ``--model-type=transformer-inference`` flag, which is now deprecated.

Bug Fixes
^^^^^^^^^

* Fixed an issue where the ``HuggingFaceGenerationModelAdapter`` class falls back to serial context encoding for models that have parallel context encoding (``GPT2ForSamplingWithContextBroadcasting``, ``LlamaForSampling``, etc.)
* [GPT2 / OPT] Fixed an issue in the parallel context encoding network where incorrect results could be generated due to incorrect masking logic.

Known Issues
^^^^^^^^^^^^

* Some configurations of Llama and Llama-2 inference models fail compilation with the error ``IndirectLoad/Save requires contiguous indirect access per partition``. This is fixed in the compiler version 2.10.0.35 (Neuron SDK 2.14.1).
* Some configurations of Llama and Llama-2 inference model fail compilation with the error ``Too many instructions after unroll for function sg0000``. To mitigate this, please try with ``-O1`` compiler option (or ``--optlevel 1``) by adding ``os.environ["NEURON_CC_FLAGS"] = "-O1"`` to your script or set in the environment. A complete fix will be coming in the future release which will not require this option. Note: Using -O1 in the Llama-2 13B tutorial results in about 50% increase in latency compared to Neuron SDK 2.13.2. If this is not acceptable, please use compiler version from Neuron SDK 2.13.2.


----

.. _pytorch-2-13-0-rn:

PyTorch Framework [2.13.0] (Neuron 2.13.0 Release)
--------------------------------------------------------

Date of Release: 08/28/2023

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Added support for Llama 2 (excluding grouped/multi-query versions, such as Llama 2 70B) [Beta]
* Improved the performance of BLOOM and Llama models [Beta]
* Reduced execution latency of token generation in tensor parallel models by improving thread synchronization (supported in Llama only)
* Added an optimized vector implementation of RoPE positional embedding (supported in Llama only)
* Added support for faster context encoding on sequences of varying lengths. This is implemented by allowing multiple buckets for parallel context encoding. During inference the best fit bucket is chosen (supported in Llama/GPT-2 only)
* Added the Neuron Persistent Cache for compilation to automatically load pre-compiled model artifacts (supported by all models)
* Improved compilation time by compiling models used for different sequence length buckets in parallel (not supported in GPT-NeoX/GPT-J)

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Added support for Llama 2 (excluding grouped/multi-query versions, such as Llama 2 70B) [Beta]
* Improved the performance of BLOOM and Llama models [Beta]
* Reduced execution latency of token generation in tensor parallel models by improving thread synchronization (supported in Llama only)
* Added an optimized vector implementation of RoPE positional embedding (supported in Llama only)
* Added support for faster context encoding on sequences of varying lengths. This is implemented by allowing multiple buckets for parallel context encoding. During inference the best fit bucket is chosen (supported in Llama/GPT-2 only)
* Added the Neuron Persistent Cache for compilation to automatically load pre-compiled model artifacts (supported by all models)
* Improved compilation time by compiling models used for different sequence length buckets in parallel (not supported in GPT-NeoX/GPT-J)

Bug Fixes
^^^^^^^^^

* [Llama] Fixed an issue in the parallel context encoding network where incorrect results could be generated if the context length is shorter than the context length estimate
* [GPT2 / OPT] Fixed an issue in the parallel context encoding network where incorrect results could be generated

Known Issues
^^^^^^^^^^^^

* The ``HuggingFaceGenerationModelAdapter`` class currently falls back to serial context encoding for models that have parallel context encoding (``GPT2ForSamplingWithContextBroadcasting``, ``LlamaForSampling``, etc.)
* Beam search can introduce memory issues for large models
* There can be accuracy issues for the GPT-J model for certain use-cases


----

.. _pytorch-2-12-0-rn:

PyTorch Framework [2.12.0] (Neuron 2.12.0 Release)
--------------------------------------------------------

Date of Release: 07/21/2023

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Added support for GPT-NeoX models [Beta]
* Added support for BLOOM models [Beta]
* Added support for Llama models [Alpha]
* Added support for more flexible tensor-parallel configurations to GPT2, OPT, and BLOOM. The attention heads doesn't need to be evenly divisible by ``tp_degree`` anymore
* Added multi-query / multi-group attention support for GPT2

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Added support for GPT-NeoX models [Beta]
* Added support for BLOOM models [Beta]
* Added support for Llama models [Alpha]
* Added support for more flexible tensor-parallel configurations to GPT2, OPT, and BLOOM. The attention heads doesn't need to be evenly divisible by ``tp_degree`` anymore
* Added multi-query / multi-group attention support for GPT2

Bug Fixes
^^^^^^^^^

* Fixed NaN issues for GPT2 model
* Fixed OPT/GPT-NeoX gibberish output
* Resolved an issue where NaN values could be produced when the context_length argument was used in GPT2/OPT

Known Issues
^^^^^^^^^^^^

* Missing cache reorder support for beam search


----

.. _pytorch-2-11-0-rn:

PyTorch Framework [2.11.0] (Neuron 2.11.0 Release)
--------------------------------------------------------

Date of Release: 06/14/2023

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Added ``int8`` weight storage for GPT2 models
* Improved prompt context encoding performance for GPT2 models
* Improved collective communications performance for tp-degrees 4, 8, and 24 on Inf2
* Improved collective communications performance for tp-degrees 8 and 32 on Trn1
* Support for the ``--model-type=transformer-inference`` compiler flag for optimized decoder-only LLM inference

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Added ``int8`` weight storage for GPT2 models
* Improved prompt context encoding performance for GPT2 models
* Improved collective communications performance for tp-degrees 4, 8, and 24 on Inf2
* Improved collective communications performance for tp-degrees 8 and 32 on Trn1
* Support for the ``--model-type=transformer-inference`` compiler flag for optimized decoder-only LLM inference

Bug Fixes
^^^^^^^^^

* Added padding to the GPT-J ``linear`` layer to correctly handle odd vocabulary sizes
* Issues where the HuggingFace ``generate`` method produces incorrect results when ``beam_search`` is used have been resolved


----

.. _pytorch-2-10-0-rn:

PyTorch Framework [2.10.0] (Neuron 2.10.0 Release)
--------------------------------------------------------

Date of Release: 05/01/2023

torch-neuronx
~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Added ``transformers-neuronx`` artifacts to PyPI repository
* Added support for the HuggingFace ``generate`` method
* Added model serialization support for GPT2 models, including model saving, loading, and weight swapping
* Added support for caching compiled artifacts
* Improved performance by removing unnecessary KV-cache tensor resetting
* Improved prompt context encoding performance (OPT, GPT2)

transformers-neuronx
~~~~~~~~~~~~~~~~~~~~

Improvements
^^^^^^^^^^^^^^^

* Added ``transformers-neuronx`` artifacts to PyPI repository
* Added support for the HuggingFace ``generate`` method
* Added model serialization support for GPT2 models, including model saving, loading, and weight swapping
* Added support for caching compiled artifacts
* Improved performance by removing unnecessary KV-cache tensor resetting
* Improved prompt context encoding performance (OPT, GPT2)

Bug Fixes
^^^^^^^^^

* Fixed the GPT-J demo to import the correct ``amp_callback`` function

Known Issues
^^^^^^^^^^^^

* When the HuggingFace ``generate`` method is configured to use ``beam_search``, this can produce incorrect results for certain configurations. It is recommended to use other generation methods such as ``sample`` or ``greedy_search``. This will be fixed in a future Neuron release.

Breaking Changes
^^^^^^^^^^^^^^^^

* None
