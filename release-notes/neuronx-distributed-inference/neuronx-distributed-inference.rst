.. _neuronx-distributed-inference-rn:


NxD Inference Release Notes (``neuronx-distributed-inference``)
=============================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for Neuronx Distributed Inference library.

Neuronx Distributed Inference [0.4.7422] (Neuron 2.24.0 Release)
-----------------------------------------------------------------------

Date: 06/24/2025

* Models

  * Qwen2.5 text models, which are tested on Trn1. Compatible models include:

    * `Qwen2.5-0.5B-Instruct <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>`__
    * `Qwen2.5-7B-Instruct <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>`__
    * `Qwen2.5-32B-Instruct <https://huggingface.co/Qwen/Qwen2.5-32B-Instruct>`__
    * `Qwen2.5-72B-Instruct <https://huggingface.co/Qwen/Qwen2.5-72B-Instruct>`__

* Features

  * Automatic Prefix Caching support (APC) through vLLM. APC
    improves efficiency by reusing KV cache from previous queries if the
    new query shares a prefix. APC can significantly improve TTFT based on how often
    different queries share the same prefixes. Performance gains are greater
    when requests have longer shared prefixes and when there is a higher
    frequency of prefix sharing across requests. For example, with Llama3.3 70B on Trn2,
    you can observe a 3.2x TTFT improvement with the math.math dataset (90% cache hit),
    a 1.6x TTFT improvement with a Sonnet dataset with 2K prompt length (25% cache hit),
    or no TTFT improvement with the HumanEval dataset (0% cache hit). For more information,
    see :ref:`nxdi-prefix-caching` and :ref:`/libraries/nxd-inference/tutorials/trn2-llama3.3-70b-apc-tutorial.ipynb`.
  * Disaggregated Inference (DI) support through vLLM (Beta). Disaggregated Inference is 
    also known as disaggregated serving, disaggregated prefill, or p/d disaggregation.
    DI separates the prefill and decode phase of inference onto different hardware resources.
    DI can improve inter token latency (ITL) by by eliminating prefill stall in
    continuous batching, where decode is paused to perform prefill for a new incoming request.
    With DI, you can also scale prefill and decode resources independently to further improve
    performance. For more information, see :ref:`nxdi-disaggregated-inference`.
  * Context parallelism in NeuronAttentionBase (Beta). Context parallelism
    distributes context processing across multiple NeuronCores. Context
    parallelism improves TTFT, particularly at higher sequence lengths where
    the number of KV heads is low. To use context parallelism, set ``cp_degree``
    in NeuronConfig.
  * Mixed-precision parameters in modeling code. This feature enables
    you to configure each module's dtype independently. To use
    mixed-precision parameters, set ``cast_type="as-declared"`` in
    NeuronConfig. Note: The default behavior (``cast_type="config"``) is
    to cast all parameters to the ``torch_dtype`` in NeuronConfig.
  * Output logits when using on-device sampling. To output logits,
    enable ``output_logits`` in NeuronConfig. Note that this flag
    impacts performance and should only be used for debugging model
    logits.

* Other changes

  * Add support for PyTorch 2.7. This release includes support for PyTorch 2.5, 2.6, and 2.7.
  * Upgrade ``transformers`` requirement from v4.48 to v4.51.
  * Re-enable warmup on Trn2. NxD Inference disabled warmup on Trn2 in the
    previous release due to an issue that prevented certain model
    configurations from loading correctly. That issue is now fixed.
  * Update the behavior of the ``attn_kernel_enabled`` attribute in
    NeuronConfig, which configures whether to use the flash attention
    kernel. Previously, ``True`` meant to enable in all cases where
    supported, and ``False`` meant to auto-enable where beneficial
    (defaults to ``False``). Now, ``attn_kernel_enabled=False`` disables
    the flash attention kernel in all cases. To use the previous
    auto-enable behavior, set ``attn_kernel_enabled=None``. The default
    value for ``attn_kernel_enabled`` is now ``None`` to retain the same
    default behavior as before.
  * Enable ``--verify-hlo`` flag during compilation. Now, if an HLO is
    invalid, compilation will fail. Previously, in certain scenarios,
    the compiler would successfully compile invalid HLOs.
  * Update the flash attention kernel strategy to use the attention
    kernel on Trn2 in all cases where it's supported. This change fixes
    an issue where certain context lengths failed to trace.
  * Add ``logical_nc_config`` as an argument to the ``build_module`` and
    ``build_function`` test utilities, so you can use these utilities to
    test modules/functions for Trn2 using LNC2.
  * Other minor fixes and improvements.


Known Issues and Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Increased Device Memory Usage for Certain Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Certain model configurations require slightly more device memory than in
previous releases. If your model used close to the maximum amount of device
memory in previous releases, this increase could cause it to fail to load after
you compile it with this release. This issue is most likely to affect
Llama3.1-405B configurations that use a large number of buckets.

If this issue occurs, you will see the following error during model load.

::

   ERROR  TDRV:log_dev_mem                             Failed to allocate 512.000MB (alignment: 4.000MB, usage: shared scratchpad) on ND14:NC 6

To avoid this error, reduce the number of buckets you use, or reduce the 
sequence lengths that you use in each bucket.

Neuronx Distributed Inference [0.3.5591] (Neuron 2.23.0 Release)
-----------------------------------------------------------------------

Date: 05/20/2025

NxD Inference is now GA and out of beta in the Neuron 2.23 release.

Features in this Release
^^^^^^^^^^^^^^^^^^^^^^^^

* Features

  * Shard-on-load for weight sharding is now enabled by default. With this change,
    end-to-end compile and load time is reduced by up to 70% when
    sharding weights. This change significantly reduces compile time by skipping
    weight sharding and serialization during compile, but may lead to
    increased load time. For example, for Llama 3.1 405B,
    end-to-end compile and load time is reduced from 40 minutes to
    12 minutes. For best load performance, you can continue to serialize
    sharded weights by enabling ``save_sharded_checkpoint`` in
    NeuronConfig. For more information, see :ref:`nxdi-weights-sharding-guide`.
  * Neuron Persistent Cache. NxD Inference now supports Neuron
    Persistent Cache, which caches compiled model artifacts to reduce
    compilation times. For more information, see :ref:`nxdi-neuron-persistent-cache`.
  * Support for an attention block kernel for token generation. This kernel
    performs QKV projections, RoPE, attention, and output projections. You can use
    this kernel with Llama3-like attention on Trn2 to improve token gen performance.
    To use this kernel, enable ``attn_block_tkg_nki_kernel_enabled`` in NeuronConfig.

    * This kernel can also update the KV cache in parallel with each layer's
      attention compute to further improve performance. This functionality hides
      the latency of the KV cache update that is otherwise done for all layers at
      once at the end of each token generation iteration. To enable in-kernel
      KV cache updates, enable ``attn_block_tkg_nki_kernel_cache_update`` in NeuronConfig.
      When in-kernel KV cache updating is enabled, you can also enable ``k_cache_transposed``
      to further improve the performance.

  * Automatically extract ``target_modules`` and ``max_lora_rank`` from
    LoRA checkpoints. You no longer need to set these arguments
    manually.
  * Support fused residual add in the QKV kernel. This feature improves
    the performance of context encoding at short sequence lengths. To
    use this feature, enable the ``qkv_kernel_fuse_residual_add`` flag
    in NeuronConfig.

* Backward incompatible changes

  * Remove ``set_async_mode(async_mode)`` from NeuronBaseForCausalLM, as
    this feature didn't work as intended. Async mode cannot be enabled or
    disabled after the model is loaded. To enable async mode, set ``async_mode=True``
    in NeuronConfig.

* Other changes

  * Disable warmup for Trn2. This change avoids an issue
    that prevents certain model configurations from loading correctly.
    When warmup is disabled, you will see lower performance on the first
    few requests to the model. This change also affects initial
    performance for serving through vLLM. Warmup will work in many cases
    where it is now disabled, so you can try to reenable warmup by
    setting ``skip_warmup=False`` in NeuronConfig. Alternatively, you
    can manually warm up the model by sending a few requests to each
    bucket after loading the model.
  * Fix an issue where when continuous batching and bucketing were
    enabled, NxDI padded each input to the largest sequence in the
    batch, rather than the next largest bucket for that input. This
    change improves performance when using continuous batching with
    bucketing, including through vLLM.
  * Add a ``num_runs`` parameter to ``benchmark_sampling``, so you can
    configure the number of runs to perform when benchmarking.
  * Silence unimportant error messages during warmup.
  * NeuronConfig now includes a ``disable_kv_cache_tiling`` flag that
    you can set to disable KV cache tiling in cases where it was
    previously enabled by default.
  * Update the package version to include additional information in the
    version tag.
  * Other minor fixes and improvements.

Neuronx Distributed Inference [0.2.0] (Beta) (Neuron 2.22.0 Release)
------------------------------------------------------------------
Date: 04/03/2025

Models in this Release
^^^^^^^^^^^^^^^^^^^^^^

* Llama 3.2 11B (Multimodal)

Features in this Release
^^^^^^^^^^^^^^^^^^^^^^^^

* Multi-LoRA serving. This release adds support for multi-LoRA serving
  through vLLM by loading LoRA adapters at server startup. Multi-LoRA
  serving is currently supported for Llama 3.1 8B, Llama 3.3 70B, and
  other models that use the Llama architecture.
* Custom quantization. You can now specify which layers or modules in
  NxDI to quantize or keep in full precision during inference. To
  configure which layers or modules to skip during quantization, use
  the ``modules_to_not_convert`` and
  ``draft_model_modules_to_not_convert`` attributes in NeuronConfig.
* Models quantized through external libraries. NxDI now supports
  inference of models that are quantized externally using quantization
  libraries such as LLMCompressor.
* Async mode. This release adds support for async mode, which improves performance
  by asynchronously preparing the next forward call to a mode. To use async mode,
  enable the ``async_mode`` flag in NeuronConfig.
* CPU inference. You can now run models on CPU and compare against output on Neuron
  to debug accuracy issues. To use this feature, enable the ``on_cpu`` flag in
  NeuronConfig.
* Unit/module testing utilities. These common utilities include
  ``build_module``, ``build_function``, and ``validate_accuracy``,
  which enable you to build a module or function and validate its
  accuracy on Neuron. You can use these utilities in unit/integration
  tests to verify your modeling code works correctly.
* Add support for models that use a custom ``head_dim`` value from InferenceConfig.
  This change enables support for models where ``head_dim`` isn't equivalent to
  ``hidden_size`` divided by ``num_attention_heads``.
* Input capture hooks. When you call the NeuronBaseForCausalLM forward function, you
  can provide an ``input_capture_hook`` function that will be called with the model
  inputs as arguments.
* Runtime warmup. To improve the performance of the first request sent to a model,
  NxD Inference now warms up the model during load. You can disable this behavior
  with the ``skip_warmup`` flag in NeuronConfig.

Backward Incompatible Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Fix the behavior of the ``do_sample`` sampling flag. Previously,
  NxDI used greedy sampling when ``do_sample=True``, which was a bug because
  ``do_sample=True`` should result in multinomial sampling.
  If you use ``do_sample=True`` in a config where you intend to use
  greedy sampling, you must change it to ``do_sample=False``. As part
  of this change, the default value for ``do_sample`` is now
  ``False``.
* Enforce that tensors in a model's state_dict don't share memory with
  other tensors. This change can cause models to fail to load if their
  tensors share memory, which now results in an error:
  ``RuntimeError: Error while trying to find names to remove to save state dict``.
  To fix this issue, apply ``.clone().detach().contiguous()`` to the
  model's state_dict, and re-shard the weights.
* Change the quantization state_dict keys from ``weight_scale`` to
  ``scale`` to match the NxD quantization scale keys and avoid any
  confusion. If you use quantization and have sharded weights from
  earlier versions of NxDI, you must re-shard the weights.
* If you use a model that skips quantization for certain modules (such
  as in Llama 3.1 405B FP8), you must now specify
  ``modules_not_to_convert`` to configure the modules that skip
  quantization.
* Validate when input size exceeds the model's maximum length (``max_context_length``
  or ``max_length``). NxD Inference now throws a ValueError if given an input that's
  too large. To enable the previous behavior, where input is truncated to the maximum
  length, enable the ``allow_input_truncation`` flag in NeuronConfig.

Other Changes
^^^^^^^^^^^^^

* Improve model performance by up to 50% (5-20% in most cases) by eliminating overheads in logging.
* Upgrade ``transformers`` from v4.45 to v4.48.
* Deprecate NeuronConfig's ``logical_neuron_cores`` attribute and replace it with
  ``logical_nc_config``. The LNC config is now automatically set from the 
  ``NEURON_LOGICAL_NC_CONFIG`` environment variable if set.
* Deprecate NeuronConfig's ``trace_tokengen_model`` attribute. This attribute is now
  determined dynamically based on other configuration attributes.
* Improve the performance of on-device sampling.
* When running Llama models with LNC2, the sharded flash attention kernel is now 
  automatically enabled when context length is 256 or greater. Previously, this kernel
  was enabled for context length of 1024 or greater. This change improves performance 
  at smaller context lengths.
* NeuronConfig now includes a ``skip_sharding`` flag that you can enable to skip weight 
  sharding during model compilation. This option is useful in cases where you have 
  already sharded weights, such as during iterative development, so you can iterate 
  without re-sharding the weights each time you compile the model.
* NeuronApplicationBase now includes a ``shard_weights`` function that
  you can use to shard weights independent of compiling the model.
* Fix vanilla speculative decoding support for models with multiple
  EOS tokens.
* Other minor fixes and improvements.

Known Issues and Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* For some configurations that use continuous batching or vLLM, model warmup can cause ``Numerical Error`` during inference. 
  If you encounter this error, set ``skip_warmup=True`` in NeuronConfig to disable warmup and avoid this issue. 
  To disable warmup in vLLM, pass ``"skip_warmup": true`` in ``override_neuron_config``. For more information about how to configure vLLM, see vLLM 
  `Model Configuration <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide.html#model-configuration>`_.
 
  ::

      RuntimeError: Failed to execute the model status=1003 message=Numerical Error

Neuronx Distributed Inference [0.1.1] (Beta) (Neuron 2.21.1 Release)
------------------------------------------------------------------
Date: 01/14/2025

Bug Fixes
^^^^^^^^^
* Fix minor issues with sampling params and add validation for sampling params.


Neuronx Distributed Inference [0.1.0] (Beta) (Neuron 2.21 Release)
------------------------------------------------------------------
Date: 12/20/2024

Features in this Release
^^^^^^^^^^^^^^^^^^^^^^^^

NeuronX Distributed (NxD) Inference (``neuronx-distributed-inference``) is
an open-source PyTorch-based inference library that simplifies deep learning
model deployment on AWS Inferentia and Trainium instances. Neuronx Distributed
Inference includes a model hub and modules that users can reference to
implement their own models on Neuron.

This is the first release of NxD Inference (Beta) that includes:

* Support for Trn2, Inf2, and Trn1 instances
* Support for the following model architectures. For more information, including
  links to specific supported model checkpoints, see :ref:`nxdi-model-reference`.

  * Llama (Text), including Llama 2, Llama 3, Llama 3.1, Llama 3.2, and Llama 3.3
  * Llama (Multimodal), including Llama 3.2 multimodal
  * Mistral (using Llama architecture)
  * Mixtral
  * DBRX
  
* Support for onboarding additional models.
* Compatibility with HuggingFace checkpoints and ``generate()`` API
* vLLM integration
* Model compilation and serialization
* Tensor parallelism
* Speculative decoding

  * EAGLE speculative decoding
  * Medusa speculative decoding
  * Vanilla speculative decoding

* Quantization
* Dynamic sampling
* Llama3.1 405B Inference Example on Trn2
* Open Source Github repository: `aws-neuron/neuronx-distributed-inference <https://github.com/aws-neuron/neuronx-distributed-inference>`_

For more information about the features supported by NxDI, see :ref:`nxdi-feature-guide`.


Known Issues and Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Longer Load Times for Large Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: Users may experience extended load times when working with large models,
particularly during weight sharding and initial model load. This is especially
noticeable with models like Llama 3.1 405B.

Root Cause: These delays are primarily due to storage performance limitations.

Recommended Workaround: To mitigate this issue, we recommend that you store
model checkpoints in high-performance storage options:

* `Instance store volumes <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ssd-instance-store.html>`_:
  On supported instances, instance store volumes offer fast, temporary block-level storage.
* `Optimized EBS volumes <https://docs.aws.amazon.com/ebs/latest/userguide/ebs-performance.html>`_:
  For persistent storage with enhanced performance.

By using these storage optimizations, you can reduce model load times and improve
your overall workflow efficiency.

Note: Load times may still vary depending on model size and specific hardware configurations.


Other Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Llama 3.2 11B (Multimodal) is not yet supported with PyTorch 2.5.
* The following model architectures are tested only on Trn1 and Inf2:

  * Llama (Multimodal)

* The following model architectures are tested only on Trn1:
  
  * Mixtral
  * DBRX

* The following kernels are tested only on Trn2:
  
  * MLP
  * QKV
  
* If you run inference with an prompt that is larger than the model's ``max_context_length``,
  the model will generate incorrect output. In a future release, NxD Inference will
  throw an error in this scenario.
* Continuous batching (including through vLLM) supports batch size up to 4.
  Static batching supports larger batch sizes.
* To use greedy on-device sampling, you must set ``do_sample`` to ``True``.
* To use FP8 quantization or KV cache quantization, you must set the
  ``XLA_HANDLE_SPECIAL_SCALAR`` environment variable to ``1``.


Neuronx Distributed Inference [0.1.0] (Beta) (Trn2)
---------------------------------------------------
Date: 12/03/2024

Features in this release
^^^^^^^^^^^^^^^^^^^^^^^^

NeuronX Distributed (NxD) Inference (``neuronx-distributed-inference``) is
an open-source PyTorch-based inference library that simplifies deep learning
model deployment on AWS Inferentia and Trainium instances. Neuronx Distributed
Inference includes a model hub and modules that users can reference to
implement their own models on Neuron.

This is the first release of NxD Inference (Beta) that includes:

* Support for Trn2 instances
* Compatibility with HuggingFace checkpoints and ``generate()`` API
* vLLM integration
* Model compilation and serialization
* Tensor parallelism
* Speculative decoding

  * EAGLE speculative decoding
  * Medusa speculative decoding
  * Vanilla speculative decoding

* Quantization
* Dynamic sampling
* Llama3.1 405B Inference Example on Trn2
* Open Source Github repository: `aws-neuron/neuronx-distributed-inference <https://github.com/aws-neuron/neuronx-distributed-inference>`_

For more information about the features supported by NxDI, see :ref:`nxdi-feature-guide`.
