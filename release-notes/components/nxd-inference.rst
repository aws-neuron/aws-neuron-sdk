.. meta::
    :description: Complete release notes for the NxD Inference component across all AWS Neuron SDK versions.
    :keywords: nxd inference, neuronx-distributed-inference, release notes, aws neuron sdk
    :date-modified: 02/26/2026

.. _nxd-inference_rn:

.. _nxd-inference-2-28-0-rn:

Component Release Notes for NxD Inference
=========================================

The release notes for the NxD Inference Neuron component. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

NxD Inference [0.8.16251] + vLLM Neuron Plugin [0.4.0] (Neuron 2.28.0 Release)
------------------------------------------------------------------------------

Date of Release: 02/26/2026

NxD Inference
~~~~~~~~~~~~~

Neuron SDK 2.28.0 includes the following updates for NxD Inference library 0.8.16251:

Improvements
^^^^^^^^^^^^
* Qwen2 VL Model Support (Beta) - NxD Inference supports Qwen2 VL vision language model which processes text and image inputs. Please refer to :doc:`Tutorial: Qwen2 VL Inference </libraries/nxd-inference/tutorials/qwen2-vl-tutorial>`.
  
  Compatible models include:
    - `Qwen2-VL-7B-Instruct <https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct>`__
* Qwen3 VL Model Support (Beta) - NxD Inference supports Qwen3 VL vision language model which processes text and image inputs. Please refer to :doc:`Tutorial: Qwen3 VL Inference </libraries/nxd-inference/tutorials/qwen3-vl-tutorial>`.
  
  Compatible models include:
    - `Qwen3-VL-8B-Thinking <https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking>`__
* Pixtral Model Support Improvements (Beta) - Adds new functionality support with batch size 32 and sequence length 10240 with vllm v1 on Trn2.
* Flux.1 Model Support Improvements (Beta) - Adds new functionality support for in-paint, out-paint, canny and depth. Please refer to :doc:`Tutorial: Flux Inpainting </libraries/nxd-inference/tutorials/flux-inpainting-inference-tutorial>`


Known Issues
^^^^^^^^^^^^
* Qwen3 MoE only supports batch size >= 16 configurations.
* Qwen3-VL only supports dynamic image resolution up to vision sequence length 16K, and total vision and text sequence length up to 32K. Qwen2-VL does not support dynamic image resolution yet.
* Qwen-VL models only support batch size 1 configuration in vision encoder. No video understanding functionality is supported yet.
* Llama 3.2 11B/90B tutorial and samples not compatible to vLLM V1 are removed.

vLLM Plugin for Neuron
~~~~~~~~~~~~~~~~~~~~~~

Neuron SDK 2.28.0 includes the following updates for the vLLM Plugin 0.4.0 for Neuron:

Improvements
^^^^^^^^^^^^
* Multi-LoRA Serving Enhancements - NxD Inference supports streaming LoRA adapters via vLLM's `load_adapter` serving API, allowing adapters to be loaded into CPU memory dynamically at runtime. This provides more flexibility as users no longer need to specify all adapter checkpoint paths before execution. Additionally, users can now run the base model alone when multi-LoRA serving is enabled. See the :ref:`Llama 3.1 8B Multi-LoRA tutorial <trn2-llama3.1-8b-multi-lora-tutorial>` for more details.
* Eagle3 Speculative Decoding - NxD Inference supports Eagle3 speculative decoding on Llama 3.1 8B.

  Supported Eagle3 draft models include:
    - `EAGLE-LLaMA3-Instruct-8B <https://huggingface.co/yuhuili/EAGLE-LLaMA3-Instruct-8B>`__
* vLLM v0.13.0 Support - vLLM Neuron Plugin supports vLLM v0.13.0 and Pytorch 2.9.


Known Issues
^^^^^^^^^^^^
* This version of the vLLM Neuron Plugin is pinned to vLLM version v0.13.0 and requires PyTorch 2.9. If you must use PyTorch 2.7 or 2.8, you may fall back to the Neuron fork of vLLM that implements a Neuron integration using the vLLM V0 architecture. However, note that this fork is no longer maintained and not all features may be available. The fork can be found at https://github.com/aws-neuron/upstreaming-to-vllm/releases/tag/2.26.1.
* Known issues for vLLM Neuron plugin are tracked in [vLLM V1 user guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide-v1.html#known-issues).

.. _nxd-inference-2-27-1-rn:

NxD Inference [0.7.15603] (Neuron 2.27.1 Release)
---------------------------------------------------

Date of Release: 01/14/2026

Bug Fixes
~~~~~~~~~

* Fixed stability issue affecting Llama 4 that may occur when changing model configuration.


----

.. _nxd-inference-2-27-0-rn:

NxD Inference [0.6.9230] (Neuron 2.27.0 Release)
-------------------------------------------------

Date of Release: 12/19/2025

Improvements
~~~~~~~~~~~~~~~

* Added support for running NxD Inference on Trn3 instances.
* Added support for vLLM V1 through vllm-neuron plugin.
* Qwen3 MoE Model Support (Beta) — NxD Inference supports Qwen3 MoE language model which supports multilingual text inputs.
* Pixtral Model Support (Beta) — NxD Inference supports Pixtral image understanding model which processes text and image inputs.

Known Issues
~~~~~~~~~~~~

* Pixtral deployment is supported up to batch size 32 and sequence length 10240 with vLLM v0. vLLM v1 deployment supports up to batch size 4 and sequence length 10240.
* The performance of Qwen3 MoE and Pixtral on Trn2 is not fully optimized.
* The vllm-neuron plugin source code in github is currently not compatible with 2.27 SDK.


----

.. _nxd-inference-2-25-0-rn:

NxD Inference [0.5.9230] (Neuron 2.25.0 Release)
-------------------------------------------------

Date of Release: 07/31/2025

Improvements
~~~~~~~~~~~~~~~

* Added support for Qwen3 dense models (0.6B to 32B parameters), which are tested on Trn1.
* Added simplified functions for validating the accuracy of logits returned by a model: ``check_accuracy_logits_v2`` and ``generated_expected_logits``.
* Added ``scratchpad_page_size`` attribute to NeuronConfig for configuring the scratchpad page size used during compilation and at runtime.
* Enabled Chunked Attention as a generic building block for any attention-based model.
* Published scripts to evaluate model accuracy and benchmark performance against Neuron.

Breaking Changes
~~~~~~~~~~~~~~~~

* Removed support for Meta checkpoint compatibility in Llama3.2 Multimodal modeling code. You can continue to use Hugging Face checkpoints.

Bug Fixes
~~~~~~~~~

* Fixed accuracy issues when using Automatic Prefix Caching (APC) with EAGLE speculation.
* Fixed continuous batching for Llama3.2 Multimodal where the input batch size is less than the compiled batch size.
* Added support for continuous batching when running Neuron modeling code on CPU.
* Set a manual seed in ``benchmark_sampling`` to improve the stability of data-dependent benchmarks like speculation.


----

.. _nxd-inference-2-24-0-rn:

NxD Inference [0.4.7422] (Neuron 2.24.0 Release)
---------------------------------------------------

Date of Release: 06/24/2025

Improvements
~~~~~~~~~~~~~~~

**Models**

* Qwen2.5 text models, which are tested on Trn1. Compatible models include:

  * `Qwen2.5-0.5B-Instruct <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>`__
  * `Qwen2.5-7B-Instruct <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>`__
  * `Qwen2.5-32B-Instruct <https://huggingface.co/Qwen/Qwen2.5-32B-Instruct>`__
  * `Qwen2.5-72B-Instruct <https://huggingface.co/Qwen/Qwen2.5-72B-Instruct>`__

**Features**

* Automatic Prefix Caching support (APC) through vLLM. APC improves efficiency by reusing KV cache from previous queries if the new query shares a prefix. APC can significantly improve TTFT based on how often different queries share the same prefixes. Performance gains are greater when requests have longer shared prefixes and when there is a higher frequency of prefix sharing across requests. For example, with Llama3.3 70B on Trn2, you can observe a 3.2x TTFT improvement with the math.math dataset (90% cache hit), a 1.6x TTFT improvement with a Sonnet dataset with 2K prompt length (25% cache hit), or no TTFT improvement with the HumanEval dataset (0% cache hit). For more information, see :ref:`nxdi_prefix_caching` and :ref:`/libraries/nxd-inference/tutorials/trn2-llama3.3-70b-apc-tutorial.ipynb`.
* Disaggregated Inference (DI) support through vLLM (Beta). Disaggregated Inference is also known as disaggregated serving, disaggregated prefill, or p/d disaggregation. DI separates the prefill and decode phase of inference onto different hardware resources. DI can improve inter token latency (ITL) by by eliminating prefill stall in continuous batching, where decode is paused to perform prefill for a new incoming request. With DI, you can also scale prefill and decode resources independently to further improve performance. For more information, see :ref:`nxdi-disaggregated-inference`.
* Context parallelism in NeuronAttentionBase (Beta). Context parallelism distributes context processing across multiple NeuronCores. Context parallelism improves TTFT, particularly at higher sequence lengths where the number of KV heads is low. To use context parallelism, set ``cp_degree`` in NeuronConfig.
* Mixed-precision parameters in modeling code. This feature enables you to configure each module's dtype independently. To use mixed-precision parameters, set ``cast_type="as-declared"`` in NeuronConfig. Note: The default behavior (``cast_type="config"``) is to cast all parameters to the ``torch_dtype`` in NeuronConfig.
* Output logits when using on-device sampling. To output logits, enable ``output_logits`` in NeuronConfig. Note that this flag impacts performance and should only be used for debugging model logits.

**Other changes**

* Add support for PyTorch 2.7. This release includes support for PyTorch 2.5, 2.6, and 2.7.
* Upgrade ``transformers`` requirement from v4.48 to v4.51.
* Re-enable warmup on Trn2. NxD Inference disabled warmup on Trn2 in the previous release due to an issue that prevented certain model configurations from loading correctly. That issue is now fixed.
* Update the behavior of the ``attn_kernel_enabled`` attribute in NeuronConfig, which configures whether to use the flash attention kernel. Previously, ``True`` meant to enable in all cases where supported, and ``False`` meant to auto-enable where beneficial (defaults to ``False``). Now, ``attn_kernel_enabled=False`` disables the flash attention kernel in all cases. To use the previous auto-enable behavior, set ``attn_kernel_enabled=None``. The default value for ``attn_kernel_enabled`` is now ``None`` to retain the same default behavior as before.
* Enable ``--verify-hlo`` flag during compilation. Now, if an HLO is invalid, compilation will fail. Previously, in certain scenarios, the compiler would successfully compile invalid HLOs.
* Update the flash attention kernel strategy to use the attention kernel on Trn2 in all cases where it's supported. This change fixes an issue where certain context lengths failed to trace.
* Add ``logical_nc_config`` as an argument to the ``build_module`` and ``build_function`` test utilities, so you can use these utilities to test modules/functions for Trn2 using LNC2.
* Other minor fixes and improvements.

Bug Fixes
~~~~~~~~~

* Other minor fixes and improvements.

Known Issues
~~~~~~~~~~~~

* Increased Device Memory Usage for Certain Configurations: Certain model configurations require slightly more device memory than in previous releases. If your model used close to the maximum amount of device memory in previous releases, this increase could cause it to fail to load after you compile it with this release. This issue is most likely to affect Llama3.1-405B configurations that use a large number of buckets.


----

.. _nxd-inference-2-23-0-rn:

NxD Inference [0.3.5591] (Neuron 2.23.0 Release)
-------------------------------------------------

Date of Release: 05/20/2025

Improvements
~~~~~~~~~~~~~~~

* NxD Inference is now GA and out of beta in the Neuron 2.23 release.

**Features**

* Shard-on-load for weight sharding is now enabled by default. With this change, end-to-end compile and load time is reduced by up to 70% when sharding weights. This change significantly reduces compile time by skipping weight sharding and serialization during compile, but may lead to increased load time. For example, for Llama 3.1 405B, end-to-end compile and load time is reduced from 40 minutes to 12 minutes. For best load performance, you can continue to serialize sharded weights by enabling ``save_sharded_checkpoint`` in NeuronConfig. For more information, see :ref:`nxdi-weights-sharding-guide`.
* Neuron Persistent Cache. NxD Inference now supports Neuron Persistent Cache, which caches compiled model artifacts to reduce compilation times. For more information, see :ref:`nxdi-neuron-persistent-cache`.
* Support for an attention block kernel for token generation. This kernel performs QKV projections, RoPE, attention, and output projections. You can use this kernel with Llama3-like attention on Trn2 to improve token gen performance. To use this kernel, enable ``attn_block_tkg_nki_kernel_enabled`` in NeuronConfig.

  * This kernel can also update the KV cache in parallel with each layer's attention compute to further improve performance. This functionality hides the latency of the KV cache update that is otherwise done for all layers at once at the end of each token generation iteration. To enable in-kernel KV cache updates, enable ``attn_block_tkg_nki_kernel_cache_update`` in NeuronConfig. When in-kernel KV cache updating is enabled, you can also enable ``k_cache_transposed`` to further improve the performance.

* Automatically extract ``target_modules`` and ``max_lora_rank`` from LoRA checkpoints. You no longer need to set these arguments manually.
* Support fused residual add in the QKV kernel. This feature improves the performance of context encoding at short sequence lengths. To use this feature, enable the ``qkv_kernel_fuse_residual_add`` flag in NeuronConfig.

Breaking Changes
~~~~~~~~~~~~~~~~

* Remove ``set_async_mode(async_mode)`` from NeuronBaseForCausalLM, as this feature didn't work as intended. Async mode cannot be enabled or disabled after the model is loaded. To enable async mode, set ``async_mode=True`` in NeuronConfig.

Bug Fixes
~~~~~~~~~

* Disable warmup for Trn2. This change avoids an issue that prevents certain model configurations from loading correctly.

Known Issues
~~~~~~~~~~~~

* None reported for this release.