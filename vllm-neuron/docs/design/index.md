# Concepts & architecture

How vLLM Neuron works under the hood — parallelism strategies, plugin integration, speculative decoding internals, and the accuracy validation framework.

## Parallelism

| Topic | Description |
| --- | --- |
| [Attention DP](parallelism/attention_dp.md) | Sharding Q/O weights across DP groups |
| [Component DP sharding](parallelism/component_dp_sharding.md) | Per-component independent sharding |
| [Data parallelism](parallelism/data_parallelism.md) | Data parallelism overview |
| [Decode Context Parallelism](parallelism/dcp.md) | KV cache sequence sharding for long contexts |
| [Encoder parallelism](parallelism/encoder_parallelism.md) | Independent TP/DP for multimodal encoders |
| [Expert parallelism](parallelism/expert_parallelism.md) | Expert parallelism for MoE |
| [Tensor parallelism](parallelism/tensor_parallelism.md) | Tensor parallelism overview |

## Speculation

| Topic | Description |
| --- | --- |
| [Speculative decoding (EAGLE3)](speculation/speculative_decoding_design.md) | EAGLE3 autoregressive drafter internals on Neuron |
| [Speculative decoding (DFlash)](speculation/dflash_speculative_decoding_design.md) | DFlash block-parallel drafter internals on Neuron |

## Multimodal

| Topic | Description |
| --- | --- |
| [Block Packing Attention](multimodal/block_packing_attention.md) | FFD block packing for multi-image attention efficiency |
| [On-Device Encoder Cache](multimodal/on_device_encoder_cache.md) | Block-based on-device cache for encoder outputs |
| [Encoder Cache Metrics](multimodal/encoder_cache_metrics.md) | mm / prefix / encoder cache layers and their Prometheus metrics |
| [M-RoPE](multimodal/mrope.md) | Spatial position embeddings for VLMs |

## vLLM integration

| Topic | Description |
| --- | --- |
| [vLLM integration design reference](vllm/vllm-integration-design-reference.md) | Plugin registration, scheduler, KV cache |
| [KV cache integration](vllm/vllm-integration-kv-cache.md) | KV cache integration points with vLLM |
| [Additional config](vllm/additional-config.md) | Additional configuration options |
| [Async scheduling and execution](vllm/async-scheduling-and-async-execution.md) | Async scheduling and execution design |
| [Context-length bucketing](vllm/decode-context-length-bucketing.md) | Reducing decode HBM reads |
| [Memory management](vllm/determine_available_memory_design.md) | Available memory determination |
| [KV cache quantization](vllm/kv-cache-quantization.md) | FP8 KV cache mechanics |
| [Metrics](vllm/metrics.md) | Production metrics design |
| [Neuron profiling](vllm/neuron-profiling.md) | Profiling integration |
| [Neuron scheduler](vllm/neuron-scheduler.md) | Holdback queue and admission control |
| [Block KV batching and padding](vllm/padding-batching-block-kv.md) | Block KV cache architecture |
| [Prefill processing and prefix caching](vllm/prefix-caching.md) | Prefill segmentation and KV reuse |
| [Pooling models](vllm/pooling-models.md) | Pooling model execution on Neuron |
| [Prompt embeddings](vllm/prompt-embeddings.md) | Prompt embedding support |
| [Structured outputs and tool calling](vllm/structured-outputs-and-tool-calling.md) | JSON, regex, grammar-constrained generation |
| [Disaggregated inference](vllm/disaggregated-inference.md) | DI architecture, NIXL transport, hybrid TP, DCP |

## Framework

| Topic | Description |
| --- | --- |
| [Async execution](framework/async_execution.md) | Async execution double buffering |
| [Model factory](framework/model_factory_design.md) | Model registry and factory pattern |

## Compilation

| Topic | Description |
| --- | --- |
| [How vLLM is integrated with TorchNeuron](compilation/how_vllm_is_integrated_with_torchneuron.md) | How vLLM Neuron uses the TorchNeuron `torch.compile` lowering |
| [Compilation cache](compilation/compilation_cache.md) | Compilation cache (hit/miss, remote store) |
| [CPU compilation](compilation/cpu_compilation.md) | Ahead-of-time CPU compilation (NEFF extraction) |
| [FX passes architecture](compilation/fx_passes_design.md) | FX passes architecture |
| [Aliasing output rewrite pass](compilation/aliasing_output_rewrite_pass.md) | Aliasing output rewrite pass |
| [Device rewriting FX pass](compilation/device_rewriting_fx_pass.md) | Device rewriting FX pass |
| [Inplace to out-of-place pass](compilation/inplace_to_outofplace_pass.md) | Inplace to out-of-place rewrite |
| [SPMD compilation](compilation/spmd_compilation.md) | Collapse per-rank compiles into a single shared NEFF |

## Accuracy

| Topic | Description |
| --- | --- |
| [Accuracy debugging design](accuracy/accuracy_debugging_design.md) | Accuracy debugging framework |
| [KV cache analysis](accuracy/kv_cache_analysis_design.md) | KV cache analysis |
| [Logit validation](accuracy/logit_validation_design.md) | Logit validation |
| [Module test guidelines](accuracy/module_test_guidelines.md) | Module test guidelines |
| [Tensor capture](accuracy/tensor_capture_design.md) | Tensor capture |
| [Tensor compare](accuracy/tensor_compare_design.md) | Tensor compare |
| [Tensor replacement](accuracy/tensor_replacement_design.md) | Tensor replacement |

:::{toctree}
:maxdepth: 1
:hidden:

parallelism/index
speculation/index
multimodal/index
vllm/index
framework/index
compilation/index
accuracy/index
:::
