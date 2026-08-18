# vLLM Neuron Plugin (Beta) Documentation

The vLLM Neuron plugin brings the full vLLM serving stack to AWS Trainium
accelerators. It supports continuous batching, speculative decoding
(EAGLE3), disaggregated inference, structured outputs, multimodal models, and
more — all accessible through the standard `vllm serve` command and
OpenAI-compatible API.

For a high-level overview of inference on Neuron and help choosing the right
inference solution, see
[Inference on Neuron](/libraries/vllm-neuron/neuron-inference-overview). The
source code for the vLLM Neuron plugin is hosted in the
[vLLM Neuron GitHub repository](https://github.com/vllm-project/vllm-neuron).

---

## Get started

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Setup guide
:link: getting-started/setup-guide
:link-type: doc

Install and configure vLLM Neuron on a Trainium instance.
:::

:::{grid-item-card} Online serving quickstart
:link: getting-started/quickstart-online-serving
:link-type: doc

Launch an OpenAI-compatible API server and send your first chat request.
:::

:::{grid-item-card} Offline serving quickstart
:link: getting-started/quickstart-offline-serving
:link-type: doc

High-throughput batch inference with the `vllm.LLM` Python API.
:::

:::{grid-item-card} Migration from NxD Inference
:link: getting-started/migration-nxdi-to-vllm-neuron
:link-type: doc

Migrate existing NxDI deployments to the vLLM Neuron plugin.
:::

::::

## Deploy & serve

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Features guide
:link: guides/features-guide
:link-type: doc

Configure and tune all serving features — bucketing, quantization, DI, speculation, and more.
:::

:::{grid-item-card} Configuration reference
:link: guides/reference-configuration
:link-type: doc

All Neuron-specific options in `additional_config` and environment variables.
:::

:::{grid-item-card} Profiling workloads
:link: guides/how-to-profile-workloads
:link-type: doc

Capture Neuron Runtime profiles via built-in profiler endpoints.
:::

:::{grid-item-card} Model recipes
:link: model-recipes/index
:link-type: doc

Supported models and their feature tables.
:::

::::

## Model Recipes

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Deploy Llama 3
:link: model-recipes/llama-3
:link-type: doc

Model recipe for the Llama 3 family (1B, 8B, 70B) on Trn2/Trn3.
:::

:::{grid-item-card} Deploy GPT-OSS
:link: model-recipes/gpt-oss
:link-type: doc

Model recipe for GPT-OSS 20B and 120B (MoE) on Trn2/Trn3.
:::

:::{grid-item-card} Deploy Qwen3-VL
:link: model-recipes/qwen3-vl
:link-type: doc

Model recipe for Qwen3-VL 32B (multimodal) on Trn2/Trn3.
:::

:::{grid-item-card} Deploy Qwen3-Embedding 8B
:link: model-recipes/qwen3-embedding-8b
:link-type: doc

Model recipe for Qwen3-Embedding 8B (pooling / embeddings) on Trn2/Trn3.
:::

::::

## Tutorials

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} EAGLE3 speculative decoding (Llama 3.1)
:link: tutorials/tutorial-eagle3-speculative-decoding-llama-3-1
:link-type: doc

Run Llama 3.1 8B with an EAGLE3 draft model for higher throughput.
:::

:::{grid-item-card} EAGLE3 speculative decoding (GPT-OSS)
:link: tutorials/tutorial-eagle3-speculative-decoding-gpt-oss
:link-type: doc

Run GPT-OSS-120B with an EAGLE3 draft model for higher throughput.
:::

:::{grid-item-card} GPT-OSS deployment tutorial
:link: tutorials/tutorial-gpt-oss
:link-type: doc

End-to-end deployment of GPT-OSS on Trn2/Trn3.
:::

:::{grid-item-card} Qwen3-VL multimodal tutorial
:link: tutorials/tutorial-qwen3-vl-32b
:link-type: doc

Deploy Qwen3-VL 32B for multimodal inference.
:::

:::{grid-item-card} Disaggregated inference: 1P1D and xPyD
:link: tutorials/tutorial-di-1p1d-xpyd
:link-type: doc

Configure disaggregated inference topologies.
:::

:::{grid-item-card} Disaggregated encoder: 1E1PD and xEyPD
:link: tutorials/tutorial-epd-1e-1pd-xeypd
:link-type: doc

Configure encoder-disaggregated (EPD) multimodal topologies.
:::

:::{grid-item-card} Prefix caching benchmark
:link: tutorials/tutorial-prefix-caching-gpt-oss-benchmarking
:link-type: doc

Measure TTFT improvement from prefix caching with GPT-OSS.
:::

:::{grid-item-card} Deploy Qwen3-Embedding-8B
:link: tutorials/tutorial-qwen3-embedding-8b
:link-type: doc

Serve embeddings via `/v1/embeddings` with a pooling model.
:::

::::

## Model development

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Onboard a new model
:link: model-dev/onboarding-models
:link-type: doc

Implement and register a new architecture with vLLM.
:::

:::{grid-item-card} Onboard a vision-language model
:link: model-dev/onboarding-vlm-models
:link-type: doc

Add a vision encoder tower on top of the text-decoder flow.
:::

:::{grid-item-card} Optimizing a vision-language model
:link: model-dev/optimizing-vlm-models
:link-type: doc

Roofline, sharding, and profiling to optimize a VLM.
:::

:::{grid-item-card} CPU development workflow
:link: model-dev/cpu-development
:link-type: doc

Develop and test without Neuron hardware.
:::

:::{grid-item-card} NKI CPU simulator
:link: model-dev/nki_cpu_simulator
:link-type: doc

Validate NKI kernel correctness on CPU.
:::

:::{grid-item-card} Debugging model code
:link: model-dev/debugging
:link-type: doc

Use pdb and print statements to inspect model execution.
:::

:::{grid-item-card} Debugging accuracy issues
:link: model-dev/accuracy-debugging-guide
:link-type: doc

Methodology for isolating where accuracy drift is introduced.
:::

:::{grid-item-card} Accuracy debugger tools
:link: model-dev/how-to-use-accuracy-debugger
:link-type: doc

Run the automated debugger pipeline and interpret results.
:::

::::

## Concepts & architecture

### Parallelism

| Topic | Description |
|---|---|
| [Attention DP](design/parallelism/attention_dp.md) | Sharding Q/O weights across DP groups |
| [Component DP sharding](design/parallelism/component_dp_sharding.md) | Per-component independent sharding |
| [Data parallelism](design/parallelism/data_parallelism.md) | Data parallelism overview |
| [Decode Context Parallelism](design/parallelism/dcp.md) | KV cache sequence sharding for long contexts |
| [Expert parallelism](design/parallelism/expert_parallelism.md) | Expert parallelism for MoE |
| [Tensor parallelism](design/parallelism/tensor_parallelism.md) | Tensor parallelism overview |
| [Vision encoder parallelism](design/parallelism/vision_encoder_parallelism.md) | Independent TP/DP for vision encoders |

### Speculation

| Topic | Description |
|---|---|
| [Speculative decoding](design/speculation/speculative_decoding_design.md) | EAGLE3 internals on Neuron |

### Multimodal

| Topic | Description |
|---|---|
| [Block packing attention](design/multimodal/block_packing_attention.md) | FFD block packing for multi-image attention efficiency |
| [On-Device Encoder Cache](design/multimodal/on_device_encoder_cache.md) | Block-based on-device cache for vision encoder outputs |
| [M-RoPE](design/multimodal/mrope.md) | Spatial position embeddings for VLMs |

### vLLM integration

| Topic | Description |
|---|---|
| [vLLM integration design reference](design/vllm/vllm-integration-design-reference.md) | Plugin registration, scheduler, KV cache |
| [KV cache integration](design/vllm/vllm-integration-kv-cache.md) | KV cache integration points with vLLM |
| [Additional config](design/vllm/additional-config.md) | Additional configuration options |
| [Async scheduling and execution](design/vllm/async-scheduling-and-async-execution.md) | Async scheduling and execution design |
| [Context-length bucketing](design/vllm/decode-context-length-bucketing.md) | Reducing decode HBM reads |
| [Memory management](design/vllm/determine_available_memory_design.md) | Available memory determination |
| [KV cache quantization](design/vllm/kv-cache-quantization.md) | FP8 KV cache mechanics |
| [Metrics](design/vllm/metrics.md) | Production metrics design |
| [Neuron profiling](design/vllm/neuron-profiling.md) | Profiling integration |
| [Neuron scheduler](design/vllm/neuron-scheduler.md) | Holdback queue and admission control |
| [Block KV batching and padding](design/vllm/padding-batching-block-kv.md) | Block KV cache architecture |
| [Prefix caching](design/vllm/prefix-caching.md) | Prefill segmentation and KV reuse |
| [Pooling models](design/vllm/pooling-models.md) | Pooling model execution on Neuron |
| [Prompt embeddings](design/vllm/prompt-embeddings.md) | Prompt embedding support |
| [Structured outputs and tool calling](design/vllm/structured-outputs-and-tool-calling.md) | JSON, regex, grammar-constrained generation |
| [Disaggregated inference](design/vllm/disaggregated-inference.md) | DI architecture, NIXL transport, hybrid TP, DCP |

### Framework

| Topic | Description |
|---|---|
| [Async execution](design/framework/async_execution.md) | Async execution double buffering |
| [Checkpoint loading](design/framework/checkpoint_loading.md) | Weight loading with parallelism |
| [Model bringup](design/framework/model_bringup.md) | Model bringup workflow |
| [Model factory](design/framework/model_factory_design.md) | Model registry and factory pattern |

### Compilation

| Topic | Description |
|---|---|
| [Compilation cache](design/compilation/compilation_cache.md) | Compilation cache (hit/miss, remote store) |
| [CPU compilation](design/compilation/cpu_compilation.md) | Ahead-of-time CPU compilation (NEFF extraction) |
| [FX passes architecture](design/compilation/fx_passes_design.md) | FX passes architecture |
| [Aliasing output rewrite pass](design/compilation/aliasing_output_rewrite_pass.md) | Aliasing output rewrite pass |
| [Device rewriting FX pass](design/compilation/device_rewriting_fx_pass.md) | Device rewriting FX pass |
| [Inplace to out-of-place pass](design/compilation/inplace_to_outofplace_pass.md) | Inplace to out-of-place rewrite |

### Accuracy

| Topic | Description |
|---|---|
| [Accuracy debugging design](design/accuracy/accuracy_debugging_design.md) | Accuracy debugging framework |
| [KV cache analysis](design/accuracy/kv_cache_analysis_design.md) | KV cache analysis |
| [Logit validation](design/accuracy/logit_validation_design.md) | Logit validation |
| [Module test guidelines](design/accuracy/module_test_guidelines.md) | Module test guidelines |
| [Tensor capture](design/accuracy/tensor_capture_design.md) | Tensor capture |
| [Tensor compare](design/accuracy/tensor_compare_design.md) | Tensor compare |
| [Tensor replacement](design/accuracy/tensor_replacement_design.md) | Tensor replacement |
