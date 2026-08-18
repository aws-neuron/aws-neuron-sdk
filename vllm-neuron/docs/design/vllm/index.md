# vLLM integration

Design documentation for how vLLM Neuron integrates with the vLLM core
framework. For user-facing configuration, see the
[features guide](../../guides/features-guide.md) and
[configuration options](../../guides/reference-configuration.md).

| Topic | Description |
| --- | --- |
| [vLLM integration design reference](vllm-integration-design-reference.md) | Plugin registration, scheduler, KV cache |
| [KV cache integration](vllm-integration-kv-cache.md) | KV cache integration points with vLLM |
| [Additional config](additional-config.md) | Additional configuration options |
| [Async scheduling and execution](async-scheduling-and-async-execution.md) | Async scheduling and execution design |
| [Context-length bucketing](decode-context-length-bucketing.md) | Reducing decode HBM reads |
| [Memory management](determine_available_memory_design.md) | Available memory determination |
| [KV cache quantization](kv-cache-quantization.md) | FP8 KV cache mechanics |
| [Metrics](metrics.md) | Production metrics design |
| [Neuron profiling](neuron-profiling.md) | Profiling integration |
| [Neuron scheduler](neuron-scheduler.md) | Holdback queue and admission control |
| [Block KV batching and padding](padding-batching-block-kv.md) | Block KV cache architecture |
| [Prefix caching](prefix-caching.md) | Prefill segmentation and KV reuse |
| [Pooling models](pooling-models.md) | Pooling model execution on Neuron |
| [Prompt embeddings](prompt-embeddings.md) | Prompt embedding support |
| [Structured outputs and tool calling](structured-outputs-and-tool-calling.md) | JSON, regex, grammar-constrained generation |
| [Disaggregated inference](disaggregated-inference.md) | DI architecture, NIXL transport, hybrid TP, DCP |

:::{toctree}
:maxdepth: 1
:hidden:

vllm-integration-design-reference
vllm-integration-kv-cache
additional-config
async-scheduling-and-async-execution
decode-context-length-bucketing
determine_available_memory_design
kv-cache-quantization
metrics
neuron-profiling
neuron-scheduler
padding-batching-block-kv
prefix-caching
pooling-models
prompt-embeddings
structured-outputs-and-tool-calling
disaggregated-inference
:::
