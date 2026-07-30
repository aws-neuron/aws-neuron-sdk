# Configuration options

<!-- meta: description: Complete reference for all Neuron-specific
configuration parameters available through additional_config and environment
variables. -->
<!-- meta: date_updated: 2026-07-20 -->
<!-- Content type: reference-general -->

## Overview

All Neuron-specific options are passed through `additional_config` with
two config namespaces:

- **`neuron_config`** — text model settings (compilation, sampling,
  quantization, KV cache, scheduling)
- **`vision_neuron_config`** — vision encoder settings (multimodal models only)

```python
from vllm import LLM

llm = LLM(
    model="openai/gpt-oss-20b",
    max_model_len=4096,
    tensor_parallel_size=8,
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096],
            "num_seqs_buckets": [1, 2, 4, 8],
        }
    }
)
```

Or via CLI:

```bash
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [256, 512, 1024]}}'
```

## Compilation options

For conceptual overview and trade-offs, see [Bucketing and dynamic shapes](features-guide.md#bucketing-and-dynamic-shapes) and [Compilation](features-guide.md#compilation).

| Option | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `num_batched_tokens_buckets` | list[int] | Power-of-2 from 128 to `max_num_batched_tokens` | Compiled prefill token counts. Inputs padded to nearest bucket. Fewer = faster startup, more padding. Largest must equal `max_num_batched_tokens`. When segmented prefill is enabled, must match `kv_segment_size_buckets` (kernel constraint). |
| `num_seqs_buckets` | list[int] | Power-of-2 from 1 to `max_num_seqs` | Compiled decode batch sizes. Requests batched to smallest bucket >= current size. Largest must equal `max_num_seqs`. |
| `kv_segment_size_buckets` | list[int] or null | null (disabled) | KV segment sizes for segmented attention kernel. Values must be in {512, 1024, 2048, 4096}, divisible by `block_size`. |
| `decode_context_length_buckets` | list[int] or null | null (disabled) | Second decode bucketing dimension. Compiles smaller NEFFs sized to typical context lengths instead of `max_model_len`. Values must be ascending, < `max_model_len`, divisible by 128. |

## KV cache options

For conceptual overview, see [Prefix caching](features-guide.md#prefix-caching).

Prefix caching is enabled by default and controlled through standard vLLM
parameters (not `neuron_config`). The options below are vLLM parameters passed
directly to `LLM(...)` or on the CLI.

| Option | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `enable_prefix_caching` (vLLM parameter) | bool | true | Reuse KV cache from shared prompt prefixes. On by default; pass `--no-enable-prefix-caching` to disable. |
| `num_gpu_blocks_override` (vLLM parameter) | int | -- | Number of KV cache blocks. More blocks retain more prefixes for reuse. Not nested under `neuron_config`. |
| `block_size` (vLLM parameter) | int | 32 | KV cache block size in tokens. Not nested under `neuron_config`. |
| `kv_cache_dtype` (vLLM parameter) | str | "auto" | KV cache data type. Set to `"fp8"` for FP8 KV cache quantization (~50% memory reduction). Not nested under `neuron_config`. Requires calibrated checkpoint scales. |

```bash
# FP8 KV cache via CLI
vllm serve openai/gpt-oss-20b --kv-cache-dtype fp8
```

## Sampling options

For conceptual overview, see [On-device sampling](features-guide.md#on-device-sampling).

| Option | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `on_device_sampling_config` | dict or null | null (CPU sampling) | Enable on-device sampling. Supported sub-parameters: `temperature`, `top_k`, `top_p`, `all_greedy`. Required for async scheduling, speculative decoding, and structured output enforcement. |

```python
# Greedy on-device sampling
"on_device_sampling_config": {"all_greedy": True}

# Temperature-based
"on_device_sampling_config": {"temperature": "0.7"}
```

## Scheduling options

For conceptual overview, see [Async scheduling](features-guide.md#async-scheduling).

| Option | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `--no-async-scheduling` (CLI flag) | bool | true (enabled) | Disable async scheduling. Async overlaps CPU work with device execution. Requires on-device sampling. |

## Prompt embedding options

For conceptual overview, see [Prompt embeddings](features-guide.md#prompt-embeddings).

| Option | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `enable_prompt_embeds` (vLLM parameter) | bool | false | Enable passing precomputed embedding tensors instead of token IDs. Not nested under `neuron_config`. Requests include `prompt_embeds` with shape `[seq_len, hidden_size]`. |

```bash
vllm serve openai/gpt-oss-20b --enable-prompt-embeds
```

## Quantization options

For conceptual overview, see [Quantization](features-guide.md#quantization).

| Option | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `quantization` | str | -- | Neuron-specific weight quantization (e.g., `"mxfp4"` for gpt-oss on Trn3, `"mxfp8"`). Not the vLLM `--quantization` flag. FP8 static weight quantization is not set here — it is driven by the checkpoint's `quantization_config`. |

:::{warning}
Do not set vLLM's `--quantization` flag to `neuron_quant`. Keep it
unset and configure Neuron-specific quantization through `neuron_config`,
or supply a calibrated checkpoint for FP8 static weight quantization.
:::

## Vision encoder options (vision_neuron_config)

For multimodal models (e.g., Qwen3-VL). See [Multimodal support](features-guide.md#multimodal-support).

| Option | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `num_vision_tokens_buckets` | list[int] | auto | Vision encoder buckets over the number of vision patches per encoder forward pass. Scales with image count and resolution. |
| `vision_attention_block_size` | int | 2048 | Attention block size for the vision encoder. Must be large enough to hold the largest single image; if `mm_processor_kwargs.max_pixels` implies more tokens, it is auto-raised to fit (with a warning). |
| `max_vision_seq_len` | int | -- | Caps the largest auto-generated bucket, limiting how many buckets compile. Use it to bound compile time/memory when your workload won't need the full token ceiling. Ignored if `num_vision_tokens_buckets` is set explicitly. |

If `num_vision_tokens_buckets` is omitted, buckets are auto-generated at
startup as a power-of-2 progression from `vision_attention_block_size` up to a
token ceiling derived from the serving config, so multimodal models serve with
no vision config; an explicit list always takes precedence. Set
`max_vision_seq_len` to cap that ceiling and compile fewer buckets.

The scheduler may batch images from multiple requests into one encoder
forward pass, so size buckets for total images processed together:

| `num_vision_tokens_buckets` | Approximate capacity |
|----|---|
| `[2048]` | 1–2 images at 448×448 px |
| `[2048, 8192]` | Up to ~8 images |
| `[2048, 8192, 20480]` | Up to ~20 images |

## Environment variables

### Compilation and caching

For conceptual overview, see [Compilation](features-guide.md#compilation).

| Variable | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `VLLM_CACHE_ROOT` | str | `~/.cache/vllm` | Root directory for all vLLM caches. Compiled model artifacts (NEFFs) are stored under `$VLLM_CACHE_ROOT/neuron/compile_cache`. Change this to persist or relocate the compile cache. |
| `VLLM_NEURON_CPU_COMPILE` | bool | 0 | Enable CPU-only compilation mode (compile NEFFs without Neuron hardware). |
| `NEURON_PLATFORM_TARGET_OVERRIDE` | str | -- | Target platform for CPU compile mode (e.g., `trn2`). Required when `VLLM_NEURON_CPU_COMPILE=1`. |
| `VLLM_NEURON_PARALLEL_COMPILE_WORKERS` | int | -- | Number of parallel compilation workers. |
| `VLLM_NEURON_REMOTE_CACHE` | str | -- | Path to NFS/FSx mount for shared persistent cache across nodes. |
| `VLLM_NEURON_DISABLE_COMPILE_CACHE` | bool | 0 | Disable compilation cache entirely. Forces recompilation on every startup. |
| `VLLM_NEURON_COMPILATION_TIMEOUT` | int | -- | Timeout in seconds for individual NEFF compilation. |
| `VLLM_NEURON_DISABLE_WARMUP_COMPILE` | bool | 0 | Treat cache miss as fatal error. Use when all graphs must be pre-compiled. |

### What triggers recompilation

Your compilation cache is invalidated (causing a full recompile on next startup) when any of the following changes:

- Neuron SDK version upgrade (`neuronxcc`, `torch_neuronx`, or NKI)
- Model architecture changes (different model code producing a different FX graph)
- Different input shapes (new bucket sizes)
- Different compiler flags (e.g., changing `--optimization-level`)
- Different hardware target (e.g., moving from trn2 to trn3)
- Different tensor parallel degree

Restarts on the same instance with the same versions will hit the local cache — no recompilation. New nodes with a remote cache configured will fetch from the remote store — also no recompilation.

### Runtime and execution

| Variable | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `VLLM_NEURON_CPU_MODE` | bool | 0 | Enable CPU fallback mode (no Neuron hardware needed). For development/testing only. |
| `VLLM_NEURON_LOG_LEVEL` | str | INFO | Logging level for vLLM Neuron components. |
| `VLLM_NEURON_DISABLE_NKI_KERNELS` | bool | 0 | Disable all NKI kernels, forcing torch fallback paths. |

### Memory management

For conceptual overview, see [Memory management](features-guide.md#memory-management).

| Variable | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` | float | 0.30 | Fraction of total GMU budget to cap KV cache allocation. |
| `VLLM_NEURON_MIN_KV_BUDGET_GIB` | float | 1.0 | Minimum KV budget in GiB. Server fails fast if below. |

### Metrics

| Variable | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `PROMETHEUS_MULTIPROC_DIR` | str | auto-created tempdir | Directory for prometheus_client multiprocess mode. Wipe between server restarts. |

## Complete configuration example

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="openai/gpt-oss-20b",
    max_model_len=4096,
    max_num_seqs=8,
    tensor_parallel_size=8,
    block_size=128,
    kv_cache_dtype="fp8",
    enable_prompt_embeds=True,
    additional_config={
        "neuron_config": {
            # Compilation buckets
            "num_seqs_buckets": [1, 2, 4, 8],

            # Segmented prefill buckets
            "kv_segment_size_buckets": [2048],
            "num_batched_tokens_buckets": [2048],

            # On-device sampling (required for async scheduling)
            "on_device_sampling_config": {
                "all_greedy": True
            },
        }
    }
)

prompts = ["What is machine learning?"]
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
outputs = llm.generate(prompts, sampling_params)
```

Equivalent CLI:

```bash
export VLLM_CACHE_ROOT=/opt/neuron-cache
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=4
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.35

vllm serve openai/gpt-oss-20b \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --tensor-parallel-size 8 \
    --block-size 128 \
    --kv-cache-dtype fp8 \
    --enable-prompt-embeds \
    --additional-config '{
        "neuron_config": {
            "num_seqs_buckets": [1, 2, 4, 8],
            "kv_segment_size_buckets": [2048],
            "num_batched_tokens_buckets": [2048],
            "on_device_sampling_config": {"all_greedy": true}
        }
    }' \
    --port 8000
```

## Configuration tips

### Startup time vs flexibility trade-off

- Fewer buckets = faster startup, but less flexibility
- More buckets = slower startup, but handles more scenarios efficiently

**Recommended approach:**

1. Start with minimal configuration (single bucket) during development
2. Profile your workload to understand prompt length distribution
3. Add buckets matching your 25th, 50th, 75th, and 95th percentile lengths
4. Remove unused buckets to speed up startup

### Memory considerations

- Each compiled bucket consumes device memory
- Monitor memory usage when adding many buckets
- KV cache FP8 halves cache footprint but requires calibrated scales
- The `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` default (0.30) is
  conservative; increase it for models with lower non-KV overhead

### Async scheduling guidance

- Keep async scheduling enabled (default) for production workloads
- Disable only for debugging or when CPU sampling is required
- Longer responses benefit more from async scheduling
- Smaller models see larger relative speedups from async overlap

### Prefix caching constraints

- `kv_segment_size_buckets` and `num_batched_tokens_buckets` must
  currently be identical (kernel constraint)
- Only a single segment size is supported at this time
- Block size must divide evenly into the segment size

## Troubleshooting

### "Request exceeds largest bucket" error

Your input exceeds the largest `num_batched_tokens_buckets` value.
Either add a larger bucket or reduce your input length.

### Compilation during inference

If you see compilation messages after warmup, you have inputs that
do not match any configured bucket. Check your bucket configuration
covers your actual workload.

### "Only support greedy sampling" error

Currently, on-device sampling only supports `temperature=0`. For
non-greedy sampling, remove `on_device_sampling_config` to use CPU
sampling.

### NCC_EVRF009 compiler error

KV cache allocation exceeds compile-safe limits. Solutions:

- Reduce `gpu_memory_utilization`
- Lower `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION`
- Reduce `max_model_len` or `max_num_seqs`

### "KV cache budget below minimum threshold" error

Computed KV budget is less than `VLLM_NEURON_MIN_KV_BUDGET_GIB`
(default 1.0 GiB). Increase `gpu_memory_utilization` or reduce
model/batch parameters.

### Async scheduling not engaging

Async scheduling requires:

- On-device sampling enabled (`on_device_sampling_config` set)
- `--no-async-scheduling` NOT passed
- Consecutive decode steps with unchanged batch composition

Check logs for "batch composition changed" if you see frequent sync fallbacks.

## Related reference

- [Features guide](features-guide.md) — feature descriptions with
  enable/disable guidance
- For supported models and features, see the [README](https://github.com/vllm-project/vllm-neuron#supported-models)
  and [model cards](../model-recipes/index.md).
