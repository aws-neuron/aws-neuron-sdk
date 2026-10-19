# Configuration options

<!-- meta: description: Complete reference for all Neuron-specific
configuration parameters available through additional_config and environment
variables. -->
<!-- meta: date_updated: 2026-10-16 -->
<!-- Content type: reference-general -->

## Overview

All Neuron-specific options are passed through `additional_config` with
two config namespaces:

- **`neuron_config`** — text model settings (compilation, sampling,
  quantization, KV cache, scheduling)
- **`encoder_neuron_config`** — multimodal encoder settings (multimodal models only)

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    max_model_len=4096,
    tensor_parallel_size=32,
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
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 32 \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [256, 512, 1024]}}'
```

## Compilation options

For conceptual overview and trade-offs, see [Bucketing and dynamic shapes](features-guide.md#bucketing-and-dynamic-shapes) and [Compilation](features-guide.md#compilation).

| Option | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `num_batched_tokens_buckets` | list[int] | Power-of-2 from 128 to `max_num_batched_tokens` | Compiled prefill token counts. Inputs padded to nearest bucket. Fewer = faster startup, more padding. Largest must equal `max_num_batched_tokens`. When segmented prefill is enabled, must match `kv_segment_size_buckets` (kernel constraint). |
| `num_seqs_buckets` | list[int] | Power-of-2 from 1 to `max_num_seqs` | Compiled decode batch sizes. Requests batched to smallest bucket >= current size. Largest must equal `max_num_seqs`. |
| `kv_segment_size_buckets` | list[int] or null | null (disabled) | KV segment sizes for segmented attention kernel. Values must be in {512, 1024, 2048, 4096, 8192}, divisible by `block_size`. |
| `decode_context_length_buckets` | list[int] or null | null (disabled) | Second decode bucketing dimension. Compiles smaller NEFFs sized to typical context lengths instead of `max_model_len`. Values must be ascending, < `max_model_len`, divisible by 128. |

## KV cache options

For conceptual overview, see [Prefix caching](features-guide.md#prefix-caching).

Prefix caching is enabled by default and controlled through standard vLLM
parameters (not `neuron_config`). The options below are vLLM parameters passed
directly to `LLM(...)` or on the CLI.

| Option | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `gpu_memory_utilization` (vLLM parameter) | float | vLLM default | Applies to default fractional-cap sizing. Explicit headroom mode sets this value to 1.0. |
| `enable_prefix_caching` (vLLM parameter) | bool | true | Reuse KV cache from shared prompt prefixes. On by default; pass `--no-enable-prefix-caching` to disable. |
| `num_gpu_blocks_override` (vLLM parameter) | int | -- | Number of KV cache blocks. More blocks retain more prefixes for reuse. Not nested under `neuron_config`. |
| `block_size` (vLLM parameter) | int | 32 | KV cache block size in tokens. Not nested under `neuron_config`. |
| `kv_cache_dtype` (vLLM parameter) | str | "auto" | KV cache data type. Set to `"fp8"` for FP8 KV cache quantization (~50% memory reduction). Not nested under `neuron_config`. Requires calibrated checkpoint scales. |

```bash
# FP8 KV cache via CLI
vllm serve meta-llama/Llama-3.3-70B-Instruct --kv-cache-dtype fp8
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
vllm serve meta-llama/Llama-3.3-70B-Instruct --enable-prompt-embeds
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

## Multimodal encoder options (encoder_neuron_config)

For multimodal models, for example Qwen3-VL and Qwen3-ASR, see
[Multimodal support](features-guide.md#multimodal-support).

| Option | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `num_encoder_input_buckets` | list[int] | auto | Compiled input-size buckets for one encoder forward pass. The model sets the unit: a vision encoder counts pre-merge tokens, and an audio encoder counts blocks. Scales with the item count and the size of each item. |
| `encoder_attention_block_size` | int | 2048 | Attention block size for the encoder. For vision, it must hold the largest image. For audio, it must hold one inference window. See the sizing instructions below. |
| `max_encoder_seq_len` | int | -- | Caps the largest auto-generated bucket, limiting how many buckets compile. Use it to bound compile time/memory when your workload won't need the full token ceiling. Ignored if `num_encoder_input_buckets` is set explicitly. |
| `tp_size` | int | 1 | Tensor parallel degree for the encoder. Whichever of `tp_size`/`dp_size` is left at 1 is inferred from `world_size`; if both are set, `tp_size * dp_size` must equal `world_size`. |
| `dp_size` | int | 1 | Data parallel degree for the encoder — encoder blocks are scattered across `dp_size` ranks. Same inference rule as `tp_size`. |
| `encoder_cache_num_blocks` | int or null | null (auto) | Block count for the on-device encoder cache. Auto-derived from the scheduler's `encoder_cache_size` budget. Acts as a floor, not an override: a value below the derived minimum is raised to it (logged). Set it above that minimum for workloads with many small images — the scheduler sizes by token count and is unaware of per-block padding waste, so each small item still consumes a whole block. |
| `encoder_cache_min_hold_time_ms` | float or null | null (auto: 0 monolithic, 100 EPD) | Minimum time a written cache block stays allocated before it can be freed. Guards against reuse while a remote reader is still pulling the data, which is why EPD defaults to 100 ms and a single-server deployment to 0. |

For vision models, buckets are auto-generated when
`num_encoder_input_buckets` is omitted. The buckets use a power-of-2 progression
from `encoder_attention_block_size` to a token ceiling from the serving
configuration. An explicit list always takes precedence. Set
`max_encoder_seq_len` to cap the ceiling and compile fewer buckets.

(audio-encoder-sizing)=
For Qwen3-ASR, `num_encoder_input_buckets` contains encoder block counts. Use
`encoder_attention_block_size: 128`, which holds one 104-token window
produced from 800 mel frames. At batch size 1, `[4]` fits one 30-second clip.
The transcription endpoint chunks longer audio automatically. Increase the
bucket for larger batches or longer offline input.

The scheduler may batch images from multiple requests into one encoder
forward pass, so size buckets for total images processed together:

| `num_encoder_input_buckets` | Approximate capacity |
|----|---|
| `[2048]` | 1–2 images at 448×448 px |
| `[2048, 8192]` | Up to ~8 images |
| `[2048, 8192, 20480]` | Up to ~20 images |

For details, see [Encoder parallelism](../design/parallelism/encoder_parallelism.md).

### EPD construction roles

These two flags select which half of a vision-language model an engine builds,
and are what split a multimodal deployment into an encoder pool and a
prefill+decode pool. See
[Encoder-disaggregated inference (EPD)](features-guide.md#encoder-disaggregated-inference-epd).

| Option | Where it goes | Type | Default | Description |
| ---- | ---- | ---- | ---- | ---- |
| `mm_encoder_only` | `--mm-encoder-only` (CLI flag) | bool | false | Build **vision-only** — the Vision Encoder (VE) pool. The language model is dropped, so no prefill/decode warmup runs. This is an upstream vLLM flag, not a `neuron_config` field. |
| `mm_language_model_only` | top level of `additional_config` | bool | false | Build **language-only** — the Prefill+Decode (PD) pool. The vision tower is dropped, so no encoder warmup runs. Plugin-specific: upstream has no equivalent because its disaggregated-encoder consumer loads the full model and leaves the vision tower unused. |

:::{important}
`mm_language_model_only` sits at the **top level** of `additional_config`, not
inside `encoder_neuron_config` or `neuron_config`, and there is no
`--mm-language-model-only` CLI flag. Nesting it produces no error — the flag is
silently ignored and the engine builds both towers, wasting device memory on a
vision tower it never runs.

```bash
# Correct — top level, sibling of neuron_config / encoder_neuron_config.
--additional-config '{"neuron_config": {...}, "encoder_neuron_config": {...}, "mm_language_model_only": true}'
```

:::

The two are mutually exclusive: a pool is either vision-only or language-only.
Setting both raises `ValueError` at startup. Leaving both unset is the default
monolithic path, where one engine builds and runs both towers.

## Environment variables

### Compilation and caching

For conceptual overview, see [Compilation](features-guide.md#compilation).

| Variable | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `VLLM_NEURON_BACKEND` | str | -- | Compilation route. Unset or `neuron_native` selects the native route. `vllm_neuron` selects the deprecated XLA route. An invalid value logs a warning and selects the native route. See [How to choose a compilation route](how-to-choose-compilation-route.md). |
| `NEURON_COMPILED_ARTIFACTS` | str | -- | Path to cache/load compiled models. Skips recompilation when valid artifacts exist. |
| `VLLM_NEURON_CPU_COMPILE` | bool | 0 | Enable CPU-only compilation mode (compile NEFFs without Neuron hardware). |
| `NEURON_PLATFORM_TARGET_OVERRIDE` | str | -- | Target platform for CPU compile mode (e.g., `trn2`). Required when `VLLM_NEURON_CPU_COMPILE=1`. |
| `NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS` | int | -- | Number of parallel compilation workers. |
| `NEURON_LIBTORCH_REMOTE_CACHE` | str | -- | Path to NFS/FSx mount for shared persistent cache across nodes. |
| `NEURON_LIBTORCH_DISABLE_COMPILE_CACHE` | bool | 0 | Disable compilation cache entirely. Forces recompilation on every startup. |
| `NEURON_LIBTORCH_COMPILATION_TIMEOUT` | int | -- | Timeout in seconds for individual NEFF compilation. |
| `VLLM_NEURON_DISABLE_WARMUP_COMPILE` | bool | 0 | Treat cache miss as fatal error. Use when all graphs must be pre-compiled. |
| `VLLM_CACHE_ROOT` | str | `~/.cache/vllm` | Root directory for vLLM cache storage. |

### What triggers recompilation

Your compilation cache is invalidated (causing a full recompile on next startup) when any of the following changes:

- Neuron SDK version upgrade (`neuronxcc`, `torch_neuronx`, or NKI)
- Model architecture changes (different model code producing a different FX graph)
- Different input shapes (new bucket sizes)
- Different compiler flags (e.g., changing `--optimization-level`)
- Different hardware target (e.g., moving from trn2 to trn3)
- Different tensor parallel degree

Restarts on the same instance with the same versions will hit the local cache — no recompilation. New nodes with a remote cache configured will fetch from the remote store — also no recompilation.

For full details on cache key composition, see [Compilation cache design](../design/compilation/compilation_cache.md#cache-key).

### Runtime and execution

| Variable | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `VLLM_NEURON_CPU_MODE` | bool | 0 | Enable CPU fallback mode (no Neuron hardware needed). For development/testing only. |
| `VLLM_NEURON_LOG_LEVEL` | str | INFO | Logging level for vLLM Neuron components. |
| `VLLM_NEURON_DISABLE_NKI_KERNELS` | bool | 0 | Disable all NKI kernels, forcing torch fallback paths. |
| `VLLM_NEURON_ATTN_MASK_TORCH` | bool | 0 | Build the decode attention mask in PyTorch, in place of the `gen_mask_tkg_hbm` NKI kernel. |
| `VLLM_USE_V2_MODEL_RUNNER` | bool | unset | Select the Neuron model runner. Unset uses Model Runner V2 (MRv2) for supported dense configurations and Model Runner V1 (MRv1) otherwise. `0` forces MRv1. `1` forces MRv2 with no fallback and logs a warning for most unsupported features. The resolved value is `VllmConfig.use_v2_model_runner`. See [Model Runner V2](features-guide.md#model-runner-v2). |

### Memory management

For conceptual overview, see [Memory management](features-guide.md#memory-management).

| Variable | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` | float | 0.30 | Fractional cap for default KV sizing. Explicit headroom bypasses this cap. |
| `VLLM_NEURON_RESERVED_HEADROOM_GIB` | float | unset | Explicit reserve per worker. Enables headroom sizing on both routes. |
| `VLLM_NEURON_MIN_KV_BUDGET_GIB` | float | 1.0 | Minimum KV budget in GiB. Server fails fast if below. |

When headroom is unset, the worker respects memory
utilization and the fractional cap. An explicit reserve, including zero,
enables headroom sizing. This mode sets `gpu_memory_utilization` to `1.0`
and bypasses the fractional cap.

For example, 4 GiB is a tested reserve for GPT-OSS 120B on Trn3
with the Native backend and MXFP4 weights.
Validate the reserve for your model and configuration.

Both backends use `VLLM_NEURON_RESERVED_HEADROOM_GIB`.
Each deployment can select a different reserve.
Use the smallest value that passes the validated workload.

### Metrics

| Variable | Type | Default | Description |
| ---- | ---- | ---- | ---- |
| `PROMETHEUS_MULTIPROC_DIR` | str | auto-created tempdir | Directory for prometheus_client multiprocess mode. Wipe between server restarts. |

## Complete configuration example

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    max_model_len=4096,
    max_num_seqs=8,
    tensor_parallel_size=32,
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
export NEURON_COMPILED_ARTIFACTS=/opt/neuron-cache/llama-70b
export NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS=4

vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --tensor-parallel-size 32 \
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
- Default sizing respects memory utilization and a 30% fractional cap
- An explicit headroom value enables the reserve policy
- In headroom mode, increase the reserve after an HBM allocation failure

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

### HBM allocation errors during load or compilation

Check the active budget policy when logs contain an HBM allocation signature:

- `NCC_EOOM002` or `Maximum peak HBM usage`
- `NRT_RESOURCE` in `nrt_load_util`
- `nrt_tensor_allocate status=4`
- `Allocation Failure` while loading a model

The KV cache can leave too little HBM for code, scratch, internal tensors,
or communication buffers. In default mode, reduce
`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` to reduce the KV budget.

In explicit headroom mode, increase the active reserve from its logged value.
For a deployment that currently reserves 4 GiB, the next step is:

```bash
export VLLM_NEURON_RESERVED_HEADROOM_GIB=5
```

Start with 1 GiB increments and rerun the same model, sharding, backend, and
hardware configuration. Each additional 1 GiB of headroom reduces the KV cache
budget by 1 GiB. Keep the smallest value that passes. If more headroom would
leave too little KV cache, reduce `max_model_len` or `max_num_seqs`.

### "KV cache budget below minimum threshold" error

The computed KV budget is less than `VLLM_NEURON_MIN_KV_BUDGET_GIB`
(default 1.0 GiB). In default mode, increase `gpu_memory_utilization` or
`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` and validate the configuration.

In explicit headroom mode, the worker computes:

```text
CPU:
available = max(host_available / local_workers - headroom_bytes, 0)

Neuron:
available = max(total_hbm - model_weight_bytes - headroom_bytes, 0)
```

CPU host available memory already reflects loaded weights and current allocations.

`headroom_bytes` comes from `VLLM_NEURON_RESERVED_HEADROOM_GIB`.
The error reports this variable and the maximum headroom that
satisfies the configured minimum.

Do not increase headroom for this error: increasing it makes the KV cache
budget smaller. Decrease the active headroom variable, or lower
`VLLM_NEURON_MIN_KV_BUDGET_GIB` only when a smaller KV cache is valid for the
workload.

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
