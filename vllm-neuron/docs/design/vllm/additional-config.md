# Additional Configuration Options

<!-- meta: description: Additional configuration options -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

vLLM Neuron extends vLLM with Neuron-specific configuration options through the `additional_config` parameter. This document describes the available options and their usage.

## Configuration Structure

All Neuron-specific options are nested under the `neuron_config` key:

``` python
from vllm import LLM

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    max_model_len=256,
    tensor_parallel_size=8,
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [256],
            "num_seqs_buckets": [1, 2, 4, 8],
            "on_device_sampling_config": {
                "temperature": "0"
            }
        }
    }
)
```

Or via command line:

``` bash
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-model-len 256 \
    --tensor-parallel-size 8 \
    --additional_config '{"neuron_config": {"num_batched_tokens_buckets": [256], "num_seqs_buckets": [8]}}' \
    --port 8000
```

## Configuration Options

### num_batched_tokens_buckets

**Purpose:** Specifies the prefill token counts for which the model will be compiled during prefill (prompt processing).

**Type:** `list[int]`

**Default:** Power-of-2 token counts from 128 up to `max_num_batched_tokens`

**Description:**

During prefill, the input segment is padded to the smallest bucket size that can accommodate it. The model is pre-compiled for each bucket size during warmup to avoid compilation latency during inference.

``` python
# Example: Only compile for 256 batched tokens
"num_batched_tokens_buckets": [256]

# Example: Multiple buckets for varying prompt lengths
"num_batched_tokens_buckets": [128, 256, 512, 1024]
```

**Considerations:**

- Each bucket adds to startup time (compilation) but reduces inference latency
- Buckets must be in strictly ascending order
- The largest bucket must equal `max_num_batched_tokens`
- Fewer buckets = faster startup, but more padding waste
- More buckets = slower startup, but less padding waste

**Example - Minimal Configuration:**

For applications with fixed-length prompts (e.g., always around 200 tokens):

``` python
"num_batched_tokens_buckets": [256]  # Single bucket, fastest startup
```

**Example - Production Configuration:**

For applications with varying prompt lengths:

``` python
"num_batched_tokens_buckets": [128, 256, 512, 1024, 2048]
```

### num_seqs_buckets

**Purpose:** Specifies the decode batch sizes (number of sequences) for which the model will be compiled during decode (token generation).

**Type:** `list[int]`

**Default:** Power-of-2 batch sizes from 1 up to `max_num_seqs`

**Description:**

During decode, each request generates one token at a time. Multiple requests can be batched together for efficiency. The model is pre-compiled for each batch size during warmup.

``` python
# Example: Only compile for batch size 8
"num_seqs_buckets": [8]

# Example: Multiple batch sizes for varying concurrency
"num_seqs_buckets": [1, 2, 4, 8, 16]
```

**Considerations:**

- Each batch bucket adds to startup time (compilation)
- Buckets must be positive integers in strictly ascending order
- The largest bucket must equal `max_num_seqs`
- The model will be compiled only for the specified batch sizes
- During inference, requests are batched to match the smallest bucket \>= current batch size
- Larger batch sizes improve throughput but may increase latency for individual requests

**Example - Fixed Concurrency:**

For applications with predictable request patterns:

``` python
"num_seqs_buckets": [8]  # Always process 8 requests at a time
```

**Example - Variable Concurrency:**

For applications with varying load:

``` python
"num_seqs_buckets": [1, 2, 4, 8]  # Handle 1-8 concurrent requests
```

### decode_context_length_buckets

**Purpose:** Opt in to a second bucketing dimension on decode so the worker compiles smaller-shape NEFFs whose attention block-table is sized to a user-specified context-length bucket instead of `max_model_len`. Cuts decode attention HBM traffic when `max_model_len` is much larger than the typical effective KV (e.g. disaggregated inference with `max_model_len=131072` but effective KV under 16K).

**Type:** `list[int]`

**Default:** `None` (feature disabled; behavior is bit-identical to today).

**Description:**

When set, the decode worker compiles one NEFF per `(num_seqs_bucket, ctx_bucket)` pair, plus an implicit `max_model_len` fallback NEFF (always compiled). At runtime each decode step picks the smallest context-length bucket that fits `max(num_computed_tokens) + 1 + num_speculative_tokens` and head-trims the block table to that many blocks; otherwise dispatches to the fallback. Per-step bucket switching is automatic — `torch.compile` routes to the cached NEFF for the chosen shape.

**Validation:** Each value must be (1) strictly ascending, (2) strictly less than `max_model_len`, and (3) divisible by `P_MAX = 128` (NKI attention tile constraint).

``` python
# Example: single bucket plus fallback (typical DI deployment)
"decode_context_length_buckets": [16384]

# Example: multiple buckets for varied effective-KV distributions
"decode_context_length_buckets": [4096, 8192, 16384]
```

**Considerations:**

- Each `(batch, ctx)` pair is a separate NEFF compile. With EAGLE3 speculative decoding, the count is multiplied by 3 (target with-spec, target without-spec, draft).
- The runtime `neff_execution_count` Prometheus label is unchanged so existing dashboards keep working; the compile-time `COMPILATION_TIME` label gains a `_ctx{S}` suffix.
- Sliding-window attention layers ignore this setting (they have their own tighter bound from the window).

See `decode-context-length-bucketing` for the full design.

### on_device_sampling_config

**Purpose:** Enables on-device sampling for async execution.

**Type:** `dict`

**Default:** `None` (sampling performed on CPU)

**Description:**

When enabled, token sampling (argmax for greedy, or probabilistic sampling for temperature \> 0) is performed on the Neuron device instead of the CPU. This enables asynchronous execution where the model can process the next batch while the previous batch's results are being transferred.

``` python
"on_device_sampling_config": {
    "temperature": "0"  # Greedy sampling (argmax)
}
```

**Parameters:**

- `temperature`: String value for sampling temperature
  - `"0"` - Greedy sampling (deterministic, fastest)
  - Other values - Not yet supported (raises `NotImplementedError`)

**Requirements:**

- On-device sampling is required for async scheduling (`async_scheduling=True`)
- Currently only greedy sampling (temperature=0) is supported

**Example:**

``` python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    additional_config={
        "neuron_config": {
            "on_device_sampling_config": {"temperature": "0"},
            "num_batched_tokens_buckets": [256]
        }
    }
)

# Use temperature=0 in SamplingParams to match on-device config
sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
```

## Complete Configuration Example

Here's a complete example showing all options:

``` python
from vllm import LLM, SamplingParams

# Create LLM with all Neuron options configured
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    max_model_len=1024,
    max_num_seqs=8,
    tensor_parallel_size=8,
    additional_config={
        "neuron_config": {
            # Compile for these prefill token counts
            "num_batched_tokens_buckets": [128, 256, 512, 1024],

            # Compile for these batch sizes during decode
            "num_seqs_buckets": [1, 2, 4, 8],

            # Enable on-device greedy sampling
            "on_device_sampling_config": {
                "temperature": "0"
            }
        }
    }
)

# Run inference
prompts = ["What is machine learning?"]
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
outputs = llm.generate(prompts, sampling_params)
```

## Configuration Tips

**Startup Time vs. Flexibility Trade-off:**

- Fewer buckets = faster startup, but less flexibility
- More buckets = slower startup, but handles more scenarios efficiently

**Recommended Approach:**

1. Start with minimal configuration (single bucket) during development
2. Profile your workload to understand prompt length distribution
3. Add buckets that match your 25th, 50th, 75th, and 95th percentile lengths
4. Remove unused buckets to speed up startup

**Memory Considerations:**

- Each compiled bucket consumes device memory
- Monitor memory usage when adding many buckets
- Consider using larger buckets with more padding vs. many small buckets

## Troubleshooting

**"Request exceeds largest bucket" Error:**

Your input exceeds the largest `num_batched_tokens_buckets` value. Either:

1. Add a larger bucket to your configuration
2. Reduce your input length

**Compilation During Inference:**

If you see compilation messages after warmup, you may have inputs that don't match any configured bucket. Check your bucket configuration covers your actual workload.

**"Only support greedy sampling" Error:**

Currently, on-device sampling only supports `temperature=0`. For non-greedy sampling, remove `on_device_sampling_config` to use CPU sampling.
