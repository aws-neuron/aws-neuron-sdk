# How to use vLLM Neuron features in your deployment

<!-- meta: description: Detailed reference companion to the quickstarts,
covering every significant feature and configuration option for vLLM on
Neuron. -->
<!-- meta: date_updated: 2026-08-13 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: NDOC-182 -->

This guide covers every significant feature the vLLM Neuron plugin
exposes, organized by feature. For each feature: what it is, when to
use it, how to enable it, and trade-offs.

For supported models and per-model feature availability, see the
[model cards](../model-recipes/index.md) and the
[Supported Models](https://github.com/vllm-project/vllm-neuron#supported-models)
section in the README.

## Prerequisites

- Completed [Set up vLLM Neuron](../getting-started/setup-guide.md) and one of the
  quickstarts
- A running vLLM server or `LLM` instance on a Trainium/Inferentia
  instance

## Compilation

vLLM Neuron uses PyTorch's `torch.compile` API with the `vllm_neuron`
backend — the same `torch.compile` compilation pattern used in vLLM
upstream. Under the hood, the `vllm_neuron` backend lowers FX graphs
through XLA and HLO to the Neuron Compiler (`neuronx-cc`), which
produces optimized NEFF binaries for Trainium/Inferentia hardware.

Compilation happens automatically during server warmup — one NEFF per
bucket (sequence length × batch size combination). Compiled NEFFs are
cached on disk and reused on subsequent restarts.

For production, you can pre-compile on a Neuron instance or a CPU
instance and distribute artifacts via a shared filesystem so production
nodes never compile at startup. See
[compilation cache](../design/compilation/compilation_cache.md) and
[CPU compilation](../design/compilation/cpu_compilation.md).

### Pre-compiled model artifacts

You can also use `NEURON_COMPILED_ARTIFACTS` to point at a directory of
pre-compiled models:

```bash
export NEURON_COMPILED_ARTIFACTS=/path/to/cache
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 32
```

If the path contains valid artifacts, they are loaded directly with no
recompilation. If the path is empty, the model is compiled and saved
there for future use.

## Bucketing and dynamic shapes

Neuron compiles models into static-shape programs (NEFFs). To handle
varying input lengths and batch sizes at runtime, vLLM Neuron uses
**bucketing** — compiling multiple NEFFs at different sizes and
selecting the smallest one that fits each request.

There are three bucketing dimensions:

- **Prefill token buckets** (`num_batched_tokens_buckets`) — compiled
  prefill token counts for prompt processing. Inputs are padded to the
  nearest bucket.
- **Decode batch buckets** (`num_seqs_buckets`) — compiled batch
  sizes for token generation. Active requests are padded to the
  nearest bucket.
- **Decode context-length buckets** (`decode_context_length_buckets`)
  — optional second dimension that compiles smaller decode NEFFs sized
  to typical KV lengths rather than `max_model_len`. Useful for
  disaggregated inference where effective context is much shorter than
  the maximum.

### Trade-offs

- More buckets = slower startup (more compilations), less padding
  waste at runtime
- Fewer buckets = faster startup, more padding waste
- Each bucket consumes device memory

### Recommended approach

1. Start with minimal buckets during development (single prefill
   bucket, single decode batch)
2. Profile your workload to understand prompt length and concurrency
   distribution
3. Add buckets matching your traffic percentiles
4. Remove unused buckets to speed up startup

For the full parameter reference, see
[Configuration options](reference-configuration.md#compilation-options).
For the decode context-length bucketing design, see
[Decode context-length bucketing](../design/vllm/decode-context-length-bucketing.md).

## Continuous batching

Continuous batching is enabled by default. The Neuron scheduler
automatically interleaves prefill and decode requests, admitting new
requests without waiting for in-flight requests to complete.

No configuration is needed -- this is the default behavior when serving
with vLLM Neuron. For details on how the scheduler works internally,
see [Scheduler design](../design/vllm/neuron-scheduler.md).

## Streaming

Streaming is supported by default via the OpenAI-compatible API. No
additional configuration is needed:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

## Segmented prefill

Segmented prefill breaks long prompts into smaller segments, processing
them over multiple iterations. This enables serving long-context models
(64K, 128K+) without compiling a single NEFF for the full sequence
length.

Segmentation is controlled by `max_num_batched_tokens` — prompts
exceeding this limit are processed in multiple iterations:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 32 \
    --max-model-len 131072 \
    --max-num-batched-tokens 8192
```

In this example, a 32K prompt is processed in 4 iterations of 8192
tokens each.

:::{note}
Unlike chunked prefill (mixed batching) in vLLM upstream, prefill and
decode always run in separate batches on Neuron — they are never mixed
in the same batch.
:::

### When to use segmented prefill

- Long-context models where compiling a single bucket for the full
  context is impractical
- Reducing peak memory during prefill by processing fewer tokens per
  iteration

For the full design, see
[Prefill processing design](../design/vllm/prefix-caching.md).

## Prefix caching

Prefix caching allows you to reuse the KV cache from shared prompt
prefixes across requests. This improves time-to-first-token (TTFT) for
workloads where many requests share common system prompts or context.

Prefix caching is **enabled by default** on vLLM Neuron, the same as vLLM
upstream. You do not need to configure anything to benefit from it — requests
that share a prompt prefix automatically reuse the cached KV.

To disable it, pass `--no-enable-prefix-caching` (CLI) or
`enable_prefix_caching=False` (`LLM` constructor).

### Sizing the KV cache

Prefix caching only helps when there are enough KV cache blocks to retain
prefixes across requests. On Neuron, block count is set explicitly via
`num_gpu_blocks_override`, and `block_size` must be specified:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 32 \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --block-size 32 \
    --num-gpu-blocks-override 4096
```

The equivalent `LLM` invocation:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=32,
    max_model_len=4096,
    max_num_seqs=8,
    block_size=32,
    num_gpu_blocks_override=4096,
)
```

For more blocks to reuse (higher hit rate on repeated prefixes), increase
`num_gpu_blocks_override`. See the
[prefix caching benchmark tutorial](../tutorials/tutorial-prefix-caching-gpt-oss-benchmarking.md)
for a worked example measuring the TTFT improvement.

### How prefix caching works

vLLM hashes each block of prompt tokens and reuses the cached KV for any
request whose leading blocks hash-match a previously computed prefix. For the
full design, see [Prefix caching design](../design/vllm/prefix-caching.md).

### When to use prefix caching

- Workloads with long, repeated system prompts shared across requests
- RAG applications where the same retrieved context appears in multiple
  queries
- Chat applications where conversation history is re-sent with each turn

## Tensor, data, and expert parallelism

vLLM Neuron exposes three parallelism dimensions:

- **Tensor parallelism (TP)** — Shards each weight matrix across ranks. Required
  to fit large models. Always used; pick the smallest size that fits the model and
  meets target latency.
- **Data parallelism (DP)** — Runs independent model replicas to increase
  concurrent throughput. Use when you have spare cores beyond what TP requires.
- **Expert parallelism (EP)** — Distributes MoE experts across ranks. Use for MoE
  models when intermediate sharding becomes too small under high TP.

```bash
# TP only, single replica
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 64

# TP + DP, multiple replicas
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --data-parallel-size 8

# TP + EP, full expert parallelism for MoE
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --enable-expert-parallel

# TP + EP with explicit ep_degree
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --additional-config '{"neuron_config": {"ep_degree": 2}}'
```

**Trade-offs:** Each dimension adds collective traffic. EP reduces per-rank memory
at the cost of routing all tokens through the EP collective. DP increases
throughput linearly with replicas, at the cost of cores per replica. Match the
parallelism to your bottleneck.

**Neuron-specific notes:**

- With `--enable-expert-parallel` and no explicit `ep_degree`, the worker defaults
  `ep_degree` to `--tensor-parallel-size` (full EP).
- Setting `ep_degree > 1` without `--enable-expert-parallel` fails at startup.
- Combining DP with EP requires `ep_degree = TP × DP`. Other combinations are
  rejected.
- Component-level data parallelism (per-layer DP for attention, MLP, embedding, or
  LM head) requires disaggregated inference — the worker rejects component DP
  without a `--kv-transfer-config`.
- **Vision encoder parallelism** — For multimodal models, all ranks run both the text model and the vision encoder — ranks
  are not partitioned between them. The vision encoder has its own TP and DP
  degrees, decoupled from the text model's parallelism, and only runs during
  prefill. EP does not apply (no MoE component). By default the encoder uses
  full DP (`tp_size=1, dp_size=world_size`) with no additional configuration.
  To override, pass `vision_neuron_config` in `--additional-config`:

  ```bash
  vllm serve Qwen/Qwen3-VL-32B-Instruct \
      --tensor-parallel-size 8 \
      --max-model-len 8192 \
      --max-num-batched-tokens 8192 \
      --additional-config '{
          "vision_neuron_config": {
              "tp_size": 1,
              "dp_size": 8
          }
      }'
  ```

  Constraint: `tp_size * dp_size` must equal `world_size`. See
  [Vision encoder parallelism design](../design/parallelism/vision_encoder_parallelism.md)
  for block-level DP semantics, process groups, and configuration details.

## Disaggregated inference

Disaggregated inference separates prefill (prompt processing) and decode
(token generation) onto different instances, allowing each phase to be
scaled and optimized independently.

Disaggregated inference transfers the KV cache over NIXL/LIBFABRIC (EFA),
which requires the `libcuda.so.1` and `libfabric.so.1` shared libraries on the
loader path. If your environment does not already ship them, see
[Install dependencies for disaggregated inference](../getting-started/setup-guide.md#install-dependencies-for-disaggregated-inference).

### Configurations

- **1P1D**: 1 prefill instance + 1 decode instance
- **xPyD**: Multiple prefill instances + multiple decode instances for
  higher throughput

### When to use disaggregated inference

- High-throughput serving where prefill and decode have different
  resource requirements
- Large-scale deployments where you need to scale prefill and decode
  independently
- Workloads with long prompts but short generations (or vice versa)

## Quantization

Quantization reduces model memory footprint and improves throughput by
using lower-precision arithmetic for inference. vLLM Neuron supports two
model-weight quantization paths — FP8 static (Trn2) and MXFP4 (gpt-oss on
Trn3) — plus FP8 KV-cache quantization. See the
[model recipes](../model-recipes/index.md)
for per-model feature support.

Always validate with the
[accuracy debugging guide](../model-dev/accuracy-debugging-guide.md) after
enabling any quantization.

### FP8 static weight quantization

FP8 static weight quantization is driven by the **checkpoint**, not a
runtime flag. Point the server at a ModelOpt-calibrated checkpoint carrying
per-tensor static FP8 scales in its `quantization_config`; vLLM Neuron
detects the format and loads the FP8 weights automatically:

```bash
vllm serve <path-to-modelopt-fp8-checkpoint> \
    --tensor-parallel-size 32 \
    --max-model-len 4096
```

Do not set vLLM's `--quantization` flag — the checkpoint's
`quantization_config` selects the path. A checkpoint whose config requests
an unsupported scheme (e.g. weight quantization from `compressed-tensors`,
which the platform validator rejects — only KV-cache `q_scale`/`k_scale`/
`v_scale` is accepted) fails at startup with a clear error rather than
silently falling back to BF16.

### MXFP4 weight quantization (gpt-oss on Trn3)

gpt-oss supports MXFP4 (mixed-precision FP4) weights on Trn3 for maximum
throughput. MXFP4 is Neuron-specific: pass it under
`neuron_config.quantization`, **not** the upstream `--quantization` flag:

```bash
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 64 \
    --additional-config '{"neuron_config": {"quantization": "mxfp4"}}'
```

MXFP4 is supported only for gpt-oss on Trn3. The worker rejects `mxfp4` on Trn2
(gpt-oss falls back to BF16 there) — this fails at startup with a clear error.

### KV cache FP8 quantization

Separate from model weight quantization, you can quantize the KV cache
to FP8 to reduce memory footprint and increase concurrent request
capacity. This is controlled through vLLM's standard `kv_cache_dtype`
parameter (not `neuron_config`).

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=32,
    max_model_len=4096,
    kv_cache_dtype="fp8",  # FP8 KV cache
)
```

Or via CLI:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 32 \
    --kv-cache-dtype fp8
```

**How it works:**

- K and V tensors are quantized to FP8 before being written to the
  cache and dequantized back to BF16 when read
- Attention matmuls still run in BF16 (memory reduction, not compute
  speedup)
- Cache footprint is roughly halved, allowing more concurrent requests

**Scale calibration.** Scales must be provided by the model checkpoint
(produced offline via `llm-compressor` calibration). Without checkpoint
scales, values default to 1.0 which degrades accuracy.

**TRN2 vs TRN3 clamp range.** TRN2 FP8 E4M3 has a max finite value of
240.0, whereas TRN3 supports up to 448.0. The correct clamp is resolved
automatically based on the detected platform.

### When to use quantization

- Large models that don't fit in available HBM at full precision
- Throughput-bound workloads where slight accuracy loss is acceptable
- Deploying on smaller instance types with limited memory
- KV cache FP8: when you need more concurrent requests without changing
  model precision

## On-device sampling

On-device sampling performs the sampling step (argmax, top-k, top-p)
directly on the Neuron device rather than transferring logits back to
CPU. This is enabled by default and reduces latency.

### Supported sampling parameters

When on-device sampling is enabled, the following parameters are
supported:

- `temperature`
- `top_k`
- `top_p`

:::{note}
When `top_k` is set to -1, it is internally limited to 256. When
`temperature` is set to 0, greedy decoding is used (equivalent to
`top_k=1`).
:::

## Async scheduling

Async scheduling overlaps CPU work (scheduling, input preparation,
output processing) with Neuron device execution (NEFF forward pass) to
reduce per-step latency.

### How async scheduling works

In synchronous mode, the CPU idles while the device runs the forward
pass, and the device idles while the CPU prepares the next step. Async
scheduling eliminates this idle time by:

1. **Batch queue (depth 2).** The engine core schedules one step ahead
   so the worker always has work ready.
2. **Device-to-device token passing.** The previous step's sampled
   token future is fed directly as `input_ids` into the next forward
   pass -- no CPU roundtrip. The Neuron async runtime enqueues NEFF
   executions ahead of time.
3. **Async output materialization.** Tokens are materialized to CPU on
   a separate thread, overlapped with scheduling the next step.

```text
Steady-state async (batch unchanged):

CPU:    | schedule(N+1) | prepare(N+1) | schedule(N+2) |
Device: |====== forward(N) ======|====== forward(N+1) ======|
Async:  |  materialize(N-1)  |      materialize(N)     |
```

### Breaking the async flow

The async flow is "broken" when:

- Batch composition changes (requests finish or new requests join)
- The previous sampled-token future shape does not match the next
  decode input shape
- The request order changed (future is remapped but still forced)

When broken, the pending device future is materialized on the critical
path, batch state is updated, and CPU-built `input_ids` are used for
that step. The async flow automatically reinstates on the next
steady-state decode step.

### Configuration

Async scheduling is **on by default** and requires on-device sampling.

```bash
# Disable async scheduling
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --no-async-scheduling
```

### When to disable

- **Debugging**: disable to simplify the execution flow for
  troubleshooting
- **Very short responses**: if most responses are 1-2 tokens, the
  overhead of managing futures may not be worth the overlap benefit
- **CPU sampling**: async scheduling requires on-device sampling; it
  cannot be used with CPU sampling

### When async scheduling helps most

- **Smaller models (faster NEFF execution)**: CPU overhead is a larger
  fraction of total step time, so hiding it yields greater relative
  speedup
- **Longer average response length**: more consecutive decode steps
  means the async path is active for a larger fraction of total
  generation

## Speculative decoding (EAGLE3)

Speculative decoding accelerates autoregressive inference without
changing the output distribution. A lightweight draft model predicts
several tokens ahead, and the target model verifies them in a single
forward pass.

:::{note}
**Mutually exclusive with async scheduling.** Setting `--speculative-config`
disables async scheduling automatically with a startup warning.

EAGLE3 and DFlash are supported speculative methods. The examples below use
EAGLE3. For per-model compatibility, see the
[model recipes](../model-recipes/index.md).
For compatible target/draft pairings and worked examples, see the EAGLE3
speculative decoding tutorials for
[Llama 3.1](../tutorials/tutorial-eagle3-speculative-decoding-llama-3-1.md) and
[GPT-OSS](../tutorials/tutorial-eagle3-speculative-decoding-gpt-oss.md).
:::

### How it works

1. The target model generates a token and produces hidden states.
2. The draft model proposes `K` speculative tokens from those hidden
   states.
3. The target model verifies all `K` tokens in one forward pass.
4. A rejection sampler accepts tokens sequentially until the first
   mismatch.
5. The process repeats.

This guarantees output is identical to the target model alone while
improving throughput by accepting multiple tokens per step.

### Enable speculative decoding (CLI)

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 32 \
    --speculative-config '{"method": "eagle3", "model": "eagle3-llama3.3-70b", "num_speculative_tokens": 5}' \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --additional-config '{"neuron_config": {"on_device_sampling_config": {"temperature": "0"}}}'
```

### Enable speculative decoding (Python)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=32,
    max_model_len=4096,
    max_num_seqs=8,
    speculative_config={
        "method": "eagle3",
        "model": "eagle3-llama3.3-70b",
        "num_speculative_tokens": 5,
    },
    additional_config={
        "neuron_config": {
            "on_device_sampling_config": {
                "temperature": "0"
            }
        }
    }
)

sampling_params = SamplingParams(max_tokens=256, temperature=0.0)
outputs = llm.generate(["Explain quantum computing"], sampling_params)
```

### Key parameters

| Parameter                   | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `--speculative-config`      | JSON config selecting the speculative method (`eagle3`), the draft `model`, and `num_speculative_tokens` |
| `method`                    | Speculative method; `eagle3` is supported in 2.31            |
| `model`                     | Path or name of the EAGLE3 draft model                       |
| `num_speculative_tokens`    | Number of tokens the draft proposes per step (typically 3-7) |
| `on_device_sampling_config` | Required for on-device rejection sampling                    |

### On-device vs CPU sampling

Speculative decoding uses **on-device sampling** for the rejection step.
The target model verifies draft tokens and the rejection sampler runs
entirely on the Neuron device, avoiding the latency of transferring
logits to CPU. This requires `on_device_sampling_config` to be set.

### When to use speculative decoding

- Latency-sensitive deployments where TTFT is acceptable but inter-token
  latency matters
- Workloads where the draft model acceptance rate is high (structured
  outputs, code generation)

## Structured outputs and tool calling

Structured outputs guarantee that model responses conform to a specified
schema (JSON schema, regex, choice enum, or BNF grammar). Tool calling
builds on the same infrastructure -- vLLM converts tool schemas into
structured output constraints internally.

### How structured outputs work

Structured outputs use the same on-device bitmask approach as vLLM
upstream — a grammar-derived bitmask is applied to logits before
sampling, ensuring only grammar-valid tokens can be selected. For
implementation details, see
[Structured outputs design](../design/vllm/structured-outputs-and-tool-calling.md).

### Performance

Structured output adds virtually no per-token overhead:

| Component                         | Latency                 |
| --------------------------------- | ----------------------- |
| Grammar bitmask computation (CPU) | ~0.12 ms                |
| Bitmask unpacking + transfer      | ~0.24 ms                |
| Forward pass delta                | ~0 ms (within variance) |
| **Total per-token overhead**      | **~0.4 ms (~1.8%)**     |

### Enable structured outputs

No additional configuration is needed. Use standard OpenAI API
parameters:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "colors",
            "schema": {
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["colors"]
            }
        }
    }
)
```

### Tool calling

Tool calling uses two paths depending on `tool_choice`:

- **auto/none**: No structured output constraints. The model generates
  free text and `Llama3JsonToolParser` extracts tool calls via regex
  post-generation.
- **required/named**: vLLM converts the tool schema into a JSON grammar
  constraint and routes it through the same bitmask pipeline as
  structured outputs.

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[
        {"role": "user", "content": "What's the weather in Seattle?"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="required"
)
```

:::{note}
`tool_choice="required"` and named tool choice require on-device
sampling. With CPU sampling, the `logit_mask` is not applied and the
model may produce tokens that violate the grammar FSM.
:::

## Multimodal support

vLLM Neuron supports multimodal inference on vision-language models,
accepting image, video, and text inputs.

For the supported models, feature matrix, and accuracy results, see the
[model cards](../model-recipes/index.md). For an end-to-end deployment walkthrough of a
multimodal model, see the [Qwen3-VL deployment tutorial](../tutorials/tutorial-qwen3-vl-32b.md).

### Multimodal Input

Multimodal requests use the standard vLLM interfaces.

**Offline (`LLM.generate`).** Build the prompt with the model's chat template
and pass the images and/or videos under `multi_modal_data`:

:::{dropdown} Offline inference example
:icon: code

```python
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")

llm = LLM(
    model="Qwen/Qwen3-VL-32B-Instruct",
    tensor_parallel_size=16,
    max_model_len=32768,
    max_num_seqs=8,
    additional_config={
        "neuron_config": {"quantization": "bf16"},
        # Vision encoder buckets are auto-generated; see the reference config
        # to tune num_vision_tokens_buckets / vision_attention_block_size.
    },
)
sampling_params = SamplingParams(max_tokens=256, temperature=0.0)

# Image input
image = ImageAsset("cherry_blossom").pil_image
messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image in detail."},
    ],
}]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": [image]}},
    sampling_params,
)

# Video input
video = VideoAsset("baby_reading", num_frames=4)
messages = [{
    "role": "user",
    "content": [
        {"type": "video"},
        {"type": "text", "text": "Describe what happens in this video."},
    ],
}]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {
            "video": (video.np_ndarrays, video.metadata),
        },
    },
    sampling_params,
)
```

:::

For multiple images, add more `{"type": "image"}` entries and pass a list under
`multi_modal_data={"image": [...]}`. For video, use a `{"type": "video"}`
content part and pass the frames-and-metadata tuple the vLLM video parser
expects (`multi_modal_data={"video": (frames, metadata)}`).

**Online (OpenAI-compatible API).** Send images as `image_url` content parts —
either a remote URL or a base64 data URL:

:::{dropdown} Online request example
:icon: code

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen3-VL-32B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"}},
            {"type": "text", "text": "What's in this image?"},
        ],
    }],
)
```

:::

Refer to the vLLM [docs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/) for more detailed examples.
For more details on how to configure the vision encoder, refer to the [vision encoder options](reference-configuration.md#vision-encoder-options-vision_neuron_config).

## Encoder-disaggregated inference (EPD)

Encoder-disaggregated inference separates the vision encoder from the language
model onto a **Vision Encoder (VE)** pool and a **Prefill+Decode (PD)** pool,
allowing each stage to be scaled and given its own parallelism, with an
OpenAI-compatible router fronting both. Unlike
[disaggregated inference](#disaggregated-inference), which moves KV cache over a
`--kv-transfer-config` connector, EPD moves vision embeddings over an
`--ec-transfer-config` connector. EPD is strictly two-way; prefill and decode
stay co-located in one PD engine.

Each engine sets a construction role (`--mm-encoder-only` for a VE,
`"mm_language_model_only": true` for a PD; see
[EPD construction roles](reference-configuration.md#epd-construction-roles)) and
a transport role (`ec_producer` or `ec_consumer`). EPD transfers embeddings over
NIXL/LIBFABRIC (EFA) as an HBM→HBM RDMA read, which requires NIXL plus the
`libcuda.so.1` and `libfabric.so.1` shared libraries on the loader path — see
[Install dependencies for disaggregated inference](../getting-started/setup-guide.md#install-dependencies-for-disaggregated-inference).

### EPD configurations

- **1E1PD**: 1 vision encoder pool + 1 prefill+decode pool
- **xEyPD**: Multiple VE and PD engines, scaled independently — add VEs for
  image-heavy traffic, PDs for token-heavy traffic

### When to use EPD

- Multimodal serving where image encoding and text generation saturate at
  different rates, so one monolithic engine leaves one of the two idle
- Workloads whose image-to-text ratio shifts over time, where you want to add
  encoder capacity without recompiling or resizing the language-model pool
- Deployments that want asymmetric parallelism between the two stages

For the full launch commands, the required EFA environment variables, core
alignment rules, and the failure modes with their error signatures, see the
[EPD tutorial](../tutorials/tutorial-epd-1e-1pd-xeypd.md). For how embeddings
are stored and read on device, see
[On-device encoder cache](../design/multimodal/on_device_encoder_cache.md).

## Prompt embeddings

Prompt embeddings allow you to pass precomputed embedding tensors
instead of token IDs. This is useful when another system has already
produced embeddings (e.g., a multimodal encoder or retrieval pipeline).

### Enable prompt embeddings

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=32,
    enable_prompt_embeds=True,
)
```

Or via CLI:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 32 \
    --enable-prompt-embeds
```

### How prompt embeddings work

1. Send a request with `prompt_embeds` (shape `[seq_len, hidden_size]`)
   instead of prompt text
2. vLLM schedules it like any other request (batching and padding
   unchanged)
3. The model merges your embeddings with token-derived hidden states
   during the forward pass

For implementation details, see
[Prompt embeddings design](../design/vllm/prompt-embeddings.md).

### When to use prompt embeddings

- Multimodal pipelines where an encoder produces embeddings
- Retrieval-augmented generation with pre-embedded passages
- Any workflow where embeddings are computed externally

## Metrics

vLLM Neuron publishes custom Prometheus metrics alongside vLLM's
built-in metrics at the `/metrics` endpoint.

### Multiprocess mode

vLLM Neuron enables `prometheus_client` multiprocess mode so that
metrics observed in the EngineCore and Worker processes are written to
shared mmap files and aggregated by the API server's `/metrics`
endpoint.

`PROMETHEUS_MULTIPROC_DIR` is set automatically at module load time. You
can override it by setting the environment variable before starting the
server. The directory should be wiped between server restarts to avoid
stale metrics.

### Neuron-specific metrics

All Neuron-specific metrics use the `vllm_neuron:` prefix.

**General metrics:**

| Metric Name                              | Type      | Description                                               |
| ---------------------------------------- | --------- | --------------------------------------------------------- |
| `vllm_neuron:num_seqs_padding`           | Histogram | Padded batch lines, by `model_name` and `bucket_name`     |
| `vllm_neuron:num_batched_tokens_padding` | Histogram | Padded sequence length, by `model_name` and `bucket_name` |
| `vllm_neuron:neff_execution_count`       | Counter   | NEFF executions, by `model_name` and `bucket_name`        |

**Server startup metrics:**

| Metric Name                            | Type  | Description                              |
| -------------------------------------- | ----- | ---------------------------------------- |
| `vllm_neuron:startup_time_seconds`     | Gauge | Total server startup time                |
| `vllm_neuron:compilation_time_seconds` | Gauge | Graph compilation time, by `bucket_name` |
| `vllm_neuron:model_load_time_seconds`  | Gauge | Weight load time (host to HBM)           |
| `vllm_neuron:model_load_size_bytes`    | Gauge | Weight size transferred to device        |

**Common labels:**

| Label         | Description                                    |
| ------------- | ---------------------------------------------- |
| `model_name`  | Model name for that server process             |
| `bucket_name` | Bucket identifier (e.g., `prefill_s1024`)      |
| `rank_id`     | Process local rank ID (relative to start rank) |

### Querying metrics

```bash
curl http://localhost:8000/metrics | grep vllm_neuron
```

### Profiling

For capturing Neuron Runtime profiles to analyze per-layer latency and
device utilization, see
[How to profile workloads](how-to-profile-workloads.md).

## Memory management

vLLM Neuron uses a heuristic to determine how much HBM to allocate
for the KV cache, balancing throughput against compile-time constraints.

### How memory management works

On Neuron, KV cache size is baked into the compiled program (NEFF)
shape. The KV sizing decision directly affects whether the model can
compile at all -- if KV allocation is too large, the compiler rejects
the program.

The memory budget is computed as:

```text
total_budget = total_hbm * gpu_memory_utilization
kv_cache_budget = max(total_budget - bytes_used, 0)
heuristic_cap = total_budget * KV_GMU_BUDGET_CAP_FRACTION
available = min(kv_cache_budget, heuristic_cap)
```

### Key environment variables

| Variable                                 | Default | Description                                       |
| ---------------------------------------- | ------- | ------------------------------------------------- |
| `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` | 0.30    | Fraction of total GMU budget to cap KV allocation |
| `VLLM_NEURON_MIN_KV_BUDGET_GIB`          | 1.0     | Minimum KV budget in GiB (fails fast if below)    |

The default cap of 0.30 is conservative, derived from worst-case
compile behavior on GPT-OSS family models. Override it when more
aggressive KV allocation is desired and compile-fit has been validated.

### When to tune memory settings

- If you see `NCC_EVRF009` compiler errors, reduce
  `gpu_memory_utilization` or lower the cap fraction
- If you need more concurrent requests, increase the cap fraction
  (validate that compilation succeeds)
- If startup fails with "KV cache budget below minimum threshold",
  your configuration is too memory-constrained

## Neuron-specific configuration

All Neuron-specific options are passed through the `additional_config`
parameter nested under `neuron_config`. For the complete parameter
reference, see [Configuration options](reference-configuration.md).

## Common issues

### Prefix caching has no effect (no cache reuse, no TTFT improvement)

- Prefix caching is on by default; the most common cause of low reuse is too
  few KV cache blocks to retain prefixes. Increase `num_gpu_blocks_override`
  (see the prefix caching section above).
- Cache hits require a **byte-for-byte identical** shared prefix across
  requests. Even a trailing whitespace difference in the system prompt prevents
  reuse.
- Confirm it was not disabled — check for `--no-enable-prefix-caching` or
  `enable_prefix_caching=False` in your launch configuration.

### Server fails with "ep_degree requires --enable-expert-parallel"

- Set both `--enable-expert-parallel` and `ep_degree` together, or remove
  `ep_degree` from `neuron_config` entirely.

### Server fails with "quantization='mxfp4' is not supported on TRN2"

- Use BF16 on Trn2, or use Trn3 for MXFP4. See the quantization section.

### Compilation takes too long at startup

- Reduce the number of `num_batched_tokens_buckets` and
  `num_seqs_buckets`
- Use `NEURON_COMPILED_ARTIFACTS` to cache compiled models between
  restarts
- Set `NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS` to increase parallel
  compilation

### Structured outputs produce unexpected results

- Verify your JSON schema is valid and complete
- Ensure `required` fields are specified in the schema
- Confirm on-device sampling is enabled (required for
  `tool_choice="required"`)

### KV cache budget too low

- Increase `gpu_memory_utilization` (default 0.9)
- Increase `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` above 0.30
- Reduce `max_model_len` or `max_num_seqs` to lower KV requirements

### Async scheduling sync fallbacks

- Check logs for frequent "batch composition changed" messages
- This is normal for workloads with many short responses; async
  benefits scale with response length

## Related information

- [Configuration reference](reference-configuration.md)
- For supported models and features, see the [README](https://github.com/vllm-project/vllm-neuron#supported-models)
  and [model cards](../model-recipes/index.md).
- [Setup guide](../getting-started/setup-guide.md) — Install and configure vLLM on
  Neuron.
- [Accuracy debugging guide](../model-dev/accuracy-debugging-guide.md) — Diagnose accuracy
  issues if feature changes affect outputs.
- [Llama 3 recipe](../model-recipes/llama-3.md) — Llama 3 family (1B, 8B, 70B)
  model recipe.
- [gpt-oss deployment tutorial](../tutorials/tutorial-gpt-oss.md) — End-to-end
  gpt-oss deployment recipe.
- [EAGLE3 speculative decoding tutorial (Llama 3.1)](../tutorials/tutorial-eagle3-speculative-decoding-llama-3-1.md)
- [EAGLE3 speculative decoding tutorial (GPT-OSS)](../tutorials/tutorial-eagle3-speculative-decoding-gpt-oss.md)
  — Worked Eagle3 deployment.
- [Disaggregated inference (1P1D / xPyD) tutorial](../tutorials/tutorial-di-1p1d-xpyd.md)
  — Worked disaggregated inference deployment.
- [Encoder-disaggregated inference (1E1PD / xEyPD) tutorial](../tutorials/tutorial-epd-1e-1pd-xeypd.md)
  — Worked EPD multimodal deployment.
