# Tutorial: Deploy Llama 3.3 70B (FP8) with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying Llama 3.3 70B in
static FP8 with vLLM Neuron on a single Trn2/Trn3 instance with tensor
parallelism. -->
<!-- meta: keywords: vLLM, Neuron, Llama 3, Llama-3.3-70B, FP8, ModelOpt,
single instance, tensor parallelism, tutorial, LLM serving, Trn2, Trn3,
Trainium -->
<!-- meta: date_updated: 2026-08-17 -->
<!-- Content type: procedural-tutorial -->
<!-- Jira: NDOC-184 -->

This tutorial is a production-ready recipe for deploying **Llama 3.3 70B Instruct
in static FP8** with vLLM Neuron on a single Trn2 or Trn3 instance using tensor
parallelism.

For the model's feature support, accuracy results, and supported checkpoints, see
the [Llama 3 model recipe](../model-recipes/llama-3.md). This tutorial assumes you
have already worked through the [setup guide](../getting-started/setup-guide.md)
and one of the serving quickstarts.

## Tested versions

This recipe was validated against the following components.

| Component | Version |
| --- | --- |
| Neuron SDK | 2.32 |
| vLLM Neuron plugin | Shipped with the Neuron 2.32 release |
| vLLM (upstream) | The version pinned by the vLLM Neuron plugin for Neuron 2.32 |
| Llama 3.3 70B checkpoint (FP8) | `nvidia/Llama-3.3-70B-Instruct-FP8` (Hugging Face) |

## Prerequisites

- **vLLM Neuron environment:** A working vLLM Neuron setup on the instance you
  deploy to. See the [setup guide](../getting-started/setup-guide.md).
- **Model access:** A Hugging Face account and an access token with read access to
  the FP8 checkpoint. Fine-grained tokens that lack this permission return HTTP 403
  on gated repos.
- **Disk:** The FP8 checkpoint is a large download (~70 GB). Provide a fast local
  path via `--download-dir` or pre-stage the checkpoint.
- **Instance:** A `trn2.48xlarge` or Trn3 instance.
- **Familiarity with vLLM serving:** See the
  [online serving quickstart](../getting-started/quickstart-online-serving.md).

## About static FP8

Llama 3.3 70B runs in **static FP8** from a ModelOpt-calibrated checkpoint such as
`nvidia/Llama-3.3-70B-Instruct-FP8`. FP8 roughly halves the weight footprint and
speeds up the compute-bound prefill phase, at a small accuracy cost you can
quantify from the [model recipe](../model-recipes/llama-3.md#accuracy-evaluation).

FP8 is **checkpoint-driven**: point the server at the FP8 checkpoint and the
Neuron backend detects the `quantization_config` and loads the FP8 weights
automatically. There is no FP8-specific launch flag, and **no difference between
Trn2 and Trn3** — both run the same static FP8 checkpoint. Do **not** set vLLM's
`--quantization` flag; the checkpoint's config selects the path. See
[FP8 static weight quantization](../guides/features-guide.md#fp8-static-weight-quantization)
in the features guide for how detection and loading work under the hood.

## Deploy on a single instance

One server serves the whole model with tensor parallelism. Use it for functional
validation, development, and serving.

### Set environment variables

```bash
# Compilation and execution timeouts for the 70B model.
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200

# Hugging Face token for the checkpoint.
export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Launch the server

```bash
vllm serve nvidia/Llama-3.3-70B-Instruct-FP8 \
    --tensor-parallel-size 16 \
    --enable-prefix-caching \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 8 \
    --block-size 128 \
    --optimization-level 2 \
    --additional-config '{
        "neuron_config": {
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [8],
            "fp8_packed_kv": true
        }
    }'
```

The server listens on port 8000. On the first launch it compiles the model graphs
before accepting traffic (roughly 15–20 minutes for the 70B model); subsequent
launches reuse cached NEFFs. See the
[setup guide](../getting-started/setup-guide.md) for cache configuration.

### Understand the key flags

- `--tensor-parallel-size 16` — shards the model across 16 NeuronCores (TP16),
  the configuration this recipe targets on a single instance.
- `--max-model-len 16384` — maximum combined prompt + generation length per
  request.
- `--max-num-batched-tokens 8192` — the maximum number of tokens processed in a
  single batched forward pass. Set it explicitly: `--enable-prefix-caching` turns
  on chunked prefill, and with chunked prefill enabled vLLM leaves
  `max_num_batched_tokens` at its 2048 default rather than raising it to
  `--max-model-len`. It must match the last entry of `num_batched_tokens_buckets`.
- `num_batched_tokens_buckets: [8192]` — the compiled prefill bucket sizes, giving
  a single, predictable compiled prefill graph. This sizes the prefill graphs; it
  does not by itself enable segmented prefill or set a segment size (that is
  `kv_segment_size_buckets`). The last entry must equal
  `--max-num-batched-tokens`.
- `kv_segment_size_buckets: [8192]` — the prefill segment size. Because
  `--max-num-batched-tokens` (8192) is below `--max-model-len` (16384), prefill is
  segmented: prompts longer than 8192 tokens are processed in 8192-token chunks.
  Prefix caching also requires segmented prefill, so this entry is what lets
  `--enable-prefix-caching` start at all. It must match
  `num_batched_tokens_buckets`.
- `fp8_packed_kv: true` — stores the FP8 KV cache in the packed (swizzled) layout.
  This checkpoint sets `kv_cache_quant_algo: FP8`, so the KV cache is FP8, and the
  segmented-prefill attention kernel can only read a *packed* FP8 K cache. Without
  this the server fails at startup with `NotImplementedError: FP8 segmented
  prefill requires a packed KV cache`.
- `--max-num-seqs` — the maximum number of concurrent sequences (the decode batch
  size). This is a **trade-off between interactivity and throughput**: a lower
  value gives each request more of the instance and lowers per-token latency
  (better interactivity), while a higher value packs more requests into each
  decode step for higher aggregate throughput at the cost of higher latency. Start
  at 8 and tune down for latency-sensitive traffic or up for throughput-oriented
  batch workloads; keep `num_seqs_buckets` in the `neuron_config` in sync with the
  value you pick.
- `--block-size 128` — the KV-cache block size in tokens. Neuron defaults
  `block_size` to 32; this recipe sets it to 128, the block size validated for the
  packed FP8 KV path used here (`fp8_packed_kv: true` with segmented prefill).
  `kv_segment_size_buckets` must stay divisible by `block_size`, so 128 divides the
  8192-token segment bucket cleanly.
- `--optimization-level 2` — raises the compiler optimization level from Neuron's
  O1 default to O2 for better serving performance on the 70B graphs. Neuron
  defaults to O1 (vLLM's own default is O2); passing this flag opts back into the
  higher optimization level. Expect a somewhat longer first compile in exchange.

### Validate the server

```bash
# Health check.
curl -i http://localhost:8000/health

# Sample completion.
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nvidia/Llama-3.3-70B-Instruct-FP8",
        "prompt": "The capital of France is ",
        "max_tokens": 16
    }'
```

You should see `HTTP/1.1 200 OK` on the health check, and an OpenAI-compatible
JSON payload with a coherent `choices[0].text` on the completion.

## Add EAGLE3 speculative decoding (optional)

Llama 3 supports EAGLE3 speculative decoding to raise decode throughput. Pair the
target with a matching EAGLE3 draft — for example
[RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3](https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3) —
via `--speculative-config`:

```bash
vllm serve nvidia/Llama-3.3-70B-Instruct-FP8 \
    --tensor-parallel-size 16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 8 \
    --block-size 128 \
    --optimization-level 2 \
    --additional-config '{
        "neuron_config": {
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [8],
            "fp8_packed_kv": true
        }
    }' \
    --speculative-config '{
        "method": "eagle3",
        "model": "RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3",
        "num_speculative_tokens": 3
      }'
```

The `neuron_config` block carries over unchanged from the launch command above —
the FP8 KV cache requirements apply here too. Expect a longer first compile,
since both the target and the draft model are compiled.

Under greedy sampling EAGLE3 is lossless — the target output is unchanged. For the
full walkthrough, including how to read the acceptance rate and tune
`num_speculative_tokens`, see the
[EAGLE3 speculative decoding tutorial](../tutorials/tutorial-eagle3-speculative-decoding-llama-3-1.md).

## Confirm your work

You are done when the server reports startup complete, `/health` returns 200, and
a completion request to port 8000 returns a coherent OpenAI-compatible response.

## Common issues

### Compilation hits the timeout

- **Possible solution:** Llama 3.3 70B is large; cold compilation can exceed the
  default 600-second budget (typically 15–20 minutes). Raise
  `VLLM_NEURON_COMPILATION_TIMEOUT` before launch. If compilation continues to
  time out, see the [setup guide](../getting-started/setup-guide.md) for cache
  configuration and shared-cache options.

### Out of memory during compilation or serving

- **Possible solution:** Reduce `--max-model-len` or `--max-num-seqs` (keeping
  `num_seqs_buckets` in sync). If you lower `--max-num-batched-tokens`, lower the
  last entry of `num_batched_tokens_buckets` to match, or startup validation
  fails.

### HTTP 403 when the server pulls the model

- **Possible solution:** Your Hugging Face token lacks read access to the
  checkpoint repository. Update the token at
  <https://huggingface.co/settings/tokens> and restart the server.

## Related information

- [Llama 3 model recipe](../model-recipes/llama-3.md) — Feature support, accuracy
  results, and supported checkpoints for Llama 3.
- [EAGLE3 speculative decoding tutorial](../tutorials/tutorial-eagle3-speculative-decoding-llama-3-1.md)
  — Add an EAGLE3 draft model for higher decode throughput.
- [Features guide](../guides/features-guide.md) — Feature configuration,
  including [FP8 static weight quantization](../guides/features-guide.md#fp8-static-weight-quantization).
- [Online serving quickstart](../getting-started/quickstart-online-serving.md) —
  Underlying online serving flow.
- [Accuracy debugging guide](../model-dev/accuracy-debugging-guide.md) —
  Investigate accuracy issues if validation completions look wrong.
