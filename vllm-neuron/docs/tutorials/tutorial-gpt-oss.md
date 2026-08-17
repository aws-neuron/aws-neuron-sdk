# Tutorial: Deploy gpt-oss with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying gpt-oss with vLLM
Neuron, covering a simple single-instance TP8 deployment for validation and a
disaggregated-inference deployment for best performance, for gpt-oss 20B and
120B on Trn3 (MXFP4) or Trn2 (BF16). -->
<!-- meta: keywords: vLLM, Neuron, gpt-oss, gpt-oss-20b, gpt-oss-120b, MoE,
MXFP4, single instance, disaggregated inference, expert parallelism, tutorial,
LLM serving, Trn2, Trn3, Trainium -->
<!-- meta: date_updated: 2026-07-15 -->
<!-- Content type: procedural-tutorial -->
<!-- Jira: NDOC-185 -->

This tutorial is a production-ready recipe for deploying gpt-oss with vLLM
Neuron. It covers **gpt-oss 20B** and **gpt-oss 120B**, MXFP4-quantized on Trn3
(both models also run on Trn2 in BF16), and presents two deployment paths:

1. **Single instance (non-DI):** one server with tensor parallelism. The easiest
   way to stand gpt-oss up.
2. **Disaggregated inference (DI):** prefill and decode served on separate
   instances. More setup, but yields the best performance under load.

For the model's feature support, accuracy results, and supported checkpoints,
see the [gpt-oss model recipe](../model-recipes/gpt-oss.md). This tutorial assumes you
have already worked through the [setup guide](../getting-started/setup-guide.md)
and one of the serving quickstarts.

## Tested versions

This recipe was validated against the following components. If you are on a newer
release, confirm the parameter set still applies before promoting it to
production.

| Component | Version |
| --- | --- |
| Neuron SDK | 2.31 |
| vLLM Neuron plugin | Shipped with the Neuron 2.31 release |
| vLLM (upstream) | The version pinned by the vLLM Neuron plugin for Neuron 2.31 |
| gpt-oss 20B checkpoint | `openai/gpt-oss-20b` (Hugging Face) |
| gpt-oss 120B checkpoint | `openai/gpt-oss-120b` (Hugging Face) |

## Choose a deployment

| | Single instance (non-DI) | Disaggregated inference (DI) |
| --- | --- | --- |
| Setup | One server, one command | Separate prefill + decode instances, a proxy, and EFA networking |
| Best for | Functional validation, development, light or non-latency-critical traffic | Production and stress/benchmark workloads |
| Performance | Adequate for simple tasks | **Best throughput and latency under load** |

Start with the single-instance deployment to validate the model and your
environment. Move to disaggregated inference when you need the best performance —
**only the DI setup delivers the full throughput and latency of gpt-oss on
Neuron.**

## Prerequisites

- **vLLM Neuron environment:** A working vLLM Neuron setup on each instance
  you deploy to. See the [setup guide](../getting-started/setup-guide.md).
- **Model access and disk:** Access to the gpt-oss checkpoint you intend to
  serve. gpt-oss 120B is a large download; provide a fast local path via
  `--download-dir` or pre-stage the checkpoint.
- **Instances:** On Trn3, both gpt-oss 20B and 120B use MXFP4 weights
  (auto-selected) — the path this recipe targets. Both models also run on Trn2 in
  BF16. The disaggregated deployment additionally requires **two instances with
  EFA connectivity** between them (see that section).
- **Familiarity with vLLM serving:** See the
  [online serving quickstart](../getting-started/quickstart-online-serving.md).
  For the disaggregated path, also work through the
  [disaggregated inference tutorial](../tutorials/tutorial-di-1p1d-xpyd.md).

## Deploy on a single instance (non-DI)

This is the simplest path: one server serving the whole model with tensor
parallelism. Use it for functional validation, development, and light workloads.

### Set environment variables

```bash
# Compilation and execution timeouts for gpt-oss.
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
```

### Launch the server

```bash
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 8 \
    --enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --hf-overrides '{"quantization_config": {}}' \
    --additional-config '{
        "neuron_config": {
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
```

The Neuron backend auto-selects the MXFP4 weights on Trn3. To serve **gpt-oss
20B** instead, swap the model identifier to `openai/gpt-oss-20b`; the same
command works on Trn3 (MXFP4) or Trn2 (BF16).

The server listens on port 8000. On the first launch it compiles the model graphs
before accepting traffic; subsequent launches reuse cached NEFFs. See the
[setup guide](../getting-started/setup-guide.md) for cache configuration.

### Validate the single-instance server

```bash
# Health check.
curl -i http://localhost:8000/health

# Sample completion (swap in openai/gpt-oss-20b if you launched 20B).
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openai/gpt-oss-120b",
        "prompt": "The capital of France is ",
        "max_tokens": 16
    }'
```

You should see `HTTP/1.1 200 OK` on the health check, and an OpenAI-compatible
JSON payload with a coherent `choices[0].text` on the completion.

## Deploy with disaggregated inference (best performance)

Disaggregated inference separates prompt processing (prefill) from token
generation (decode) so you can size and scale each phase independently, which is
what delivers the best throughput and latency for gpt-oss. This section pins the
tested topology and parameter set; for the mechanics of DI on Neuron — the NIXL
KV connector, the proxy router, and how to scale from 1P1D to xPyD — see the
[disaggregated inference tutorial](../tutorials/tutorial-di-1p1d-xpyd.md) and the
[disaggregated inference design](../design/vllm/disaggregated-inference.md).

### Target configuration

This deploys a **1P1D** topology: one prefill instance (`kv_producer`) and one
decode instance (`kv_consumer`), fronted by a proxy router. KV cache is
transferred from prefill to decode over NIXL using the `LIBFABRIC` backend (EFA
on AWS). Both instances serve MXFP4 weights.

The prefill configuration is identical for both models. The decode configuration
differs only in its data-parallel degree, which sets the expert-parallel width
and the global decode batch size.

| Role | Parallelism | `max_num_seqs` | Global decode batch |
| --- | --- | --- | --- |
| Prefill (20B and 120B) | attention TP4 (DP1), MoE EP2 | 1 | — |
| Decode, gpt-oss 20B | TP8 DP4 EP32 (1 expert/rank) | 4 | 16 (4 × DP4) |
| Decode, gpt-oss 120B | TP8 DP8 EP64 (2 experts/rank) | 4 | 32 (4 × DP8) |

`max_num_seqs` is the batch bucket **per data-parallel replica**, so the global
decode batch is `max_num_seqs × data-parallel degree` — for the 120B TP8 DP8 EP64
setup above, `4 × 8 = 32`. The `max_num_seqs 4` in this recipe is a conservative
starting point. For a throughput-optimized deployment, raise `max_num_seqs` so
the global decode batch reaches **128 or more** (for 120B at DP8, that is
`max_num_seqs 16`), then confirm the larger batch fits in HBM at your
`max_model_len`.

Parameters shared by all roles:

| Setting | Value | Notes |
| --- | --- | --- |
| Quantization | MXFP4 | Pass `--dtype bfloat16` and `neuron_config.quantization: "mxfp4"`; the Neuron backend loads the published MXFP4 weights. |
| `max_model_len` | 16384 | Set explicitly on every role. |
| `max_num_batched_tokens` | 8192 | Prefill segment size; matches the single kv-segment and batched-tokens bucket. |
| Bucketing | `kv_segment_size_buckets: [8192]`, `num_batched_tokens_buckets: [8192]` | One bucket per role for a single, predictable compiled graph. |
| Prefix caching | On (default) | See [prefix caching benchmark tutorial](../tutorials/tutorial-prefix-caching-gpt-oss-benchmarking.md) for tuning. |
| On-device sampling | top-k (default) | Set `all_greedy: true` in `on_device_sampling_config` for deterministic outputs. |

### DI prerequisites

- **Two instances:** One for prefill and one for decode, each with all 64
  NeuronCores available.
- **EFA connectivity between the instances:** DI transfers KV cache over NIXL on
  the `LIBFABRIC` backend. The two instances must reach each other over EFA. See
  the [disaggregated inference tutorial](../tutorials/tutorial-di-1p1d-xpyd.md)
  for network setup.
- **The checkpoint staged on both instances.**

### 1. Set environment variables on both instances

gpt-oss compilation and multi-node startup run longer than the vLLM defaults. Set
these on both the prefill and decode instances before launching.

```bash
# Compilation and startup timeouts for gpt-oss.
export VLLM_NEURON_COMPILATION_TIMEOUT=2400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_READY_TIMEOUT_S=5400

# NIXL side channel: listen on all interfaces so the peer can reach it over EFA.
export VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0
```

### 2. Launch the decode server

On the **decode instance**, bind all 64 cores and set the decode-side NIXL side
channel port. This is the `kv_consumer`.

```bash
export NEURON_VISIBLE_DEVICES="0-63"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5659

vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 8 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --optimization-level 2 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --max-logprobs 0 \
    --no-disable-hybrid-kv-cache-manager \
    --port 8200 \
    --hf-overrides '{"quantization_config": {}}' \
    --kv-transfer-config '{"kv_connector": "NeuronNixlConnector", "kv_role": "kv_consumer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "mxfp4",
            "embedding_dp_size": 8,
            "lm_head_dp_size": 8,
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
```

For **gpt-oss 20B**, serve `openai/gpt-oss-20b` and drop the decode-side
data-parallel degree to 4 (`--data-parallel-size 4`, `embedding_dp_size: 4`,
`lm_head_dp_size: 4`, and omit `--optimization-level 2`). This yields TP8 DP4
EP32 and a global decode batch of 16.

The decode server reports `Application startup complete` on port 8200 once it is
ready. It has no dependency on the prefill server at startup, so you can launch
the prefill server (next step) at the same time rather than waiting for decode to
finish; the proxy in step 4 is the only component that requires both to be up.

### 3. Launch the prefill server

On the **prefill instance**, bind four cores and set the prefill-side NIXL side
channel port. This is the `kv_producer`. The prefill configuration is the same
for both models — only the model identifier changes.

```bash
export NEURON_VISIBLE_DEVICES="0-3"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5559

vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 1 \
    --no-disable-hybrid-kv-cache-manager \
    --port 8100 \
    --hf-overrides '{"quantization_config": {}}' \
    --kv-transfer-config '{"kv_connector": "NeuronNixlConnector", "kv_role": "kv_producer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "mxfp4",
            "ep_degree": 2,
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [1]
        }
    }'
```

The prefill server reports `Application startup complete` on port 8100 once it is
ready.

:::{note}
On the first launch each server compiles the model graphs before it accepts
traffic; this is why the timeouts in step 1 are raised. Subsequent launches reuse
cached NEFFs unless you invalidate the cache. See the
[setup guide](../getting-started/setup-guide.md) for cache configuration.
:::

#### Scale prefill to fully utilize the instance

A single TP4 prefill server uses only 4 of the instance's 64 cores, leaving the
other 60 idle and capping end-to-end throughput. For best performance, tile up to
16 TP4 prefill servers across those 64 cores to match one TP8 DP8 EP64 decode
server — a **16P1D** topology that leaves no idle cores. Because prefill and
decode scale independently under DI, adding prefill servers does not change the
decode topology.

Launch each additional prefill server with the same command above, but give each
one a **distinct 4-core slice** via `NEURON_VISIBLE_DEVICES` so the servers do not
contend for the same cores, along with a unique HTTP port and a unique
`VLLM_NIXL_SIDE_CHANNEL_PORT`:

```bash
# Prefill server 0
export NEURON_VISIBLE_DEVICES="0-3"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5559
# vllm serve ... --port 8100

# Prefill server 1
export NEURON_VISIBLE_DEVICES="4-7"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5560
# vllm serve ... --port 8101

# ... and so on through server 15 (NEURON_VISIBLE_DEVICES="60-63", port 8115)
```

Then list every prefill port on the proxy in the next step, and it round-robins
requests across them:

```bash
    --prefiller-host PREFILL_HOST --prefiller-port 8100 8101 8102 ...
```

Run these as **16 separate prefill servers**, not one server with a data-parallel
degree of 16. With expert parallelism enabled, raising the data-parallel degree
turns the MoE layers into a cross-DP expert-parallel sharding — a different (and
here undesirable) expert layout. Keeping each server at TP4 DP1 preserves the
tested prefill sharding and simply replicates it.

For the general xPyD scaling procedure, see the
[disaggregated inference tutorial](../tutorials/tutorial-di-1p1d-xpyd.md).

### 4. Launch the proxy router

Launch the proxy only after **both** the decode and prefill servers have reported
`Application startup complete`. The proxy accepts client requests and coordinates
the prefill → decode handoff. Run it on the prefill instance (or any instance
that can reach both). Replace `PREFILL_HOST` and `DECODE_HOST` with the
instances' addresses.

```bash
python3 examples/vllm_neuron/vllm/disaggregated_inference/toy_proxy_server.py \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-host PREFILL_HOST --prefiller-port 8100 \
    --decoder-host DECODE_HOST --decoder-port 8200
```

The proxy listens on port 8000 and routes each request to the prefill server for
prompt processing, then to the decode server, which pulls the KV cache from
prefill over NIXL and streams tokens back. `toy_proxy_server.py` ships in the
vLLM Neuron repository. See the
[disaggregated inference tutorial](../tutorials/tutorial-di-1p1d-xpyd.md) for
details on the proxy and the request lifecycle.

If you launched multiple prefill servers (see
[Scale prefill to fully utilize the instance](#scale-prefill-to-fully-utilize-the-instance)),
pass all their ports to `--prefiller-port`.

### 5. Understand the gpt-oss-specific flags

A few flags warrant explicit explanation because they are gpt-oss specific, DI
specific, or because the defaults are not what you want for this recipe.

- `--enable-expert-parallel` with `ep_degree` / data parallelism — gpt-oss is a
  Mixture-of-Experts model. On decode, expert parallelism is derived from the
  tensor- and data-parallel degrees (TP8 × DP4 = EP32 for 20B; TP8 × DP8 = EP64
  for 120B). On prefill, `ep_degree: 2` sets the MoE expert-parallel width.
- `embedding_dp_size` / `lm_head_dp_size` — set to the decode data-parallel
  degree so the embedding and LM-head layers are replicated across the DP
  replicas.
- `--max-logprobs 0` (decode) — disables logprob support so the decode graph
  skips gathering the full logits tensor before sampling, which saves decode
  time and memory. Set it only on the decode server; leave it off if your
  workload requests logprobs.
- `--optimization-level 2` (120B decode) — raises the compiler optimization
  level for the larger model's decode graph.

### 6. Validate the deployment

Send requests **to the proxy** (port 8000), not to the prefill or decode servers
directly. With all three processes running, confirm the response shape.

```bash
# Health check against the proxy.
curl -i http://PREFILL_HOST:8000/health
```

You should see `HTTP/1.1 200 OK`.

Next, send a sample completion request. Replace `openai/gpt-oss-120b` with
`openai/gpt-oss-20b` if you launched the 20B variant.

```bash
curl http://PREFILL_HOST:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openai/gpt-oss-120b",
        "prompt": "The capital of France is ",
        "max_tokens": 16
    }'
```

The response is an OpenAI-compatible JSON payload. Confirm the following:

- `choices[0].text` contains a coherent continuation of the prompt.
- `choices[0].finish_reason` is `length` (because the request hit `max_tokens`)
  or `stop`.
- `usage.prompt_tokens` and `usage.completion_tokens` are populated.

:::{note}
With default top-k sampling, completion content varies across runs. For
deterministic outputs, add `"on_device_sampling_config": {"all_greedy": true}`
to the `neuron_config` on both roles. For substantive accuracy validation, see
the [accuracy debugging guide](../model-dev/accuracy-debugging-guide.md) and the
[accuracy debugger tool](../model-dev/how-to-use-accuracy-debugger.md).
:::

### 7. Benchmark and tune prefix caching

Before promoting this configuration to production, run a workload-shaped
benchmark. The companion tutorial uses gpt-oss as its subject, so its setup,
datasets, and tuning guidance apply directly to this recipe.

- [Prefix caching benchmark tutorial](../tutorials/tutorial-prefix-caching-gpt-oss-benchmarking.md)
  — Quantify the throughput and time-to-first-token impact of prefix caching for
  your prompt distribution and tune accordingly.

## Confirm your work

For the **single-instance** deployment, you are done when the server reports
startup complete, `/health` returns 200, and a completion request to port 8000
returns a coherent OpenAI-compatible response.

For the **disaggregated** deployment, you have a successful deployment when:

- The decode and prefill servers each report `Application startup complete`, and
  the proxy's `/health` endpoint returns 200.
- A completion request sent through the proxy returns an OpenAI-compatible
  response on the validation prompt with the expected shape.
- Server logs show the decode server pulling KV cache from prefill over NIXL /
  LIBFABRIC (no NIXL handshake errors).
- A short representative-traffic run produces stable, on-spec completions, and
  throughput and latency on your benchmark traffic meet the targets you set.

## Common issues

### Compilation hits the timeout

- **Possible solution:** gpt-oss 120B is large; cold compilation can exceed the
  default 600-second budget. Raise `VLLM_NEURON_COMPILATION_TIMEOUT` (and, for
  the disaggregated path, `VLLM_ENGINE_READY_TIMEOUT_S`) before launch. If
  compilation continues to time out, see the
  [setup guide](../getting-started/setup-guide.md) for cache configuration and
  shared-cache options.

### NIXL handshake fails or the decode server cannot reach prefill (DI)

- **Possible solution:** Confirm `VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0` on both
  roles, that the prefill and decode side-channel ports differ (5559 and 5659 in
  this recipe), and that the two instances can reach each other over EFA.

### Degenerate or garbled decode output (DI)

- **Possible solution:** Keep the decode topology at the tested expert-parallel
  degree for the model (EP32 for 20B, EP64 for 120B). Changing the
  data-parallel degree changes the expert-parallel width, and untested EP degrees
  have produced degenerate output. Match `embedding_dp_size` and `lm_head_dp_size`
  to `--data-parallel-size`.

## Related information

- [gpt-oss model recipe](../model-recipes/gpt-oss.md) — Feature support, accuracy
  results, and supported checkpoints for gpt-oss.
- [Disaggregated inference tutorial](../tutorials/tutorial-di-1p1d-xpyd.md) —
  DI mechanics: NIXL KV connector, proxy router, and 1P1D → xPyD scaling.
- [Disaggregated inference design](../design/vllm/disaggregated-inference.md) —
  How DI control and data flow are implemented on Neuron.
- [Prefix caching benchmark tutorial](../tutorials/tutorial-prefix-caching-gpt-oss-benchmarking.md)
  — Benchmark prefix caching with gpt-oss across open-source datasets.
- [Features guide](../guides/features-guide.md) — Feature configuration for
  gpt-oss and other models.
- [Online serving quickstart](../getting-started/quickstart-online-serving.md) —
  Underlying online serving flow.
- [Accuracy debugging guide](../model-dev/accuracy-debugging-guide.md) —
  Investigate accuracy issues if validation completions look wrong.
