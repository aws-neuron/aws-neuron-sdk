# Tutorial: Deploy Qwen3-Embedding with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying Qwen3-Embedding-8B with
vLLM Neuron, covering an online OpenAI-compatible /v1/embeddings server and
offline batch embedding with LLM.embed(), plus variable output dimensions, on
Trn2 or Trn3. -->

<!-- meta: keywords: vLLM, Neuron, Qwen3-Embedding, Qwen3-Embedding-8B,
embeddings, pooling, /v1/embeddings, LLM.embed, Matryoshka, retrieval, RAG,
tutorial, embedding serving, Trn2, Trn3, Trainium -->

<!-- meta: date_updated: 2026-08-04 -->

<!-- Content type: procedural-tutorial -->

This tutorial is a production-ready recipe for deploying Qwen3-Embedding with
vLLM Neuron. It covers **Qwen3-Embedding-8B** in BF16 on Trn2 or Trn3, and
presents two deployment paths:

1. **Online server:** an OpenAI-compatible `/v1/embeddings` endpoint. The way to
   serve embeddings to an application.
2. **Offline batch:** the `LLM.embed()` Python API in a single process. The
   higher-throughput path for building an index over a corpus you already have.

For the model's feature support, accuracy results, and supported checkpoints,
see the
[Qwen3-Embedding model recipe](../model-recipes/qwen3-embedding-8b.md). This
tutorial assumes you have already worked through the
[setup guide](../getting-started/setup-guide.md) and one of the serving
quickstarts.

## Tested versions

This recipe was validated against the following components. If you are on a newer
release, confirm the parameter set still applies before promoting it to
production.

| Component                     | Version                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| Neuron SDK                    | 2.32                                                         |
| vLLM Neuron plugin            | Shipped with the Neuron 2.32 release                         |
| vLLM (upstream)               | The version pinned by the vLLM Neuron plugin for Neuron 2.32 |
| Qwen3-Embedding-8B checkpoint | `Qwen/Qwen3-Embedding-8B` (Hugging Face)                   |

## Choose a deployment

|           | Online server                                          | Offline batch                           |
| --------- | ------------------------------------------------------ | --------------------------------------- |
| Setup     | One`vllm serve` command                              | One Python script                       |
| Interface | `POST /v1/embeddings` (OpenAI-compatible)            | `LLM.embed()`                         |
| Best for  | Applications embedding queries and documents on demand | Bulk index building over a fixed corpus |

The two paths are the same model reached through different APIs — the online
endpoint is a network wrapper around the offline `LLM.embed()` call, so both take
the same configuration and produce identical vectors. Start with the online
server to validate the model and your environment; use the offline path when you
have a corpus to embed in one pass.

## Prerequisites

- **vLLM Neuron environment:** A working vLLM Neuron setup on the instance you
  deploy to. See the [setup guide](../getting-started/setup-guide.md).
- **Model access and disk:** Access to the `Qwen/Qwen3-Embedding-8B` checkpoint
  (~16 GB in BF16). Provide a fast local path via `--download-dir` or pre-stage
  the checkpoint.
- **Instances:** A Trn2 or Trn3 instance with enough NeuronCores for your tensor
  parallel size — the examples below use `--tensor-parallel-size 8`, which fits on
  a `trn2.12xlarge`. Qwen3-Embedding-8B runs in BF16 on both Trn2 and Trn3.
- **Familiarity with vLLM serving:** See the
  [online serving quickstart](../getting-started/quickstart-online-serving.md).

## What to know before you deploy

Qwen3-Embedding runs as a pooling model, which affects the commands below in a
few practical ways:

- Launch with `--runner pooling`. It is required — the checkpoint declares
  `architectures: ["Qwen3ForCausalLM"]`, so without it the model loads as a
  generative model — and it makes the server mount `/v1/embeddings`.
- Sampling, speculative decoding, and structured-output flags do not apply, since
  the model emits a vector rather than generated tokens.
- Output vectors are L2-normalized, so on the client a dot product is already the
  cosine similarity.

For why the pooling path behaves this way (the prefill-only execution model and
its consequences), see [Pooling models on Trainium](../design/vllm/pooling-models.md).

## Deploy an online embedding server

### Set environment variables

```bash
# Compilation and execution timeouts for Qwen3-Embedding.
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
```

### Launch the server

```bash
vllm serve Qwen/Qwen3-Embedding-8B \
    --runner pooling \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --tensor-parallel-size 8 \
    --additional-config '{
        "neuron_config": {
            "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096]
        }
    }'
```

The server listens on port 8000. On the first launch it compiles the prefill
graphs before accepting traffic; subsequent launches reuse cached NEFFs. See the
[setup guide](../getting-started/setup-guide.md) for cache configuration.

`num_batched_tokens_buckets` sets the compiled prefill shapes; see
[bucketing and dynamic shapes](../guides/features-guide.md#bucketing-and-dynamic-shapes)
for how to size them for your traffic.

For short-sequence workloads with no shared prefix — the common embedding case —
add `--no-enable-prefix-caching` for better performance: disabling it avoids the segmented-prefill
overhead. See
[prefix caching](../guides/features-guide.md#prefix-caching) for when it helps.

### Validate the online server

```bash
# Health check.
curl -i http://localhost:8000/health

# Sample embedding request.
curl http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": ["What is the capital of France?", "Paris is the capital of France."]
    }'
```

You should see `HTTP/1.1 200 OK` on the health check, and an OpenAI-compatible
JSON payload with one `data[i].embedding` array of 4096 floats per input.

> **Note:** For retrieval workloads, prefix each **query** with a task
> instruction; embed documents without one. Qwen3-Embedding is trained for the
> template `Instruct: {task description}\nQuery: {query text}`, and using it
> measurably improves retrieval quality over sending the bare query.

## Deploy offline batch embedding

For bulk index building, `LLM.embed()` runs the same model in your own process
and avoids per-request HTTP overhead.

```python
import os

os.environ["VLLM_NEURON_COMPILATION_TIMEOUT"] = "1200"
os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1200"

from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-Embedding-8B",
    runner="pooling",
    dtype="bfloat16",
    max_model_len=4096,
    tensor_parallel_size=8,
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096]
        }
    },
)

docs = [
    "Retrieval-augmented generation grounds answers in retrieved documents.",
    "def add(a, b):\n    return a + b",
    "The quarterly earnings report exceeded analyst expectations by 12%.",
    "Photosynthesis converts sunlight, water, and CO2 into glucose.",
]

for doc, out in zip(docs, llm.embed(docs)):
    vec = out.outputs.embedding
    print(f"dim={len(vec)}  first3={[round(x, 4) for x in vec[:3]]}  | {doc[:40]}")
```

`llm.embed()` requires `runner="pooling"` and raises on a generative runner.

## Request a smaller embedding (optional)

Qwen3-Embedding is trained with Matryoshka Representation Learning (MRL), so a
truncated prefix of the output vector is still a usable embedding — letting you
trade a little quality for a much smaller index. Because the published
`config.json` does not declare that support, opt in at launch:

```bash
vllm serve Qwen/Qwen3-Embedding-8B \
    --runner pooling \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --tensor-parallel-size 8 \
    --hf-overrides '{"is_matryoshka": true}'
```

Then pass `dimensions` per request. Requests in one batch may ask for different
widths:

```bash
curl http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": "What is the capital of France?",
        "dimensions": 256
    }'
```

The truncated vector is re-normalized, so it remains unit-norm and directly
comparable by dot product. For the measured quality-versus-width trade-off, see
the [model recipe](../model-recipes/qwen3-embedding-8b.md#accuracy-evaluation).

## Confirm your work

For the **online** deployment, you are done when the server reports startup
complete, `/health` returns 200, and an embeddings request to port 8000 returns
one 4096-float vector per input with L2 norm ≈ 1.0.

For the **offline** deployment, you are done when `llm.embed()` returns one
embedding per input document, each of the expected dimension.

Either way, sanity-check that the vectors are meaningful, not just well-formed:
embed one related pair and one unrelated sentence — for example
`"What is the capital of France?"`, `"Paris is the capital of France."`, and
`"The mitochondrion is the powerhouse of the cell."` — and take the dot product of
each pair. The related pair should score visibly higher (typically > 0.6) than the
unrelated pair (typically < 0.4). If the two are close, the embeddings are not
capturing meaning and something is wrong with the deployment.

## Common issues

### Compilation hits the timeout

- **Possible solution:** Raise `VLLM_NEURON_COMPILATION_TIMEOUT` and
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` before launch. If compilation continues to
  time out, see the [setup guide](../getting-started/setup-guide.md) for cache
  configuration and shared-cache options.

### The server mounts chat endpoints, or `/v1/embeddings` returns 404

- **Possible solution:** Add `--runner pooling`. Qwen3-Embedding-8B declares
  `architectures: ["Qwen3ForCausalLM"]`, so without that flag the checkpoint is
  loaded as a generative model.

### Startup fails on FP8 KV cache or async scheduling

- **Possible solution:** An FP8 KV cache is not supported on the embedding path;
  use the default. Async scheduling is not supported for pooling models; pass
  `--no-async-scheduling`.

### A `dimensions` request is rejected

- **Possible solution:** Launch with `--hf-overrides '{"is_matryoshka": true}'`.
  The published checkpoint config does not declare truncation support, so vLLM
  rejects the parameter until you opt in.

### Prefix caching is enabled but throughput does not improve

- **Possible solution:** The shared prefix is shorter than one whole KV segment,
  so padding returns the saved tokens to the same compiled shape. Either lengthen
  the shared prefix or lower the segment size. Note that segmented prefill also
  pads short requests up to the segment size, so size segments to your traffic
  rather than to the model's maximum context.

### Latency grows with client concurrency while throughput stays flat

- **Possible solution:** Expected behavior — prefill processes one request at a
  time. Raise throughput by running more server replicas behind a load balancer,
  not by raising concurrency.

## Related information

- [Qwen3-Embedding model recipe](../model-recipes/qwen3-embedding-8b.md) —
  Feature support, accuracy results, and supported checkpoints.
- [Pooling models design](../design/vllm/pooling-models.md) — How the pooling
  execution path is implemented on Neuron.
- [Features guide](../guides/features-guide.md) — Feature configuration for
  Qwen3-Embedding and other models.
- [Online serving quickstart](../getting-started/quickstart-online-serving.md) —
  Underlying online serving flow.
- [vLLM pooling models](https://docs.vllm.ai/en/stable/models/pooling_models.html)
  — Upstream documentation for `--runner pooling`.
- [vLLM embeddings API](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#embeddings-api)
  — Request and response schema for `/v1/embeddings`.
