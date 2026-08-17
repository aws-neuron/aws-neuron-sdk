# Pooling models on Trainium

<!-- meta: description: How embedding (pooling) models run on AWS Trainium with
vLLM Neuron — the prefill-only execution path, what that means for batching,
memory, and prefix caching, and which serving features apply. -->

<!-- meta: content_type: conceptual-overview -->

<!-- meta: date_updated: 2026-08-04 -->

**Pooling models** are models that return a vector per request instead of
generated text. vLLM groups them by task: `embed` (embeddings), `classify` and
`score` (sequence classification and scoring), and the token-wise tasks
`token_embed` and `token_classify`. On Neuron, **`embed` is the task supported
today** — see [Supported tasks](#supported-tasks) for the full matrix.

This topic covers how pooling models execute on AWS Trainium with vLLM Neuron and
why their behavior differs from generative models in ways that affect how you
configure and size a deployment.

## Applies to

This concept is applicable to:

- Serving embedding models for semantic search, retrieval, RAG, clustering, and
  deduplication
- Understanding why sampling, speculative decoding, and disaggregated inference
  do not apply to embedding endpoints
- Sizing and tuning an embedding deployment for throughput or latency

## What a pooling model does differently

A pooling model shares the same transformer backbone as a generative model. The
two diverge only at the output stage:

```text
Generative:  forward -> logits        -> sampler -> next token   (prefill + decode)
Pooling:     forward -> hidden states -> pooler  -> embedding    (prefill only)
```

A generative model projects the final hidden state through its language-model
head to vocabulary logits, samples a token, and then repeats — appending to the
KV cache once per generated token. A pooling model instead applies a **pooler** to
the backbone's hidden states to produce one fixed-size vector per request, and
stops. There is no token to generate, so there is no decode phase.

vLLM Neuron selects this path from the `--runner pooling` flag. The endpoints a
pooling server exposes depend on the task its pooler reports; because `embed` is
the task supported on Neuron today, the server exposes the OpenAI-compatible
`/v1/embeddings` endpoint in place of the chat and completions endpoints, and the
offline `LLM.embed()` API becomes available.

Everything is shared with the generative path: the same attention kernels, the
same weight loading and tensor-parallel sharding, the same compilation pipeline,
and the same KV cache implementation. Only the output stage is different, which is
why an embedding model inherits the backbone's performance characteristics on
Trainium.

## How pooling produces one vector per request

The pooler decides which of a request's token positions contribute to its vector:

| Pooling method | Reads                             | Typical model                                      |
| -------------- | --------------------------------- | -------------------------------------------------- |
| Last-token     | the final token's hidden state    | decoder-based embedders, including Qwen3-Embedding |
| CLS            | the first token's hidden state    | encoder models (BERT family)                       |
| Mean           | the average over all token states | symmetric embedders                                |

Which method applies is determined by the checkpoint, so you do not normally
configure it.

Because the output is L2-normalized, the dot product of two embedding vectors
**is** their cosine similarity. No further normalization is needed on the client.

Some embedding models additionally support returning a **truncated** vector,
which trades a small amount of quality for a much smaller index. Request it per
call with the `dimensions` parameter; the truncated vector is re-normalized, so it
remains unit-norm and directly comparable.

## What prefill-only means in practice

Three consequences follow, and they explain most of the configuration differences
between an embedding deployment and a generative one.

**Startup is shorter.** Only prefill graphs are compiled and warmed up. Neuron
compiles fixed-shape programs ahead of serving, and an embedding model needs no
decode graphs at all, so there are fewer to build.

**Features that act on generated tokens do not apply.** Sampling parameters,
on-device sampling, speculative decoding, and structured outputs all operate on a
sampled token, and a pooling model emits none. Disaggregated inference likewise
presupposes a decode phase to separate from prefill. Async scheduling, which
overlaps host work with the decode loop, has nothing to overlap and is not
supported. Requesting an FP8 KV cache is rejected at startup: a prefill-only
workload never reads the KV cache back, so there is no bandwidth benefit to gain.

> **Note:** "Prefill-only" means no decode step, not no KV cache. The prompt's
> own causal self-attention reads the keys and values of earlier tokens, and that
> storage is the KV cache. What an embedding model skips is the autoregressive
> loop in which the cache grows one token at a time.

## Converting a generative model to a pooling model

You do not need a purpose-built embedding checkpoint to produce embeddings on
Neuron. Passing `--convert embed` turns any single-tower text generative model
into an embedding model: it drops the language-model head and attaches a pooler,
while the backbone, its weights, and its compiled graphs are reused unchanged.
This makes embeddings available for architectures that ship no dedicated
embedding checkpoint, and for quick prototyping on a model you already have.

`--convert embed` is the only conversion supported on Neuron today; see
[Supported tasks](#supported-tasks) for the full matrix.

## Supported tasks

vLLM defines pooling tasks in two families: **sequence-wise** tasks, which reduce
a request to one vector, and **token-wise** tasks, which return one vector per
token. vLLM Neuron officially supports the sequence-wise `embed` task today.

| Task               | Family        | API                                         | Status on Neuron |
| ------------------ | ------------- | ------------------------------------------- | ---------------- |
| `embed`          | sequence-wise | `/v1/embeddings`, `LLM.embed()`         | ✅ Supported     |
| `classify`       | sequence-wise | `/classify`, `LLM.classify()`           | ❌ Not supported |
| `score`          | sequence-wise | `/score`, `/v1/rerank`, `LLM.score()` | ❌ Not supported |
| `token_embed`    | token-wise    | `/pooling`, `LLM.encode()`              | ❌ Not supported |
| `token_classify` | token-wise    | `/pooling`, `LLM.encode()`              | ❌ Not supported |

## Related Information

- [Qwen3-Embedding model recipe](../../model-recipes/qwen3-embedding-8b.md) —
  Feature support and accuracy results for the reference embedding model.
- [Deploy Qwen3-Embedding tutorial](../../tutorials/tutorial-qwen3-embedding-8b.md)
  — End-to-end online and offline embedding deployment.
