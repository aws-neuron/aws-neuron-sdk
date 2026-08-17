# Qwen3-Embedding Model Recipe

<!-- meta: description: Model recipe for deploying Qwen3-Embedding with vLLM on
Neuron, including supported checkpoints, feature support, accuracy results, and a
link to the end-to-end deployment tutorial for embedding serving on Trn2 or
Trn3. -->

<!-- meta: keywords: vLLM, Neuron, Qwen3-Embedding, Qwen3-Embedding-8B,
embeddings, pooling, retrieval, RAG, MTEB, Matryoshka, model recipe, model card,
embedding serving, Trn2, Trn3, Trainium -->

<!-- meta: date_updated: 2026-08-04 -->

<!-- Content type: model-card -->

## Introduction

[Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-8B) is an
open-weight text-embedding model family released by the Qwen team, built on the
Qwen3 backbone and designed for semantic search, retrieval, RAG, clustering, and
deduplication. Given a text input it returns a single fixed-size vector instead
of generated text, so each request runs one forward pass over the prompt.

Qwen3-Embedding is supported for inference serving with
[vLLM](https://github.com/vllm-project/vllm) using the Neuron SDK on AWS
Trainium2 (`trn2`) and Trainium3 (`trn3`) hardware.

**Compatible model checkpoints:**

| Model                | HuggingFace                                                                  | Hardware   | Embedding dim | Quantization | Validated |
| -------------------- | ---------------------------------------------------------------------------- | ---------- | ------------- | ------------ | --------- |
| Qwen3-Embedding-8B   | [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)     | Trn2, Trn3 | 4096          | BF16         | ✅        |
| Qwen3-Embedding-4B   | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)     | Trn2, Trn3 | 2560          | BF16         | ❌        |
| Qwen3-Embedding-0.6B | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Trn2, Trn3 | 1024          | BF16         | ❌        |

> The 4B and 0.6B checkpoints share the same Qwen3 architecture as the 8B and are
> expected to work through the same pooling path, but have not yet been validated
> on Neuron; the accuracy and performance results below are for the 8B.
>
> Embedding serving requires `--runner pooling`. All three checkpoints declare
> `architectures: ["Qwen3ForCausalLM"]` in their `config.json` — the same value as
> the generative Qwen3 models — so without that flag the checkpoint is loaded as
> a generative model.

## Features

Per-model feature availability for Qwen3-Embedding. See the
[features guide](../guides/features-guide.md) for configuration details and the
cross-model feature compatibility matrix.

| Category               | Feature                                          | Status |
| ---------------------- | ------------------------------------------------ | ------ |
| **Inputs**       | Text                                             | ✅     |
| **Tasks**        | Embeddings (`/v1/embeddings`, `LLM.embed()`) | ✅     |
|                        | Classification / scoring / reranking             | ❌     |
| **Quantization** | BF16 weights                                     | ✅     |
|                        | FP8 / MXFP4 weights                              | ❌     |
|                        | KV cache FP8                                     | ❌     |
| **Parallelism**  | Tensor parallelism (TP)                          | ✅     |
|                        | Data parallelism (DP)                            | ✅     |
|                        | Pipeline parallelism (PP)                        | ❌     |
|                        | Context parallelism (CP)                         | ❌     |
| **Performance**  | Segmented prefill                                | ✅     |
|                        | Prefix caching (APC)                             | ✅     |
|                        | Matryoshka (variable output dimensions)          | ✅     |
|                        | Async scheduling                                 | ❌     |
|                        | Disaggregated inference (1P1D / xPyD)            | ❌     |
| **Compilation**  | torch.compile (XLA backend)                      | ✅     |
|                        | CPU mode (testing)                               | ✅     |

**Status legend:**

- ✅ Supported: integrated and tested for Qwen3-Embedding
- ❌ Not supported: may be considered for future releases

The [deployment tutorial](../tutorials/tutorial-qwen3-embedding-8b.md) walks
through deploying Qwen3-Embedding-8B via two paths: an online OpenAI-compatible
`/v1/embeddings` server and offline batch embedding with `LLM.embed()`.

Because an embedding model emits a vector rather than generating tokens, there is
no decode phase. Features that act on generated tokens — sampling, speculative
decoding, structured outputs — do not apply, and the decode graph is neither
compiled nor warmed up, which shortens startup.

Tensor parallelism is validated at `tensor_parallel_size` ∈ {2, 4, 8, 16, 32} on
both Trn2 and Trn3, and in combination with data parallelism.

Prefix caching requires segmented prefill: set `--max-num-batched-tokens` to one
of `{512, 1024, 2048, 4096, 8192}` and below `--max-model-len`. Enabling it with
single-shot prefill fails at startup with an explanatory error. It benefits
workloads whose requests share a long leading prefix, such as a fixed task
instruction, a RAG template head, or a constant document preamble. See
[prefix caching](../guides/features-guide.md#prefix-caching) for
configuration details.

Matryoshka support lets a client request a shorter, re-normalized vector by
passing `dimensions` per request. Because Qwen3-Embedding-8B's published
`config.json` does not declare Matryoshka support, opt in at launch with
`--hf-overrides '{"is_matryoshka": true}'`.

## Accuracy Evaluation

Accuracy measured on real hardware with **BF16** weights (Trn2), using the
[MTEB](https://github.com/embeddings-benchmark/mteb) harness with the official
per-task Qwen3 query instruction.

| Benchmark                                                                                                                                                              | Neuron (BF16) | Published |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----------: | :-------: |
| [STS12 (Spearman)](https://github.com/embeddings-benchmark/results/blob/main/results/Qwen__Qwen3-Embedding-8B/4e423935c619ae4df87b646a3ce949610c66241c/STS12.json)      |    0.8639    |  0.8614  |
| [NFCorpus (NDCG@10)](https://github.com/embeddings-benchmark/results/blob/main/results/Qwen__Qwen3-Embedding-8B/4e423935c619ae4df87b646a3ce949610c66241c/NFCorpus.json) |    0.4143    |  0.4145  |
| [SciFact (NDCG@10)](https://github.com/embeddings-benchmark/results/blob/main/results/Qwen__Qwen3-Embedding-8B/4e423935c619ae4df87b646a3ce949610c66241c/SciFact.json)   |    0.7839    |  0.7846  |

The **Published** column is the Qwen3-Embedding-8B score from the MTEB results
repository ; our BF16-on-Neuron scores match it to
within 0.001.

Requesting a truncated embedding trades a small amount of quality for a much
smaller vector. Measured on SciFact (NDCG@10), relative to the full 4096
dimensions:

| Output dimensions |  32  |  64  |  128  |  256  | 2048 | 4096 |
| ----------------- | :---: | :---: | :---: | :---: | :---: | :---: |
| NDCG@10           | 0.542 | 0.669 | 0.736 | 0.752 | 0.783 | 0.784 |
| % of full width   |  69%  |  85%  |  94%  |  96%  | 99.9% | 100% |

## Tutorials

- [Tutorial: Deploy Qwen3-Embedding-8B with vLLM Neuron](../tutorials/tutorial-qwen3-embedding-8b.md)
  — End-to-end online (`/v1/embeddings`) and offline (`LLM.embed()`) embedding
  deployment recipe, including variable output dimensions and prefix caching.
