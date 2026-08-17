# Llama 3 Model Recipe

<!-- meta: description: Model recipe for deploying the Llama 3 family (1B, 8B,
70B) with vLLM Neuron, including supported checkpoints, feature support,
static FP8 and EAGLE3 support, accuracy results, and a link to the end-to-end
deployment tutorial for Llama 3.3 70B on Trn2/Trn3. -->
<!-- meta: keywords: vLLM, Neuron, Llama 3, Llama-3.2-1B, Llama-3.1-8B,
Llama-3.3-70B, FP8, ModelOpt, EAGLE3, speculative decoding, model recipe,
model card, LLM serving, Trn2, Trn3, Trainium -->
<!-- meta: date_updated: 2026-08-17 -->
<!-- Content type: model-card -->
<!-- Jira: NDOC-184 -->

## Introduction

[Llama 3](https://huggingface.co/collections/meta-llama/llama-3) is a family of
open-weight, decoder-only text language models from Meta, instruction-tuned for
chat and general task-following. This recipe covers three sizes across the Llama
3.1 / 3.2 / 3.3 releases — a 1B, an 8B, and a 70B — which share the same
architecture and serving path and differ only in scale and the parallelism
degree used to serve them.

Llama 3 is supported for inference serving with
[vLLM](https://github.com/vllm-project/vllm) using the Neuron SDK on AWS
Trainium2 (`trn2`) and Trainium3 (`trn3`) hardware.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Llama 3.2 1B Instruct | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Trn2, Trn3 | BF16, FP8 (static) |
| Llama 3.1 8B Instruct | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Trn2, Trn3 | BF16, FP8 (static) |
| Llama 3.3 70B Instruct | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Trn2, Trn3 | BF16, FP8 (static) |

> The three sizes are byte-identical in serving path; only the tensor-parallel
> degree changes. All three run in BF16 out of the box, and all three accept a
> static FP8 checkpoint (see below). There is **no quantization difference
> between Trn2 and Trn3** — both run the same static FP8 checkpoints.

### Quantization

Llama 3 supports **tensor-wide (per-tensor) static FP8** checkpoints in the
[NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) format,
for example
[nvidia/Llama-3.3-70B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8).
The checkpoint carries per-tensor static weight, activation, and KV-cache scales
in its `quantization_config` (`quant_method: "modelopt"`, `quant_algo: "FP8"`).
The Neuron backend detects the format and loads the FP8 weights automatically —
you do **not** set vLLM's `--quantization` flag. See the
[FP8 static weight quantization](../guides/features-guide.md#fp8-static-weight-quantization)
section of the features guide for details.

Both Trn2 and Trn3 run the same static FP8 checkpoints. Internally, Trn3 routes
FP8 matmuls through STATIC_MX kernels for a prefill speedup, but the checkpoint
and the way you launch the server are identical on both platforms.

### Speculative decoding

Llama 3 supports **EAGLE3** speculative decoding. Pair a target checkpoint with
a matching EAGLE3 draft, for example
[RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3](https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3)
as the draft for `meta-llama/Llama-3.3-70B-Instruct`. Under greedy sampling
EAGLE3 is a lossless acceleration — the target model's output is unchanged. See
the [EAGLE3 speculative decoding tutorial](../tutorials/tutorial-eagle3-speculative-decoding-llama-3-1.md)
for an end-to-end walkthrough and acceptance-rate tuning.

## Features

Per-model feature availability for Llama 3. See the
[features guide](../guides/features-guide.md) for configuration details and the
[cross-model feature compatibility matrix](../guides/reference-feature-model-compatibility.md).

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| **Quantization** | BF16 weights | ✅ |
| | FP8 static (per-tensor, ModelOpt) | ✅ |
| | KV cache FP8 | ✅ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Data parallelism (DP) | ✅ (see [Known issues](#known-issues)) |
| | Expert parallelism (EP) | N/A |
| | Pipeline parallelism (PP) | ❌ |
| **Performance** | Continuous batching | ✅ |
| | Segmented prefill | ✅ |
| | Prefix caching (APC) | ✅ |
| | Speculative decoding (EAGLE3) | ✅ |
| | Disaggregated inference (1P1D / xPyD) | ✅ |
| | On-device sampling (greedy, top-k, top-p) | ✅ |
| **Serving** | Structured outputs / tool calling | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Llama 3
- ❌ Not supported: may be considered for future releases
- N/A Not applicable to this (dense) architecture

The [deployment tutorial](../tutorials/tutorial-llama3-70b.md) walks through
deploying Llama 3.3 70B in static FP8 on a single Trn2 or Trn3 instance with
tensor parallelism.

## Known issues

- **Data parallelism (DP) throughput does not scale as expected**, due to a vLLM
  DP load-balancing issue. **Mitigation:** run multiple DP1 servers (e.g. 4× DP1
  in place of one DP4) and let a higher-level router balance requests across them.

## Accuracy Evaluation

Accuracy measured on real hardware against the Neuron 2.32 release build
(plugin sha `f9548908`). GSM8K scores use `flexible-extract` (n=100).

| Model | Precision | GSM8K |
|-------|-----------|:-----:|
| Llama 3.3 70B | BF16 | 0.95 |
| Llama 3.3 70B | Static FP8 | 0.95 |

FP8 is within measurement noise of BF16 on GSM8K at 70B — there is no
meaningful accuracy cost from quantization at this scale.

**Reproduce:** Serve the checkpoint following the
[tutorial](../tutorials/tutorial-llama3-70b.md), then run the evaluation harness
against the running server over its OpenAI-compatible endpoint. See the
[accuracy debugging guide](../model-dev/accuracy-debugging-guide.md) for the
evaluation workflow.

## Tutorials

- [Tutorial: Deploy Llama 3.3 70B (FP8) with vLLM Neuron](../tutorials/tutorial-llama3-70b.md)
  — End-to-end single-instance deployment recipe for Llama 3.3 70B in static FP8.
- [EAGLE3 speculative decoding tutorial](../tutorials/tutorial-eagle3-speculative-decoding-llama-3-1.md)
  — Add an EAGLE3 draft model for higher decode throughput.
