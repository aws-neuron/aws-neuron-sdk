# GPT-OSS Model Recipe

<!-- meta: description: Model recipe for deploying gpt-oss with vLLM on Neuron,
including supported checkpoints, feature support, accuracy results, and a link to
the end-to-end deployment tutorial for gpt-oss 20B and 120B on Trn3 (MXFP4) or
Trn2 (BF16). -->
<!-- meta: keywords: vLLM, Neuron, gpt-oss, gpt-oss-20b, gpt-oss-120b, MoE,
MXFP4, model recipe, model card, LLM serving, Trn2, Trn3, Trainium -->
<!-- meta: date_updated: 2026-10-16 -->
<!-- Content type: model-card -->
<!-- Jira: NDOC-185 -->

## Introduction

[gpt-oss](https://huggingface.co/openai/gpt-oss-120b) is an open-weight
Mixture-of-Experts (MoE) language model family released by OpenAI, designed for
strong reasoning, tool use, and long-context generation. The published
checkpoints ship with MXFP4-quantized expert weights.

gpt-oss is supported for inference serving with
[vLLM](https://github.com/vllm-project/vllm) using the Neuron SDK on AWS
Trainium2 (`trn2`) and Trainium3 (`trn3`) hardware.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| gpt-oss-20b | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | Trn2, Trn3 | BF16 (Trn2), MXFP4 (Trn3) |
| gpt-oss-120b | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | Trn2, Trn3 | BF16 (Trn2), MXFP4 (Trn3) |

> On Trn3, MXFP4-quantized weights are auto-selected from the published
> checkpoints. On Trn2, both models run in BF16. MXFP4 is Trn3 only.

## Features

Per-model feature availability for gpt-oss. See the
[features guide](../guides/features-guide.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| **Quantization** | MXFP4 weights (Trn3) | ✅ |
| | BF16 weights (Trn2) | ✅ |
| | KV cache FP8 | ✅ (see [Known issues](#known-issues)) |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Data parallelism (DP) | ✅ |
| | Expert parallelism (EP) | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| **Performance** | Continuous batching | ✅ |
| | Segmented prefill | ✅ |
| | Prefix caching (APC) | ✅ |
| | Speculative decoding (EAGLE3) | ✅ |
| | Speculative decoding (P-EAGLE) | ✅ |
| | Disaggregated inference (1P1D / xPyD) | ✅ |
| | Decode megakernel at LNC=1 (Trn3, DI decode, 120B) | ✅ |
| | On-device sampling (greedy, top-k, top-p) | ✅ |
| **Serving** | Structured outputs / tool calling | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |
| **Runtime** | Model Runner V2 | ❌ (runs on MRv1) |

**Status legend:**

- ✅ Supported: integrated and tested for gpt-oss
- ❌ Not supported: may be considered for future releases

This model runs on Model Runner V1 (MRv1) because it is a Mixture-of-Experts model with sliding-window attention. [Model Runner V2](../guides/features-guide.md#model-runner-v2) supports dense text-generation models only.

The [deployment tutorial](../tutorials/tutorial-gpt-oss.md) walks through
deploying gpt-oss 20B and 120B on Trn3 (MXFP4) or Trn2 (BF16) via two paths: a
single-instance server and a disaggregated-inference (separate prefill and
decode) deployment.

## Known issues

- **Intermittent HTTP 500 errors with an FP8 KV cache on Trn3.** A race
  condition can cause some requests to fail with an HTTP 500 error. The server
  log shows an `openai_harmony.HarmonyError` for each failed request. A retry
  of the same request succeeds. The issue occurs on Trn3 with
  `--kv-cache-dtype fp8` when the decode does not use the decode megakernel.
  Trn2 is not affected. The configuration in the
  [8K/1K Trn3 tutorial](../tutorials/tutorial-gpt-oss-120b-8k1k-trn3.md) uses
  the decode megakernel and is not affected. **Mitigation:** set
  `VLLM_NEURON_ATTN_MASK_TORCH=1` in the server environment. For a
  disaggregated deployment, set it on the decode server. This flag builds the
  decode attention mask in PyTorch, in place of an NKI kernel.

## Accuracy Evaluation

Accuracy measured on real hardware with **MXFP4** weights (Trn3), medium
reasoning effort.

| Benchmark | gpt-oss-120b (MXFP4) |
|-----------|:--------------------:|
| GSM8K-CoT | 88.8% |
| AIME25 (avg@8, medium) | 78.75% |
| GPQA-diamond (medium) | 72.22% |

## Tutorials

- [Tutorial: Deploy gpt-oss with vLLM Neuron](../tutorials/tutorial-gpt-oss.md)
  — End-to-end single-instance and disaggregated-inference deployment recipe for
  gpt-oss 20B and 120B.
- [Prefix caching benchmark tutorial](../tutorials/tutorial-prefix-caching-gpt-oss-benchmarking.md)
  — Benchmark prefix caching with gpt-oss across open-source datasets.
- [High-performance GPT-OSS 120B implementation for 8K/1K serving on Trn3](../tutorials/tutorial-gpt-oss-120b-8k1k-trn3.md)
  — The 6P1D configuration on two Trn3 instances with the decode megakernel,
  and the scripts that run it and its benchmark.
