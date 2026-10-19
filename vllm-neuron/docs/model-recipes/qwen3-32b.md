# Qwen3-32B Model Recipe

<!-- meta: description: Model recipe for deploying Qwen3-32B with vLLM on Neuron,
including supported checkpoints, feature support, validation coverage, and
native MXFP8 limitations on Trn3. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3, Qwen3-32B, dense, MXFP8, compressed-tensors,
model recipe, model card, LLM serving, Trn2, Trn3, Trainium -->
<!-- meta: date_updated: 2026-10-13 -->
<!-- Content type: model-card -->

## Introduction

[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) is a dense decoder-only
language model from the Qwen family. In vLLM Neuron, the dense 32B path supports
the published BF16 checkpoint and a native MXFP8 path for compressed-tensors
checkpoints on Trn3.

Qwen3-32B is supported for inference serving with
[vLLM Neuron Plugin](https://github.com/vllm-project/vllm-neuron) using the
Neuron SDK on AWS Trainium2 (`trn2`) and Trainium3 (`trn3`) hardware.

**Compatible model checkpoints:**

| Model | Checkpoint | Hardware | Quantization |
|-------|------------|----------|--------------|
| Qwen3-32B | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Trn2, Trn3 | BF16 |
| Qwen3-32B (native MXFP8) | Built locally from `Qwen/Qwen3-32B` with `examples/vllm_neuron/models/qwen3/build_mxfp8_checkpoint.py` | Trn3 | MXFP8 |

> The repository includes `run.py` for the BF16 path and
> `build_mxfp8_checkpoint.py` plus `run_mxfp8.py` for the native MXFP8 path.
> MXFP8 is Trn3-only.

## Features

Per-model feature availability for Qwen3-32B dense. See the
[features guide](../guides/features-guide.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| **Quantization** | BF16 | ✅ |
| | MXFP8 weights (Trn3) | ✅ |
| | KV cache FP8 — packed swizzled layout (Trn3, MXFP8) | ✅ (V1) |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Replica data parallelism (DP) | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| | Sequence parallelism (SP) — prefill | ✅ |
| **Performance** | Continuous batching | ✅ |
| | Segmented prefill | ✅ |
| | Prefix caching (APC) | ✅ |
| | On-device sampling (greedy, top-k, top-p) | ✅ |
| | Async scheduling | ✅ |
| **Serving** | Structured outputs / tool-calling enforcement | ✅ (V1) |
| | Disaggregated inference (DI), MXFP8 packed FP8 KV | ✅ (V1) |
| | EAGLE3 speculative decoding, single server | ✅ (V1) |
| **Compilation** | torch.compile (native `neuron_native` and XLA `vllm_neuron` backends) | ✅ |
| | CPU mode (testing) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Qwen3-32B dense
- ✅ (V1) Supported: requires Model Runner V1 (MRv1) in this release
- ❌ Not supported: may be considered for future releases

The plugin selects one model runner per server from the complete configuration.
By default, it automatically selects MRv1 for the rows marked **(V1)**.
Common features, such as TP, continuous batching, and async scheduling, also
work on MRv1.

The DI configuration in this recipe uses MRv1 on both the prefill and decode
servers because it uses packed FP8 KV. CPU tests with `enforce_eager` also
select MRv1. Feature support does not guarantee support for every combination.
See the [runner selection guide](../guides/features-guide.md#model-runner-v2)
and the [validated deployment configurations](#deployment-configurations).

## Deployment configurations

Use the [deployment tutorial](../tutorials/tutorial-qwen3-32b.md) to prepare a
checkpoint and launch a TP8 single server. The tutorial also describes a
packed-FP8-KV 1P1D deployment with TP4/LNC2 prefill and TP8/LNC1 decode at bs64.

| Weight format | Hardware | Checkpoint |
|---|---|---|
| BF16 | Trn2, Trn3 | `Qwen/Qwen3-32B` |
| Native MXFP8 | Trn3 | Convert the BF16 source with `build_mxfp8_checkpoint.py` |

### Checkpoint provenance

Pin the BF16 checkpoint to an immutable Hugging Face revision. For MXFP8,
record the source revision, converter commit, conversion options, dependency
versions, and checksums. Keep `lm_head` in BF16. The
[tutorial](../tutorials/tutorial-qwen3-32b.md#step-1-record-the-environment-and-prepare-checkpoints)
shows the download, conversion, and checksum commands.

### Tensor parallelism and KV cache

BF16 weights use BF16 KV. The BF16 baseline uses TP8, and BF16 examples and accuracy
tests cover TP8, TP16, and TP32. MXFP8 weights are validated with both BF16 KV and packed FP8 KV.
MXFP8 tests cover TP8 single-server inference and
TP4/LNC2 prefill → TP8/LNC1 decode through real NIXL transfer. The native MXFP8
example (`run_mxfp8.py`) uses `tensor_parallel_size=8`, `max_model_len=16384`, and
`num_batched_tokens_buckets=[8192]`.

The MXFP8 performance grid includes TP4/DP16, TP8/DP8, TP16/DP4, and TP32/DP2.
The DI target uses TP4 for prefill and TP8 for decode, with packed FP8 KV on both roles.
Prefill uses O2; decode uses O3, block size 128, and a pinned 5232-block cache.

Use the same backend, TP/DP, LNC, optimization level, context buckets, and KV
dtype when comparing accuracy or performance. Keep per-replica batch and node
concurrency distinct when sizing a DP deployment.

Component data parallelism also selects MRv1. See the
[runner selection guide](../guides/features-guide.md#model-runner-v2) for details.

## Accuracy Evaluation

Logit checks compare greedy device output with a Hugging Face reference. GSM8K
and LongBench-v2 provide task-level checks. The table lists the existing CI
coverage and acceptance floors. These floors are test criteria, not measured accuracy scores.

| Evaluation | Existing coverage | CI floor |
|---|---|---|
| BF16 GSM8K | TP8/16/32, bs8, 50 examples, max length 10,240 | 0.85 |
| MXFP8 GSM8K | TP8/DP1, 200 examples. Packed FP8 KV: bs8 at 32k, bs64/128 at 10,240. BF16 KV: bs8 at 32k | 0.94 |
| BF16 LongBench-v2 | TP8, bs8, 32k | 0.378 |
| MXFP8 LongBench-v2 | TP8, bs8, 32k, BF16 KV | 0.378 |
| Greedy logits | BF16 TP8/16/32; MXFP8 TP8 with BF16 and packed FP8 KV; MXFP8 DI | Per-target numerical tolerances |

### Mixed-LNC DI accuracy measurements

These measurements use MXFP8, TP4/LNC2 prefill, TP8/LNC1 decode, packed FP8 KV,
and real NIXL transfer. GSM8K evaluates 200 examples with `flexible-extract` scoring.

| Evaluation | Native `neuron_native` | XLA `vllm_neuron` | Method |
|---|---|---|---|
| Greedy logits | Pass | Pass | Device logits compared with the Hugging Face reference |
| GSM8K | 98.0% | 97.0% | 200 examples; 94% acceptance floor |

The logit and GSM8K measurements use pre-release builds of the
Neuron 2.33 vLLM Neuron plugin. These are measured accuracy snapshots.
Use the same checkpoint and serving settings
to reproduce these measurements.

Both evaluations use a maximum sequence count of 64, the 5232-block cache
allocation, and a maximum model length of 10,240.

When reporting a measured score, include the dataset revision, sample count,
scoring method, checkpoint identity, and software versions. Record the backend,
TP/DP, LNC, batch, context length, and KV dtype with the result.

## Common Issues

| Issue | Cause | Resolution |
|-------|-------|------------|
| Native MXFP8 TP64 is not supported | The native MXFP8 MLP scale layout requires `intermediate_size / tensor_parallel_size` to be a multiple of `32`. For Qwen3-32B, `25600 / 64 = 400`, so the per-rank MLP intermediate does not satisfy the MX scale-packing contract. | Use TP8, TP16, or TP32 for native MXFP8. The BF16 path does not have this specific restriction. |
| No published Qwen3-32B MXFP8 checkpoint | The dense MXFP8 path serves a self-quantized compressed-tensors checkpoint rather than a public upstream MXFP8 release. | Generate the checkpoint locally with `examples/vllm_neuron/models/qwen3/build_mxfp8_checkpoint.py`, then serve it with `run_mxfp8.py` or `vllm serve ... --additional-config '{"neuron_config":{"quantization":"mxfp8"}}'`. |
| Native MXFP8 on Trn2 is not supported | The native MXFP8 path depends on Trn3 support. | Use BF16 on Trn2, or move the MXFP8 deployment to Trn3. |

## Related

- [Dense Qwen3-32B deployment tutorial](../tutorials/tutorial-qwen3-32b.md)
- [Features guide](../guides/features-guide.md)
- [Feature and model compatibility matrix](../guides/reference-feature-model-compatibility.md)
