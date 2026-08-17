# Qwen3-VL (Dense) Model Recipe

<!-- meta: description: Model recipe for deploying Qwen3-VL with vLLM on Neuron,
including supported checkpoints, feature support, accuracy results, and a link to
the end-to-end deployment tutorial for Qwen3-VL 32B (multimodal) on Trn2/Trn3. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3-VL, Qwen3-VL-32B, multimodal, VLM,
vision-language, model recipe, model card, LLM serving, Trn2, Trn3, Trainium -->
<!-- meta: date_updated: 2026-08-13 -->
<!-- Content type: model-card -->
<!-- Jira: NDOC-183 -->

## Introduction

[Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl) is a multimodal language model developed by the Qwen team. It supports image-text understanding, visual reasoning, OCR, document analysis, and multi-image comparison. The `-Instruct` variant is instruction-tuned for chat and task-following, while the `-Thinking` variant adds extended chain-of-thought reasoning.

Qwen3-VL is now supported for inference serving with [vLLM Neuron Plugin](https://github.com/vllm-project/vllm-neuron) using the Neuron SDK on AWS Trainium2 (`trn2`) and Trainium3 (`trn3`) hardware.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Qwen3-VL-32B-Instruct | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | Trn2, Trn3 | BF16, MXFP8 (Trn3 only) |
| Qwen3-VL-32B-Thinking | [Qwen/Qwen3-VL-32B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking) | Trn2, Trn3 | BF16, MXFP8 (Trn3 only) |
| Qwen3-VL-8B-Instruct | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Trn2, Trn3 | BF16, MXFP8 (Trn3 only) |
| Qwen3-VL-8B-Thinking | [Qwen/Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) | Trn2, Trn3 | BF16, MXFP8 (Trn3 only) |

> The `-Thinking` checkpoint shares the same architecture and serving path as `-Instruct` and runs with the identical configuration. The accuracy numbers below were measured on the `-Thinking` variant.

<!-- -->

> Qwen3-VL is not published with MXFP8 weights, follow the [tutorial](../tutorials/tutorial-qwen3-vl-32b.md#step-3-quantize-the-text-model-to-mxfp8-mxfp8-only) to quantize the BF16 checkpoint offline.

## Features

Per-model feature availability for Qwen3-VL. See the [features guide](../guides/features-guide.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Multimodal Inputs** | Text | ✅ |
| | Single image + text | ✅ |
| | Multi-image + text | ✅ |
| | Video + text | ✅ |
| **Quantization** | BF16 | ✅ |
| | MXFP8 (Trn3 only) | ✅ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Vision encoder parallelism | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| | Context parallelism (CP) | ❌ |
| **Performance** | On-device sampling (greedy, top-k, top-p) | ✅ |
| | Disaggregated encoder (EPD) (1E1PD / xEyPD) | ✅ |
| | Segmented prefill | ❌ |
| | Chunked prefill (mixed batching) | ❌ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Qwen3-VL
- ❌ Not supported: may be considered for future releases

Tensor parallelism is recommended at `tensor_parallel_size=16` on a `trn2.48xlarge` (16 NeuronCores), matching the configuration used throughout the [tutorial](../tutorials/tutorial-qwen3-vl-32b.md).

Multi-image input is validated up to 30 images at 512x512 resolution.

Vision encoder parallelism shards the encoder by the `num_blocks` dimension of the block-packed images (unlike text DP, which shards by request batch), configurable via `vision_neuron_config`. See [vision encoder parallelism](../design/parallelism/vision_encoder_parallelism.md) for details.

## Known limitations

- **Segmented prefill and prefix caching are not supported yet.** The validated workload (30 images at 512x512, 8K sequence length) is served without segmented prefill enabled. Prefix caching requires segmented prefill, so it is unavailable here as well.

## Accuracy Evaluation

**Benchmark:** POPE (Polling-based Object Probing Evaluation): 5,000 binary Yes/No questions about object presence in images across three difficulty subsets: popular, adversarial, random. Results are from the open-source [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) harness.

| Metric | 32B-Thinking, Neuron Trn2 BF16 |
|--------|--------------------------------|
| Overall F1 | 85.94 |
| Adversarial F1 | 84.86 |
| Popular F1 | 85.63 |
| Random F1 | 87.38 |

**Reproduce:** Serve the `-Thinking` checkpoint following the [tutorial](../tutorials/tutorial-qwen3-vl-32b.md), then run [VLMEvalKit](https://github.com/open-compass/vlmevalkit)'s `run.py` against the running server over its OpenAI-compatible endpoint:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit && git checkout fd3884b && pip install . && cd ..
python VLMEvalKit/run.py --data POPE --model Qwen3-VL-32B-Thinking \
    --base-url http://localhost:8000/v1 --max-tokens 256
```

## Tutorials

- [Tutorial: Deploy Qwen3-VL-32B with vLLM Neuron](../tutorials/tutorial-qwen3-vl-32b.md)
- [Tutorial: Configure disaggregated encoder inference with 1E1PD and xEyPD](../tutorials/tutorial-epd-1e-1pd-xeypd.md)
- [Optimizing a Vision-Language Model](../model-dev/optimizing-vlm-models.md)
