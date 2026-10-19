# Qwen3-ASR Model Recipe

<!-- meta: description: Model recipe for deploying Qwen3-ASR-1.7B with vLLM
Neuron. The recipe includes the supported checkpoint, features, deployment
tutorial, and word error rate results on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3-ASR, Qwen3-ASR-1.7B, speech
recognition, audio transcription, ASR, model recipe, model card, Trn2,
Trainium -->
<!-- meta: date_updated: 2026-09-30 -->
<!-- Content type: model-card -->

## Introduction

[Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) is a speech
recognition model family from the Qwen team. It converts speech into text and
supports 52 languages and dialects. vLLM Neuron supports Qwen3-ASR-1.7B in
BF16 on AWS Trainium2 (`trn2`) and Trainium3 (`trn3`).

**Compatible model checkpoint:**

| Model | Hugging Face | Hardware | Quantization | Validated |
|---|---|---|---|---|
| Qwen3-ASR-1.7B | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Trn2, Trn3 | BF16 | ✅ |
| Qwen3-ASR-0.6B | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Trn2, Trn3 | BF16 | ❌ |

## Features

The following table shows the Qwen3-ASR feature status. See the
[features guide](../guides/features-guide.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Audio with a text instruction | ✅ |
| | Text-only input | ✅ |
| **Serving** | OpenAI-compatible `/v1/audio/transcriptions` endpoint | ✅ |
| | Offline `LLM.generate()` transcription | ✅ |
| **Quantization** | BF16 | ✅ |
| | FP8 or MXFP8 | ❌ |
| **Parallelism** | Text decoder tensor parallelism (TP) | ✅ |
| | Audio encoder data parallelism (DP) | ✅ |
| | Audio encoder tensor parallelism (TP) | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| **Performance** | Greedy decoding | ✅ |
| | Prefix caching (APC) | ❌ |
| | Disaggregated encoder inference | ❌ |
| **Compilation** | `torch.compile` | ✅ |
| | CPU mode for module tests | ✅ |

**Status legend:**

- ✅ Supported: The release integrates and tests this feature for Qwen3-ASR.
- ❌ Not supported: The current release does not support this feature.

CI validates the text decoder at TP1, TP2, TP4, and TP8 on Trn2. The audio
encoder uses data parallelism in the end-to-end tests. Module tests also
validate audio encoder tensor parallelism.

## Accuracy evaluation

The benchmark uses the
[ArtificialAnalysis/Earnings22-Cleaned-AA-chunked](https://huggingface.co/datasets/ArtificialAnalysis/Earnings22-Cleaned-AA-chunked)
dataset. Lower word error rate (WER) values are better.

| Compile backend | TP1 | TP2 | TP4 | TP8 |
|---|---:|---:|---:|---:|
| Native (`neuron_native`) | 6.04% | 6.11% | 6.06% | 6.01% |
| XLA (`vllm_neuron`) | 6.03% | 6.12% | 6.01% | 6.05% |

These results use BF16 on Trn2. For GPU accuracy results, see the
[Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).
To reproduce them, use its
[GitHub repository](https://github.com/huggingface/open_asr_leaderboard).

**Reproduce:** Serve the model by using the
[deployment tutorial](../tutorials/tutorial-qwen3-asr-1-7b.md). Send each
dataset audio chunk to the `/v1/audio/transcriptions` endpoint. Join the
transcriptions for each call. Normalize each reference and transcription with
[EnglishTextNormalizer](https://github.com/huggingface/open_asr_leaderboard).
Calculate WER with `kaldialign` by using the
[public leaderboard method](https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/eval_utils.py#L507-L529).

## Common issues

| Issue | Cause | Resolution |
|---|---|---|
| The server rejects a long audio request. | The request needs more blocks than the largest encoder input bucket. | Add a larger value to `num_encoder_input_buckets`. |
| Audio decoding fails in the offline example. | The environment does not have the optional vLLM audio dependencies. | Install them with `pip install "vllm[audio]"`. |
| The server rejects an FP8 configuration. | Qwen3-ASR has a BF16 implementation only. | Set `quantization` to `bf16`. |

## Related resources

- [Tutorial: Deploy Qwen3-ASR-1.7B with vLLM Neuron](../tutorials/tutorial-qwen3-asr-1-7b.md)
- [Offline Qwen3-ASR example](https://github.com/vllm-project/vllm-neuron/blob/main/examples/vllm_neuron/models/qwen3_asr/run.py)
- [Features guide](../guides/features-guide.md)
- [Accuracy debugging guide](../model-dev/accuracy-debugging-guide.md)
