# Wan2.2-I2V-A14B Model Card

<!-- meta: description: Model card for Wan2.2-I2V-A14B on AWS Trainium with the
vLLM Omni Neuron plugin — supported features (image-to-video, first-last-frame
conditioning, resolutions, frame counts, BF16, TP/CP/CFG parallelism, VAE
tiling and patch parallelism), the recommended 64-core 480P and 32-core 720P
configurations, VBench accuracy on Neuron, and known issues. -->
<!-- meta: keywords: Wan2.2, Wan2.2-I2V-A14B, model card, image-to-video, I2V,
FLF2V, first-last-frame, video generation, diffusion, MoE, vLLM, vLLM Omni,
Neuron, Trainium, trn2, trn3, BF16, tensor parallelism, context parallelism,
Megatron sequence parallelism, CFG parallelism, VAE tiling, VAE patch parallelism, VBench -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-09-23 -->

## Introduction

[Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) is an image-to-video diffusion model developed by Wan-AI. Given a conditioning image (the first frame) and a text prompt, it animates the image into a short video that stays faithful to the input frame. Like its text-to-video sibling, it uses a Mixture-of-Experts (MoE) architecture with two experts — a high-noise expert for overall layout during early denoising stages and a low-noise expert for detail refinement during later stages. The model has ~27B total parameters but only 14B active parameters per inference step, keeping computation and memory roughly equivalent to a single 14B dense model. It generates 5-second videos (81 frames at 16fps) at 480P and 720P resolutions with cinematic-level aesthetics and complex motion.

Wan2.2-I2V-A14B is now supported for inference serving with [vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on AWS Trainium2 (`trn2`) and Trainium3 (`trn3`) hardware. For how the image conditioning path works on Neuron, see [Image conditioning on Neuron](#image-conditioning-on-neuron).

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Wan2.2-I2V-A14B | [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | Trn2, Trn3 | BF16 |

> I2V runs in BF16. FP8 is not currently supported for image-to-video.

## Features

Per-model feature availability for Wan2.2-I2V-A14B. See the [README](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/README.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Generation** | Image-to-Video | ✅ |
| | 832x480 resolution (480p) | ✅ |
| | 1280x720 resolution (720p) | ✅ |
| | Up to 81 frames | ✅ |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ |
| | Context Parallelism (CP) | ✅ |
| | Megatron Sequence Parallelism (SP) | ✅ |
| | CFG Parallelism | ✅ |
| | [VAE Patch Parallelism](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/vae_parallel/) | ✅ |
| **Performance** | Classifier-Free Guidance | ✅ |
| | Spatial Tiling (VAE) | ✅ |
| | Temporal Chunking (VAE) | ✅ |
| | Continuous request batching | Limited |
| **Compilation** | torch.compile | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Wan2.2-I2V-A14B
- Limited: accepted, but concurrent requests run serially rather than as a batched forward pass

### Recommended configurations

The recommended **480P** configuration runs on 64 NeuronCores (e.g., `trn3`), matching the [stage config](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_i2v_stage.yaml): `tensor_parallel_size=4` × `ring_degree=8` (context parallelism) × `cfg_parallel_size=2` = 64.

The recommended **720P** configuration runs on 32 NeuronCores with `tensor_parallel_size=8` × `ring_degree=4` (context parallelism) × `cfg_parallel_size=1` = 32, with a higher `vae_patch_parallel_size` to shard the large-resolution tiled VAE encode/decode.

The I2V stage config carries a few model-specific engine args beyond the T2V stage:

- `boundary_ratio: 0.9` — the denoising-step fraction at which the pipeline switches from the high-noise to the low-noise expert.
- `flow_shift: 5.0` — the flow-matching timestep shift.
- `vae_use_tiling: true` — required so the image-condition VAE encode and the final VAE decode fit within per-device HBM at production shapes.

CFG parallelism (size=2) runs the conditional and unconditional denoising passes across separate replicas in parallel, reducing wall-clock time per diffusion step.

Context parallelism shards the sequence dimension (temporal-spatial latent tokens) across up to 8 ranks, enabling generation of longer videos within per-device HBM limits. The CP degree is configured with **`ring_degree`, not `sequence_parallel_size`**: vLLM Omni enforces `sequence_parallel_size = ulysses_degree * ring_degree`, so the stage config sets `ring_degree` and leaves `ulysses_degree` (default 1) and `sequence_parallel_size` unset — vLLM Omni then derives `sequence_parallel_size = 1 * ring_degree`. `ring_degree` sizes the sequence-parallel group; the CP self-attention itself runs ring attention (K/V stay sharded and are streamed around the CP group), falling back to all-gather + flash only where the ring kernel cannot run. See the [context parallelism design doc](../design/context_parallelism.md#configuration) for details.

Megatron sequence parallelism (SP) is layered **within** the TP group (orthogonal to CP): the normalization and MLP regions that TP would otherwise replicate are instead sharded along the sequence dimension across the TP ranks, cutting activation memory. It is enabled with `model_config.tp_sequence_parallel: true` and requires `tensor_parallel_size > 1`.

### Image conditioning on Neuron

The conditioning image is passed through `multi_modal_data` (`{"image": <PIL.Image>}`); an optional `last_image` enables FLF2V. On Neuron the DiT runs SPMD on every rank, but the VAE is materialized only on the VAE rank(s), so the pipeline VAE-encodes the image on the VAE rank and broadcasts the fixed-shape condition latent to all ranks before the denoise loop. The seeded noise latents are RNG-reproducible and identical on every rank, so they need no broadcast.

## Accuracy Evaluation

**Benchmark:** [VBench-I2V](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v) is the image-to-video track of [VBench](https://github.com/Vchitect/VBench), a comprehensive benchmark suite for video generation models. It scores generation quality (subject/background consistency, motion smoothness, dynamic degree, aesthetic quality, imaging quality) alongside how faithfully the video preserves the conditioning frame and camera motion (the I2V dimensions).

See the [VBench paper](https://arxiv.org/abs/2311.17982) for the original benchmark
and the [VBench++ paper](https://arxiv.org/abs/2411.13503) for its image-to-video extension.

**Wan2.2-I2V-A14B-Diffusers (BF16), 480p**

| Platform | Total Score | Quality Score | I2V Score |
|----------|-------------|---------------|-----------|
| Trn2 | 88.14% | 79.49% | 96.78% |

- **Quality Score** dimensions: `subject_consistency`, `background_consistency`, `motion_smoothness`, `dynamic_degree`, `aesthetic_quality`, `imaging_quality`.
- **I2V Score** dimensions: `i2v_subject`, `i2v_background`, `camera_motion`.
- **Total Score** is the simple average of the Quality and I2V scores.
- Scores use VBench's normalization and weights (see [`vbench2_beta_i2v`](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v)).

For externally published results, see the image-to-video results on the
[VBench Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard).
Results from different evaluation configurations are not directly comparable.

**Reproduce:** Serve the model following the [quickstart](../tutorials/tutorial-wan22-14b.md), then run VBench-I2V evaluation:

```bash
git clone https://github.com/Vchitect/VBench.git
cd VBench && pip install . && cd ..
python VBench/evaluate_i2v.py \
    --videos_path wan_i2v_output_dir \
    --custom_image_folder wan_i2v_input_images \
    --dimension i2v_subject i2v_background camera_motion \
        subject_consistency background_consistency motion_smoothness \
        dynamic_degree aesthetic_quality imaging_quality \
    --ratio 16-9 \
    --mode=custom_input
```

## Known limitations

- **Batch size is limited to one.** The current Wan2.2 pipeline generates one video per request; concurrent requests run serially rather than as a batched forward pass.
- **CFG parallelism (`cfg_parallel_size=2`) is not validated at 720P.** Use the recommended TP8CP4CFGP1 config for the I2V 720P case.

## Tutorials

- [Tutorial: Deploy Wan2.2-A14B with vLLM Omni Neuron](../tutorials/tutorial-wan22-14b.md)
- [Optimizing High-Quality Offline Video Generation](../model-dev/optimizing-offline-video-generation.md)
