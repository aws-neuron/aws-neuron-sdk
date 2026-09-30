# Tutorial: Deploy Wan2.2-A14B with vLLM Omni Neuron

<!-- meta: description: End-to-end tutorial for deploying the Wan2.2-A14B
diffusion video models with vLLM Omni Neuron, covering environment setup
(manual or DLC installation), model download, stage
configuration, and offline text-to-video (T2V) and image-to-video (I2V)
generation at 480P and 720P on Trn2 and Trn3, plus optional FP8 for T2V. -->
<!-- meta: keywords: vLLM Omni, Neuron, Wan2.2, Wan2.2-T2V-A14B, Wan2.2-I2V-A14B,
text-to-video, image-to-video, T2V, I2V, diffusion, video generation, MoE, BF16,
FP8, tensor parallelism, context parallelism, CFG parallelism, VAE tiling,
tutorial, Trn2, Trn3, Trainium -->
<!-- meta: content_type: procedural-tutorial -->
<!-- meta: date_updated: 2026-09-23 -->

This tutorial walks through deploying the Wan2.2-A14B diffusion video models with the
vLLM Omni Neuron plugin. It covers environment setup, model download, stage configuration,
and offline video generation. Wan2.2 comes in two variants, both served by this plugin:

- **[Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)** —
  text-to-video. See its [model card](../models/wan22-t2v-14b.md).
- **[Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)** —
  image-to-video. See its
  [model card](../models/wan22-i2v-14b.md).

Both are Mixture-of-Experts models (~27B total, 14B active per step) that generate
5-second clips (81 frames at 16fps) at 480P and 720P. Generation runs **offline** through
the vLLM Omni Neuron entrypoint — there is no online chat endpoint. BF16 is the default precision
throughout; FP8 (`fp8_row_mx`) is an optional T2V path covered in the optional step.

| Variant | Entry script | Default resolution | Cores (recommended) |
|---------|--------------|--------------------|---------------------|
| T2V | `examples/wan22/run.py` | 832x480 | 64 (TP4 × CP8 × CFG2) |
| I2V | `examples/wan22/run_i2v.py` | 832x480 | 64 (TP4 × CP8 × CFG2) |

**Prerequisites:**

- One SSH-accessible `trn2.48xlarge` or `trn3` instance. Step 1 prepares it.

## Step 1: Set up your environment

Complete the [setup guide](../getting-started/setup-guide.md), choosing manual or
DLC installation. It configures the dependencies, device access, and persistent
caches, then verifies the environment.

Run these commands in an SSH session on the instance host, using its DLC container
`vllm-omni-neuron`. For manual installation,
run the Python commands in the activated environment, omit
`sudo docker exec vllm-omni-neuron`, and replace `/workspace` with
`$VLLM_OMNI_HOME`.

## Step 2: Download the model (optional)

```bash
sudo docker exec vllm-omni-neuron hf download \
    Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --local-dir /workspace/huggingface/Wan2.2-T2V-A14B-Diffusers
```

:::{note}
This step is optional. You can pass the Hugging Face model ID (e.g.
`Wan-AI/Wan2.2-T2V-A14B-Diffusers`) directly via `--model-path`, and the weights are
downloaded automatically on first run into the mounted cache. For I2V, download
`Wan-AI/Wan2.2-I2V-A14B-Diffusers` instead.
:::

## Step 3: Review the stage configuration

The parallelism layout and diffusion settings live in a stage-config YAML, not on the
command line. The entry scripts default to
[`examples/wan22/wan22_stage.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_stage.yaml) (T2V) and
[`examples/wan22/wan22_i2v_stage.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_i2v_stage.yaml) (I2V);
override with `--stage-config`. The default 64-core T2V config is:

```yaml
# engine_args in the stage config
dtype: bfloat16
boundary_ratio: 0.875          # I2V uses 0.9 — the MoE high→low-noise expert switch point
flow_shift: 5.0
vae_use_tiling: true           # tile the VAE decode to fit 720P within HBM
model_config:
  tp_sequence_parallel: true   # Megatron sequence parallelism (720P headroom)
parallel_config:
  tensor_parallel_size: 4      # TP — shards the model
  ring_degree: 8               # CP degree — sets sequence_parallel_size = 1 * 8 = 8
  cfg_parallel_size: 2         # runs the two guidance passes in parallel
  vae_patch_parallel_size: 16  # default; the 720P config uses 32 ranks
```

- **`ring_degree`** (not `sequence_parallel_size`) sets the context-parallel degree; vLLM
  Omni derives `sequence_parallel_size = ulysses_degree(1) * ring_degree`. CP self-attention
  runs ring attention. See the [context parallelism design doc](../design/context_parallelism.md).
- **`boundary_ratio`** should be `0.9` for I2V and `0.875` for T2V — the I2V stage config
  already sets this. A wrong value switches MoE experts at the wrong step.

## Step 4: Run inference

### Text-to-video (T2V)

Quick smoke test (5 frames, low resolution) to validate the setup:

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run.py --dev
```

Full 480P generation (81 frames, 832x480, 40 steps):

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run.py \
  --num-frames 81 --height 480 --width 832 --guidance-scale 5.0 \
  --prompt "A fluffy orange cat walking gracefully across a sunny garden path, high quality, detailed" \
  --output 480P.mp4
```

720P generation (1280x720) uses the [720P stage config](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_stage_tp4cp8cfg2_720p.yaml), with VAE tiling and 32-way VAE patch parallelism:

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run.py \
  --stage-config /workspace/plugin/examples/wan22/wan22_stage_tp4cp8cfg2_720p.yaml \
  --num-frames 81 --height 720 --width 1280 --guidance-scale 5.0 \
  --prompt "A fluffy orange cat walking gracefully across a sunny garden path, high quality, detailed" \
  --output 720P.mp4
```

### Image-to-video (I2V)

I2V animates a conditioning first frame passed with `--image`:

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run_i2v.py \
  --image /workspace/plugin/examples/wan22/i2v_input.JPG \
  --num-frames 81 --height 480 --width 832 --guidance-scale 5.0 \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  --output i2v_480P.mp4
```

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run_i2v.py \
  --stage-config /workspace/plugin/test/neuron/configs/wan22_i2v_stage_tp8cp4cfg1_720p.yaml \
  --image /workspace/plugin/examples/wan22/i2v_input.JPG \
  --num-frames 81 --height 720 --width 1280 \
  --output i2v_720P.mp4
```

:::{note}
CFG parallelism (`cfg_parallel_size=2`) is not yet supported for I2V at 720P.
:::

vLLM Omni Neuron compiles the model on the first run.
With the DLC flow, copy the output with
`docker cp vllm-omni-neuron:/workspace/480P.mp4 .`; with manual installation, it is in your working directory.

## Optional: FP8 for T2V

T2V can run its QKV, output, and FFN projections in FP8 (`fp8_row_mx`) from a
ComfyUI-repackaged FP8 checkpoint. **Skip this for the default BF16 path.**

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run.py \
  --quantization fp8_row_mx \
  --comfyui-fp8-model-path Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  --num-frames 81 --height 480 --width 832 \
  --output 480P_fp8.mp4
```

`--modules-to-not-convert` selects which FP8 modules to CPU-dequant; pass
`attn1 attn2 ffn` for the full CPU-dequant reference.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `NRT_FAILURE` in `nrt_init()` / "Logical Neuron Core(s) not available" | Driver/runtime incompatibility or unavailable devices | Follow the host prerequisites and device checks in the setup guide. |
| Silent crash / `SIGBUS` on output | `/dev/shm` too small for the returned video | Start the container with `--shm-size=2g`. |
| OOM during VAE decode at 720P | Single-shot decode exceeds HBM | Set `vae_use_tiling: true` in the stage config or increase tp size. |
| Very long first-run latency | Compiler generating NEFFs on first inference | Wait for the first compilation to complete. |

## Conclusion

You have deployed Wan2.2-A14B on Trainium — text-to-video and image-to-video, at 480P and
720P, in BF16 (and optionally FP8 for T2V). Generation runs offline through the vLLM Omni
Neuron entrypoint.

## Next steps

- **Tune throughput and memory** — choosing TP/CP/CFG degrees with roofline analysis and
  profiling: [Optimizing offline video generation](../model-dev/optimizing-offline-video-generation.md).