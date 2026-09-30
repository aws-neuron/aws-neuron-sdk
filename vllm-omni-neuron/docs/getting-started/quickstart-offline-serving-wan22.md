# Quickstart: Offline video generation with Wan2.2 on Neuron

<!-- meta: description: Generate a video clip offline from a text prompt with the
Wan2.2-T2V-A14B model on AWS Trainium using the vLLM Omni Neuron plugin. Covers a
quick smoke test and a full 480x832, 81-frame generation run. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, Wan2.2, Wan2.2-T2V-A14B,
text-to-video, video generation, offline, diffusion, AWS Trainium, trn2, trn3,
NeuronCores, quickstart -->
<!-- meta: date_updated: 2026-09-04 -->
<!-- meta: content_type: procedural-quickstart -->
<!-- Jira: NMI-414 -->

This quickstart shows you how to generate a video from a text prompt on AWS
Trainium with the vLLM Omni Neuron plugin. When you finish, you have an MP4 clip
produced offline by the [Wan2.2-T2V-A14B](../models/wan22-t2v-14b.md) text-to-video
model.

## Prerequisites

Before you start, make sure that you have the following:

- One SSH-accessible `trn2.48xlarge` or `trn3` instance.
- The environment prepared with either flow in the [setup guide](setup-guide.md).
  The DLC commands below use the `vllm-omni-neuron` container, with device
  access, persistent caches, and `--shm-size=2g`.
- Network access to download the Wan2.2 weights
  (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`) from Hugging Face.

Run the following commands in an SSH session on the instance host. Each command uses
`docker exec` to run the example script inside the local `vllm-omni-neuron` container.

For manual installation, run the Python commands in the activated environment,
omit `sudo docker exec vllm-omni-neuron`, and replace `/workspace` with
`$VLLM_OMNI_HOME`.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`), not the
> `vllm.LLM` API. It reads its configuration from
> `/workspace/plugin/examples/wan22/wan22_stage.yaml`.

## Step 1: Run a quick smoke test

To verify the pipeline end to end, run the script in dev mode. Dev mode uses low
resolution, few frames, and few denoising steps, so it finishes quickly.

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run.py --dev
```

The command writes an MP4 file to the container working directory (`/workspace`).
When the smoke test passes, continue to the full run.

## Step 2: Run a full offline generation

To generate the default clip, run the script with an output path. The default run produces
81 frames at 480x832 over 40 denoising steps.

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run.py \
  --output /workspace/output/wan_output.mp4
```

The command writes the clip to `/workspace/output/wan_output.mp4`, which is in
the host-mounted output directory created by the setup guide.

To set your own prompt and output file, add the `--prompt` and `--output` flags:

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run.py \
  --prompt "A red hot air balloon rising over green hills at sunrise" \
  --output /workspace/output/my_video.mp4
```

### Change the output shape and sampling controls

`examples/wan22/run.py` sets the default height, width, frame count, and number of denoising
steps. Override the output shape with `--height`, `--width`, and
`--num-frames`. For example, to generate an 81-frame 720p clip:

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run.py \
  --stage-config /workspace/plugin/examples/wan22/wan22_stage_tp4cp8cfg2_720p.yaml \
  --height 720 \
  --width 1280 \
  --num-frames 81 \
  --output /workspace/output/wan_720p.mp4
```

Changing the output shape triggers compilation for that shape. Use frame counts of
the form `4k + 1`, such as 5 or 81. The example runner fixes full-mode generation
at 40 denoising steps; use the `OmniDiffusionSamplingParams` Python API or adapt the
runner to change `num_inference_steps`. More steps increase latency approximately
linearly and can improve detail or motion, but do not guarantee higher quality. See
the [feature and configuration guide](../guides/features-guide.md#generation-controls)
for the complete control and trade-off table.

The run uses `/workspace/plugin/examples/wan22/wan22_stage.yaml` by default and
the [720P stage config](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_stage_tp4cp8cfg2_720p.yaml)
for the 720P command above. For parallelism, 720P, and
quality tuning, see
[Optimizing offline video generation](../model-dev/optimizing-offline-video-generation.md).

To copy the clip to the host, use `docker cp`:

```bash
sudo docker cp vllm-omni-neuron:/workspace/output/wan_output.mp4 .
```

> **Note:** The first run for a new output shape is slow because Neuron compiles
> shape-specific NEFF graphs and loads large model weights (~116 GB).

## Common issues

- **Silent crash or `SIGBUS` on output:** The container `/dev/shm` is too small
  for the returned video. Start the container with `--shm-size=2g`, as shown in
  the setup guide.
- **`NRT_FAILURE` in `nrt_init()`, or Neuron cores not available:** The host
  driver or device access is incompatible with the runtime. Follow the host
  prerequisites and device checks in the setup guide.
- **Very long first-run latency:** The Neuron compiler builds NEFF graphs on the
  first run. This is expected.
- **Weights download fails:** The run cannot reach the Wan2.2 weights. Confirm
  network access and Hugging Face permission for
  `Wan-AI/Wan2.2-T2V-A14B-Diffusers`.

## Clean up

To stop and remove the container, run the following commands:

```bash
sudo docker stop vllm-omni-neuron
sudo docker rm vllm-omni-neuron
```

## Next steps

- [Optimizing offline video generation](../model-dev/optimizing-offline-video-generation.md) —
  parallelism selection, 720p and VAE tiling, and quality tuning.
- [Wan2.2-T2V-A14B model card](../models/wan22-t2v-14b.md) — architecture, feature
  support, and [VBench](https://arxiv.org/abs/2311.17982) accuracy.
- [Entrypoint script (`run.py`)](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/run.py) — the offline
  entrypoint and its flags.
- [Stage config (`wan22_stage.yaml`)](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_stage.yaml) — the
  configuration the run reads.
