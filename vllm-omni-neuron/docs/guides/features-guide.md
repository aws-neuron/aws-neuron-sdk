# vLLM Omni Neuron features guide

<!-- meta: description: Configure generation, compilation, parallelism,
quantization, memory, caching, and latency measurement for diffusion inference
with vLLM Omni on AWS Trainium. -->
<!-- meta: keywords: vLLM Omni, Neuron, Trainium, diffusion, Wan2.2,
text-to-video, image-to-video, tensor parallelism, context parallelism,
CFG parallelism, VAE tiling, FP8, Cache-DiT -->
<!-- meta: date_updated: 2026-08-25 -->
<!-- Content type: procedural-how-to -->

This guide covers the significant inference features exposed by the vLLM Omni
Neuron plugin. It explains what each feature does, when to use it, how to
configure it, and the main trade-offs.

vLLM Omni owns request processing, diffusion scheduling, and output
post-processing. The plugin supplies the Neuron platform, distributed worker,
model runner, compiled model components, and Neuron-specific parallelism.
Features are configured primarily through stage YAML, with request-level
controls available through the example runners and Python API.

For model architecture, accuracy results, and known model-specific limitations,
see the [Wan2.2 model card](../models/wan22-t2v-14b.md).

## Prerequisites

Before using this guide, you need:

- A supported AWS Trainium instance with the Neuron devices required by your
  parallel configuration.
- vLLM Omni Neuron installed through the manual or DLC flow, with device access.
- Storage for model weights and compilation work files.
- At least 2 GB of shared memory for full-size video output (`--shm-size=2g` for Docker).

See [Set up vLLM Omni Neuron](../getting-started/setup-guide.md) for host,
driver, package, container, and cache setup.

## Feature support

The current release supports the Wan2.2 A14B model family.

| Category | Feature | Availability |
| --- | --- | --- |
| Generation | Text-to-video (T2V) | Wan2.2-T2V-A14B |
| Generation | Image-to-video (I2V) | Wan2.2-I2V-A14B |
| Output | 832 x 480, up to 81 frames | T2V and I2V |
| Output | 1280 x 720, up to 81 frames | T2V and I2V; VAE tiling required |
| Compilation | `torch.compile` with reusable Neuron artifacts | T2V and I2V |
| Parallelism | Tensor, context, sequence, CFG, and VAE patch parallelism | T2V and I2V |
| Precision | BF16 | T2V and I2V |
| Precision | FP8 (ROW_MX Projections) | T2V; Trainium3 only |
| Acceleration | Cache-DiT | T2V; approximate, opt-in |
| Measurement | Warm end-to-end generation latency | T2V and I2V |

Feature availability is model-specific
and may differ as additional diffusion pipelines are added.

## Compilation and artifact reuse

During model initialization, the plugin wraps the text encoder, diffusion
transformer (DiT), and VAE with PyTorch `torch.compile` and the vLLM Neuron
backend. Compilation is lazy: graph capture and Neuron executable file format (NEFF)
compilation happen when each compiled component first runs for a new input
shape.

The first request can therefore take substantially longer than later requests.
`VLLM_CACHE_ROOT` sets the compiler work directory; it does not set the NEFF cache.

The compiled graph depends on its shape and compile configuration. Expect a new
compilation after changing:

- Model or checkpoint.
- Output height, width, or frame count.
- Parallel topology.
- Precision or quantization mode.
- A model option that changes the compiled graph.

Prompts, seeds, guidance scale, and inference-step count do not change the DiT
input shape or require a new graph for that shape.

### Static output shapes

Neuron compiles fixed-shape graphs. For Wan2.2, the requested spatial
dimensions are rounded down to a multiple of 16 before latent processing:

```text
H_lat = ((height // 16) * 16) / 8
W_lat = ((width  // 16) * 16) / 8
T_lat = (num_frames - 1) / 4 + 1
S = T_lat * (H_lat / 2) * (W_lat / 2)
```

`S` is the DiT token sequence length. It must be divisible by the configured
context-parallel degree. Use frame counts of the form `4k + 1`, such as 5, 81,
or another model-supported value.

Keep the output shape fixed while tuning prompts, guidance, or denoising steps
to avoid unnecessary recompilation.

## Generation modes

### Text-to-video

Run the T2V example with the default 832 x 480, 81-frame, 40-step
configuration:

```bash
sudo docker exec vllm-omni-neuron \
  python /workspace/plugin/examples/wan22/run.py \
  --prompt "A lighthouse above a stormy sea at sunrise" \
  --output /workspace/output/wan-t2v.mp4
```

For a short development run with a smaller shape:

```bash
sudo docker exec vllm-omni-neuron \
  python /workspace/plugin/examples/wan22/run.py \
  --dev \
  --output /workspace/output/wan-t2v-dev.mp4
```

Development mode uses 208 x 128, 5 frames, and 10 denoising steps. It compiles
a different shape from the full configuration.

### Image-to-video

The I2V pipeline conditions generation on a first-frame image. The example uses
the repository's sample image unless `--image` is provided:

```bash
sudo docker exec vllm-omni-neuron \
  python /workspace/plugin/examples/wan22/run_i2v.py \
  --prompt "The subject turns toward the camera as waves move in the background" \
  --output /workspace/output/wan-i2v.mp4
```

To use your own image, make it available inside the container and pass its
container path:

```bash
sudo docker cp first-frame.png vllm-omni-neuron:/workspace/first-frame.png

sudo docker exec vllm-omni-neuron \
  python /workspace/plugin/examples/wan22/run_i2v.py \
  --image /workspace/first-frame.png \
  --output /workspace/output/wan-i2v.mp4
```

I2V uses the model's VAE encoder to create the image condition
before the denoising loop. At 720p, both VAE encode and decode require tiling
and benefit from VAE patch parallelism.

## Generation controls

The principal output-quality controls are:

| Control | Effect | Trade-off |
| --- | --- | --- |
| `height`, `width` | Spatial output resolution | Higher resolution increases attention cost and VAE memory; changing it recompiles |
| `num_frames` | Video duration at the output frame rate | More frames increase sequence length, latency, and memory; changing it recompiles |
| `num_inference_steps` | Number of denoising iterations | More steps usually improve detail and motion at approximately linear latency cost |
| `guidance_scale` | Strength of prompt adherence | Higher values can improve adherence but can introduce artifacts |
| `seed` | Seeds initial latent noise | Reproduces the same sampling trajectory for a fixed configuration |

The example runner exposes height, width, frame count, prompt, and guidance
through command-line options. It uses 40 steps and seed 42 in full mode. Use
the `OmniDiffusionSamplingParams` API or adapt the example when you need to
control the step count or seed:

```python
params = OmniDiffusionSamplingParams(
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=40,
    guidance_scale=5.0,
    seed=42,
)
result = omni.generate({"prompt": prompt}, params)
```

### Classifier-free guidance

Classifier-free guidance (CFG) evaluates conditioned and unconditioned noise
predictions and combines them:

```text
prediction = unconditioned + guidance_scale * (conditioned - unconditioned)
```

The default guidance scale is 5.0. Use `--no-cfg` to generate without CFG:

```bash
python /workspace/plugin/examples/wan22/run.py --no-cfg
```

This sets `guidance_scale=1.0` and skips the unconditioned pass. Use a validated
`cfg_parallel_size: 1` stage configuration for no-CFG generation to avoid
reserving cores for CFG parallelism. Disabling CFG weakens prompt adherence
and removes negative-prompt guidance.
This is different from CFG parallelism, which preserves both passes and runs
them concurrently.

Wan2.2 also exposes `boundary_ratio` and `flow_shift` in stage configuration.
These control the high-noise/low-noise expert handoff and flow-matching
schedule. Keep the model-tuned defaults (`boundary_ratio: 0.875` for T2V,
`0.9` for I2V, and `flow_shift: 5.0`) unless you are deliberately evaluating a
different schedule.

## Parallelism

Wan2.2 composes three parallel dimensions for the DiT. The number of worker
ranks is:

```text
world_size = tensor_parallel_size * context_parallel_size * cfg_parallel_size
```

The VAE uses a separate subgroup selected by `vae_patch_parallel_size`; it does
not multiply the DiT world size.

The shipped 64-core T2V configuration is:

```yaml
stage_args:
  - runtime:
      devices: "0-63"
    engine_args:
      model_class_name: Wan22Pipeline
      dtype: bfloat16
      boundary_ratio: 0.875
      flow_shift: 5.0
      vae_use_tiling: true
      model_config:
        tp_sequence_parallel: true
      parallel_config:
        tensor_parallel_size: 4
        ring_degree: 8
        cfg_parallel_size: 2
        vae_patch_parallel_size: 16
```

Keep `runtime.devices` consistent with the calculated world size. Parallel
configuration comes from the stage YAML; changing only an example runner's
thread settings does not change the worker topology.

### Tensor parallelism

Tensor parallelism (TP) shards the DiT and text-encoder weights, attention
heads, and feed-forward dimensions. It is the primary way to reduce per-rank
weight memory.

Wan2.2 has 40 attention heads, so the TP degree must divide 40. The validated
configurations use TP=4 or TP=8:

- TP=4 leaves more cores available for context or CFG parallelism.
- TP=8 reduces weight memory per rank but increases the relative cost of TP
  collectives.

Sequence parallelism keeps transformer activations partitioned over the TP
group between row-parallel operations. It replaces selected TP all-reduces
with reduce-scatter and all-gather collectives, reducing replicated activation
residency. Enable it in the model configuration:

```yaml
engine_args:
  model_config:
    tp_sequence_parallel: true
```

This is independent of the context-parallel dimension described below. The
shipped T2V and I2V stage configurations enable it.

Choose the smallest TP degree that fits the model and leaves a useful amount
of work per rank.

### Context parallelism

Context parallelism (CP) shards the spatial-temporal token sequence across
ranks. It reduces per-rank DiT compute and shardable activation memory, making
longer or higher-resolution video shapes practical.

Configure the CP degree with `ring_degree`:

```yaml
parallel_config:
  tensor_parallel_size: 4
  ring_degree: 8
```

vLLM Omni derives `sequence_parallel_size` from
`ulysses_degree * ring_degree`. The plugin leaves `ulysses_degree` at 1, so
`ring_degree` is the CP degree. Do not also set `sequence_parallel_size`.

The DiT token length `S` must be divisible by the CP degree. If it does not,
adjust the output shape or use a smaller `ring_degree`.

#### Context-parallel attention

Context-parallel self-attention uses ring attention: it rotates local K/V chunks
around the CP group instead of materializing full K/V on every rank. Where the
ring kernel cannot run (CPU mode, fake-tensor tracing, NKI kernels disabled) the
runtime falls back to all-gather K/V with flash attention automatically.

Because the ring kernel has no online-max pass, the softmax maximum is bounded up
front, at one value per query row. The bound is static across ring steps, so merging
each rank's partial attention is pure addition — no running max and no correction
factors travel around the ring.

### CFG parallelism

CFG parallelism assigns the conditioned and unconditioned passes to two model
replicas:

```yaml
parallel_config:
  cfg_parallel_size: 2
```

The replicas run concurrently and gather their predicted-noise tensors for
the guidance combine. This preserves CFG output while reducing wall-clock
denoising time. It doubles the cores assigned to the DiT and does not reduce
per-rank memory because each replica holds the model.

Use CFG=2 for final guided generation when the extra cores are available. Use
CFG=1 for sequential guidance or when those cores are needed for TP or CP.
Values greater than 2 provide no benefit because CFG has only two branches.

### VAE patch parallelism

VAE patch parallelism distributes spatial tiles across a subgroup of ranks:

```yaml
engine_args:
  vae_use_tiling: true
  parallel_config:
    vae_patch_parallel_size: 32
```

For T2V, it parallelizes tiled decode. For I2V, it parallelizes both
conditioning-image encode and output decode. The provided configurations use a
VAE patch-parallel size of 16 for 832 x 480 and 32 for 1280 x 720. The VAE
patch-parallel group cannot exceed the DiT world size, and useful parallelism is
further limited by the number of VAE tiles; additional ranks may receive no
tile.

### Choosing a topology

Start from a validated stage configuration:

| Topology | Workers | Typical use |
| --- | ---: | --- |
| TP4 x CP8 x CFG2 | 64 | Recommended final T2V/I2V generation with parallel guidance |
| TP4 x CP8 x CFG1 | 32 | Same TP/CP memory layout with sequential guidance |
| TP8 x CP4 x CFG1 | 32 | Alternative 32-core layout; used for 720p I2V |

Then account for these constraints:

- TP degree must divide the model's attention-head count.
- CP degree must divide DiT token length.
- CFG parallelism degree is either 1 or 2.
- The product must equal the number of worker ranks in `runtime.devices`.
- Collective groups must map to a topology supported by the target hardware.

For a detailed performance and memory treatment, see
[Optimizing high-quality offline video generation](../model-dev/optimizing-offline-video-generation.md).

## VAE memory features

VAE processing can be the peak-memory part of a request. T2V decodes the final
latents, while I2V also encodes the conditioning image before denoising.

### Spatial tiling

Spatial tiling divides the latent into overlapping tiles, decodes each tile,
and blends the overlap regions into the output. Enable it in stage YAML:

```yaml
engine_args:
  vae_use_tiling: true
```

Use tiling for 1280 x 720 generation. It lowers peak HBM usage at the cost of
multiple VAE executions and additional stitching work. Tile shapes are part of
the compiled workload, so the first tiled run must compile them.

### Temporal chunking

The Wan VAE processes frames in temporal chunks while carrying causal
convolution state between chunks. This behavior is automatic rather than a
user-facing configuration option. It bounds intermediate memory while
preserving temporal continuity.

### CPU offload during decode

Set `enable_cpu_offload: true` under `engine_args` to move DiT weights off the
VAE rank during decode and restore them afterward:

```yaml
engine_args:
  enable_cpu_offload: true
```

This frees HBM for the VAE on memory-constrained configurations. It adds
host-device transfer time to every request, so prefer tiling and VAE patch
parallelism when they provide enough memory headroom.

## FP8 quantization

Wan2.2 T2V supports the ComfyUI per-tensor FP8-scaled checkpoint through
`quantization: fp8_row_mx`. This is separate from the BF16 checkpoint selected
by `--model-path`.

On Trainium3, the default FP8 configuration runs the self-attention and
cross-attention QKV/output projections and the FFN up/down projections with
native ROW_MX kernels. The attention operation, normalization, and projection
biases remain BF16. Enable it with:

```bash
sudo docker exec vllm-omni-neuron \
  python /workspace/plugin/examples/wan22/run.py \
  --quantization fp8_row_mx \
  --comfyui-fp8-model-path Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  --output /workspace/output/wan-fp8.mp4
```

Add any combination of `attn1`, `attn2`, and `ffn` to `modules_to_not_convert`
to dequantize those projection groups to BF16 at load time. This is useful for
comparing execution paths, isolating quality differences, or using the FP8
checkpoint without native ROW_MX compute; specify all three to disable it
entirely. Dequantization preserves the quantized checkpoint values rather
than restoring the original BF16 weights.

FP8 can improve latency and reduce projection-weight bandwidth, but it changes
numeric behavior and requires a compatible scaled checkpoint. Validate video
quality against the BF16 configuration before deploying it.

## Cache-DiT

Cache-DiT reuses intermediate DiT residuals across nearby denoising steps and
skips selected block computation. Enable the example configuration with:

```bash
sudo docker exec vllm-omni-neuron \
  python /workspace/plugin/examples/wan22/run.py \
  --cache-backend cache_dit \
  --output /workspace/output/wan-cache-dit.mp4
```

The example sets warmup count, cache duration, residual threshold, and maximum
continuous skipped steps in `cache_config`.

Unlike parallelism or compilation caching, Cache-DiT is approximate: it can
change the denoising trajectory and final video. The performance gain and
quality impact depend on the prompt, output shape, step count, and thresholds.
Treat the supplied values as a starting point, compare cached and uncached
outputs, and evaluate quality over a representative prompt set.

## Measuring generation latency

The example runners can measure warm end-to-end generation latency:

```bash
sudo docker exec vllm-omni-neuron \
  python /workspace/plugin/examples/wan22/run.py \
  --profile \
  --output /workspace/output/wan-profile.mp4
```

The first generation compiles and warms the graphs. The runner then times an
additional generation and reports its latency. Keep the model, prompt, seed,
output shape, denoising steps, and guidance fixed when comparing
configurations, and validate output quality as well as latency.

## Common issues

| Symptom | Likely cause | Resolution |
| --- | --- | --- |
| First generation takes a long time | Neuron is compiling a new graph or loading uncached weights | Compare warm runs after compilation completes |
| Changing resolution triggers compilation | Height and width are compiled shapes | Reuse a small set of production output shapes |
| `Sequence length ... is not divisible by cp_size` | The output shape is incompatible with `ring_degree` | Change frame count/resolution or reduce CP |
| OOM during 720p VAE encode/decode | Non-tiled VAE exceeds per-rank HBM | Enable `vae_use_tiling` and increase `vae_patch_parallel_size` |
| Worker exits with `SIGBUS` while returning video | Container shared memory is too small | Recreate the container with `--shm-size=2g` |
| Native ROW_MX raises a platform error | Native ROW_MX projection kernels are selected on a non-Trn3 target | Use Trainium3 or dequantize `attn1 attn2 ffn` |
| Cache-DiT output quality changes | Cached residual reuse is approximate | Tighten cache thresholds, reduce skipped steps, or disable Cache-DiT |

## Related information

- [Wan2.2 model card](../models/wan22-t2v-14b.md)
- [Optimizing high-quality offline video generation](../model-dev/optimizing-offline-video-generation.md)
- [Context parallelism design](../design/context_parallelism.md)
- [Kernel implementations](../model-dev/kernels/index.md)
- [vLLM Omni Neuron architecture](../design/vllm_omni_neuron_overview.md)
- [Onboard a model to vLLM-Omni Neuron](../model-dev/onboarding-models.md)
- [Accuracy evaluation and debugging](../model-dev/accuracy-evaluation-debugging.md)
