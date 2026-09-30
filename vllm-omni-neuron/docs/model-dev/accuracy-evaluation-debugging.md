<!-- meta: description: Evaluate output quality, detect numerical regressions,
and isolate accuracy issues in Wan2.2 running through the vLLM Omni Neuron
plugin on AWS Trainium. -->
<!-- meta: keywords: vLLM, Neuron, Omni, Wan2.2, diffusion, accuracy,
regression, BF16, SSIM, Trainium -->
<!-- meta: content_type: procedural-how-to -->

# Accuracy evaluation and debugging

## Task overview

This guide describes how to evaluate output quality, detect regressions, and
isolate numerical accuracy issues in diffusion inference through the vLLM Omni
Neuron plugin.

Diffusion models generate their output through an iterative denoising loop and a
final decode to pixels, so accuracy debugging differs from the token-by-token
workflow used for large language models. If you are debugging an LLM (token divergence,
logit drift, KV-cache consistency), see the
[vLLM Neuron accuracy debugging guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/model-dev/accuracy-debugging-guide.html)
instead.

The examples use [Wan2.2 (text-to-video)](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers), the reference model that ships with this
plugin, but the methodology applies to any diffusion model you run on Neuron.

## Prerequisites

- **A reproducible deployment.** A running configuration where you can observe the
  accuracy issue, and reproduce it deterministically (fixed seed, fixed number of
  denoising steps, fixed guidance scale).
- **A reference implementation.** Known-good outputs from the same model through the
  upstream implementation (for example, Hugging Face `diffusers`). Use CPU FP32 and
  BF16 references for numerical comparisons, or another hardware platform you are
  evaluating for an end-to-end comparison.
- **Familiarity with the model.** An understanding of the model's pipeline stages
  (text encoder, the denoising transformer, and the decoder) and its expected
  input and output shapes.
- **A Python environment** with the reference framework and the model checkpoint, so
  you can generate reference outputs to compare against.

## Choose the right evaluation

Different questions require different evidence:

| Goal | Comparison | Recommended evidence |
|---|---|---|
| Measure output quality | Neuron output against a quality baseline | Representative prompts, a video benchmark such as [VBench](https://arxiv.org/abs/2311.17982), and human review |
| Detect a software regression | Current output against a reviewed Neuron golden | Per-frame Structural Similarity Index Measure (SSIM) and saved video artifacts |
| Validate a numerical component | Equivalent inputs through FP32 reference, BF16 reference, and BF16 Neuron | Three-way L-inf, L2, sigma-ratio, and BC metrics |
| Investigate timestep-dependent behavior | The same latent, timestep, and conditioning at selected steps | Fixed-input step-level comparison |

Do not use tensor metrics alone to determine video quality. Conversely, visual
inspection alone cannot identify a numerical defect.

## Validation levels

Accuracy validation operates at several levels, from end-to-end output quality
to individual model computations. Start at the level that matches the reported
symptom, then move to narrower levels to isolate the cause.

1. **Quality validation**.
Evaluate whether generated videos meet the quality requirements of the intended
workload. Use a representative prompt set covering supported resolutions, frame counts,
motion patterns, CFG configurations, and relevant subject matter. Evaluate the
results with human review and, where appropriate, a video-generation benchmark
such as VBench.

2. **End-to-end regression validation**.
Compare the current output against a previously reviewed and versioned golden
output using the same checkpoint, initial latent, prompt, scheduler
configuration, and execution settings.
For decoded video, use per-frame metrics such as SSIM together with visual
inspection. SSIM compares luminance, contrast, and structure; higher values mean
the frames are more similar, and `1.0` means they are identical. A regression
failure means the output changed; it does not determine whether the new output
is worse or identify which component caused the change.

3. **Pipeline-stage validation**. Validate complete pipeline components independently:

    - Text encoder
    - Denoising transformer
    - Scheduler update
    - VAE decoder
    - Frame post-processing

   For numerical components, compare the same inputs and weights in FP32 reference,
BF16 reference, and BF16 Neuron configurations. Use real weights and production
shapes whenever practical. Stop at the first component that shows unexpected
divergence.

4. **Module-level validation**.
Narrow a failing component to modules such as attention, normalization,
feed-forward, convolution, positional encoding, or residual blocks.
Begin with modules in execution order and compare their outputs using equivalent
inputs. If modules pass independently but the complete component fails, test
progressively larger groups of modules. Confirm small-shape or random-input
results with real weights and production inputs.

5. **Step-level validation**.
Use step-level validation when a problem appears to depend on the denoising
timestep or develops during the denoising loop.
For a meaningful comparison, provide each implementation with the same latent
entering the step, scheduler timestep, prompt conditioning, CFG behavior, and
model preprocessing. Compare the transformer noise prediction before applying
the scheduler update. Test scheduler arithmetic separately with the same latent
and noise prediction.

   Begin with one fixed-input step. If the problem appears timestep-dependent, save
reference inputs from several representative steps and replay each independently.
Do not use later outputs from independently evolving trajectories to localize a
defect because those calls may already be receiving different latents.

## Three-way comparison

Numerical tests in this repository use a three-way comparison:

| Leg | Precision and device | Meaning |
|---|---|---|
| FP32 reference | FP32 in the reference implementation | Higher-precision reference computation |
| BF16 reference | BF16 in the reference implementation | Expected low-precision behavior of the reference implementation |
| BF16 Neuron | BF16 on Neuron | Target computation under test |

The BF16 reference distinguishes expected low-precision behavior from additional
Neuron error. Comparing BF16 Neuron directly against FP32 can incorrectly classify
ordinary low-precision differences as a Neuron defect.

All three legs must use equivalent weights and logical inputs. Detailed reference
generation, metrics, and thresholds are covered in
{ref}`Step 3 <step3-fixed-input-three-way-comparison>`.

## Debug an accuracy issue

The following steps describe the typical investigation order. If an existing
test already identifies a failing component or module, begin at the corresponding
step rather than repeating broader evaluations.

### Step 0: Prepare a reproducible comparison

#### Record the comparison configuration

An accuracy comparison is meaningful only when both runs use equivalent inputs
and settings. Record:

- Model ID and immutable checkpoint revision
- Plugin, container, Neuron SDK, and compiler versions
- Trainium generation and instance type
- Stage configuration and TP, CP, and CFG parallelism
- Prompt and negative prompt
- Height, width, and frame count
- Scheduler, flow shift, inference steps, and guidance scale
- Cache and decoder-tiling configuration
- Random seed and, for cross-implementation comparisons, the saved initial
  latent or its hash

If the backend, compiler flags, or platform overrides differ from the documented
defaults, record those changes as well. Align configurations before treating a
difference as an accuracy defect, and change only one setting at a time during
diagnosis.

#### Use identical initial latents

A fixed seed is useful for rerunning one implementation, but it does not prove
that two implementations generated the same initial noise. Random-number
generation can depend on device and dtype.

For cross-implementation comparisons, create one FP32 latent tensor on CPU and
inject it into every run. For the supported Wan2.2 model:

```python
import torch

height = 128
width = 208
num_frames = 5
latent_seed = 42

# Wan2.2 constants: 16 latent channels, temporal scale 4, spatial scale 8.
latent_frames = (num_frames - 1) // 4 + 1
latent_shape = (1, 16, latent_frames, height // 8, width // 8)

generator = torch.Generator(device="cpu").manual_seed(latent_seed)
initial_latents = torch.randn(
    latent_shape,
    generator=generator,
    dtype=torch.float32,
    device="cpu",
)
torch.save(initial_latents, "/tmp/wan22_initial_latents.pt")
```

Start with [`examples/wan22/run.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/run.py), retain its setup,
and replace its sampling-parameter block with this excerpt:

```python
initial_latents = torch.load(
    "/tmp/wan22_initial_latents.pt",
    map_location="cpu",
    weights_only=True,
)

params = OmniDiffusionSamplingParams(
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=1,
    guidance_scale=1.0,
    latents=initial_latents,
)

result = omni.generate(
    {"prompt": "A cat sitting on a windowsill watching rain"},
    params,
)
```

The injected tensor defines the initial noise, so no seed is required for this run.

### Step 1: Confirm the issue (quality and end-to-end)

First determine whether the report concerns output quality, a software regression,
or a numerical component.

#### Evaluate output quality

Use a prompt set that represents the intended workload. Include short and detailed
prompts, different motion speeds, relevant subjects, CFG and no-CFG configurations,
and every supported resolution and frame count.

Review the generated videos and run a suitable video-generation benchmark. Keep this
result separate from deterministic regression results. Two videos can differ
pixel-by-pixel while remaining similar in quality, and a video can match a low-quality
golden exactly.

[VBench](https://github.com/Vchitect/VBench) provides public evaluation workflows for
text-to-video (T2V) and image-to-video (I2V) outputs. See the
[VBench paper](https://arxiv.org/abs/2311.17982) and
[VBench++ paper](https://arxiv.org/abs/2411.13503) for the benchmark methodology.
Install it by following its upstream requirements:

```bash
git clone https://github.com/Vchitect/VBench.git
cd VBench
pip install .
```

For T2V videos generated from your own prompts, run the custom-input workflow with the
dimensions supported by VBench:

```bash
python evaluate.py \
    --videos_path <generated-videos> \
    --dimension subject_consistency background_consistency motion_smoothness \
        dynamic_degree aesthetic_quality imaging_quality \
    --mode=custom_input
```

For I2V outputs, follow [VBench-I2V naming requirements](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v)
 and provide both the generated videos and their corresponding input images.

```bash
python evaluate_i2v.py \
    --videos_path <generated-videos> \
    --custom_image_folder <input-images> \
    --dimension i2v_subject i2v_background camera_motion \
        subject_consistency background_consistency motion_smoothness \
        dynamic_degree aesthetic_quality imaging_quality \
    --ratio 16-9 \
    --mode=custom_input
```

These custom-input workflows are useful for evaluating arbitrary generated outputs, but
they do not by themselves reproduce published or leaderboard-comparable scores. For a
controlled comparison, use the official VBench prompt or image suite and follow its
sampling and file-naming requirements. Record the VBench commit, evaluated dimensions,
resolution, frame count, FPS, generation configuration, and seeds with the results.

#### Detect an end-to-end regression

Compare decoded output against a reviewed, versioned golden using identical inputs
and configuration. Version the golden with the comparison configuration, initial
latent, decoded frame tensor, viewable video artifact, prompt, and negative prompt.

Per-frame SSIM can help detect output changes, but it is sensitive to small spatial
and temporal differences and can produce false alarms. Calibrate thresholds for
the workload and use SSIM alongside visual inspection or a quality benchmark. A
low SSIM result confirms a difference, not its cause or whether output quality
degraded.

### Step 2: Isolate the first failing pipeline stage (pipeline-stage level)

Compare pipeline boundaries in execution order and stop at the first unexpected
divergence:

```text
tokenization and text encoding
    -> denoising transformer
    -> scheduler update
    -> latent unscaling and VAE decode
    -> frame post-processing
```

Use the fixed-input three-way method in Step 3 for numerical component outputs.

**Text encoder:** Start with identical token IDs and attention masks. Compare
the post-processed prompt embeddings, not only the raw encoder output. This
separates tokenization and padding differences from encoder arithmetic.

**Denoising transformer:** Supply the same latent, timestep, prompt embeddings,
attention masks, and CFG inputs to all three legs. Begin with one step and one
transformer forward pass.

**Scheduler:** Supply the same latent, timestep, and model output. Do not compare
scheduler results produced from already divergent noise predictions and attribute
the difference to the scheduler.

**VAE decoder:** Supply the same unscaled latent and use the same tiling setting.
Compare the decoder tensor before frame post-processing, then compare normalized
frames.

(step3-fixed-input-three-way-comparison)=
### Step 3: Run a fixed-input three-way comparison (shared method)

Use the same reference implementation, weights, and input tensors for both CPU
legs. Load reference models sequentially to limit peak host memory, run in
inference mode, and convert outputs to FP32 before comparison:

```python
import gc

import torch
from diffusers import WanTransformer3DModel


def run_cpu_reference(dtype):
    model = WanTransformer3DModel.from_pretrained(
        checkpoint_path,
        subfolder="transformer",
        torch_dtype=dtype,
    ).eval()
    with torch.inference_mode():
        output = model(
            hidden_states=hidden_states.to(dtype),
            timestep=timestep.to(dtype),
            encoder_hidden_states=encoder_hidden_states.to(dtype),
            return_dict=False,
        )[0]
    output = output.float().cpu()
    del model
    gc.collect()
    return output


fp32_reference = run_cpu_reference(torch.float32)
bf16_reference = run_cpu_reference(torch.bfloat16)
```

Create `hidden_states`, `timestep`, and `encoder_hidden_states` once on CPU. Do
not regenerate them between calls. The exact arguments differ by component.


After producing all three tensors, use the assertion provided by `vllm_neuron`.
This numerical comparison helper is shared with the LLM accuracy tooling; the
LLM-specific token, logit, and KV-cache debugging workflows do not apply directly
to diffusion inference.

```python
from vllm_neuron.accuracy.testing import assert_close_three_way

assert_close_three_way(
    fp32_reference,
    bf16_reference,
    bf16_neuron,
    name="wan22_transformer",
)
```

The helper reports:

- **L-inf ratio:** target worst-case relative error divided by BF16-reference
  worst-case relative error
- **L2 ratio:** target relative L2 error divided by BF16-reference relative L2
  error
- **Sigma-ratio:** RMS target error divided by RMS BF16-reference error over the
  aggregated tensors
- **Bhattacharyya Coefficient (BC):** overlap between histograms of absolute
  errors

The current default pass rule requires:

- Aggregate BC of at least `0.99` or aggregate sigma-ratio of at most `1.0`
- Worst L-inf ratio below `5.0`
- Worst L2 ratio below `3.0`

Treat helper defaults as a starting point for diagnosis. Calibrate the
criteria with known-good and known-bad results for the component and shape under
test.

### Step 4: Narrow the failure to a module (module level)

Use the failing component's real input as the fixture for smaller comparisons.
Random inputs can miss failures that depend on activation range or trained weight
distributions.

1. Compare major modules independently in execution order.
2. Stop at the first unexpected output difference.
3. Narrow that module to operations such as attention, normalization,
   feed-forward, convolution, or positional encoding.
4. If individual modules pass but the component fails, test progressively larger
   groups or adjacent modules.
5. Reproduce the result with real weights and production shapes.

When parallelism changes the result, verify weight sharding, collective groups,
padding removal, reduction order, and reconstruction of the logical output. A
per-rank tensor is not directly comparable to an unsharded CPU tensor unless the
model contract says it is replicated.

### Step 5: Check for timestep-dependent behavior (step level)

Begin with one fixed-input transformer call. Supply the same latent, timestep,
and conditioning to the FP32 reference, BF16 reference, and Neuron transformer,
then compare their noise predictions before the scheduler update.

If the mismatch appears timestep-dependent, save the reference input at several
representative timesteps and replay each input independently. This prevents an
earlier output difference from changing the inputs to later comparisons.

A free-running trajectory, where each implementation feeds its own output into
the next step, can measure total output drift. It cannot reliably identify the
step or operation that introduced a numerical defect after the trajectories have
diverged.

### Step 6: Validate the fix

After identifying the root cause, apply your fix and validate systematically:
1. Rerun the smallest fixed-input component or module reproducer.
2. Verify the original production shape and parallel configuration.
3. Test another timestep and at least one additional prompt or conditioning
   input.
4. Run the trusted end-to-end golden regression.
5. Run the representative quality prompt set.
6. Confirm that the fix does not introduce non-finite values, compilation
   failures, or unacceptable performance regressions.

A fix is not complete merely because one tensor passes or one prompt looks
better.

## Quick troubleshooting

| Check | Action |
|---|---|
| Initial noise | Load one saved FP32 latent into every implementation; matching seeds alone are insufficient across devices and dtypes |
| Configuration | Match model revision, scheduler, flow shift, steps, guidance, prompts, cache, tiling, resolution, and frame count |
| Dtype | Use the three-way comparison, then inspect casts around normalization, attention, residual paths, and scheduler arithmetic |
| Shape | Reproduce the actual production dimensions, check non-finite indices, and include non-aligned shapes in coverage |
| Parallelism | Compare TP and CP configurations and verify sharding, groups, padding removal, reduction order, and output reconstruction |
| Cache | Disable caching with the same latent, then evaluate quality before deciding whether cache-induced drift is acceptable |
| Compiler or platform | Record SDK and compiler versions, target, optimization, and auto-cast settings; preserve the error and minimal graph for compile failures |

Once a check changes the result, hold all other variables fixed and reduce that
difference to a component or module reproducer.
