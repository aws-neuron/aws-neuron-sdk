# Optimizing High-Quality Offline Video Generation

<!-- meta: description: How to reason about high-quality offline text-to-video
generation with diffusion models on vLLM Omni Neuron — the three-stage cost model, the
denoising loop as the quality lever, a formal treatment of the parallelism-selection
problem (why CFG parallelism, tensor parallelism, and context parallelism differ in
efficiency and how to allocate a fixed core budget across them), the per-core HBM
budget that decides feasibility, fitting the VAE decode in memory, and ready-to-run
deployment commands. -->
<!-- meta: keywords: vLLM, Neuron, diffusion, text-to-video, video generation,
Wan2.2, DiT, denoiser, VAE, MoE, guidance scale, CFG, inference steps, flow
matching, tensor parallelism, context parallelism, CFG parallelism, VAE tiling,
NKI kernels, Trainium -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-07-28 -->

## Table of contents

1. [Overview](#overview)
2. [Grounding in an output spec](#grounding-in-an-output-spec)
3. [The denoising loop](#the-denoising-loop-low-quality-to-high-quality)
4. [Choosing a parallelism scheme](#choosing-a-parallelism-scheme)
5. [NKI kernels](#nki-kernels)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)
8. [Related information](#related-information)

## Overview

Offline video generation is a *quality-first* workload: no user is waiting on a
stream, so the goal is the best clip for a fixed compute budget, not the lowest
latency. This document shows how to reason from that premise on AWS
Trainium (`trn2` / `trn3`) with the `vllm-omni-neuron` plugin.
**[Wan2.2-T2V-A14B](../models/wan22-t2v-14b.md)** is the running example, but the
reasoning applies to any latent text-to-video diffusion model.

The expected customer use case is to run this high-quality Wan2.2 model as a
**teacher for distillation**: its slow, bidirectional, high-fidelity outputs become
the training signal for smaller, faster student models — causal or few-step
distilled variants such as [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
and [Rolling Forcing](https://github.com/TencentARC/RollingForcing) — that meet
real-time latency budgets. Because the teacher runs
offline, maximizing its quality per clip — the focus of this tutorial — directly
raises the ceiling on what the distilled student can learn.

A text-to-video diffusion model runs in three on-device stages, each bound by a
different resource:

| Stage | What it does | Bound by |
| --- | --- | --- |
| **Text encoder** | Encodes the prompt (and negative prompt) into conditioning | Compute (one-shot, negligible) |
| **DiT denoising loop** | Iteratively denoises the latent over `num_inference_steps` | Compute, **repeated per step** |
| **VAE decoder** | Decodes the final latent into RGB frames | **HBM capacity** |

The **DiT loop dominates** both quality and cost — it is one transformer run tens of
times over, so its per-step latency is multiplied by the step count, and it is where
composition, motion, and detail are decided. The **VAE** runs once but is the memory
hotspot: a single-shot high-resolution decode can exceed per-device HBM.

The rest of this document follows that cost model: **[pick a fixed output
spec](#grounding-in-an-output-spec)**, **[set quality via the denoising
loop](#the-denoising-loop-low-quality-to-high-quality)**, and **[choose a parallelism
scheme](#choosing-a-parallelism-scheme)**. It assumes the model is already running —
see the [model card](../models/wan22-t2v-14b.md),
[`run.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/run.py), and [`README.md`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/README.md).

## Grounding in an output spec

Analysis is only meaningful against the clip you want to ship. Two properties drive
everything: **resolution** (`width × height`) and **length** (`num_frames`). They
set the latent tensor the DiT denoises and the VAE decodes — a latent diffusion
model never works on pixels:

```text
z_dim = 16                          # Wan2.2 VAE latent channels
W_lat = ((width  // 16) * 16) / 8   # VAE spatial compression 8x
H_lat = ((height // 16) * 16) / 8
T_lat = (num_frames - 1) / 4 + 1    # VAE temporal compression 4x
```

The pipeline first rounds `height`/`width` **down to a multiple of 16**
(`vae_scale_factor_spatial × patch_size`), so a spec that is not already a multiple
of 16 silently generates at a slightly smaller frame size.

The DiT then patchifies the latent into a token sequence. Its length `S` is the one
number every parallelism and memory decision keys off of:

```text
post_patch_w = W_lat / 2                  # DiT patch size 2 per spatial dim
post_patch_h = H_lat / 2
S = T_lat * post_patch_w * post_patch_h   # DiT token sequence length
```

For the default **832×480, 81 frames**: `W_lat=104, H_lat=60, T_lat=21`, so
`S = 21 · 52 · 30 = 32,760`. Watch this number: DiT attention cost scales as `S²`,
context parallelism must divide `S` evenly, and `S` decides whether the clip fits in
HBM.

**Keep the spec fixed while comparing options.** Changing `height`, `width`, or
`num_frames` changes `S` and forces a **recompile** (a new NEFF). Changing step
count, guidance, seed, or the prompt does not — so iterate on those freely once a
shape is compiled.

## The denoising loop: low-quality to high-quality

Quality is decided mainly in the DiT loop, which integrates a flow-matching ordinary
differential equation (ODE) from noise to a clean latent over `num_inference_steps`.
Four levers control it, in rough order of impact.

### `num_inference_steps` — the primary quality dial

Each step is one ODE integration step; more steps mean smoother motion, cleaner
textures, and fewer temporal artifacts, with diminishing returns. This tutorial uses
the Wan2.2 default of 40.

Step count costs latency linearly (every step is a full DiT pass), so it is the
first quality/latency trade to reason about.

(guidance-scale)=
### `guidance_scale` — prompt adherence vs. artifacts

Classifier-Free Guidance (CFG) runs a conditioned and an unconditioned pass per step
and extrapolates between them by `guidance_scale`. This tutorial uses the Wan2.2
default of 5.0.

Every denoising step evaluates the same DiT **twice on the same latent** —
once with the prompt embedding as conditioning, once with the negative-prompt (or empty)
embedding — and combines the two velocity predictions:

```text
v = v_uncond + guidance_scale * (v_cond - v_uncond)
```

Only the conditioning tensor differs between the two evaluations; the latent, the
timestep, the weights, and therefore the shapes and FLOPs are identical.

Two ways to spend or avoid that cost:

- **{ref}`CFG parallelism <cfg-parallelism>`**
  exists precisely because the two passes are *independent* — nothing in the conditioned
  pass feeds the unconditioned one, so they can run concurrently on separate replicas and
  recover most of the wall-clock time. This costs 2× the cores, not 2× the time.
- **`--no-cfg`** sets `guidance_scale=1.0`, which collapses the formula above to
  `v = v_cond` and lets the pipeline skip the unconditioned pass entirely — a genuine ~2×
  speedup, but a preview-only one: it weakens prompt adherence and removes
  negative-prompt control, since there is no longer an unconditioned baseline to
  extrapolate away from.

### `boundary_ratio` — the MoE expert boundary

Wan2.2's DiT is a Mixture-of-Experts denoiser (~27B total, **14B active per step**)
with two experts: a **high-noise expert** for the early steps that set layout and
motion, and a **low-noise expert** for the later steps that refine detail.
`boundary_ratio` (default **0.875**) is the fraction of the trajectory the high-noise
expert handles before the handoff. It is a Wan-AI-tuned default that interacts with
step count and schedule — **leave it at 0.875 unless deliberately experimenting.**

### `flow_shift` — the noise schedule

`flow_shift` (default **5.0**) shifts the schedule toward high-noise timesteps, which
aids temporal coherence. Like `boundary_ratio`, it reshapes the whole trajectory;
treat it as fixed unless running a deliberate sweep.

## Choosing a parallelism scheme

Parallelism runs your chosen quality settings within HBM, as fast as the hardware
allows, without changing the result. Wan2.2 composes three DiT axes — the core count
is their product (`TP × CP × CFG`); the VAE shards separately. Picking the three
degrees is not guesswork: it is a small constrained-optimization problem, and this
section states it formally, analyzes each axis, and then derives the recommended
configuration.

### The optimization problem

Fix the output spec (so `S`, the [DiT token sequence length](#grounding-in-an-output-spec),
is fixed) and a **core budget** `N` — the NeuronCores available for the DiT stage
(`N=64` when `LNC=2`, `N=128` when `LNC=1` on one `trn2.48xlarge` / `trn3`). Choose the three degrees to minimize
the per-step wall-clock time `t_step`:

```text
minimize    t_step(TP, CP, CFG)
subject to  TP · CP · CFG = N              # spend the whole budget
            TP  ∈ divisors of 40           # head count (see TP below)
            S mod CP = 0                    # sequence divisibility (see CP below)
            CFG ∈ {1, 2}                    # only two guidance passes exist
            peak per-core HBM ≤ capacity    # the clip must fit
```

The two knobs on that problem are **speed** (minimize `t_step`) and **feasibility**
(stay under HBM). Feasibility comes first: a fast configuration that does not fit is not
a configuration. Per-core HBM holds four things, and each parallelism axis acts on them
differently:

```text
peak_per_core ≈ W/TP           # weights — TP shards; CP and CFG replicate
              + A_shard/(TP·CP) # activations that shard: Q, K/V, FFN, hidden — TP and CP shard
              + workspace       # kernel scratch, compiler-managed
```

- **Weights `W`.** At the default `boundary_ratio=0.875` **both experts are resident**
  (~27 B params ≈ 54 GB in BF16; only one is *active* per step, but both sit in HBM).
  Only **TP** shards weights — CP and CFG each hold a full copy per replica. This is the
  largest fixed term and the reason TP cannot go too low on a memory-tight spec.
- **Shardable activations `A_shard`.** Q, K/V, FFN intermediates, and hidden states shard
  on **both TP and CP** (`1/(TP·CP)` per core). K/V shard because CP self-attention uses
  {ref}`ring attention <context-parallelism>`, which keeps
  each core's K/V local instead of gathering the full sequence — so `A_shard` is rarely the
  binding term once CP ≥ 4. (The all-gather fallback, used only where the ring kernel cannot
  run, would instead pin full-sequence K/V on every core.)
- **Workspace.** Compiler-managed kernel scratch; not a lever you tune directly.

CFG is absent from the levers above on purpose: a CFG replica is a full copy of the model
and its activations, so `cfg_parallel_size` adds cores without lowering `peak_per_core` at
all. The consequence: **CP is the only axis that reduces the activation footprint**, so
raising CP is the primary way to make a long or high-resolution clip feasible; TP reduces
the weight footprint; CFG buys speed, never headroom. This is why the memory constraint —
not raw throughput — is the first thing to check for 720P and long sequences, and it drives
the deviation rules in [Selecting the degrees](#selecting-the-degrees).

With feasibility framed, the per-axis analysis below covers how each degree trades compute
for cores.

### Tensor parallelism (TP) — shard the model

TP shards attention heads and MLP dimensions, so each rank does `1/TP` of the matmul
and holds `1/TP` of the weights. It speeds up the projection and FFN matmuls — the bulk
of the per-step work — directly. Its efficiency is sub-linear, though: every attention
and FFN block ends in an **all-reduce over the TP group** whose volume (`S · hidden` per
layer) is *independent of TP*, so the fixed communication cost is amortized over shrinking
compute — the larger the TP degree, the worse the collective overheads.

**TP is capped by the head count.** Each rank takes `num_heads // TP` heads, and no head
is ever split or replicated across ranks. Wan2.2's DiT has **40 heads of 128 dims**
(`hidden = 40 × 128 = 5120`), so TP must divide 40 evenly: TP=4 gives 10 heads/rank and
TP=8 gives 5, but TP=16 would need 2.5 heads per rank and is not expressible without
duplicating heads — which would mean redundant compute.

Because efficiency erodes as TP grows, the useful range is small: **TP=4 is
recommended** on Trn2/Trn3 (10 heads/rank), with TP=8 (5 heads/rank) the alternative
when you want to spend more cores on the model and fewer on the sequence — the TRN3
layout. Scale beyond TP with **CP** and **CFG parallelism**, which are not bounded by
the head count.

(context-parallelism)=
### Context parallelism (CP / `ring_degree`) — shard the sequence

CP shards the token sequence `S` across ranks. It is the only axis that reduces **both**
per-rank compute *and* the activation-memory footprint, which is why it is the lever for
long or high-resolution clips. Wan2.2 uses **ring-attention CP**: each rank keeps its local
`S/CP` shard of Q *and* K/V, and the ring-attention kernel streams K/V chunks around the CP
group (via `collective_permute`) so local Q still attends the full sequence — without any
rank materializing full K/V. (Where the ring kernel cannot run — CPU mode, fake-tensor
tracing, NKI disabled — it falls back to all-gathering full K/V + flash.)

- **Compute** per rank scales as `~1/CP`: each rank owns only `S/CP` query tokens, so both
  the projection/FFN matmuls and the attention itself shrink with CP. The attention saving
  matters most for long or high-resolution clips, where all-pairs attention over the
  sequence is the largest cost.
- **Memory** shrinks by `1/CP` for **all** the shardable activations — Q, FFN, *and* K/V —
  because ring attention never gathers the full K/V. This is what makes CP the axis that
  buys HBM headroom for long or high-resolution clips.
- **Communication** is the counter-pressure. Ring attention exchanges a `S/CP` K/V chunk per
  round over `CP` rounds per layer; the rounds are overlapped with attention compute but
  their count grows with CP, so past a point the ring latency dominates the savings. This is
  the diminishing-returns knee — the reason to stop at **CP=8**, not push to 16 or 32.

The current constraint is that `S` must be divisible by `ring_degree`:

> `Sequence length {S} is not divisible by cp_size {N}. Choose a resolution/frame
> count that yields a divisible patch sequence length.`

The default spec's `S = 32,760` is divisible by 8. If a new spec hits this error,
adjust frames/resolution until `S` divides evenly, or lower `ring_degree`. See the
[CP design doc](../design/context_parallelism.md) for the mechanics.

Note vLLM Omni enforces `sequence_parallel_size = ulysses_degree * ring_degree`, so the
stage configs set `ring_degree` and leave `ulysses_degree` (default 1) and
`sequence_parallel_size` unset — vLLM Omni then derives
`sequence_parallel_size = 1 × ring_degree`. Despite the name, `ring_degree` only sizes
the sequence-parallel group; it does not necessarily select a ring-attention algorithm.

(cfg-parallelism)=
### CFG parallelism (`cfg_parallel_size`) — shard the two guidance passes

CFG parallelism is the most efficient of the three axes because it exploits an
**independence** the other two cannot: classifier-free guidance evaluates the same DiT
twice per step — a conditioned pass and an unconditioned pass — and nothing in one pass
feeds the other. The only coupling is the final combine `v = v_uncond + scale·(v_cond −
v_uncond)`. So the two passes can run on separate replicas concurrently and be fused by
a single collective.

**Why it is nearly ideal, and how to estimate the gain.** Sequential CFG costs two DiT
passes per step; CFG-parallel costs one pass plus one gather-combine:

```text
t_seq   = 2 · t_dit
t_cfgp  = t_dit + t_gather
speedup = t_seq / t_cfgp = 2 / (1 + ε),   where ε = t_gather / t_dit
```

The overhead `ε` is small by construction. The gather-combine
([`cfg_parallel.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/vllm_omni_neuron/diffusion/distributed/cfg_parallel.py))
all-gathers the predicted-noise **latent** — shape `[B, C, T_lat, H_lat, W_lat]`, ≈ 4.2 MB
at 480P in BF16 — **once per step**, in its own fullgraph NEFF. Compare that to one DiT
pass: 40 layers, each moving `S · hidden` activations and running the per-layer ring K/V
exchange. The gather is orders of magnitude cheaper and happens once per step, not once per layer, so
`ε ≪ 0.1` and the expected DiT speedup is **≈ 1.8–2.0×**. To size `ε` for a new spec,
divide the latent bytes by the per-step DiT traffic — both scale with the spec, but the
ratio stays tiny.

CFG parallelism is capped at **CFG=2**: there are exactly two passes, so `CFG > 2` has
nothing to parallelize. It also produces a **bit-identical** result to sequential CFG
(same combine, different placement) — unlike `--no-cfg`, which *skips* the unconditioned
pass for a genuine ~2× but weakens prompt adherence
({ref}`denoising loop <guidance-scale>`). Use CFG=2 for a final
render; use `--no-cfg` only for fast previews.

### Selecting the degrees

The three axes have a clear efficiency ranking, which gives a greedy allocation rule:
**spend cores on the highest-efficiency axis first, up to its cap, then move down.**

| Axis | Efficiency | Cap | Also reduces memory? |
| --- | --- | --- | --- |
| **CFG** | Highest — one cheap gather/step, independent passes | 2 (hard) | No (replicates) |
| **TP** | High at small degree; eroded by the per-layer all-reduce | 8 (head count) | Weights + some activations |
| **CP** | Good; comm-bound at high degree | `S`-divisibility | Yes — the memory lever |

Apply the rule to a 64-core budget:

1. **CFG = 2.** Nearest-to-ideal 2× for two cores, bit-identical output. Take it for any
   final render. (Budget left: 32 cores.)
2. **TP = 4.** The largest head-dividing degree that keeps per-rank matmuls large; TP=8
   would expose more of the fixed per-layer all-reduce. (Budget left: 8.)
3. **CP = 8.** Absorb the remainder. CP also does double duty as the memory lever, and
   `S = 32,760` is divisible by 8. ✅

That derivation lands on `TP=4 × CP=8 × CFG=2 = 64` — the base
[`wan22_stage.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_stage.yaml):

```yaml
parallel_config:
  tensor_parallel_size: 4
  ring_degree: 8          # CP degree — sets sequence_parallel_size = 1 * 8 = 8
  cfg_parallel_size: 2    # world = 4 * 8 * 2 = 64 cores
```

## NKI kernels

Wan2.2 uses both kernels vendored under `vllm_omni_neuron/kernels/nkilib/`
and kernels imported from the installed [NKI Library](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/index.html).
The vendored kernels are adapted for the DiT; the installed library supplies
additional attention, output-projection, and MLP paths:

| NKI kernel | Source | Where it runs | Role |
| --- | --- | --- | --- |
| [`ring_attention_const_max_fwd`](kernels/ring-attention-const-max.md) | Vendored | DiT self-attention with context parallelism | Rotates sharded K/V and overlaps the exchange with const-max attention. |
| [`adaln_quant_kernel`](kernels/adaln-quant.md) | Vendored | DiT normalization | Fuses residual addition, modulation, and optional row FP8 quantization. |
| [`qkv_cte`](kernels/qkv-cte.md) | Vendored | DiT FP8 self- and cross-attention projections | Fuses QKV projection, distributed QK RMSNorm, and RoPE. |
| `attention_cte` | Installed `nkilib.core.attention` | DiT local attention and VAE decoder attention | Flash attention when the ring path does not apply. |
| `output_projection_cte` | Installed `nkilib.core.output_projection` | DiT BF16 and FP8 attention output | Output projection. |
| `mlp` | Installed `nkilib.core.mlp` | DiT BF16 feed-forward | Gate-less GELU up/down projections. |
| `mlp` | Vendored `vllm_omni_neuron.kernels.nkilib.core.mlp` | DiT FP8 feed-forward | ROW_MX up/down projections. |

See the [vendored kernel reference](kernels/index.md) for implementation details
and adaptation guidance. Rows marked "Installed" refer to imports from the
installed NKI Library, not the copies under `vllm_omni_neuron/kernels/nkilib/`.

## Deployment

These commands use the DLC container from the [setup guide](../getting-started/setup-guide.md).
For manual installation, omit `sudo docker exec vllm-omni-neuron` and replace
`/workspace` with `$VLLM_OMNI_HOME`.

### 480P video generation (recommended default)

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run.py \
  --num-frames 81 --height 480 --width 832 --guidance-scale 5.0 \
  --prompt "A fluffy orange cat walking gracefully across a sunny garden path, high quality, detailed" \
  --output 480P.mp4
```

### 720P video generation

A single-shot 1280×720 VAE decode exceeds per-device HBM, so 720P needs **VAE spatial
tiling**: the decoder splits each frame into overlapping spatial tiles, decodes them
across ranks, and stitches the result — capping the HBM peak at the cost of a fresh
compile at this shape. Enable it in the stage config by setting `vae_use_tiling: true`,
and set `vae_patch_parallel_size` > 1 to distribute the tile decode across ranks.

```yaml
# engine_args in the stage config
vae_use_tiling: true   # decode in overlapping spatial tiles to cap the HBM peak
parallel_config:
  vae_patch_parallel_size: 32   # ranks the VAE tiles are sharded across
```

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/wan22/run.py \
  --stage-config /workspace/plugin/examples/wan22/wan22_stage_tp4cp8cfg2_720p.yaml \
  --num-frames 81 --height 720 --width 1280 --guidance-scale 5.0 \
  --prompt "A fluffy orange cat walking gracefully across a sunny garden path, high quality, detailed" \
  --output 720P.mp4
```

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| OOM during VAE decode at 720P | Single-shot decode exceeds HBM | Set `vae_use_tiling: true` in the stage config. |
| Very long first-run latency | Compiler generating NEFFs on first inference | Wait for compilation to complete. |
| `Sequence length S not divisible by cp_size N` | `S` not divisible by `ring_degree` | Adjust `num_frames`/resolution to a divisible shape, or lower `ring_degree`. |
| Silent crash / `SIGBUS` on output | `/dev/shm` too small for the returned video | Start the container with `--shm-size=2g`. |

## Related information

- [Wan2.2-T2V-A14B model card](../models/wan22-t2v-14b.md) — architecture, feature
  matrix, [VBench](https://arxiv.org/abs/2311.17982) accuracy baseline.
- [Deploy Wan2.2-A14B with vLLM Omni Neuron](../tutorials/tutorial-wan22-14b.md) — end-to-end setup,
  model download, stage configuration, and offline T2V/I2V generation.
