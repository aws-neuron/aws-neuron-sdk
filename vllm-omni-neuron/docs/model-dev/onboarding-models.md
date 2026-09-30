# How to onboard a model to vLLM-Omni Neuron

<!-- meta: description: Onboard a new diffusion / omni model to the vLLM-Omni Neuron
plugin so it runs on AWS Trainium. Implement the Neuron pipeline and model
components, let the registry auto-discover them, compile, validate accuracy, and
benchmark — using Wan2.2-T2V-A14B as the worked reference blueprint. -->
<!-- meta: keywords: vLLM-Omni, Neuron, Trainium, model onboarding, diffusion,
video generation, DiT, VAE, text encoder, pipeline registry, tensor parallelism,
context parallelism, CFG parallelism, torch.compile, NEFF -->
<!-- meta: content_type: procedural-how-to -->
<!-- meta: date_updated: 2026-09-11 -->

## Introduction

This guide explains how to onboard a new model to the vLLM-Omni Neuron plugin so it can be
served on AWS Trainium: implement the Neuron model components, register the pipeline, compile
and smoke-test on device, validate accuracy, and benchmark.

Everything below `Pipeline.forward()` is modeling code you implement for Neuron; everything
above it — input processing, scheduling, and output processing — stays with vLLM-Omni. Read
the [design overview](../design/vllm_omni_neuron_overview.md) first for how the
plugin plugs into the engine → pipeline execution path.

**Wan2.2 is the reference blueprint.** Two Wan2.2 pipelines are registered today —
text-to-video and image-to-video — and between them they exercise the full surface: a
two-expert MoE DiT, a 3D causal-conv VAE (decode, plus encode for image conditioning), a UMT5
text encoder, TP + CP + CFG parallelism, and a scheduler patch. Each step links the file to
copy from; read those files rather than a prose description of them.

## Prerequisites

- A Trainium instance with the plugin installed editable (`pip install -e .`) — see
  the [README](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/README.md) for setup and usage. `Dockerfile.plugin` is the
  source of truth for the runtime image.
- Ideally, an upstream `vllm_omni.diffusion` pipeline for your model to subclass; note the
  architecture key it registers under. If there is none, contribute one upstream or carry
  standalone modeling code in the plugin (Step 1).
- The runnable reference: [`examples/wan22/`](https://github.com/aws-neuron/vllm-omni-neuron/tree/release-0.24.0.0.1.0/examples/wan22) and
  `vllm_omni_neuron/diffusion/models/wan2_2/`.

## Onboarding flow

Onboarding is linear: **1.** implement the components → **2.** register the pipeline
(zero-config) → **3.** add the stage config + runner and smoke-test → **4.** validate
accuracy → **5.** benchmark & tune → **6.** wire up CI and docs. Each step is a section
below.

### Step 1 — Implement the model components

A generation runs in three phases: the **text encoder** runs once over the prompt, the **DiT**
denoises a fixed-size latent over N scheduler steps, and the **VAE decoder** turns the final
latent into pixels. Every denoising step has the same tensor shapes and carries no state
forward, so one compiled graph is replayed N times. An image-conditioned model adds a **VAE
encode** of the conditioning image before the loop. Your components map onto those phases, and
the **pipeline** class is the one the engine resolves and calls.

**Start from upstream.** Subclass the upstream vLLM-Omni pipeline and reuse its pure-math
modeling code (RoPE, embeddings, block structure, scheduler), overriding only the
Neuron-specific parts — device handling, TP/CP weight loading, compilation. This keeps you
aligned with upstream fixes. If a model genuinely cannot subclass the upstream contract, you
can instead carry its standalone modeling code inside the plugin behind a thin delegation
pipeline: keep it in its own subpackage, record the upstream revision and any local patches
alongside it, and add an entry to [`NOTICE`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/NOTICE). You then own its
maintenance, so prefer the subclass. Steps 2–6 apply to either path.

Model implementations follow the upstream package layout, with one subpackage per model and
shared components in the corresponding directories under `diffusion/`:

```text
vllm_omni_neuron/diffusion/
├── distributed/autoencoders/
│   └── autoencoder_kl_<model>.py    # distributed VAE                  (1c)
├── models/
│   ├── <model>/
│   │   ├── __init__.py              # package-level PIPELINE_REGISTRY  (1a)
│   │   ├── pipeline_<model>.py       # pipeline class                   (1a)
│   │   └── <model>_transformer.py    # transformer / DiT                (1b)
│   └── <encoder>/
│       └── <encoder>.py              # text encoder                     (1d)
└── quantization/                     # shared diffusion quantization
```

Parallelism and performance features already exist in the plugin — you get them by mixing in a
class or setting a config field, not by writing them. Scope your work to your model's math and
reach for these rather than reimplementing:

| Feature | How you get it | Symbol / field |
|---|---|---|
| Tensor parallelism | `parallel_config` | `tensor_parallel_size` (1b) |
| CFG parallelism | Mix in, ahead of the upstream pipeline | `NeuronCFGParallelMixin` (1a) |
| Context / sequence parallelism | `parallel_config` + `model_config` | `ring_degree`, `tp_sequence_parallel` (1b) |
| Ring attention for CP self-attention | `model_config`, **on by default** | `enable_ring_attention` (Step 5) |
| Row-MX FP8 quantization | `model_config` | `quantization: fp8_row_mx`, `modules_to_not_convert` |
| VAE spatial tiling | `engine_args` | `vae_use_tiling: true` |
| VAE patch parallelism | Mix in + `parallel_config` | `DistributedVaeMixin`, `vae_patch_parallel_size` (1c) |
| Attention backend | Platform default, no code | `NeuronSDPABackend` (1e) |

If your model needs a custom NKI kernel, wire it in behind the same shape as ring attention: a
`model_config` field that selects the kernel and falls back to a torch path when the kernel
cannot run, so CPU mode and fake-tensor tracing keep working.

Subsections 1a–1f walk the Wan2.2 text-to-video reference component by component. Attention
(1e) and the scheduler (1f) are usually *reused*, not written.

#### 1a. Pipeline class

Reference:
[`pipeline_wan2_2.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/vllm_omni_neuron/diffusion/models/wan2_2/pipeline_wan2_2.py).
Subclass the upstream pipeline and add one override at a time. Start by inheriting everything:

```python
class Neuron<Model>Pipeline(<Upstream>Pipeline):
    """Inherits forward(), encode_prompt(), prepare_latents(), check_inputs()."""
```

**`__init__`** — the parent's init is GPU-centric, so call `nn.Module.__init__` directly and
build the components yourself from `od_config`:

```python
    def __init__(self, *, od_config, prefix: str = ""):
        nn.Module.__init__(self)                 # skip the parent's GPU init
        self.od_config = od_config
        self.device = get_local_device()         # this rank's Neuron core

        tf_config = load_transformer_config(model, "transformer", local_files_only)
        tf_config.update(dict(od_config.model_config or {}))   # stage-YAML model_config
        self.transformer = _create_transformer_from_config(tf_config)
        self.text_encoder = Neuron<Enc>Wrapper(model_path=model, dtype=dtype, tp_group=tp_group)
        if self.is_vae_rank:                     # the VAE is not TP-sharded (1c)
            self.vae = Neuron<Model>VAE.from_pretrained(model, subfolder="vae", ...)
```

Owning construction has two consequences: `to(...)` must move each component explicitly, and
the merge above is how the stage YAML's `model_config` reaches your DiT.

**`load_weights`** — route each component to its own TP-sharded loader; the pipeline itself
loads no tensors:

```python
    def load_weights(self, weights=None):
        self.text_encoder.load_weights(os.path.join(model_path, "text_encoder"))
        for name, t in (("transformer", self.transformer),
                        ("transformer_2", self.transformer_2)):
            if t is not None:
                t.load_weights(os.path.join(model_path, name))
```

**`compile`** — one graph per component, each with its own `model_name` (the compile cache key)
and `neuronx-cc` arguments:

```python
    def compile_transformer(self, t, *args, **kwargs):
        options = {**kwargs.pop("options", {}), "model_name": "<model>_transformer"}
        options["compiler_args"] = [
            "--model-type=transformer", "--auto-cast=none", "-O1",
            "--hbm-scratchpad-page-size=2048",
        ]
        kwargs.setdefault("fullgraph", True)     # a graph break becomes a hard error
        t.compile(*args, options=options, **kwargs)
```

The VAE differs: `--model-type=unet-inference`, a raised `--internal-max-instruction-limit`,
and no `fullgraph` when spatial tiling is on.

**`prepare_latents`** — draw the initial noise in float32 to match the diffusers/GPU RNG
trajectory for a seed, then cast. Inherit everything else.

**Classifier-free guidance.** For CFG-parallel execution, mix in `NeuronCFGParallelMixin`
*ahead of* the upstream pipeline in the MRO —
`class Neuron<Model>Pipeline(NeuronCFGParallelMixin, <Upstream>Pipeline)`. With
`cfg_parallel_size > 1` each replica runs one CFG branch and a fullgraph-compiled `all_gather`
+ combine fuses them; at `== 1` the override defers to the base sequential CFG. Reference:
`vllm_omni_neuron/diffusion/distributed/cfg_parallel.py`.

For the same overrides at minimum size, read
[`pipeline_wan2_2_i2v.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/vllm_omni_neuron/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py):
it adds image-to-video by subclassing a *different* upstream pipeline while reusing the same
DiT, VAE, and text encoder, and reuses the T2V pipeline's Neuron work by assignment rather than
copying it:

```python
    encode_prompt = NeuronWanPipeline.encode_prompt
    compile = NeuronWanPipeline.compile
    load_weights = NeuronWanPipeline.load_weights
```

#### 1b. Transformer / DiT

Reference:
[`wan2_2_transformer.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/vllm_omni_neuron/diffusion/models/wan2_2/wan2_2_transformer.py).
Implement the transformer as raw `nn.Parameter` tensors with the weight loaders in
`vllm_neuron.utils.weight_loader`, following the vLLM Neuron LLaMA3 pattern (no
`ColumnParallelLinear` / `RowParallelLinear` module wrappers in the DiT itself), and import
the pure-math helpers (rotary/time/text embeddings) from the upstream
`vllm_omni.diffusion.models.<model>` module rather than reimplementing them.

Two things that are easy to get wrong:

- **`load_weights()`** needs an explicit `{model_param_name → checkpoint_key(s)}` mapping,
  passed to `SafetensorsCheckpoint(...).load_sharded_pipelined(...)`. A fused QKV maps to a
  *list* of keys:

  ```python
          mappings[f"blocks.{i}.attn1.qkv_proj_weight"] = [   # fused -> a LIST
              f"blocks.{i}.attn1.to_q.weight",
              f"blocks.{i}.attn1.to_k.weight",
              f"blocks.{i}.attn1.to_v.weight",
          ]
          mappings[f"blocks.{i}.ffn.up_proj_weight"] = f"blocks.{i}.ffn.net.0.proj.weight"
  ```
- **Register the TP/CP process groups** in the model's `__init__` with
  `register_replica_groups(tp_size=..., cp_size=...)` from
  `vllm_omni_neuron.diffusion.distributed.parallel_state`, so the collectives in `forward()` can be
  legalized at compile time (see [Common issues](#common-issues) for the failure it causes).

For context parallelism, split the token sequence across the SP ranks before the blocks and
`all_gather` it back after, guarded on `cp_size > 1`. The algorithm and its configuration
are in [Context parallelism design](../design/context_parallelism.md).

#### 1c. VAE

Reference:
`vllm_omni_neuron/diffusion/distributed/autoencoders/autoencoder_kl_wan.py`.
Subclass the diffusers VAE, rewrite the ops that do not lower or that force a CPU↔device
round-trip, and expose your own `compile()` / `decode()`. The VAE is not TP-sharded — only
rank 0 instantiates and compiles it, and other ranks skip it entirely, which is what avoids
the redundant NEFF compilation. For resolutions that exceed per-core HBM, add a patch-parallel
subclass mixing in `DistributedVaeMixin`; the pipeline selects it when
`vae_patch_parallel_size > 1`.

If your model conditions on an image, the **encoder** needs a compiled graph of its own and
`encode()` alongside `decode()`. Wan2.2 gates it on `vae.compile(..., compile_encoder=True)`,
which is **off by default** so the text-to-video path pays no encoder compilation cost — the
image-to-video pipeline is what turns it on. Leave it off in an image-conditioned pipeline and
the encode still works but silently falls back to an uncompiled CPU path.

#### 1d. Text encoder

Reference:
`vllm_omni_neuron/diffusion/models/umt5_encoder/umt5_encoder.py`.
Add a TP-sharded encoder behind a thin wrapper that mirrors `NeuronTextEncoderWrapper`'s
contract — `load_weights(model_path)`, `compile(...)`, and
`forward(input_ids, attention_mask)` returning an object with `.last_hidden_state`.

#### 1e. Attention backend

Usually no new code — the platform already returns `NeuronSDPABackend` from
`NeuronOmniPlatform.get_diffusion_attn_backend_cls()`. Only if your model needs a different
attention path, subclass `SDPABackend` / `SDPAImpl`
([`sdpa.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/vllm_omni_neuron/diffusion/attention/backends/sdpa.py), ~30 lines) and
wire it through that method.

#### 1f. Scheduler patch

Only if the reference scheduler has a Neuron-specific dtype or tracing bug — Wan2.2
subclasses `FlowUniPCMultistepScheduler` solely to fix a float32/float64 mismatch in
`torch.linalg.solve` at `solver_order=2`. If the upstream scheduler traces cleanly, use it
directly.

### Step 2 — Register the model (zero-config)

The plugin **auto-discovers** your pipeline; there is **no `pyproject.toml` edit**. The
`vllm_omni.general_plugins` entry point imports each immediate child package under
`vllm_omni_neuron/diffusion/models/`, reads its package-level `PIPELINE_REGISTRY`, and registers
each entry with the package name as `module_name`. The package must therefore re-export every
class and named processing function referenced by its registry. For one pipeline, expose this
from your model package's `__init__.py`:

```python
from .pipeline_<model> import (
    PIPELINE_REGISTRY,
    Neuron<Model>Pipeline,
    get_<model>_post_process_func,
    get_<model>_pre_process_func,
)

__all__ = [
    "PIPELINE_REGISTRY",
    "Neuron<Model>Pipeline",
    "get_<model>_post_process_func",
    "get_<model>_pre_process_func",
]
```

If the package contains multiple pipelines, import each registry under a private name and expose
one combined list. See
[`vllm_omni_neuron/diffusion/models/wan2_2/__init__.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/vllm_omni_neuron/diffusion/models/wan2_2/__init__.py),
which combines the T2V and I2V registries and re-exports both pipeline classes and their processing
functions.

`model_arch` must match the architecture key that the stage YAML's `model_class_name`
resolves to upstream. Wan2.2 sets `model_class_name: Wan22Pipeline` and registers
`model_arch: "Wan22Pipeline"`, so the engine resolves it to `NeuronWanPipeline`.

> **Note:** Discovery is one level deep. Each model subpackage must expose its combined
> `PIPELINE_REGISTRY` from `__init__.py`; nested implementation modules are not scanned directly.

### Step 3 — Add the stage config and runner, then smoke-test

Copy [`examples/wan22/`](https://github.com/aws-neuron/vllm-omni-neuron/tree/release-0.24.0.0.1.0/examples/wan22) to `examples/<model>/` and adapt:

- **`<model>_stage.yaml`** — the stage, runtime, and engine args, including
  `model_class_name`, `dtype`, any model-specific fields, and the `parallel_config` mesh.
- **`run.py`** — must `import vllm_omni_neuron.bootstrap` **first** (before any vllm
  import), set the Neuron env vars, construct `Omni(model=..., stage_configs_path=...)`,
  call `omni.generate(...)`, and export the output.

> **Note:** Context parallelism is sized with **`ring_degree`, not
> `sequence_parallel_size`**. vLLM-Omni enforces
> `sequence_parallel_size = ulysses_degree * ring_degree`, so set `ring_degree` and leave
> `ulysses_degree` (default 1) and `sequence_parallel_size` unset. See
> [Context parallelism configuration](../design/context_parallelism.md#configuration).

Then smoke-test on device. The first run compiles NEFFs, which is slow; later runs
cache-hit.

```bash
python examples/<model>/run.py --dev
```

### Step 4 — Validate accuracy

Add hardware tests under `test/neuron/` (auto-skipped off-device by the
`require_neuron_device` fixture in `test/neuron/conftest.py`), in three tiers, cheapest
first. Reference the Wan2.2 suite: `test/neuron/test_wan22_*.py` and
`test/neuron/test_wanpipeline_accuracy.py`, plus the per-config YAMLs in
`test/neuron/configs/`.

1. **Component tests** (`test_wan22_*_modules.py` and `test_wan22_*_accuracy.py`) — each
   module and component against its diffusers CPU counterpart via `assert_close_three_way`
   from `vllm_neuron.accuracy.testing`: FP32 CPU baseline, BF16 CPU expected (isolates dtype
   quantization error), BF16 Neuron actual (isolates Neuron-specific error).
2. **Single-step tests** (`test_wanpipeline_accuracy.py`,
   `test_wan_i2v_pipeline_accuracy.py`) — the whole pipeline for one denoising step
   (`num_steps = 1`) against the diffusers CPU denoised latent, excluding only VAE decode.
   These use the same three-way comparison as tier 1; what separates the tiers is scope, not
   method.
3. **End-to-end output** (`test_wan22_e2e_accuracy.py`) — a full generation scored against a
   cached golden video with `compute_ssim_per_frame` from `test/utils/accuracy`. This is a
   regression check against a previous Neuron output, not a comparison with the reference
   implementation.

Tier 3 is not optional — a per-step error too small for tier 2 to catch can still compound into
a collapsed video (see [Common issues](#common-issues)). When a tier fails, the methodology for
isolating *where* the drift is introduced — validation levels, the three-way comparison, and the
stage/module/step bisection — is in
[Evaluating and debugging model accuracy](./accuracy-evaluation-debugging.md).

`VLLM_NEURON_CPU_MODE=1` runs the init/model path on CPU without a device — useful for
tracing checks and for the CPU-only unit tests under `test/unit/`.

### Step 5 — Benchmark and tune performance

- **Profile & benchmark** — profile the compiled graphs on device to find hot spots, and
  compare latency/throughput against a comparable GPU baseline to validate your targets.
- **Parallelism** — size the TP × CP × CFG mesh to your instance (Wan2.2's release config is
  TP=4 × CP=8 × CFG=2 = 64 cores). CFG parallelism halves per-step wall clock; CP lets longer
  sequences fit within per-core HBM. Spending a fixed core budget across the three, and the
  per-core HBM budget that decides feasibility, are worked through in
  [Optimizing high-quality offline video generation](./optimizing-offline-video-generation.md).
- **VAE tiling** — set `vae_use_tiling: true` (or VAE patch parallelism) for large
  resolutions that would otherwise OOM at decode.
- **Ring attention** — the CP self-attention path is selected by the stage config's
  `model_config.enable_ring_attention`, not an environment variable. It is a tri-state and
  **ring is the default**: omit it to run the ring NKI kernel, falling back to all-gather K/V
  + flash only where the kernel cannot run (CPU mode, fake-tensor tracing); set `false` to
  always take all-gather + flash; set `true` to require ring and raise instead of falling
  back. See the commented block in
  [`wan22_stage.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_stage.yaml).

### Step 6 — Wire up CI and docs

- **CI** — add correctness and latency coverage to the development repository's
  test manifest; the release checkout does not include CI infrastructure.
- **Model card** — add `docs/models/<model>.md`; use the
  [Wan2.2-T2V-A14B model card](../models/wan22-t2v-14b.md) as an example. A
  [model tutorial](../tutorials/tutorial-wan22-14b.md) is optional.
- **README** — add the model to the Supported Models section of the
  [README](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/README.md).

The model is onboarded when it compiles and cache-hits on a second run, all three accuracy
tiers are within tolerance, its benchmark score and per-video latency match the targets
recorded in its model card, and the development CI suite is green.

## Common issues

### Graph break / non-traceable op in `forward`

- **Possible solution**: `torch.compile` traces `forward()` into a static graph for the Neuron
  compiler, so anything that materializes tensor data to Python — `.item()`, `.tolist()`, a
  data-dependent `if tensor.sum() > 0`, or logging that references tensor values — breaks the
  graph. The `torch._dynamo` error points at the offending line: restructure to pure tensor
  ops and move logging out of the traced region.

### `KeyError` / missing key during weight loading

- **Possible solution**: The `mappings` dict in `load_weights` is incomplete, or a fused
  weight (separate Q/K/V → fused QKV) maps to a single checkpoint key where it needs a
  **list** of keys.

### `replica id #N not seen in replica groups`

- **Possible solution**: The process groups' full partitions were not registered before the
  first collective. Call `register_replica_groups(tp_size=..., cp_size=...)` in the model's
  `__init__` (Step 1b).

### Compile fails blaming a custom-call, but the kernel is fine

- **Possible solution**: A stale NKI compile cache can replay a path from a previous session and
  surface as what looks like a kernel bug. Its location is set by `NKI_COMPILE_CACHE_URL` —
  clear that directory and retry. `test/neuron/conftest.py` treats this as a real corruption
  hazard, giving each pytest-xdist worker its own cache under `/tmp/vllm_wan_<worker>/nki`. Do
  not confuse it with the NEFF cache under `/root/.cache/vllm/<hash>/`, or with the `neuronx-cc`
  scratch directories written into the working directory — both are described in the
  [README](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/README.md). `VLLM_CACHE_ROOT` is a third thing again: in this plugin
  it only supplies the compiler's `compiler_workdir`.

### Cold-compile timeout

- **Possible solution**: First-run NEFF compilation exceeds vLLM-Omni's default handshake and
  init timeouts. `examples/wan22/run.py` already raises `stage_init_timeout` / `init_timeout`
  on the `Omni(...)` call and sets `VLLM_NEURON_COMPILATION_TIMEOUT`; carry those settings into
  your runner. Persisting compiled graphs across runs is a separate concern that `run.py` does
  **not** handle: either mount the cache directory into the container, as the
  [README](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/README.md) setup does, or point `TORCH_NEURONX_NEFF_CACHE_DIR` at a
  directory that outlives the container, as the
  [Wan2.2-T2V-A14B model card](../models/wan22-t2v-14b.md) recommends.

### Single-step accuracy passes but end-to-end output is washed out / collapsed

- **Possible solution**: A small per-step approximation (a loose attention-kernel bound, a
  dtype mismatch) compounds across denoise steps. Tighten the offending kernel or dtype and
  re-check end-to-end output accuracy; use
  [Evaluating and debugging model accuracy](./accuracy-evaluation-debugging.md) to find
  which stage introduces the drift.

## Related information

- [Design: Engine, Worker, and Model Integration](../design/vllm_omni_neuron_overview.md)
  — the architecture behind every step in this guide.
- [Context parallelism design](../design/context_parallelism.md) — CP algorithm and
  configuration.
- [Kernel implementations](./kernels/index.md) — per-kernel design references, including
  the const-max ring attention kernel and how to adapt its pattern to another model.
- [Evaluating and debugging model accuracy](./accuracy-evaluation-debugging.md) —
  validation levels, the three-way comparison, and how to isolate a failing accuracy tier.
- [Optimizing high-quality offline video generation](./optimizing-offline-video-generation.md)
  — allocating a fixed core budget across CFG/TP/CP, and the per-core HBM budget.
- [Wan2.2-T2V-A14B model card](../models/wan22-t2v-14b.md) — a filled model card and
  the reference blueprint's feature/accuracy status.
- [`examples/wan22/`](https://github.com/aws-neuron/vllm-omni-neuron/tree/release-0.24.0.0.1.0/examples/wan22) — the runnable reference: `run.py` and
  `wan22_stage.yaml`.
- [README](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/README.md) — setup, usage, and supported models.
- [vLLM-Omni architecture](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/architecture_overview/)
