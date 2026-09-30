# Architecture: Engine, Worker, and Model Integration in vLLM-Omni Neuron

## Overview

`vllm-omni-neuron` is the AWS Neuron backend plugin for running
[vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) diffusion models on AWS
Trainium. It registers through vLLM-Omni's
[plugin entry points](https://github.com/vllm-project/vllm-omni/blob/v0.24.0/vllm_omni/plugins/__init__.py).
vLLM-Omni keeps
ownership of input processing, scheduling, and output processing. The plugin owns only the
Neuron-specific parts: device and distributed setup, model loading, compilation, and
execution.

For deployment and application configuration, see the
[setup guide](../getting-started/setup-guide.md),
[feature and configuration guide](../guides/features-guide.md), and
[offline quickstart](../getting-started/quickstart-offline-serving-wan22.md).

This document describes how the plugin plugs into vLLM-Omni's diffusion engine and how a
concrete diffusion model (Wan2.2-T2V-A14B) is integrated. It focuses on three things:

1. The **engine → executor → worker → model runner → pipeline** execution path and where the
   plugin substitutes its own classes.
2. How a **pipeline** and its **model components** (transformer/DiT, VAE, text encoder) are
   registered and loaded.
3. The **call stacks** for the two flows that matter: startup/model-load and per-request
   execution.

The runtime dependency relationship is:

```text
vllm_omni_neuron
├── vllm_omni ─────┐
├── vllm_neuron ───┼── vllm
└── libtorch-neuronx-lite
```

vLLM-Omni supplies the engine, executor, scheduler, and abstract worker and model-runner base
classes. vLLM Neuron supplies the Neuron runtime, device, distributed, and worker helpers;
vLLM supplies the shared configuration and execution primitives below both projects. This
plugin connects those layers with Neuron subclasses of the worker and model runner,
`NeuronOmniPlatform` for Trainium-specific device and runtime behavior, a Neuron attention
backend, and the Wan2.2 modeling code.

## How the plugin attaches to vLLM-Omni

The plugin declares two entry points in `pyproject.toml`:

```toml
[project.entry-points."vllm_omni.platform_plugins"]
neuron = "vllm_omni_neuron:neuron_omni_platform_plugin"

[project.entry-points."vllm_omni.general_plugins"]
neuron = "vllm_omni_neuron:register_neuron_pipelines"
```

**Platform plugin — `neuron_omni_platform_plugin()`.** vLLM-Omni calls this during platform
resolution. It returns the class path `vllm_omni_neuron.platform.NeuronOmniPlatform` when the
host has a Neuron device or CPU mode is active (`_is_neuron_dev()` / `_is_cpu_mode()` from
`vllm_neuron`); otherwise it returns `None` and the plugin stays inactive. The platform is the
single object the diffusion engine asks for hardware-specific decisions.

**General plugin — `register_neuron_pipelines()`.** It scans every module under
`vllm_omni_neuron/diffusion/models/` for a module-level `PIPELINE_REGISTRY` list and calls
`vllm_omni.diffusion.registry.register_diffusion_model(...)` for each entry. This is how a
Neuron pipeline becomes discoverable by architecture name (for example, `Wan22Pipeline` →
`NeuronWanPipeline`).

### NeuronOmniPlatform

`NeuronOmniPlatform` multiply-inherits vLLM-Omni's `OmniPlatform` and vLLM-Neuron's
`NeuronPlatform`, so it is both an Omni platform (the engine talks to it) and a Neuron platform
(it inherits Neuron device/runtime helpers):

```python
class NeuronOmniPlatform(OmniPlatform, NeuronPlatform):
    _omni_enum = OmniPlatformEnum.OOT  # out-of-tree platform
    device_type = "neuron"
    device_control_env_var = "NEURON_VISIBLE_DEVICES"
```

The diffusion-engine extension points it supplies:

| Method | Returns | Purpose |
|---|---|---|
| `get_diffusion_worker_cls()` | `NeuronDiffusionWorker` | Worker the executor spawns per core |
| `get_diffusion_model_runner_cls()` | `NeuronDiffusionModelRunner` | Runner that loads/compiles/runs the model |
| `get_compile_backend()` | Neuron backend name (`None` in CPU mode) | Backend passed to `torch.compile` |
| `supports_torch_inductor()` | `False` | The plugin uses the Neuron backend, not inductor |

## Execution architecture

The diffusion engine keeps vLLM-Omni's layering:
`Engine → Executor → Scheduler → Worker → ModelRunner → Model (Pipeline)`. The plugin
substitutes only the worker, the model runner, and the model; everything above the worker is
reused unchanged.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Omni.generate(prompt, sampling_params)              vLLM-Omni entrypoint │
│  └─ OmniDiffusion / DiffusionEngine                 vLLM-Omni  (reused)  │
│      └─ Multiproc executor + Scheduler              vLLM-Omni  (reused)  │
│          │  shared-memory broadcast to every core; rank 0 returns result │
│          ▼                                                               │
│      ┌───────────────────────────────────────────────────────────────┐   │
│      │ NeuronDiffusionWorker           (one process per Neuron core) │◄──┼── plugin
│      │  init_device(): runtime + distributed + mesh group setup      │   │
│      │   └─ NeuronDiffusionModelRunner                               │◄──┼── plugin
│      │       load_model(): load → capture hooks → compile            │   │
│      │       execute_model(req):                                     │   │
│      │        └─ NeuronWanPipeline.forward(req)                      │◄──┼── plugin
│      │            ├─ text encoder  (UMT5)                            │   │
│      │            ├─ denoising loop (DiT WanTransformer3DModel)      │   │
│      │            │    └─ CFG-parallel predict_noise (compiled)      │   │
│      │            └─ VAE decode    (rank 0 only)                     │   │
│      └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

The executor and scheduler are vLLM-Omni's single-node, shared-memory multiproc
implementation: the scheduler broadcasts a request to every core, all cores run the same
forward (they participate in the same TP/CP/CFG collectives), and rank 0 returns the result.

### NeuronDiffusionWorker

`NeuronDiffusionWorker(DiffusionWorker, NeuronWorker)` reuses the Neuron runtime bootstrap from
`NeuronWorker` (EFA affinity, CPU affinity, visible-core mapping) and overrides `init_device()`.
`init_device()` runs once per core at startup and performs the following steps in order:

1. Build a `VllmConfig` carrying the parallel sizes from `od_config`.
2. Initialize the distributed environment and model-parallel groups using **vLLM-Omni's**
   diffusion `parallel_state` (not vLLM's) — see
   [Collectives under compilation](#collectives-under-compilation).
3. Restore `torch.nn.functional.gelu` to the aten op so Dynamo can trace it in fullgraph mode.

`init_lora_manager()`, `sleep()`, and `wake_up()` are no-ops or raise `NotImplementedError`
(these are memory/adapter features the diffusion path does not use on Neuron).

### NeuronDiffusionModelRunner

`NeuronDiffusionModelRunner(DiffusionModelRunner)` differs from the base CUDA runner in three
ways:

- **CPU-first load.** `load_model()` loads weights to CPU via `DiffusersPipelineLoader`, then
  moves the pipeline to the Neuron device.
- **Tensor capture.** It registers capture hooks on the transformer(s) *before* compilation so
  they are traced into the compiled graph, then patches `predict_noise` *after* compilation.
  This supports step-level accuracy debugging.
- **Neuron compile.** It calls `torch.compile(..., backend=<neuron>)` when compilation is enabled.
  The actual NEFF compilation happens lazily on the first forward pass.

The runner defaults seed-created generators to CPU because
`torch.Generator(device="neuron")` is not available in this SDK version. An
explicit caller-supplied generator or `generator_device` remains authoritative;
the pipelines consume the resolved generator without rebuilding it.

### Attention backend

`NeuronSDPABackend` / `NeuronSDPAImpl` (`diffusion/attention/sdpa.py`) is a thin subclass of
vLLM-Omni's SDPA backend running with `mask_mode="broadcast_k"`.

## Integrating a pipeline and a diffusion model

vLLM-Omni treats each diffusion model as a `Pipeline` class whose `forward()` encapsulates the
whole inference (text encode → latent prep → denoising loop → VAE decode). Everything above
`Pipeline.forward()` is engine orchestration; everything below it is modeling code owned by the
plugin. Integrating a model therefore means: (a) register a pipeline class, and (b) implement
its model components for Neuron.

### Registration

A model module exposes a `PIPELINE_REGISTRY` list. For Wan2.2 (`neuron_wan_pipeline.py`):

```python
PIPELINE_REGISTRY = [
    {
        "model_arch": "Wan22Pipeline",  # HF/diffusers architecture name
        "class_name": "NeuronWanPipeline",  # plugin pipeline class
        "pre_process_func_name": "get_wan22_pre_process_func",
        "post_process_func_name": "get_wan22_post_process_func",
    },
]
```

`register_neuron_pipelines()` reads this and calls `register_diffusion_model(...)`, which binds
the architecture name, the pipeline class, and the pre/post-process functions together. The
engine later resolves `model_class_name: Wan22Pipeline` (from the stage YAML) to
`NeuronWanPipeline`.

### The pipeline — `NeuronWanPipeline`

`NeuronWanPipeline(NeuronCFGParallelMixin, Wan22Pipeline)` subclasses the upstream Wan 2.2
pipeline and inherits `forward()`, `encode_prompt()`, `prepare_latents()`, and `predict_noise()`
where the base behavior is already correct. The registered pipeline class is independent of
the request entry point: direct `Omni.generate` calls and serving integrations built on the same
vLLM-Omni diffusion engine resolve the same `NeuronWanPipeline`. The current Wan2.2 user guide
exposes offline generation. The class overrides:

- `__init__()` — skips the GPU-centric parent init; sets up the tokenizer, the Neuron text
  encoder, the VAE (rank 0 only), one or two DiT transformers, and the scheduler. It reads
  `boundary_ratio` to decide which transformers to load (Wan 2.2 uses two experts: one for
  the high-noise and one for the low-noise timestep range).
- `load_weights()` — routes each component to its TP-sharded loader.
- `prepare_latents()` — draws the initial noise in float32 to match the diffusers/GPU RNG
  trajectory for a given seed, then casts to bfloat16 for the compiled transformer.
- `forward()` — wraps the parent forward with per-stage timing and, on Trn2, offloads the DiT
  weights to CPU on the VAE rank during VAE decode to free HBM, then restores them.
- `compile_*()` — sets per-component `neuronx-cc` compiler arguments (`--model-type`,
  `--auto-cast=none`, `-O1`, `--hbm-scratchpad-page-size=2048`; the VAE adds
  `--internal-max-instruction-limit`).

It also defines `NeuronFlowUniPCMultistepScheduler`, which fixes a float32/float64 dtype
mismatch in `torch.linalg.solve` at `solver_order=2`.

### The model components

The Wan2.2 pipeline has three components, each re-implemented for Neuron.

**DiT — `WanTransformer3DModel`** (`neuron_wan_model.py`). A custom tensor-parallel transformer
built from raw `nn.Parameter` tensors with weight loaders (the vLLM-Neuron LLaMA3 pattern),
importing pure-math helpers (rotary/time/text/image embeddings) from vLLM-Omni. Key pieces:
`DistributedRMSNorm` (global RMS across the TP group via all-reduce), `WanFeedForward` (NKI MLP
kernel + all-reduce), `WanSelfAttention` (fused QKV, then ring or all-gather attention), and
`WanCrossAttention` (text + optional image). `load_weights()` uses
`SafetensorsCheckpoint.load_sharded_pipelined` with an explicit param → checkpoint-key mapping
and a three-stage load pipeline (page-cache prefetch, rank-shard read, host-to-device transfer).

**VAE — `NeuronAutoencoderKLWan`** (`autoencoders/`). A Neuron-tuned re-implementation of the
diffusers Wan 3D causal-conv VAE. The changes exist to make the decoder compile well and run
without CPU↔device transfers: a manual RMS norm (`x * rsqrt(mean(x²))`) to drop the NaN guard,
`repeat_interleave` instead of `nn.Upsample`, keeping the feature cache on-device, and padding
the first-frame output so the first-chunk and rest-of-frames paths share one compiled graph.
The VAE is not TP-sharded; only rank 0 instantiates and compiles it, and other ranks read from
the compile cache.

**Text encoder — `NeuronTextEncoderWrapper`** (`text_encoders/`). A standalone TP-sharded UMT5
encoder (T5 family, encoder-only) built from config: vocab-sharded embedding, head-sharded
relative position bias, TP-sharded attention and gated FFN. The compiled graph zeroes padded
positions in-graph, so there is no CPU round trip. Sequence length is 512, so the encoder does
not use sequence parallelism.

## Call stacks

### Startup / model load (once per core)

```
Omni(model=..., stage_configs_path=...)
 └─ DiffusionEngine
     └─ MultiprocExecutor  (spawns one worker process per core)
         └─ NeuronDiffusionWorker.init_device()
             ├─ set NEURON_RT_* / distributed env vars
             ├─ init_distributed_environment()          # vLLM-Omni diffusion parallel_state
             ├─ initialize_model_parallel(tp, sp, cfg, …)
             └─ init_workspace_manager(device)
         └─ NeuronDiffusionModelRunner.load_model()
             ├─ DiffusersPipelineLoader.load_model()      # → NeuronWanPipeline (weights on CPU)
             │   └─ NeuronWanPipeline.__init__()           # tokenizer, text encoder, VAE(rank0), DiT×{1,2}
             ├─ pipeline.to(neuron)                        # move weights to device
             ├─ NeuronWanPipeline.load_weights()           # TP-sharded per component
             │   └─ WanTransformer3DModel.__init__() → register_replica_groups(tp, cp)
             ├─ _setup_tensor_capture()                    # hooks before compile
             └─ pipeline.compile(backend=<neuron>)         # text encoder, VAE, DiT(s)
```

### Per-request execution

```
omni.generate({"prompt": ...}, OmniDiffusionSamplingParams(...))
 └─ DiffusionEngine → Scheduler.add_req()                  # broadcast to all cores
     └─ NeuronDiffusionWorker (each core) → generate()
         └─ NeuronDiffusionModelRunner.execute_model(req)
             ├─ default seed-created RNG generators to CPU
             └─ NeuronWanPipeline.forward(req)
                 ├─ encode_prompt()                        # UMT5 text encoder (compiled)
                 ├─ prepare_latents()                      # float32 noise → bf16
                 ├─ for t in timesteps:                    # denoising loop
                 │   └─ predict_noise_maybe_with_cfg()     # NeuronCFGParallelMixin
                 │       ├─ predict_noise(branch)          # DiT NEFF (compiled)
                 │       │   └─ WanTransformer3DModel.forward()
                 │       │       ├─ CP: slice sequence per rank
                 │       │       ├─ blocks: self-attn (ring|all-gather) + cross-attn + FFN
                 │       │       └─ CP: all-gather full sequence
                 │       └─ _cfg_gather_combine()          # compiled all-gather + combine
                 ├─ (Trn2) offload DiT → CPU on VAE rank
                 └─ VAE decode (rank 0)                     # → DiffusionOutput
     ◄─ rank 0 returns result to the engine → post_process_func
```

## Parallelism

Rank layout follows vLLM-Omni's `RankGenerator` order `tp-sp-pp-cfg-dp`. Common
configurations: TP=4 × CP=8 = 32 cores (Trn2), TP=8 × CP=4 = 32 cores (Trn3), and
TP=4 × CP=8 × CFG=2 = 64 cores (Trn2 release).

**Tensor parallelism (TP).** The DiT and text encoder shard attention heads and FFN dimensions
across ranks, with an all-reduce after the output/down projections. Weight loaders implement
column-parallel, row-parallel, fused-QKV, and a scaled-bias pattern (each rank adds
`bias / tp_size` before the sum-reduce so the net bias is exact).

**Context parallelism (CP).** CP uses vLLM-Omni's sequence-parallel group.
`WanTransformer3DModel.forward` splits the patch sequence across CP ranks before the blocks and
all-gathers the full sequence after. Self-attention has two paths:

- *Default — ring attention:* K/V stay local and the NKI `ring_attention_const_max_fwd` kernel
  (vendored in `vllm_omni_neuron/kernels/nkilib/`) drives its own `collective_permute` ring
  across the CP group, merging partial results with a per-query-row constant-max softmax bound.
- *Capability fallback — all-gather:* each rank all-gathers K/V across the CP group to the full
  sequence, then runs local flash attention (the "Naive AllGather CP" in
  [Context Parallelism in vLLM Omni Neuron](context_parallelism.md)). Taken automatically, and only, where the
  ring kernel cannot run (CPU mode, fake-tensor tracing, NKI disabled) — a capability check, not
  a setting.

**CFG parallelism.** `NeuronCFGParallelMixin` overrides `predict_noise_maybe_with_cfg`.
Classifier-free guidance normally runs the model twice per step (a conditional "positive" and an
unconditional "negative" pass) and combines them as `neg + scale * (pos − neg)`. Under
CFG-parallel, each replica runs one branch (rank 0 → positive, rank 1 → negative) reusing the
compiled transformer NEFF, then a fullgraph `torch.compile` region all-gathers the two branch
outputs across the CFG group and applies the combine, so the collective stays on the Neuron
collective-communication fabric. The combine is deterministic, so all ranks obtain the same
result and the output matches sequential CFG. The override defers to the base implementation for
the no-CFG, `cfg_size == 1`, `cfg_normalize`, and `output_slice` cases.

### Collectives under compilation

The `neuron_native` backend from `libtorch-neuronx-lite` lowers c10d collectives to StableHLO
and must derive HLO `replica_groups` that cover every rank an op participates in.
`register_replica_groups(tp_size, cp_size)` in `distributed/parallel_state.py` makes this work
through the Lite mesh registry, keyed by each c10d process group's `group_name`.
`WanTransformer3DModel.__init__` calls it.

## Configuration

- **Runtime** is driven by `vllm_neuron.envs`:

  | Env var | Default | Purpose |
  |---|---|---|
  | `VLLM_NEURON_BACKEND` | *(unset)* | Selects the execution backend; `neuron_native` uses the native compile backend supplied by `libtorch-neuronx-lite`. |
  | `VLLM_NEURON_CPU_MODE` | `0` | Runs the init/model path on CPU without a Neuron device (limited functionality; used by unit tests). |
  | `NEURON_PLATFORM_TARGET_OVERRIDE` | *(auto-detect)* | Overrides hardware detection, e.g. `trn2` or `trn3`, so compilation targets the intended instance. |
  | `VLLM_NEURON_DEBUG_MODE` | `0` | Enables verbose logging and additional diagnostics. |
  | `VLLM_NEURON_SWITCH_CC` | `0` | Uses contiguous collective-communication groups for the 8×8 topology instead of the custom Trn2 mesh mapping. |

- **Model** configuration is the `WanConfig` dataclass plus vLLM-Omni's `OmniDiffusionConfig`
  (`od_config`): `model`, `dtype`, `boundary_ratio`, `flow_shift`, the `parallel_config`
  (tensor/data/cfg/sequence-parallel sizes and ulysses/ring degrees), whether to compile,
  and a `tensor_capture` block.
- **Stage config (YAML)** declares the pipeline. `examples/wan22/wan22_stage.yaml` defines one
  `diffusion` stage with `model_class_name: Wan22Pipeline`, `dtype`, `boundary_ratio`,
  `flow_shift`, and a `parallel_config`.

## References

- [Context Parallelism in vLLM Omni Neuron](context_parallelism.md) — context-parallelism design.
- [Kernel implementations](../model-dev/kernels/index.md) — per-kernel design references.
- [Setup guide](../getting-started/setup-guide.md) — manual and DLC installation on Trainium.
- [Feature and configuration guide](../guides/features-guide.md) — generation, compilation, and parallelism controls.
- [Offline quickstart](../getting-started/quickstart-offline-serving-wan22.md) — runnable Wan2.2 example.
- [`examples/wan22/`](https://github.com/aws-neuron/vllm-omni-neuron/tree/release-0.24.0.0.1.0/examples/wan22) — entrypoints and stage configs.
- vLLM-Omni architecture: <https://docs.vllm.ai/projects/vllm-omni/en/latest/design/architecture_overview/>
