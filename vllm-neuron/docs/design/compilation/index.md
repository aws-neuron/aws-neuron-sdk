# Compilation

vLLM Neuron compiles models using `torch.compile` with the `vllm_neuron`
backend — the same `torch.compile` API used in vLLM upstream. This
backend uses XLA to lower FX graphs into HLO representations, then
invokes `neuronx-cc` to produce hardware-optimized NEFF binaries.

## How compilation works

When vLLM Neuron starts, every model goes through this pipeline before serving requests:

``` text
Model code
    │
    ▼
torch.compile(backend=envs.get_compile_backend_name())
    │
    ▼
FX Tracing (captures Python model → FX graph)
    │
    ▼
FX Passes (graph rewrites: aliasing, device, inplace→outofplace)
    │
    ▼
HLO Lowering (FX graph → XLA HLO)
    │
    ▼
neuronx-cc (HLO → NEFF binary, one per bucket)
    │
    ▼
NEFF loaded to device → ready to serve
```

Each bucket (sequence length × batch size combination) produces a separate NEFF. More buckets = longer startup, less padding waste at runtime. The [compilation cache](compilation_cache.md) eliminates redundant compilations across restarts and nodes.

## Topics

| Topic | Description |
| --- | --- |
| [How vLLM is integrated with TorchNeuron](how_vllm_is_integrated_with_torchneuron.md) | How vLLM Neuron uses the TorchNeuron `torch.compile` lowering |
| [Compilation cache](compilation_cache.md) | Compilation cache (hit/miss, remote store) |
| [CPU compilation](cpu_compilation.md) | Ahead-of-time CPU compilation (NEFF extraction) |
| [FX passes architecture](fx_passes_design.md) | FX passes architecture |
| [Aliasing output rewrite pass](aliasing_output_rewrite_pass.md) | Aliasing output rewrite pass |
| [Device rewriting FX pass](device_rewriting_fx_pass.md) | Device rewriting FX pass |
| [Inplace to out-of-place pass](inplace_to_outofplace_pass.md) | Inplace to out-of-place rewrite |
| [SPMD compilation](spmd_compilation.md) | Collapse per-rank compiles into a single shared NEFF |
| [Compile-artifact manifest](compile_artifact_manifest.md) | The per-engine `manifest.json` — compiled graphs, the config that produced them, and how it's laid out on disk |

:::{toctree}
:maxdepth: 1
:hidden:

how_vllm_is_integrated_with_torchneuron
compilation_cache
cpu_compilation
fx_passes_design
aliasing_output_rewrite_pass
device_rewriting_fx_pass
inplace_to_outofplace_pass
spmd_compilation
compile_artifact_manifest
:::
