# CPU Compilation

<!-- meta: description: Ahead-of-time CPU compilation and NEFF extraction -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-22 -->

CPU Compilation (`VLLM_NEURON_CPU_COMPILE=1`) compiles model graphs on a CPU
instance using the `neuron_libtorch_graph_capture` backend but does **not** execute
them. This eliminates the need to use Neuron instances for compilation and
enables scaling compilation across many parallel CPU instances.

Beyond validating that the compilation pipeline succeeds, CPU Compilation can
be used to extract compiled graph artifacts (NEFFs), save them to a remote
cache, and reuse those artifacts for later execution on Neuron devices. This
decouples compilation cost from execution cost and enables workflows where
compilation happens once on CPU infrastructure and the resulting NEFFs
are consumed by multiple Neuron instances without recompilation.

## How It Works

CPU Compile mode exercises the full `torch.compile` path (tracing, FX passes,
graph lowering) without needing a Neuron device. It catches issues in graph
capture, shape specialization, and backend passes that would otherwise only
surface during on-device compilation.

You **cannot** run inference or validate numerical outputs in this mode, it
only confirms that the compilation pipeline completes without error.

## Requirements

- `NEURON_PLATFORM_TARGET_OVERRIDE` is required to specify the target platform
  for compilation, since no hardware is present to auto-detect it. Valid list
  of target values can be found as part of the `--target` flag in the output of `neuronx-cc compile --help` command for the installed version of neuron compiler.
- `VLLM_NEURON_CPU_COMPILE` and `VLLM_NEURON_CPU_MODE` cannot be enabled
  together, they are orthogonal modes. CPU Mode runs inference on CPU; CPU
  Compilation only compiles graphs.

## When to Use

| Use case | Example |
| ---------- | --------- |
| Validating graph capture succeeds for a model | New model bringup on a CPU dev instance |
| Testing FX passes and backend transformations | Verifying custom passes don't break compilation |
| Pre-compiling NEFFs on CPU for later device execution | Compile on CPU instance, save to remote cache, execute on Neuron with cache hits |

## Parallel Compilation Workers

`NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS` controls the number of parallel
compilation workers. It defaults to 8. On instances with more CPU cores and
higher memory, this value can be increased to reduce total compilation time.
Speedups are only realized when the number of graphs to compile exceeds the
worker count — if there are fewer graphs than workers, additional workers
provide no benefit. Even with many graphs, scaling is bounded by available CPU
and memory resources.

## Using Remote Cache for Production Deployments

CPU Compilation integrates with the two-tier compilation cache to enable a
compile-once, deploy-everywhere workflow. Compiled NEFF artifacts are saved
to a remote cache (NFS or FSx mount) during the CPU compile phase and later
consumed by production Neuron instances without any recompilation.

### Workflow

1. **Compile on CPU instance** — Run with `VLLM_NEURON_CPU_COMPILE=1`,
   `NEURON_PLATFORM_TARGET_OVERRIDE` set as the target neuron instance family and
   `NEURON_LIBTORCH_REMOTE_CACHE` pointing to a shared filesystem (NFS/FSx).
   Compiled artifacts are saved to this remote location.

2. **Sync to production** — On production Neuron instances, ensure the remote
   cache artifacts are available on the local filesystem (either via the same
   NFS/FSx mount or by syncing to the local cache directory).

3. **Run on production with no compilation** — The `vllm_neuron` backend
   should not recompile these graphs on production instances. Once the remote
   artifacts are synced with the local cache, all graphs should be cache hits.

### Preventing Compilation on Production Instances

To guarantee that no compilation occurs on production machines, set:

```bash
NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND=1
VLLM_NEURON_DISABLE_WARMUP_COMPILE=1
```

- `NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND=1` disables the graph capture
  backend entirely.
- `VLLM_NEURON_DISABLE_WARMUP_COMPILE=1` treats any cache miss by
  vllm_neuron backend as a fatal error, raising `RuntimeError`
  instead of silently compiling. This ensures production instances never
  spend time on unexpected compilation.

Together these flags enforce that production deployments are strictly
cache-hit-only, catching any mismatch between the CPU compile phase and
production execution immediately at startup rather than degrading latency.

## Limitations

- Speculative decoding is not supported with `VLLM_NEURON_CPU_COMPILE` mode.
