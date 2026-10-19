# How to use the Precompile Cache

<!-- meta: description: Use the torch caching-precompile cache on the native
route. It is on when VLLM_NEURON_BACKEND is neuron_native or neuron. Confirm a
warm reuse, turn it off, and understand what invalidates it. -->
<!-- meta: date_updated: 2026-10-13 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: No Jira ticket found -->

## Task overview

This guide explains the Precompile Cache on the native route: how it works, how
to confirm that it works, how to turn it off, and what makes it start cold
again.

The native route traces the model to build the graph that the compiler needs.
A trace runs on the CPU and is slow. Without the cache, the route traces the
model again at every start. The Precompile Cache saves the trace and reuses it.
A warm start then loads the saved trace and skips the trace step, so start-up is
much faster. The result does not change, because the reused graph is the same
graph the first start compiled.

Lite (`libtorch-neuronx-lite`) turns the cache on when it imports, but only when
`VLLM_NEURON_BACKEND` is `neuron_native` or `neuron`. When the variable is not
set, the server uses the native route, but the cache is **off**. To turn the
cache off on the native route, set `TORCH_CACHING_PRECOMPILE=0`. The XLA route
does not use the cache.

## Prerequisites

- A vLLM Neuron server on the native route, with
  `VLLM_NEURON_BACKEND=neuron_native` set. The native route is the default when
  the variable is not set, but the cache is then off. The cache does not apply
  to the XLA route.
- A persistent `VLLM_CACHE_ROOT`. A warm start reads the cache from this path,
  so it must survive between starts.
- Device mode, not CPU mode. The cache does not apply in CPU mode.

## Instructions

### 1. Keep the cache on, or turn it off

To keep the cache on, set the route explicitly before you start the server:

```bash
export VLLM_NEURON_BACKEND=neuron_native
```

To turn the cache off, export this variable in the shell before you start the
server:

```bash
export TORCH_CACHING_PRECOMPILE=0
```

Torch reads this variable once, at module import. Set it in the environment
before Python starts. A value that you set inside Python, after import, has no
effect. This is the most common reason a change to the cache appears to do
nothing.

The second cache, the AOTAutograd cache, is on by default in Torch 2.13, so you
need no variable for it. To be explicit, set `TORCHINDUCTOR_AUTOGRAD_CACHE=1`,
also before Python starts.

### 2. Point the cache at a persistent path

```bash
export VLLM_CACHE_ROOT=/path/to/persistent/cache
```

The first start is a cold start. It traces, compiles, and saves the packages
under this path. The next start on the same path is a warm start. It reuses the
saved packages.

### 3. (Optional) Choose a guard filter

A guard decides whether a saved package is still valid for the current inputs.
Fewer guards make a warm start faster. Fewer guards are safe on Neuron, because
warmup runs every shape bucket.

Set `VLLM_NEURON_PRECOMPILE_GUARD_FILTER` to one of:

- `off` — keep every guard. This is the default.
- `portable` — keep the tensor-metadata, dynamic-shape, and global-context
  guards. Drop the rest.
- `skip_all` — drop every guard.

An unknown value keeps every guard and logs a warning.

### 4. Start the server

Start the server as usual. The first start compiles and saves. A later start on
the same `VLLM_CACHE_ROOT` reuses the saved packages.

## Confirm your work

The save hook prints the Dynamo and AOTAutograd counts on both a cold start and
a warm start: hit, miss, error, and save.

- On a **cold start**, expect misses and a save count greater than zero.
- On a **warm start**, expect Dynamo hits greater than zero and a save count of
  zero. A warm start that still saves did not reuse the cache.

Read `dynamo_cache_error` first. It separates a package that failed to load from
a genuine cold start. Without it, the two look the same, because both show a
miss. A nonzero `dynamo_cache_error` means the package did not load. The most
common cause is a changed fingerprint (see below).

The cache activates only when the backend is native, CPU mode is off, and
precompile is on. It is off in all other configurations.

## Cache layout and invalidation

The cache lives under:

```text
$VLLM_CACHE_ROOT/neuron/torch_compile/<fingerprint>/dp_<index>/rank_<rank>
```

Each parallel configuration, data-parallel replica, and tensor-parallel rank
gets its own directory. A different `<fingerprint>` is a different directory, so
it reads as a cold start with no error.

The fingerprint changes with:

- the vLLM configuration hash,
- the resolved bucket sets,
- the platform target,
- the effective compiler options,
- the Torch, Lite, compiler, and plugin versions, and
- the NEFF cache directory.

Come back to this section after an upgrade. A version change moves the
fingerprint, so the first start after an upgrade is a cold start. A warm miss
after an upgrade is expected. It is not a defect, and it does not mean the
feature is broken.

A saved package refers to its NEFFs by cache key, but it does not hold them. The
NEFFs stay in the NEFF cache directory, which `TORCH_NEURONX_NEFF_CACHE_DIR`,
`NEURON_LIBTORCH_CACHE_ROOT`, or `VLLM_CACHE_ROOT` sets. If you move the NEFF
cache, the fingerprint changes, and the next start is a cold start.

## Limitations

- **Enabled in Lite, only for an explicit route value.** Lite turns the cache on
  when it imports and `VLLM_NEURON_BACKEND` is `neuron_native` or `neuron`. When
  the variable is not set, the cache is off, although the server uses the native
  route. No image sets `TORCH_CACHING_PRECOMPILE`; Lite sets it. To opt out, set
  `TORCH_CACHING_PRECOMPILE=0`.
- **Multi-node is uncoordinated.** Each node resolves its own cache. Multi-node
  precompile does not coordinate the saved packages across nodes yet. This is
  the sharpest limit today. On a multi-node deployment, set
  `TORCH_CACHING_PRECOMPILE=0` until the coordination lands.
- **EPD wants the cache off.** On encoder-disaggregated inference (EPD), the
  time to first token is lower with the cache off. Set
  `TORCH_CACHING_PRECOMPILE=0` on each VE and PD engine. The reference launcher
  (`epd_common.py`) does this for you.

## Known issues

### `KeyError: __builtins_dict___N` after `torch._dynamo.reset()`

If your code calls `torch._dynamo.reset()` and then compiles again in the same
process, a warm start can fail with this error:

```text
torch._dynamo.exc.InternalTorchDynamoError: KeyError: __builtins_dict___N
```

`torch._dynamo.reset()` clears the Dynamo state in memory, but it does not
remove the saved packages on disk. A package that loads after the reset refers
to a global that the reset removed. Torch 2.13 does not include the upstream
fix.

If your code calls `torch._dynamo.reset()`, or if you see this error, turn off
the cache for that process before Python starts:

```bash
export TORCH_CACHING_PRECOMPILE=0
```

## Related information

- [Configuration reference](reference-configuration.md) for every environment
  variable and `additional_config` option.
- [Features guide](features-guide.md) for bucketing, quantization, and the other
  serving features.
