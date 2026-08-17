# Compilation Cache

<!-- meta: description: Two-tier compilation cache for vLLM Neuron -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-22 -->

## Overview

vLLM Neuron provides a two-tier compilation cache that eliminates redundant `neuronx-cc` compilations:

1. **Local cache** — coordinates parallel processes on the same node so that each unique graph is compiled exactly once.
2. **Remote cache** (optional) — an NFS or FSx mount that shares compiled artifacts across nodes and survives restarts.

The cache is **enabled by default**. When a model is compiled with `torch.compile(model, backend="vllm_neuron")`, the resulting NEFF artifacts are stored locally and can optionally be promoted to a shared remote store for other nodes to consume.

### Why a Compilation Cache?

Neuron compilation (FX → HLO → NEFF via `neuronx-cc`) is expensive — sometimes exceeding several minutes for large graphs. Without a cache:

- **Multi-process waste**: In tensor-parallel inference, every rank compiles the same graph independently, creating a CPU bottleneck proportional to the number of ranks.
- **Multi-node waste**: Each node's local filesystem is isolated, so every node recompiles from scratch on cold start.
- **No persistence**: Restarting a server means recompiling everything, even when the model and dependency versions haven't changed.

The compilation cache solves all three problems.

## Architecture

### Cache Tiers

``` text
Local cache:   $VLLM_CACHE_ROOT/neuron/compile_cache/<hash>/
Remote cache:  $NEURON_LIBTORCH_REMOTE_CACHE/<hash>/          (optional)
```

**Local cache** (always active):

- Stores compiled artifacts on the local filesystem.
- Uses file locks to coordinate parallel processes on the same node — one process compiles while others wait and reuse the result.
- Default location: `~/.cache/vllm/neuron/compile_cache`. If the home directory is on NFS, automatically falls back to `/tmp/vllm_neuron_wdir_$USER/neuron/compile_cache`.

**Remote cache** (opt-in via `NEURON_LIBTORCH_REMOTE_CACHE`):

- Points to an NFS or FSx mount visible to all nodes.
- On a local cache miss, the system fetches from the remote store before falling back to compilation.
- Artifacts are promoted to the remote store explicitly via `save_cache()`.
- All compilation still happens locally — the remote store is a read/write-back cache layer, never a compilation target.

:::{note}
S3 is supported indirectly by mounting an FSx for Lustre volume backed by S3, which exposes it as a POSIX filesystem path.
:::

### Compilation Decision Flow

When `torch.compile` is called, each process follows this flow:

``` text
1. Generate cache key
   Hash(FX graph + input metadata + versions + compiler args + platform target)
   → 32-character key

2. Fast local check (no lock)
   Local cache hit? → Load artifacts and return

3. Acquire lock
   ├─ Won lock:
   │   ├─ Re-check local cache (another process may have finished)
   │   ├─ NEURON_LIBTORCH_REMOTE_CACHE set? → Fetch from remote
   │   └─ Still miss → Compile locally, store artifacts
   │
   └─ Lost lock:
       └─ Wait for lock holder to finish, then load shared artifacts
```

### Parallel Coordination

**SPMD** (same graph on all ranks):

``` text
Rank 0: [Acquire Lock] → [Compile] → [Signal Complete]
Rank 1: [Wait] ──────────────────────→ [Reuse NEFF]
Rank 2: [Wait] ──────────────────────→ [Reuse NEFF]
Rank 3: [Wait] ──────────────────────→ [Reuse NEFF]

Result: 1× compilation time instead of 4×
```

**MPMD** (different graphs, e.g. prefill/decode):

``` text
Rank 0: [GraphA Lock] → [Compile A] → [Complete A]
Rank 1: [GraphA Wait] ──────────────→ [Reuse A]
Rank 2: [GraphB Lock] → [Compile B] → [Complete B]
Rank 3: [GraphB Wait] ──────────────→ [Reuse B]

Result: Graphs A and B compile in parallel; each compiled once
```

Each cache key gets its own lock file, so different graphs never contend.

### Multi-Node Scenarios

**Cold start (no remote cache)**:

Each node compiles independently. Within each node, only one rank compiles per unique graph. After compilation, call `save_cache()` to promote artifacts to the remote store.

**Cold start (with remote cache, first time)**:

Same as above. After `save_cache()`, the remote store is populated for future use.

**Warm start (remote cache populated)**:

``` text
Node C, Rank 0: [Local miss] → [Acquire lock] → [Fetch from remote] → [Done]
Node C, Rank 1: [Local miss] → [Wait] ────────────────────────────→ [Reuse]
```

No compilation occurs. One rank per node fetches from the remote store; all other ranks wait on the local sentinel.

``` text
Timeline:
Node A  ├─ compile ──────┤ save_cache() ┤
Node B  ├─ compile ──────┤ save_cache() ┤  (EEXIST: A already promoted → skip)
                                         ↓
Node C                   ├─ fetch ───────┤  (no compile)
Node D                   ├─ fetch ───────┤  (no compile)
```

## Cache Key

The 32-character MD5 cache key is derived from:

| Component | Purpose |
| ---- | ---- |
| FX graph structure | Different model architectures produce different keys |
| Input tensor metadata (dtype, shape, stride) | Different input configurations produce different keys |
| `torch_neuronx` version | Prevents reuse across incompatible framework versions |
| `neuronxcc` version | Prevents reuse across incompatible compiler versions |
| NKI version | Prevents reuse across incompatible kernel library versions |
| Compiler args (canonicalized) | Different compiler flags produce different keys |
| Platform target (e.g. `trn2`) | Different hardware targets produce different keys |
| Collective replica groups | Different process group configurations produce different keys |

In multi-process setups, Neuron device indices are normalized (`neuron:X` → `neuron:0`) so that all ranks produce the same key for the same graph.

## Environment Variables

| Variable | Default | Description |
| ---- | ---- | ---- |
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | vLLM cache root. Neuron compile cache is stored at `$VLLM_CACHE_ROOT/neuron/compile_cache`. |
| `NEURON_LIBTORCH_REMOTE_CACHE` | (unset) | Path to NFS/FSx mount for shared persistent cache. When unset, only the local cache is used. |
| `NEURON_LIBTORCH_DISABLE_COMPILE_CACHE` | `0` | Set to `1` to disable the compilation cache entirely. Each process compiles independently. |
| `NEURON_LIBTORCH_COMPILATION_TIMEOUT` | `600` | Seconds to wait for another process to finish compiling before timing out. |
| `VLLM_NEURON_DISABLE_WARMUP_COMPILE` | `0` | Set to `1` to treat a local cache miss as a fatal error. Use when all graphs must be pre-compiled. Incompatible with `NEURON_LIBTORCH_DISABLE_COMPILE_CACHE`. |

## Usage Guide

### Basic Usage (Local Cache Only)

The local cache is active by default. No configuration needed:

``` bash
# Compiles locally, caches in ~/.cache/vllm/neuron/compile_cache/
# Persists across restarts
python your_inference_script.py
```

Override the cache location:

``` bash
export VLLM_CACHE_ROOT="/data/my_cache"
# Cache stored at /data/my_cache/neuron/compile_cache/
```

### Using a Remote Cache (NFS/FSx)

#### Step 1: Compile and promote

``` python
from vllm import LLM, SamplingParams
from libtorch_neuronx_lite.compile import save_cache
from vllm_neuron.envs import get_neuron_compile_cache_dir

# Compile the model (artifacts go to local cache)
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", ...)
outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=10))

# Promote to shared remote cache
local_cache_dir = get_neuron_compile_cache_dir()
save_cache(local_cache_dir, "/mnt/fsx/neuron-cache", hash_key)
```

#### Step 2: Consume on other nodes

``` bash
export NEURON_LIBTORCH_REMOTE_CACHE="/mnt/fsx/neuron-cache"
# torch.compile auto-fetches from remote on local miss — no compilation
python your_inference_script.py
```

### Build Team Workflow: Compile, Validate, Distribute

``` python
from vllm import LLM, SamplingParams
from libtorch_neuronx_lite.compile import save_cache
from vllm_neuron.envs import get_neuron_compile_cache_dir

llm = LLM(model="meta-llama/Llama-3.1-70B-Instruct", ...)
results = run_tests(llm)  # validate before distributing

if results.success:
    local_cache_dir = get_neuron_compile_cache_dir()
    save_cache(local_cache_dir, "/mnt/efs/neuron-cache", hash_key)
```

### Consumer Workflow: Pre-compiled Only

When artifacts are pre-compiled and distributed, consumers can enforce that no compilation occurs at runtime:

``` bash
export NEURON_LIBTORCH_REMOTE_CACHE="/mnt/efs/neuron-cache"
export VLLM_NEURON_DISABLE_WARMUP_COMPILE=1
# Raises RuntimeError on cache miss instead of compiling
python your_inference_script.py
```

### Multi-Node with FSx / EFS

``` bash
export NEURON_LIBTORCH_REMOTE_CACHE="/mnt/fsx/neuron-cache"
# First run on any node: compile locally, then call save_cache() once
# All subsequent nodes/restarts: local miss → remote hit → no compilation
```

## API Reference

### `save_cache(local_cache_dir, remote_cache_dir, hash_key)`

Promote a locally-compiled cache entry to a shared remote cache directory.

- Copies the local entry to a staging directory on the remote store, then atomically renames it to the final location.
- Safe to call concurrently from multiple nodes: if another node already promoted the same entry, this call is a no-op.
- Raises `RuntimeError` if the local entry is incomplete.

``` python
from libtorch_neuronx_lite.compile import save_cache
from vllm_neuron.envs import get_neuron_compile_cache_dir

save_cache(
    local_cache_dir=get_neuron_compile_cache_dir(),
    remote_cache_dir="/mnt/fsx/neuron-cache",
    hash_key="<cache_key>",
)
```

### `get_neuron_compile_cache_dir()`

Returns the local compile cache directory path. Respects `VLLM_CACHE_ROOT` if set; falls back to `/tmp` if the home directory is on NFS.

``` python
from vllm_neuron.envs import get_neuron_compile_cache_dir

cache_dir = get_neuron_compile_cache_dir()
# e.g. "/home/user/.cache/vllm/neuron/compile_cache"
```

## Disabling the Cache

``` bash
export NEURON_LIBTORCH_DISABLE_COMPILE_CACHE=1
```

When disabled, each process compiles independently with no coordination. This is useful for debugging compilation issues.
