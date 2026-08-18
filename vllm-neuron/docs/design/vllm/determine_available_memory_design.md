# determine_available_memory design

<!-- meta: description: Available memory determination design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## 1) How GMU works

`gpu_memory_utilization` (GMU) is a vLLM configuration parameter that defines
the fraction of total device memory to budget for model execution.

```text
total_budget = total_hbm * gmu
kv_cache_budget = total_budget - bytes_used
```

Interpretation:

- `total_budget` is the memory budget requested by GMU.
- `kv_cache_budget` is what remains for KV after resident memory (`bytes_used`)
  is already on device.

GMU semantics are platform-agnostic in vLLM: apply GMU to total device memory
first, then subtract non-KV resident memory.

## 2) Why we needed the heuristic

### The problem

On Neuron, KV cache size is baked into the compiled program (NEFF) shape. During
warmup, each NEFF is compiled with KV tensors sized by the allocated token
count. The Neuron compiler validates that all tensors referenced by a NEFF fit
within the per-HBM-bank limit. If the KV allocation is too large, the total
tensor footprint exceeds this limit and the compiler rejects the program
(`NCC_EVRF009`).

This creates a constraint that doesn't exist on platforms where KV capacity is
only a post-compile allocator/runtime decision: the KV sizing decision directly
affects whether the model can compile at all.

Terminology used in this doc:

- compile-time fit: whether NCC can materialize a valid NEFF within HBM limits
- post-compile runtime/allocator fit: whether already-compiled KV blocks fit in
  the runtime memory allocator during execution

### The chicken-and-egg constraint

This is a compile-time chicken-and-egg problem:

- We need a KV budget to decide tensor shapes for compilation.
- But whether that KV budget is safe is only known after compilation checks the
  full tensor footprint.

In short: KV budget determines compile footprint, and compile footprint
determines safe KV budget.

Current implementation note:

- The current cap-based control safely increases KV while reducing compile-fit
  failures.

### Why GMU alone is not enough

Applying `gpu_memory_utilization` to compute a KV budget works when the only
constraint is post-compile allocator/runtime fit. On Neuron, the compile-time
tensor footprint includes model weights, KV tensors, intermediate activations,
and outputs — all of which must fit simultaneously. The non-KV portion of this
footprint varies by model architecture and cannot be precisely known before
compilation. This means the GMU-derived budget can exceed the compile-safe
limit for models with large non-KV tensor overhead.

A fixed GiB reserve also doesn't work because the compile overhead scales with
the model — it's not a stable constant across architectures.

### How we selected the current cap policy

We ran `gpu_memory_utilization` sweeps across multiple models and found there is
no single global compile-safe formula that generalizes cleanly across model
families and hardware generations. As an interim safety policy, we cap KV to a
fraction of GMU total budget (`0.30` by default), using GPT-OSS worst-case
behavior as the primary guardrail.

This cap is intentionally conservative. It safely allows more KV than legacy
clamp behavior in many cases, while reducing NCC compile-fit failures.

## 3) What the heuristic is and how we derived it

### Heuristic formula

Constants (defaults):

```python
VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION = 0.30
```

Computation:

`bytes_used` and `bytes_free` are obtained from Neuron runtime memory stats
(`Runtime().get_vnc_memory_stats()`).

```python
total_budget = int((bytes_used + bytes_free) * gpu_memory_utilization)
kv_cache_budget = max(total_budget - bytes_used, 0)

heuristic_cap = int(total_budget * VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION)

available = max(min(kv_cache_budget, heuristic_cap), 0)
```

### Why `0.30` by default

The default is a safety-first value derived from worst-case compile behavior
observed on GPT-OSS family runs. It can be overridden with
`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` when more aggressive KV is desired.

> **Limitation:** The global `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` is an interim
> policy for safe KV enablement. The `0.30` cap is conservative and not optimal
> for all architectures, TP sizes, and hardware generations.

### Minimum KV guard

Env knob:

- `VLLM_NEURON_MIN_KV_BUDGET_GIB` (default `1.0`)

If `available` is below this threshold, worker raises:

- `RuntimeError: Computed KV cache budget is below minimum threshold...`

This fails fast on configurations that are too constrained to serve reliably.

## 4) How this is implemented (flow)

Code path:

- `vllm_neuron/vllm/worker/neuron_worker.py`

Runtime flow:

```text
determine_available_memory()
  |
  +-- VLLM_NEURON_CPU_COMPILE == 1 ?
  |      |
  |      +-- yes: _estimate_available_memory_neuron()
  |           total_hbm_bytes = get_total_available_memory()  (static HBM size from platform)
  |           bytes_used = _get_byte_used_from_model()
  |                        (sum of model parameter + buffer sizes)
  |           return _compute_kv_budget(total_hbm_bytes, bytes_used, gmu)
  |
  +-- VLLM_NEURON_CPU_MODE == 1 ?
  |      |
  |      +-- yes: _determine_available_memory_cpu()
  |           bytes_free = _query_host_runtime_memory()  (fair-share host memory)
  |           total_budget = int(bytes_free * gmu)
  |           cap_fraction = _get_kv_cap_fraction()
  |           available = int(total_budget * cap_fraction)
  |           return available
  |
  +-- no (Neuron mode): _determine_available_memory_neuron()
         bytes_used, bytes_free = _query_runtime_memory_stats()
         total_hbm_bytes = bytes_used + bytes_free
         bytes_used = _get_byte_used_from_model()
                      (sum of model parameter + buffer sizes after loading)
         return _compute_kv_budget(total_hbm_bytes, bytes_used, gmu)
```

Shared KV budget logic (`_compute_kv_budget`):

```text
_compute_kv_budget(total_hbm_bytes, bytes_used, gmu):
    total_budget = int(total_hbm_bytes * gmu)
    user_budget = max(total_budget - bytes_used, 0)
    heuristic_cap = int(total_budget * VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION)
    available = max(min(user_budget, heuristic_cap), 0)
    if available < VLLM_NEURON_MIN_KV_BUDGET_GIB: raise RuntimeError
    return available
```

Key points:

- **CPU Compile only mode** uses the static HBM size for the target platform
  (looked up via `NEURON_PLATFORM_TARGET_OVERRIDE`) and computes `bytes_used`
  from model parameter and buffer sizes. No device is present — the goal is to
  produce the same KV block count that the execution phase will use, ensuring
  compilation cache hits.
- **Neuron mode (compile/execute)** also uses `_get_byte_used_from_model()` (model parameter +
  buffer sizes after model loading) rather than raw
  `runtime.get_vnc_memory_stats()` for `bytes_used`. This ensures the KV block
  count is identical between CPU compile and on-device execution. The runtime
  stats are only used to derive `total_hbm_bytes`.
- **CPU mode (no compilation)** is a separate path that uses host memory fair-share — it does not
  go through `_compute_kv_budget` and is unrelated to the compile/execute flow.
- GMU sets the total budget.
- Heuristic cap keeps budget inside compile-safe limits.
- Minimum-KV guard fails fast for unusable configs.
