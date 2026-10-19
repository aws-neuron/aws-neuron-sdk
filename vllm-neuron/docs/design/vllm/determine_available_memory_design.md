# determine_available_memory design

<!-- meta: description: Available memory determination design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-10-07 -->

## 1) KV cache budget

By default, vLLM Neuron uses the memory-utilization and fractional-cap policy:

```text
logical_core_hbm = hbm_stack_bytes / logical_cores_per_hbm_stack
total_budget = int(logical_core_hbm * gpu_memory_utilization)
user_budget = max(total_budget - model_weight_bytes, 0)
heuristic_cap = int(total_budget * cap_fraction)
available = max(min(user_budget, heuristic_cap), 0)
```

The default `cap_fraction` is `0.30`. The worker respects
`gpu_memory_utilization` and `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION`.
This conservative policy limits the KV cache to a fraction of the memory-utilization budget.

An explicit headroom value enables the reserve policy:

```text
available = max(logical_core_hbm - model_weight_bytes - headroom_bytes, 0)
```

- `hbm_stack_bytes` is the HBM capacity reported for one physical stack.
- `logical_cores_per_hbm_stack` is 1 at LNC 2 and 2 at LNC 1.
- `model_weight_bytes` includes model parameters and buffers.
- `headroom_bytes` reserves HBM per logical core for all other allocations.

The headroom includes internal tensors, scratchpad, DMA rings, runtime data,
compiled code, and a safety margin.

Only headroom mode sets `gpu_memory_utilization` to `1.0` and bypasses the
fractional cap. An explicit zero also enables headroom mode.

## 2) Why headroom is necessary

The compiler includes the KV cache shape in each compiled program. A larger KV
cache changes the graph and its memory requirements.

The compiler gives the best estimate of non-KV memory after it receives a KV
shape. However, the worker must select the KV shape before compilation. The
worker therefore uses a configurable estimate for non-weight and non-KV memory.

In CPU inference, the default policy uses:

```text
total_budget = int(host_available / local_workers * gpu_memory_utilization)
available = int(total_budget * cap_fraction)
```

Explicit headroom uses:

```text
available = max(host_available / local_workers - headroom_bytes, 0)
```

Host available memory already excludes loaded weights and current allocations,
so CPU mode does not subtract model weights again.

The reserve has no default. For example, 4 GiB is a tested reserve for
GPT-OSS 120B on Trn3 with the Native backend and MXFP4 weights.
Validate the reserve for your model and configuration.

## 3) Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` | `0.30` | Fractional cap for default sizing. Headroom mode bypasses this value. |
| `VLLM_NEURON_RESERVED_HEADROOM_GIB` | unset | Explicit reserve per worker. Enables headroom sizing on both routes. |
| `VLLM_NEURON_MIN_KV_BUDGET_GIB` | `1.0` | Minimum KV budget. The server stops if the budget is smaller. |

In headroom mode, increase the active reserve if model loading or compilation reports
`NCC_EOOM002`, `Maximum peak HBM usage`, `NRT_RESOURCE` in `nrt_load_util`, or
`Allocation Failure`. First confirm that the same configuration passes with a
smaller KV cache or with the default policy. Start with 1 GiB increments.
Each additional 1 GiB of headroom reduces the KV cache budget by 1 GiB. Rerun
the same model, sharding, backend, LNC degree, and hardware configuration, and
keep the smallest value that passes. Each backend deployment can use a
different value for the same headroom variable.

Do not increase headroom when the worker reports that the KV cache budget is
below its minimum threshold. Decrease the active headroom variable for that
error. A validated configuration can also decrease headroom to use a larger KV
cache. At LNC 1, calculate the limit from half of the HBM stack:

```text
maximum_headroom =
    hbm_stack_bytes / 2 - model_weight_bytes - minimum_kv_budget
```

The startup log reports the logical-core HBM, model weights, active headroom,
and resulting KV budget. An override at or above the logical-core HBM share
leaves no usable KV cache.

Both backends and CPU use `VLLM_NEURON_RESERVED_HEADROOM_GIB`.
When this variable is unset, they use default fractional-cap sizing.
CPU compilation and device execution select the same budget policy.

In default mode, reduce the fractional cap to reduce KV capacity after an
HBM allocation failure. An explicit headroom value changes the budget policy;
validate the resulting KV capacity before deployment.

## 4) Implementation flow

The implementation is in
`vllm_neuron/vllm/worker/neuron_worker.py`.

```text
determine_available_memory()
  |
  +-- VLLM_NEURON_CPU_COMPILE == 1
  |      total_hbm = get_total_available_memory()
  |      model_weight_bytes = _get_byte_used_from_model()
  |      return _compute_kv_budget(total_hbm, model_weight_bytes)
  |
  +-- VLLM_NEURON_CPU_MODE == 1
  |      use the selected policy with the host memory fair share
  |
  +-- Neuron device
         total_hbm = runtime_bytes_used + runtime_bytes_free
         model_weight_bytes = _get_byte_used_from_model()
         return _compute_kv_budget(total_hbm, model_weight_bytes)
```

CPU compilation and device execution use the same model weight calculation.
This keeps the KV block count equal and permits compilation cache hits.

```text
_compute_kv_budget(hbm_stack_bytes, model_weight_bytes):
    logical_core_hbm = (
        hbm_stack_bytes // get_logical_cores_per_hbm_stack()
    )
    if headroom is explicit:
        available = max(
            logical_core_hbm - model_weight_bytes - headroom_bytes, 0
        )
    else:
        total_budget = int(logical_core_hbm * gpu_memory_utilization)
        available = max(min(
            max(total_budget - model_weight_bytes, 0),
            int(total_budget * cap_fraction),
        ), 0)
    if available < VLLM_NEURON_MIN_KV_BUDGET_GIB: raise RuntimeError
    return available
```

The device path uses runtime statistics only to get total HBM. It does not use
runtime memory usage as the model weight size.

The CPU inference path uses current available host memory, which already
reflects model weights and other allocations. Neither CPU inference policy
subtracts model weights again.

- **HBM stack sharing (LNC 1)** scales the total first. One HBM stack serves 2
  physical NeuronCores. At LNC 2 a logical core owns the stack. At LNC 1 two
  logical cores share it. Both inputs still report the whole stack. The runtime
  reports the stack to each core that holds a part of it, and the static table is
  per stack. `get_logical_cores_per_hbm_stack()` gives the divisor,
  which is 1 at LNC 2 and 2 at LNC 1. Without it, two siblings each size a KV
  cache against the same memory, and the allocation fails at load
  (`nrt_tensor_allocate status=4`). The scaling is in `_compute_kv_budget`, so
  both paths that call it get the same budget. This keeps the KV block count
  equal, which the compile cache needs.
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
- An explicit headroom value reserves non-KV memory after the LNC share is known.
- Minimum-KV guard fails fast for unusable configs.

## 5) Minimum budget check

The worker compares both CPU and Neuron results with
`VLLM_NEURON_MIN_KV_BUDGET_GIB`. It stops early if the result is too small.

In default mode, the error names memory utilization and the fractional cap.
In headroom mode, it names the active reserve and reports its maximum value.
Increasing headroom makes a minimum-KV-budget error worse.

Known HBM allocation exceptions raised while loading, initializing the KV
cache, compiling, or warming up retain their original exception and add a note
with the active reserve and the 1 GiB tuning procedure in headroom mode.
