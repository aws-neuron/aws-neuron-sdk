# NKI CPU Simulator

<!-- meta: description: NKI CPU simulator for kernel testing -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

The NKI CPU simulator allows NKI kernels to execute on CPU using NumPy, without requiring Neuron hardware. This enables local development, unit testing, and CI validation of NKI kernel correctness on CPU.

vllm-neuron uses the built-in `nki.simulator.simulate_kernel` API from NKI. No separate simulator package is required.

**The simulator is off by default in CPU mode.** It must be explicitly enabled by setting `NKI_SIMULATOR=1`. This keeps CPU mode fast for general development. The simulator is best suited for validating NKI kernel correctness on single functions, modules, or layers with small shapes or tiny model configs (\<10M params).

## How it works

When `VLLM_NEURON_CPU_MODE=1` and `NKI_SIMULATOR=1` are both set, the simulator is integrated through the NKI HOP (HigherOrderOperator) dispatch chain, supporting both eager and `torch.compile` (dynamo) execution modes.

``` text
VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1

wrap_nki(kernel) → NKIHOPCaller
NKIHOPCaller[grid](**kwargs)
  → nki_kernel_wrapper HOP (NKIKernelWrapper.__call__)
    → DispatchKey.CPU dispatch (_cpu_impl)
      → simulate_nki_kernel(func, lnc, kwargs)
        → torch tensors → numpy arrays
        → nki.simulator.simulate_kernel(func, args, kwargs, _lnc)
        → numpy arrays → torch tensors
```

When `NKI_SIMULATOR` is not set (the default), NKI kernels use their PyTorch fallback implementations instead. This is faster and sufficient for many development tasks.

## torch.compile / dynamo support

The NKI CPU simulator works with both eager mode (`enforce_eager=True`) and `torch.compile` (dynamo) mode. No special flags are needed.

**Eager mode**: `torch.compiler.set_stance("force_eager")` makes `torch.compile` a no-op. The HOP fires directly at runtime, and the `DispatchKey.CPU` dispatch runs the simulator immediately.

**Dynamo mode**: Dynamo traces the model into an FX graph. When it encounters the `nki_kernel_wrapper` HOP (marked `allow_in_graph`), it treats it as an opaque node and calls the `FakeTensorMode` dispatch to determine output shapes. Since the NKI compiler is not available in CPU mode, the `FakeTensorMode` dispatch runs the simulator on ones-filled tensors to infer output shape and dtype. At runtime, the `DispatchKey.CPU` dispatch runs the simulator with real data.

``` text
Dynamo tracing (one-time):
  FakeTensorMode dispatch
    → simulate_nki_kernel(func, lnc, ones)  # shape inference
    → returns FakeTensor with correct shape/dtype

Runtime (every call):
  DispatchKey.CPU dispatch
    → simulate_nki_kernel(func, lnc, real_data)  # actual execution
```

## Performance guidance

The simulator is **slow** — it runs NKI kernels through a Python-based NumPy backend. It is suitable for:

- Single functions, modules, or layers in isolation
- Small tensor shapes
- Tiny model configs (\<10M params)

For larger models or full-model iteration, use CPU mode without the simulator (the default). NKI kernels will use their PyTorch fallback paths, which are much faster.

Always use a timeout (e.g. `--timeout 60`) when running tests with the simulator to avoid long-running processes.

## Environment variables

| Variable | Default | Description |
|----|----|----|
| `VLLM_NEURON_CPU_MODE` | `0` | Enables CPU mode. NKI kernels use torch fallback paths. |
| `NKI_SIMULATOR` | unset (off) | Set to `1` to enable the NKI CPU simulator. Must be set explicitly — not auto-enabled by CPU mode. |
| `NKI_PRECISE_FP` | `1` (when simulator on) | Enables ml_dtypes for low-precision accuracy (bfloat16, float8). Auto-set when `NKI_SIMULATOR=1`. Set to `0` to use float32 approximations. |

## Running tests

``` bash
# NKI unit tests (dtype conversion, can_run_kernel)
VLLM_NEURON_CPU_MODE=1 pytest test/vllm_neuron/nki/test_nki_cpu_sim.py -v --timeout=60

# Functional tests in CPU mode (no simulator — fast)
VLLM_NEURON_CPU_MODE=1 pytest test/vllm_neuron/functional/ -v --timeout=60

# Functional tests with simulator (slow — small shapes only)
VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 pytest test/vllm_neuron/functional/ -v --timeout=60
```

## Limitations

- **Numerical differences**: CPU float arithmetic differs from NeuronCore arithmetic. Tests should use appropriate tolerances.
- **Performance**: Simulator execution is not representative of hardware performance. It is for correctness validation only.
- **Kernel coverage**: Some kernels may not yet be supported by the simulator. Unsupported operations will raise errors at simulation time.
