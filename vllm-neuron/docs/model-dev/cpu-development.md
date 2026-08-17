# How to develop and test vLLM Neuron in CPU mode

<!-- meta: description: Set up CPU-mode development, compilation, debugging,
and NKI simulation workflows for vLLM Neuron without Neuron hardware. -->
<!-- meta: date_updated: 2026-06-09 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: NDOC-186 -->

## Task overview

This guide covers the full CPU-based development workflow for vLLM Neuron,
including CPU development mode, CPU compilation mode, the NKI CPU simulator,
and debugging techniques. These modes enable rapid iteration without Neuron
hardware.

## Prerequisites

- Python 3.10+
- Git access to the `vllm-neuron` repository
- Docker (optional, for the dev container workflow)
- No Neuron hardware required

## When to use each mode

| Mode                 | Use case                                                           | Speed             |
| -------------------- | ------------------------------------------------------------------ | ----------------- |
| CPU mode (sim off)   | General development, feature iteration, most debugging             | Fast              |
| CPU mode + simulator | Kernel development, accuracy debugging, small reproducing examples | Slow              |
| CPU compilation      | Pre-compiling NEFFs without hardware, validating graph capture     | Medium            |
| On-device            | Final validation, performance testing, large model scale-up        | Requires hardware |

## Instructions

### 1. Clone and install in development mode

```bash
git clone git@github.com:aws-neuron/vllm-neuron.git
cd vllm-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com \
    -e ".[dev]"
```

### 2. Enable CPU development mode

Set the environment variable to run without Neuron hardware:

```bash
export VLLM_NEURON_CPU_MODE=1
```

In CPU mode:

- Tensor operations run on CPU (no NeuronCores needed)
- Model compilation is simulated (no NEFF generation)
- Functional correctness can be verified without hardware
- NKI kernels fall back to PyTorch reference implementations
- Catches wrong weight mappings, shape mismatches, incorrect collectives,
  bad RoPE variants, and transposition errors

### 3. Run unit tests

```bash
VLLM_NEURON_CPU_MODE=1 pytest test/unit -v --timeout=300
```

### 4. Run functional tests in CPU mode

```bash
VLLM_NEURON_CPU_MODE=1 pytest test/vllm_neuron/functional/ -v --timeout=60
```

### 5. Use the NKI CPU simulator (optional)

The NKI CPU simulator executes NKI kernels on CPU using NumPy, enabling
correctness validation without Neuron hardware. It must be explicitly
enabled — it is not auto-activated by CPU mode.

```bash
VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 pytest \
    test/vllm_neuron/nki/test_nki_cpu_sim.py -v --timeout=60
```

:::{note}
The simulator is slow — it runs NKI kernels through a Python-based NumPy
backend. Use it only for single functions, modules, or layers with small
tensor shapes or tiny model configs (<10M params). Always set `--timeout`.
:::

#### How the simulator works

When both `VLLM_NEURON_CPU_MODE=1` and `NKI_SIMULATOR=1` are set, the
simulator integrates through the NKI HOP dispatch chain:

```text
wrap_nki(kernel) → NKIHOPCaller
NKIHOPCaller[grid](**kwargs)
  → nki_kernel_wrapper HOP (NKIKernelWrapper.__call__)
    → DispatchKey.CPU dispatch (_cpu_impl)
      → simulate_nki_kernel(func, lnc, kwargs)
        → torch tensors → numpy arrays
        → nki.simulator.simulate_kernel(func, args, kwargs, _lnc)
        → numpy arrays → torch tensors
```

The simulator supports both eager mode (`--enforce-eager`) and
`torch.compile` (dynamo) mode:

- **Eager mode**: `torch.compiler.set_stance("force_eager")` makes
  `torch.compile` a no-op. The HOP fires directly.
- **Dynamo mode**: Dynamo traces the FX graph, uses `FakeTensorMode`
  dispatch with ones-filled tensors for shape inference, then runs the
  simulator with real data at runtime.

#### NKI simulator limitations

- Numerical differences between CPU float and NeuronCore arithmetic
  require appropriate tolerances in tests
- Simulator execution is not representative of hardware performance
- Some kernels may not yet be supported by the simulator

### 6. Use CPU compilation mode

CPU Compilation (`VLLM_NEURON_CPU_COMPILE=1`) compiles model graphs on
a CPU instance without executing them. This eliminates the need for
Neuron instances during compilation and enables pre-compiling NEFFs for
later device execution.

```bash
VLLM_NEURON_CPU_COMPILE=1 \
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 32 \
    --max-model-len 4096
```

:::{warning}
`VLLM_NEURON_CPU_COMPILE` and `VLLM_NEURON_CPU_MODE` cannot be enabled
together. CPU Mode runs inference on CPU; CPU Compilation only compiles
graphs without execution.
:::

#### Remote cache workflow for production

1. **Compile on CPU instance** — set `NEURON_LIBTORCH_REMOTE_CACHE` to a
   shared filesystem (NFS/FSx):

   ```bash
   VLLM_NEURON_CPU_COMPILE=1 \
   NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
   NEURON_LIBTORCH_REMOTE_CACHE=/shared/neff_cache \
   NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS=16 \
   python3 -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3.3-70B-Instruct \
       --tensor-parallel-size 32
   ```

2. **Deploy to production Neuron instances** — ensure the remote cache
   is accessible, then disable compilation entirely:

   ```bash
   NEURON_LIBTORCH_REMOTE_CACHE=/shared/neff_cache \
   NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND=1 \
   VLLM_NEURON_DISABLE_WARMUP_COMPILE=1 \
   python3 -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3.3-70B-Instruct \
       --tensor-parallel-size 32
   ```

The `VLLM_NEURON_DISABLE_WARMUP_COMPILE=1` flag treats any cache miss as
a fatal `RuntimeError`, ensuring production never silently compiles.

#### CPU compilation limitations

- Speculative decoding is not supported in CPU compile mode
- Cannot validate numerical outputs — only confirms compilation
  succeeds

### 7. Use the dev container (alternative)

For flows that require `torch_neuronx` import paths (XLA tracing,
specific backend code):

```bash
ci/dev-container.sh login          # One-time: refresh ECR creds
ci/dev-container.sh pull           # Pull latest postmerge image
ci/dev-container.sh shell          # Interactive shell
ci/dev-container.sh exec pytest test/unit/ -x -q --no-cov
ci/dev-container.sh stop           # Remove container
```

The container bind-mounts the host repo — edits on your host appear
immediately inside the container.

## Debugging techniques

### Debugging with torch eager on CPU mode

With `VLLM_NEURON_CPU_MODE=1` and `--enforce-eager`, print statements
work for all models. For pdb support with `world_size=1`:

```bash
VLLM_NEURON_CPU_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python3 -m vllm.entrypoints.openai.api_server \
    --model <model> --enforce-eager
```

For pdb with `world_size > 1`, install forked-pdb:

```bash
pip install fpdb
```

Then insert in your code:

```python
__import__('fpdb').ForkedPdb().set_trace()
```

### Debugging with torch.compile (pdb)

When using `torch.compile`, dynamo runs in child processes. Use the
`original_stdio` context manager to redirect I/O for debugging:

```python
def forward(self, hidden_states, positions, ...):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    with original_stdio():
        breakpoint()
    # Inspect variables here
```

This lets you inspect tensor shapes, values, and TP ranks:

```text
(Pdb) p hidden_states.shape
torch.Size([19, 4096])
(Pdb) get_tensor_model_parallel_rank()
1
```

:::{warning}
Stepping through code (next/step) is not supported in torch.compile
debug mode. Dynamo generates resume functions that cause `BdbQuit`
exceptions when the debugger attempts to step. Use breakpoints only
for inspection, not stepping.
:::

### Printing during tracing

Print statements work in CPU mode for inspecting tensor shapes and
values during `torch.compile` tracing:

```python
q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
print(f'--- q  {q[:, 0]}')
```

:::{warning}
On Neuron devices (not CPU mode), printing tensor shapes inside
`torch.compile` causes dynamo graph breaks that trigger:

- `NCC_ITEN406` errors from neuronxcc (strided access patterns)
- `config.recompile_limit` errors leading to dtype mismatch
  `RuntimeError`

Only use print debugging in CPU mode. Remove all prints before
running on device.
:::

## Confirm your work

Your CPU development environment is working when:

- Unit tests pass with `VLLM_NEURON_CPU_MODE=1`
- You can import `vllm_neuron` without errors
- Functional tests detect CPU mode and skip hardware-specific
  assertions
- Code changes are reflected immediately (editable install)

## Environment variables reference

| Variable                                    | Default           | Description                                 |
| ------------------------------------------- | ----------------- | ------------------------------------------- |
| `VLLM_NEURON_CPU_MODE`                      | `0`               | Enable CPU development mode                 |
| `NKI_SIMULATOR`                             | unset             | Enable NKI CPU simulator (must be explicit) |
| `NKI_PRECISE_FP`                            | `1` (when sim on) | Enable ml_dtypes for low-precision accuracy |
| `VLLM_NEURON_CPU_COMPILE`                   | `0`               | Enable CPU compilation mode (no execution)  |
| `NEURON_PLATFORM_TARGET_OVERRIDE`           | —                 | Target platform for CPU compilation         |
| `NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS`      | `8`               | Number of parallel compile workers          |
| `NEURON_LIBTORCH_REMOTE_CACHE`                  | —                 | Path to shared remote NEFF cache            |
| `NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND` | `0`               | Disable graph capture backend entirely      |
| `VLLM_NEURON_DISABLE_WARMUP_COMPILE`        | `0`               | Treat cache miss as fatal error             |
| `VLLM_ENABLE_V1_MULTIPROCESSING`            | `1`               | Set to 0 for single-process pdb             |

## Common issues

### "No module named torch_neuronx"

- **Possible solution**: Use the dev container
  (`ci/dev-container.sh shell`) which has it pre-installed. In pure
  CPU mode, most tests skip these paths automatically.

### Tests hang indefinitely

- **Possible solution**: Always use `--timeout` flags. CPU-mode
  simulation can be slow for large shapes. Reduce model size or use
  smaller test configs.

### "NeuronCore not available" errors

- **Possible solution**: Ensure `VLLM_NEURON_CPU_MODE=1` is set. Some
  test fixtures require this variable to bypass hardware checks.

### Recompilation limit errors with print statements

- **Possible solution**: Remove print statements from model code
  before running with `torch.compile`. Dynamo treats each print as a
  graph break, triggering recompilation until the limit is hit.

### BdbQuit when stepping in pdb

- **Possible solution**: Only use `breakpoint()` for inspection (print
  variables, check shapes). Do not use `next`/`step` commands.
  Dynamo's resume functions are incompatible with the debugger's
  line-tracing mechanism.

## Related information

- [Onboarding models guide](onboarding-models.md) — full model porting
  workflow
- [Accuracy debugging guide](accuracy-debugging-guide.md) — debugging
  output differences
- [Profiling workloads](../guides/how-to-profile-workloads.md) — NRT profiling
  on live instances
