# How to choose a compilation route

<!-- meta: description: The native route is the default compilation route in
vLLM Neuron. Learn how to keep it, how to go back to the XLA route, and how to
confirm which route runs. -->
<!-- meta: date_updated: 2026-10-08 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: NDOC-314 -->

## Task overview

The native route is the default compilation route in vLLM Neuron. If you do
not set anything, your server uses it. You do not need to change your launch
command.

vLLM Neuron has two compilation routes:

- **Native route (default).** It lowers the model through PyTorch native
  compilation (AOTAutograd and torch-mlir) to StableHLO, then to `neuronx-cc`.
- **XLA route.** It lowers the model through PyTorch/XLA to HLO, then to
  `neuronx-cc`. This route is deprecated, and a future release removes it.

Both routes start from the same `torch.compile` graph, and both run the compiled
NEFFs on the same runtime. Your model code, your serving API, and your requests
stay the same. Only the lowering step between them is different.

Use this guide to keep the default, to go back to the XLA route, and to confirm
which route your server uses.

## Prerequisites

- A vLLM Neuron server that you start from a shell, a container, or a job script.
- Access to the server environment, so that you can set an environment variable
  before the server starts.

## Instructions

### 1. Keep the native route

You do not need to do anything. When `VLLM_NEURON_BACKEND` is not set, the server
uses the native route.

If you set the route in a script, you can make the choice explicit:

```bash
export VLLM_NEURON_BACKEND=neuron_native
```

Some container images set `VLLM_NEURON_BACKEND=neuron`. This value also selects
the native route.

### 2. Go back to the XLA route

Use the XLA route if a model gives a result on the native route that you cannot
accept. Set the variable before the server starts:

```bash
export VLLM_NEURON_BACKEND=vllm_neuron
```

Then restart the server. The server reads the variable once, at start-up. A change
to a server that is running has no effect.

### 3. Return to the native route

Remove the variable, or set it to `neuron_native`. Then restart the server.

```bash
unset VLLM_NEURON_BACKEND
```

## Confirm your work

Look for one of these lines in the server log at start-up:

| Route | Log line |
| --- | --- |
| Native | `Using default optimization level O2 on the Neuron native route` |
| XLA | `Defaulting optimization level to O1 on the XLA route` |

The server writes these lines only when you do not set `--optimization-level`.
When you set it, the log shows `Explicit --optimization-level: keeping O<N>`, and
the line does not name the route.

A spelling error in the value does not stop the server. The server writes a
warning and uses the native route:

```text
VLLM_NEURON_BACKEND='<value>' is not a valid route. The native route runs now.
```

If you expect the XLA route and see this warning, correct the value and restart.

## What is different on the native route

### The default optimization level is O2

vLLM uses O2 as its default optimization level. The native route keeps O2. The
XLA route lowers the default to O1.

Thus the same command can compile a different graph on each route. To use the
same level on each route, set it explicitly:

```bash
vllm serve <model> --optimization-level 2
```

The server always honors an explicit level, on each route.

### The Precompile Cache needs the route variable

The Precompile Cache saves the trace and reuses it on the next start. The cache
is on when `VLLM_NEURON_BACKEND` is `neuron_native` or `neuron`. When the
variable is not set, the server uses the native route but the cache is off. To
use the cache, set the variable before you start the server:

```bash
export VLLM_NEURON_BACKEND=neuron_native
```

For more information, see
[How to use the Precompile Cache](how-to-precompile-cache.md).

## Related information

- [Features guide](features-guide.md#compilation) for compilation, bucketing,
  and the compilation cache.
- [Configuration reference](reference-configuration.md) for every environment
  variable.
- [How to use the Precompile Cache](how-to-precompile-cache.md) to make a warm
  start faster on the native route.
