# Per-Graph Compiler Arguments

<!-- meta: description: Per-graph neuronx-cc arguments resolved by the runner and carried through a libtorch scope -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-09-09 -->

## Overview

A model runs many graphs. A prefill graph, a decode graph, an encoder graph,
and a jump-forward graph are examples. Each graph compiles to its own NEFF. This
document explains how a model gives each graph its own neuronx-cc compiler
arguments, for example a different `-O<n>` optimization level.

The mechanism has three parts:

- A **model policy**, `compiler_args_for(graph_type, example_inputs, base)`. It
  returns the full argument list for one graph type. The model owns this policy.
- A **runner resolver** in vLLM-Neuron, `resolve_graph_compiler_args(graph_type)`.
  It calls the model policy eagerly, one time per graph, and returns the concrete
  argument list, or `None` for no per-graph override.
- A **libtorch scope**, `scoped_compiler_args(args)`. The runner opens it —
  directly for an in-process compile, or by riding the resolved args on the
  parallel-trace job — and every backend seam reads it before the cache hash.

The graph type is the label the runner passes to the policy at each compile
site: `"decode"`, `"encoder"`, and so on, or `None` for an untagged graph.

## Problem Statement

vLLM-Neuron runs multiple graph types (prefill, decode, encoder), and multiple
buckets within each type. Each
graph/bucket compiles to its own executable NEFF. Because the graphs differ in
size and in performance profile (compute- vs IO-bound), model developers often
need to set different neuronx-cc arguments for each compiled NEFF. These
arguments control multiple performance and accuracy features, including the
graph's model-type (transformer vs U-Net), optimization level, auto-casting and
precision, and compute-to-collective tiling options. We need a mechanism to set
different compiler args per HLO in vLLM-Neuron.

## Alternatives Considered

We looked at four ways to give each graph its own arguments. The sections below
describe each option. The table at the end shows why the runner resolves the
arguments and carries them to libtorch through a scope.

### Option 1 — A separate dynamo model per graph type

The first design made a second `torch.compile` model for the decode phase. It
held a `self.model_decode` field and a separate compile-options dictionary. A
reviewer found three problems:

1. Each new model type (vision, audio, draft) needs its own compiled copy.
2. Each new compiler flag needs another compiled copy.
3. A single model type cannot give different arguments to different graphs.

A separate compiled model per case is not scalable. It also makes torch-export
serialization harder, because there are more graph objects. The number of
compiled copies grows with the number of graph types times the number of flag
sets.

### Option 2 — A static per-graph object in the compile options

This option passes a fixed data object, for example a `{graph_type: args}`
dictionary, in the compile options. libtorch reads the object and picks the
arguments for the graph.

A static object cannot express model-developer logic. It cannot read the FX
graph (`gm`) or the example inputs at resolve time. So it cannot select the
arguments by input shape or by a graph property.

The object also does not fit the two separate flows. The capture (trace) and
the compile are two different flows. The trace runs in forked child processes.
The NEFF build runs later in `parallel_compile`. A static object must reach the
correct resolve point in **each** flow, and the framework must enumerate every
graph type up front. This moves the model policy into the framework.

### Option 3 — A callable resolved by libtorch

This option passes a callable, `compiler_args_fn`, in the compile options.
libtorch calls it at one access point in every flow, passing `gm`,
`example_inputs`, and `options`, and writes the resolved concrete arguments (not
the callable) to the cache hash.

The callable works, but it couples libtorch to a model callback: libtorch has to
invoke model code on its compile path, including inside the forked trace
children. A child does not inherit the caller's context, so the framework has to
make the graph type cross the fork — an earlier version bound a tag to the trace
model through a proxy (`bind_graph_type`) that re-set a thread-local inside each
child. That indirection is fragile, and it puts model code on libtorch's compile
path. We replaced it with Option 4.

### Option 4 — The runner resolves eagerly, libtorch reads a scope (the chosen option)

The **runner** resolves the per-graph arguments eagerly, one time per graph,
through the model's `compiler_args_for` policy, and hands libtorch the concrete
list — never a callable. libtorch exposes one context manager,
`scoped_compiler_args(args)`, backed by a contextvar. Each backend seam calls
`apply_scoped_compiler_args(options)` before the cache hash; the scope value
replaces `compiler_args`, or the base arguments apply when no scope is set.

libtorch stays ignorant of models: it never calls model code and never carries a
model concept across the fork. The runner carries the resolved list to the
compiler two ways, one per flow:

- **In-process compile**: the call site opens `scoped_compiler_args(...)` around
  the forward or the warmup.
- **Parallel trace (deferred compile)**: the resolved list rides on the trace
  job as its third element `(model, kwargs, compiler_args)`. libtorch's
  `parallel_trace` opens the scope from that element inside each fork child.

Either way libtorch sees only a concrete argument list in a scope, and the same
list keys the cache in both flows.

### Comparison

| Option | How the model gives arguments | Main drawback |
| --- | --- | --- |
| 1. Separate dynamo model | One `torch.compile` per graph type | Copies grow per type and per flag; duplicate NEFFs; more device memory and compile time; harder serialization; one type still cannot vary its own graphs |
| 2. Static per-graph object | A fixed `{graph_type: args}` map in options | No access to `gm` or inputs, so no logic; must reach each flow's resolve point; framework must list every graph type |
| 3. Callable resolved by libtorch | A model callback in the options that libtorch calls per graph | Couples libtorch to model code on the compile path; the callback must run inside the fork children, so the graph type has to cross the fork through a fragile proxy |
| 4. Runner resolves, libtorch reads a scope (chosen) | The runner calls the model policy eagerly and carries the concrete list through a libtorch scope | Needs a resolve/scope point at every compile site: an in-process scope, or the trace-job element on the parallel path |

We picked Option 4. It keeps one `torch.compile` per backend. The model still
owns the policy, but the framework — not libtorch — runs it, so libtorch never
calls model code or carries a model concept across the fork. The concrete
arguments key the cache in both the in-process and the deferred-compile flows,
so the NEFF stays stable.

## Framework Solution: One Model Policy, One `torch.compile`

The design keeps **one** `torch.compile` per backend. The model supplies one
policy that returns the arguments for each graph type. The runner calls it
eagerly at each compile site and hands the result to libtorch through a scope.

### The model policy

A model defines `compiler_args_for`. It returns the **full** argument list for
one graph type:

``` python
# vllm_neuron/model/qwen3_vl/model_common.py
def compiler_args_for(self, graph_type, example_inputs, base):
    """Give each graph its own neuronx-cc compiler arguments."""
    from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

    base = list(base or [])                        # caller level, all flags
    if graph_type in ("decode", "encoder"):
        return with_optimization_level(base, 3)   # -O3 for decode and encoder
    return base                                    # prefill / default
```

Notes:

- `base` holds the base arguments at the caller level, with every platform flag
  (fp8, target, hlo2tensorizer) already set. The model does not rebuild them. It
  derives the per-graph arguments from `base`, for example with
  `with_optimization_level`.
- The policy returns the full list. So a model can add or remove any flag, not
  only the `-O` level.
- `example_inputs` is in the signature for future conditions. The runner passes
  `None` today, so a policy selects by graph type. See
  [Write a complex condition](#write-a-complex-condition).

### How the runner wires the policy

At `load_model` the runner sets the base arguments, then enables the policy only
when the caller did not set `--optimization-level`:

``` python
# vLLM-Neuron: neuron_model_runner.load_model()
self.compile_options["compiler_args"] = [...]   # base args at the caller level
policy = getattr(base_model_for_opt, "compiler_args_for", None)
per_graph = not opt_level_explicit and policy is not None
self._per_graph_compiler_args = per_graph
self._compiler_args_policy = policy if per_graph else None
if per_graph:
    # stamp the base args on the model so its own in-process sites can resolve
    base_model_for_opt._neuron_base_compiler_args = list(
        self.compile_options["compiler_args"]
    )
```

Opt-level precedence, highest first: (1) an explicit `--optimization-level` from
the caller, (2) the model policy, (3) the neuronx-cc default (`-O1` on Neuron).
An explicit caller level wins: it disables the policy, so every graph keeps the
caller level.

### The runner resolver

The runner turns a graph type into a concrete list one time per graph:

``` python
# vLLM-Neuron: neuron_model_runner.resolve_graph_compiler_args()
def resolve_graph_compiler_args(self, graph_type):
    if not self._per_graph_compiler_args or graph_type is None:
        return None                              # no override -> base args apply
    base = list((self.compile_options or {}).get("compiler_args") or [])
    return self._compiler_args_policy(graph_type, None, base)
```

`None` means "no per-graph override": the scope leaves the base `compiler_args`
in the options untouched.

### The libtorch scope

libtorch exposes a context manager and one access point in
`libtorch_neuronx_lite/compile/compiler_args.py`:

``` python
@contextlib.contextmanager
def scoped_compiler_args(compiler_args):
    """Set the concrete args for the graph compiled in this scope (None = no override)."""
    token = _scoped_compiler_args.set(compiler_args)
    try:
        yield
    finally:
        _scoped_compiler_args.reset(token)

def apply_scoped_compiler_args(options):
    """Return options with the scoped per-graph compiler_args applied."""
    options = options or {}
    scoped = _scoped_compiler_args.get()
    return options if scoped is None else {**options, "compiler_args": scoped}
```

libtorch has three backend seams:

| Backend | Entry point |
| --- | --- |
| `neuron_libtorch` (XLA compile) | `backend.py` |
| `neuron_libtorch_graph_capture` (capture) | `capture_backend.py` |
| `neuron_native_lite` (native) | `native_backend.py` |

All three call `apply_scoped_compiler_args(options)` before
`apply_platform_compiler_args` and before the cache hash. The scoped
`compiler_args` then flows to `create_cache_hash` and to neuronx-cc.
`create_cache_hash` keys only on `compiler_args` (plus the input tensor specs and
the tool versions). So each distinct argument set gets its own NEFF.

### The two flows

The runner opens the scope one of two ways, one per flow.

**In-process compile.** The site wraps the forward or the warmup in the scope:

``` python
# plain decode, in neuron_model_runner (_plain_decode_forward / _plain_decode_capture)
with scoped_compiler_args(self.resolve_graph_compiler_args("decode")):
    compiled_model(**kwargs)
```

The model exposes its own `_scoped_compiler_args(graph_type)` helper for the
in-process sites it owns (the encoder execute in `embed_multimodal`). It reads the
base args the runner stamped on the model, delegates to `compiler_args_for`, and
opens the libtorch scope.

**Parallel trace (deferred compile).** vLLM-Neuron can defer the NEFF build: it
captures every graph to the cache directory first, then `parallel_compile`
builds the NEFFs in parallel. The capture runs in forked child processes. The
runner resolves the args before the fork and rides them on the trace job as the
third element:

``` python
# neuron_worker._build_trace_job()
compiler_args = self.model_runner.resolve_graph_compiler_args(graph_type)
return target_model, kwargs, compiler_args
```

libtorch's `parallel_trace` opens `scoped_compiler_args(compiler_args)` from that
third element inside each child. The capture seam then resolves the args and
records them in the artifact metadata. `parallel_compile` reads them back from
the metadata for each graph, so a per-graph `-O` level survives the deferred
build. An old cache directory has no recorded arguments; `parallel_compile` then
uses the base arguments for that graph. The native path forks in the native-lite
wheel and keeps two-element jobs; it resolves in-process instead.

### Data flow

``` text
load_model:
  compile_options["compiler_args"] = base args at the caller level
  per-graph policy enabled only when the caller level is not explicit
  torch.compile(model, options=compile_options)   # ONE compile per backend

in-process compile of the plain decode graph:
  with scoped_compiler_args(resolve_graph_compiler_args("decode")):   # -> with_optimization_level(base, 3)
      compiled_model(**kwargs)
        └─> libtorch backend seam
              └─> apply_scoped_compiler_args(options)      # compiler_args <- decode args
              └─> create_cache_hash(... compiler_args=<decode args> ...)   # unique NEFF
              └─> neuronx-cc <decode args>

parallel-trace compile of the encoder graph:
  job = (encoder_model, kwargs, resolve_graph_compiler_args("encoder"))  # third element
        └─> parallel_trace child: with scoped_compiler_args(job[2])
              └─> capture seam -> record args in metadata -> unique NEFF
        └─> parallel_compile reads the args from the metadata -> neuronx-cc

any other graph (no scope / None):
  compiled_model(**kwargs)
        └─> apply_scoped_compiler_args -> no scope -> base args
```

## Cache-Key Stability

The runner resolves from the graph type at **every** compile site — warmup, the
in-process forward, and the parallel trace. So the plain decode graph resolves to
the same decode arguments each time. The warmup and a later serve-time recompile
then produce the same cache key.

**Warning**: A serve-time recompile of the plain decode graph outside the scope
falls back to the base arguments. This gives a different NEFF. To prevent this,
wrap every plain decode site with
`scoped_compiler_args(resolve_graph_compiler_args("decode"))`. You can also set
`NEURON_LIBTORCH_ASSERT_CACHE_HIT` to fail fast on a cache miss after warmup.

Keep the model policy deterministic. The same graph type must always give the
same arguments. If the arguments change between warmup and serve, the cache key
changes and the graph recompiles. The native backend additionally asserts this:
`_assert_cache_key_args` raises if one cache key is compiled with two different
argument sets, because the NEFF cache keys on graph structure and I/O shapes, not
on the compiler flags.

## Write a Complex Condition

`compiler_args_for` receives the graph type, the example inputs, and the base
list, so a model developer can write a complex condition.

**Select by graph type (the common case):**

``` python
if graph_type == "decode":
    return with_optimization_level(base, 3)
if graph_type == "jf":
    return with_optimization_level(base, 2)   # tune the jump-forward graph
return base
```

**Add or remove a flag (not only the level):**

``` python
if graph_type == "decode":
    return [*with_optimization_level(base, 3), "--some-decode-only-flag"]
```

`example_inputs` is in the signature for a future by-shape or by-graph-property
condition; the runner passes `None` today, so selection is by graph type.

To tune the jump-forward or the spec-verify graph, first resolve its warmup and
execute sites in `neuron_model_runner` with
`scoped_compiler_args(resolve_graph_compiler_args("jf"))` or `("spec_verify")`.
The code marks these sites with a `TODO(compiler-args)` comment. Then add the
matching branch to `compiler_args_for`.

## Files

| File | Role |
| --- | --- |
| `vllm_neuron/model/qwen3_vl/model_common.py` | `compiler_args_for` — the per-model policy; `_scoped_compiler_args` — the model's in-process encoder scope |
| `vllm_neuron/utils/compiler_args_ctx.py` | `with_optimization_level` — the pure helper the policy uses to set the `-O` level |
| `vllm_neuron/vllm/worker/neuron_model_runner.py` | Wires the policy; `resolve_graph_compiler_args`; scopes the plain decode sites |
| `vllm_neuron/vllm/worker/neuron_worker.py` | Rides the resolved args on the trace job (parallel path); scopes the encoder warmup and execute |
| `libtorch_neuronx_lite/compile/compiler_args.py` | `scoped_compiler_args` / `apply_scoped_compiler_args` — the scope and the access point |
| `libtorch_neuronx_lite/compile/backend.py` | XLA compile seam |
| `libtorch_neuronx_lite/compile/capture_backend.py` | Capture seam — records the resolved args in the metadata |
| `libtorch_neuronx_lite/compile/native_backend.py` | Native seam; `_assert_cache_key_args` guard |
| `libtorch_neuronx_lite/compile/parallel_compile.py` | Deferred parallel compile — reads the per-graph args from the metadata |
| `libtorch_neuronx_lite/compile/parallel_trace.py` | Opens the scope from the trace job's third element in the fork child |
| `libtorch_neuronx_lite/compile/schema.py` | Artifact metadata — carries the per-graph `compiler_args` |
