# Aliasing Output Rewrite Pass

<!-- meta: description: Aliasing output rewrite pass for Neuron compilation -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-22 -->

> This pass runs as part of the [FX passes stage](index.md) of the compilation pipeline, between FX tracing and HLO lowering. It detects in-place buffer mutations and produces aliasing metadata for the XLA/HLO backend.

## Problem Statement

The XLA/HLO backend used by Neuron allocates separate memory buffers for every input and output of a compiled graph. When a model mutates an input buffer in-place (e.g. updating KV-cache state), the runtime must know that a particular output is the *same* buffer as a particular input so it can reuse the allocation instead of copying.

HLO expresses this through `input_output_alias` entries. The aliasing pass is responsible for analyzing the FX graph, detecting every case where an output derives from a mutated input, and producing the `io_map` metadata that downstream HLO conversion uses to emit those alias entries.

### Sources of Aliasing

The pass handles four categories:

1. **In-place torch operations** — `x.add_(y)`, `operator.setitem`, etc.
2. **NKI kernel mutations** — `wrap_nki` HOP nodes whose `operand_output_aliases` dict declares which kernel inputs are written back to kernel outputs.
3. **Custom ops with mutable arguments** — ops whose schema contains `Tensor(a!)` annotations (`alias_info.is_write=True`).
4. **Pass-through / view aliases** — an output that is a direct reference to an input, or reaches one through a chain of view ops (`view`, `reshape`, `transpose`, …).

### Pipeline Position

``` text
DeviceRewriterPass
     │
     ▼
AliasingOutputRewritePass   ◄── this pass
     │
     ▼
InPlaceToOutOfPlacePass
     │
     ▼
NkiKernelWriteBackendConfigPass
     │
     ▼
CollectiveReplicaGroupsPass
```

The aliasing pass runs **before** the in-place-to-out-of-place pass. It analyzes the graph while in-place operations are still present, records which inputs are mutated, and stashes `root_input` metadata on mutating nodes so the subsequent `InPlaceToOutOfPlacePass` can name replacement nodes correctly.

## Algorithm Design

### Overview

The pass performs a single linear scan of the graph followed by an output rewrite step. It produces an `io_map` dict (`{output_idx: input_idx}`) and an `original_output_count` integer that are returned as metadata alongside the transformed `GraphModule`.

### Phase 1: Collect Input Placeholders

All `placeholder` nodes are collected in order. Their positional index is significant because `io_map` uses it to pair outputs with inputs. Dynamo's mangled names (e.g. `L_self_modules_layer_parameters_weight_`) are cleaned up for debug logging.

### Phase 2: Build the Alias Chain

A single forward pass over every node classifies it into one of:

- **Mutating op** — the mutated tensor is traced back through the alias chain to its root input placeholder. The node is recorded in `mutated_inputs` and added to `alias_chain` so downstream nodes can be traced back too. `node.meta["root_input"]` is set for the `InPlaceToOutOfPlacePass`.
- **Aliasing op** — view/reshape/transpose/etc. The node is added to `alias_chain` pointing at its source tensor (first argument).
- **Custom op with mutations** — schema arguments with `alias_info.is_write=True` are resolved to their root input placeholders.
- **NKI kernel with mutations** — `operand_output_aliases` entries are resolved. For tuple-returning kernels, `getitem` nodes are found or created and registered in `alias_chain`.

### Phase 3: Extend Outputs and Build io_map

Mutated inputs that are not already represented in the output tuple are appended as new output entries. For NKI kernels, the `getitem` node carrying the post-mutation value is preferred over the raw input placeholder.

A second scan over the (now extended) output tuple detects pass-through and view-alias relationships that allow buffer reuse even without mutations. Slice views are excluded because they change shape and cannot alias the full input buffer in HLO.

### Data Structures

``` text
alias_chain : dict[Node, Node]
    Maps each node to the node it aliases.  Following the chain
    eventually reaches a graph-level input placeholder.

mutated_inputs : dict[int, Node]
    Maps input placeholder index → placeholder node for every
    input that is mutated in-place.

io_map : dict[int, int]
    Maps output index → input index.  Forwarded to HLO conversion
    which writes input_output_alias entries.
```

### Algorithm Flow

``` text
┌──────────────────────┐
│  Collect placeholders│
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  For each node:      │
│  • Mutating op?      │──▶ Record in mutated_inputs + alias_chain
│  • Aliasing op?      │──▶ Record in alias_chain
│  • Custom op w/mut?  │──▶ Resolve schema args → mutated_inputs
│  • NKI kernel w/mut? │──▶ Resolve aliases → mutated_inputs + alias_chain
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Extend output tuple │
│  with mutated inputs │
│  not already present │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Scan outputs for    │
│  pass-through/view   │
│  aliases → io_map    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Return (gm, {       │
│    io_map,            │
│    original_output_   │
│    count              │
│  })                   │
└──────────────────────┘
```

## Example

Before the pass, a graph that mutates `kv_cache` in-place:

``` text
placeholder  x
placeholder  kv_cache
call_method  add_        (kv_cache, x)
output       (result,)
```

After the pass, `kv_cache` is appended to the output tuple and `io_map` records the relationship:

``` text
placeholder  x
placeholder  kv_cache
call_method  add_        (kv_cache, x)
output       (result, kv_cache)

io_map = {1: 1}   # output[1] aliases input[1]
```

The `InPlaceToOutOfPlacePass` then rewrites `add_` → `add` and updates references, but the `io_map` remains valid because the output position is stable.

## Module-Level Constants

Three sets classify operations for the analysis:

- `ALIASING_METHODS` — method-call names that produce views (`view`, `reshape`, `transpose`, `permute`, …).
- `ALIASING_ATEN_OPS` — ATen-level equivalents seen after `torch.export` lowering (includes `_unsafe_view`, `view_as`, `reshape_as`, …).
- `INPLACE_METHODS` — in-place method names used for mutation detection (`add_`, `copy_`, `scatter_`, …).

## Source Location

`vllm_neuron/fx_passes/aliasing_pass.py`
