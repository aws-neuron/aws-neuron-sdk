# In-Place to Out-of-Place Rewrite Pass

<!-- meta: description: In-place to out-of-place rewrite pass -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-22 -->

> This pass runs as part of the [FX passes stage](index.md) of the compilation pipeline. It rewrites in-place operations (e.g., `add_`) to out-of-place equivalents (e.g., `add`) because XLA/HLO does not support in-place semantics.

## Problem Statement

The XLA/HLO backend does not support in-place tensor semantics. Operations like `x.add_(y)` or `operator.setitem(buf, idx, value)` mutate a tensor in-place, which has no direct HLO equivalent. These operations must be converted to out-of-place equivalents that produce new tensors, and all downstream references to the original tensor must be rewritten to use the new result.

### Pipeline Position

``` text
DeviceRewriterPass
     │
     ▼
AliasingOutputRewritePass
     │
     ▼
InPlaceToOutOfPlacePass     ◄── this pass
     │
     ▼
NkiKernelWriteBackendConfigPass
     │
     ▼
CollectiveReplicaGroupsPass
```

This pass runs **after** the aliasing pass. By the time it executes, the aliasing pass has already:

- Identified which inputs are mutated.
- Built the `io_map` for HLO aliasing.
- Stashed `root_input` in `node.meta` for mutating nodes that trace back through views to an input placeholder.

This pass consumes that `root_input` metadata to name replacement nodes (e.g. `kv_cache_modified`) so debug output and later passes can identify which input was modified.

## Algorithm Design

### Overview

The core method `_convert_inplace_ops` takes a snapshot of the node list (because the graph is mutated during iteration) and processes each node in forward order. It handles three categories of in-place operations:

1. `operator.setitem` — converted to scatter ops.
2. `copy_` — converted to `expand_as` + `slice_scatter`.
3. General in-place methods (`add_`, `mul_`, …) — target name is stripped of the trailing underscore.

After each conversion, two things happen:

- The replacement node is renamed to `<original>_modified` (using `root_input` from the aliasing pass when available).
- `_update_subsequent_ops` rewrites all downstream references.

### Node Filtering

The pass only targets `call_method` nodes whose target ends with a single trailing underscore (e.g. `add_`, `copy_`). Double-underscore dunder methods like `__setitem__` are excluded. `operator.setitem` nodes (`call_function`) are handled separately via `_convert_setitem`.

## General In-Place Methods

For operations like `add_`, `sub_`, `mul_`, `div_`, `fill_`, `zero_`, `clamp_`, `relu_`, `pow_`, `abs_`, etc., the conversion is straightforward: strip the trailing underscore to get the out-of-place equivalent.

The pass verifies the equivalent exists on `torch.Tensor` via `hasattr` before rewriting. If no equivalent exists, a `NotImplementedError` is raised.

``` text
Before:  call_method  add_  (x, y)       # mutates x in-place
After:   call_method  add   (x, y)       # produces new tensor x_modified
```

The node is rewritten in-place (its `target` attribute is changed) rather than being replaced with a new node. This preserves the node's position in the graph and its metadata.

## copy\_ Handling

`Tensor.copy_` is special because it accepts a source tensor that is broadcastable with the destination. For example, copying a `[1, 64]` source into a `[32, 64]` destination is valid — PyTorch broadcasts the source automatically.

Since XLA/HLO cannot lower `copy_` directly, the pass replaces it with two operations:

1. `expand_as(source, dest)` — broadcasts the source to match the destination shape. This is necessary because `slice_scatter` requires the source and destination to have compatible shapes.
2. `slice_scatter(dest, expanded_source)` — produces a new tensor with the same content as the expanded source, typed as an update of `dest`.

``` text
Before:
  call_method  copy_  (dest, src)

After:
  call_method    expand_as      (src, dest)        # broadcast src → dest shape
  call_function  slice_scatter  (dest, expanded)   # out-of-place "copy"
```

The original `copy_` node is erased from the graph after all its uses are redirected to the `slice_scatter` result.

## setitem Handling

`operator.setitem(buf, idx, value)` is the most complex case because the index can be an `int`, `slice`, `tuple`, or tensor (FX Node). The `_convert_setitem` method delegates to `_build_scatter` which dispatches based on index type.

If `_build_scatter` returns a valid scatter node, the original setitem is erased and replaced. If it returns `None` (unsupported index type), the setitem is kept but downstream references are still rewritten.

### Scalar-to-Tensor Promotion

Many setitem calls use scalar values (e.g. `buf[0] = 0.0`). Scatter operations require tensor operands, so two helpers handle promotion:

- `_ensure_tensor_node(gm, value, buf)` — wraps a scalar into `torch.full_like(buf, fill_value=value)`. If `value` is already an FX Node, it is returned unchanged.
- `_ensure_select_src(gm, value, buf, dim, idx)` — for `select_scatter`, the source must have the shape of `buf` with dimension `dim` removed. When `value` is a scalar, this creates a reference tensor via `torch.select(buf, dim, idx)` and then broadcasts the scalar into that shape with `torch.full_like`.

### Integer Index

A single integer index (e.g. `buf[3] = value`) maps to `torch.select_scatter(buf, value, dim=0, index=3)`. The value is promoted via `_ensure_select_src` if it is a scalar.

``` text
Before:  setitem(buf, 3, value)
After:   select_scatter(buf, value, dim=0, index=3)
```

### Slice Index

A slice index (e.g. `buf[2:8] = value`) is handled by `_slice_scatter`, which chooses between three strategies based on the slice parameters:

1. **start == 0, step == 1** — direct `torch.slice_scatter`. This is the common fast path.

2. **Non-zero start with step \> 1, or negative start** — direct `torch.slice_scatter` with explicit `start`/`end`/`step` kwargs. XLA handles these offsets correctly.

3. **Non-zero positive start, step == 1** — the value and a boolean mask are padded to the full buffer shape, then `torch.where` selects between the padded value and the original buffer. This mirrors the HLO pad+select pattern that XLA produces for setitem.

    The padding is constructed in `F.pad` format (pairs in reverse dimension order). Only dimensions from the last back to the scatter dimension need entries; earlier dimensions are unaffected.

    If no `example_value` shape metadata is available on the buffer node, the pass falls back to `slice_scatter` with explicit kwargs.

When the value is a scalar, the sliced region of the buffer is first extracted via `aten.slice.Tensor` to obtain a shape reference, then the scalar is broadcast into that shape via `_ensure_tensor_node`.

### Tuple Index

Tuple indices (e.g. `buf[2:4, :, 1:3]`) are handled by `_tuple_scatter`. The tuple is first normalized by `_resolve_tuple_index`:

- Trivial `slice(None)` entries (full-dimension slices) are dropped.
- If an `Ellipsis` is present, elements before it get positive dimension indices (0, 1, …) and elements after it get negative indices (-N, …, -1).

After normalization, the non-trivial entries are processed:

- **Tensor/bool mask in tuple** — falls back to `index_put` since scatter ops cannot handle advanced indexing.

- **Single non-trivial dimension** — delegates to `select_scatter` (for int) or `_slice_scatter` (for slice).

- **Multiple non-trivial dimensions** — uses a two-phase slice-scatter chain:

  1. **Forward pass**: slice the buffer down through each outer dimension using `aten.slice.Tensor`, keeping a reference to each intermediate sliced buffer.
  2. **Inner scatter**: scatter the value into the innermost dimension using `select_scatter` or `_slice_scatter`.
  3. **Reverse pass**: scatter the modified inner region back up through each outer dimension using `slice_scatter`, using the intermediate sliced buffer (not the original full buffer) as the parent at each level.

  This chaining lowers correctly to HLO `dynamic-update-slice`.

``` text
Example: buf[2:4, 1:3] = value  (2 non-trivial dims)

Forward:
  sliced_0 = aten.slice(buf, dim=0, start=2, end=4)

Inner scatter:
  scattered = slice_scatter(sliced_0, value, dim=1, start=1, end=3)

Reverse:
  result = slice_scatter(buf, scattered, dim=0, start=2, end=4)
```

### Tensor Index (FX Node)

When the index is an FX Node (a tensor computed at runtime):

- **Boolean mask** — `torch.where(mask, value, buf)` is used instead of `index_put`, because XLA cannot lower `index_put` with boolean indices. The dtype is checked via `example_value` metadata on the index node.
- **Integer tensor** — `buf.index_put((idx,), value)` is emitted directly.

In both cases, scalar values are first promoted to tensors via `_ensure_tensor_node`.

## Downstream Reference Rewriting

After every conversion, `_update_subsequent_ops` ensures the graph maintains valid SSA form. It walks all nodes that appear *after* the replacement node and substitutes every reference to the original tensor with the new result.

The method only rewrites `call_method`, `call_function`, and `output` nodes — `placeholder` and `get_attr` nodes cannot reference other nodes and are skipped.

`_replace_in_structure` performs the substitution recursively through nested `args` and `kwargs`, handling tuples, lists, and dicts at any depth. It uses identity comparison (`is`) rather than equality to avoid false matches between distinct nodes.

``` text
Before _update_subsequent_ops:
  x = placeholder
  x_modified = add(x, y)       # just converted from add_(x, y)
  z = mul(x, w)                # still references old x
  output(x)                    # still references old x

After _update_subsequent_ops:
  x = placeholder
  x_modified = add(x, y)
  z = mul(x_modified, w)       # updated to x_modified
  output(x_modified)           # updated to x_modified
```

Only nodes *after* the replacement in graph order are rewritten. Nodes *before* it (including the replacement's own arguments) are left untouched. This preserves SSA dominance: every use of a value is dominated by its definition.

When multiple in-place operations target the same tensor, each conversion chains correctly because the previous conversion already rewrote downstream references. The second in-place op now references the first replacement, and after its own conversion, downstream references are updated again:

``` text
Original:
  x = placeholder
  add_(x, y)
  mul_(x, z)
  output(x)

After converting add_:
  x = placeholder
  x_modified = add(x, y)
  mul_(x_modified, z)          # already updated by _update_subsequent_ops
  output(x_modified)

After converting mul_:
  x = placeholder
  x_modified = add(x, y)
  x_modified_2 = mul(x_modified, z)
  output(x_modified_2)
```

## Node Naming

Replacement nodes are named `<root>_modified` where `<root>` is:

- `node.meta["root_input"]` if set by the aliasing pass (this happens when the mutation traces back through view ops to an input placeholder), or
- `original_input.name` as a fallback.

This naming convention makes it easy to identify which input was modified when inspecting graph dumps or debug logs.

## Example

A model that updates a KV-cache buffer via setitem:

``` text
Before:
  placeholder  kv_cache
  placeholder  new_keys
  setitem      (kv_cache, slice(0, 32), new_keys)
  output       (kv_cache,)

After:
  placeholder  kv_cache
  placeholder  new_keys
  slice_scatter kv_cache_modified  (kv_cache, new_keys, dim=0, end=32)
  output       (kv_cache_modified,)
```

The `io_map` from the aliasing pass (`{0: 0}`) remains valid because the output still corresponds to the same input — it is just the out-of-place version.

## Source Location

`vllm_neuron/fx_passes/inplace_rewrite_pass.py`
