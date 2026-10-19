# SPMD Compilation

<!-- meta: description: How vLLM Neuron collapses per-rank compiles into a single shared NEFF via SPMD tracing -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-07-29 -->

## Overview

vLLM Neuron traces each model forward into an FX graph, lowers it to HLO, and compiles it to a NEFF with `neuronx-cc`. In a distributed (tensor / expert / data parallel) deployment every rank runs this pipeline. Whether the compiler runs **once** for the whole world or **once per rank** depends on whether the ranks trace an *identical* graph.

- **MPMD (Multiple Program, Multiple Data)** — the legacy behavior. Each rank bakes rank-specific values (its expert-parallel rank, tensor-parallel rank, collective channel ids) into the traced graph as compile-time constants. The graph text — and therefore the compile-cache key — differs per rank, so `neuronx-cc` is invoked once per rank (e.g. 64 NEFFs on a 64-rank decode).
- **SPMD (Single Program, Multiple Data)** — the target behavior. Rank-specific values are computed *in-graph* from a runtime `rank` tensor rather than baked in. All ranks trace a byte-identical graph, produce the same cache key, and share a single NEFF. The compiler runs once.

This document describes how vLLM Neuron achieves SPMD compilation, how to keep model code SPMD-clean, and the two guard rails that assert SPMD actually holds.

> **Note:** SPMD support is **additive**. Every SPMD code path keeps a `rank=None` / eager fallback that reproduces the original MPMD behavior bit-for-bit. Callers opt in by threading the runtime `rank` tensor; nothing is forced onto SPMD.

## Why MPMD happens

Two independent sources make a traced graph diverge across ranks:

1. **Baked rank constants.** Model code that reads its rank as a Python `int` at trace time and uses it to build a tensor or slice offset — e.g. `rank_id = torch.tensor([[self.ep_rank]])` or `x[:, rank * shard : (rank + 1) * shard]` — embeds a different literal in each rank's graph.
2. **Collective channel ids / replica groups.** `_c10d_functional` collective ops reference their process group by an opaque `group_name` whose resolved channel id and replica-group membership differ per rank. Left unnormalized, these values also make the graph text (and the cache key) rank-dependent.

Both must be neutralized for a graph to be SPMD.

## The runtime rank tensor

The model runner threads the current rank into every forward as a **runtime tensor** kwarg (`rank`), specifically so it does not become a trace-time constant. Model code derives any per-rank quantity from this tensor with in-graph arithmetic. The canonical idiom mirrors `get_neuron_ep_rank()`:

``` python
# SPMD: rank_id derived from the runtime `rank` tensor (traces identically
# on every rank). ep_degree / tp_degree are static and identical world-wide.
ep_rank = (rank % (self.ep_degree * self.tp_degree)) // self.tp_degree
rank_id = ep_rank.view(1, 1).to(torch.int32)

# MPMD (avoid): bakes a per-rank literal into the graph.
rank_id = torch.tensor([[self.ep_rank]], dtype=torch.int32)
```

The vocabulary-embedding path is a good reference: `vocab_start_idx = rank * vocab_per_rank` is computed from the runtime tensor and used for masking arithmetic, so it traces identically on every rank.

### Static slices must become `index_select`

A Python slice `x[:, start:start + n]` requires **compile-time** bounds — it cannot take a runtime `start`. When `start` is rank-derived, rewrite the slice as an `index_select` over a runtime-derived index range:

``` python
# MPMD: static slice with a baked rank offset.
col_start = rank_in_group * shard          # python int
local = x[:, col_start : col_start + shard]

# SPMD: index_select over a runtime-derived ascending range.
col_start = (rank % world_size) * shard    # runtime tensor
idx = torch.arange(shard, device=x.device) + col_start
local = x.index_select(1, idx)
```

This rewrite is value-identical (`arange(n) + start` enumerates the same indices the slice would) but keeps the graph rank-invariant.

## Collective lowering

Collective divergence is handled by the mesh registry and the `CollectiveReplicaGroupsPass` FX pass (see [FX passes design](fx_passes_design.md)):

- `NeuronGroupCoordinator` registers each process group's **full mesh** (the complete partition of the world into replica groups) globally at group creation time.
- `CollectiveReplicaGroupsPass` resolves each collective's `group_name` to that full mesh and writes it into `node.meta["replica_groups"]`.
- The compile cache normalizes the `group_name` against the mesh when computing the hash, so 2D-mesh collectives hash identically across ranks.

Because the mesh is the same object on every rank, the resolved channel id and replica groups are SPMD-invariant.

## Compile-cache effect

With both divergence sources neutralized, all ranks produce the same 32-char cache hash. The compile cache then collapses N per-rank compiles into one (see [Compilation cache](compilation_cache.md), *Parallel Coordination*): rank 0 compiles and signals completion; the other ranks wait and reuse the shared NEFF.

## Guard rails

Two opt-in checks assert that SPMD actually holds, both gated by the `NEURON_LIBTORCH_SPMD_STRICT=1` environment variable. They **warn / no-op by default** (MPMD is tolerated) and only fail hard under the strict flag, so they can gate a CI lane or an isolated test without disturbing normal runs.

### `RankConstantCheckPass` (graph level)

An FX pass (runs last in the default pass manager) that scans the fully-lowered graph for **baked rank constants**: small integer-constant tensors (`torch.tensor([[c]])` / `torch.full`) constructed while a runtime `rank` placeholder exists in the graph — the signature of a rank value that should have been derived from the runtime tensor. Under the strict flag it raises; otherwise it logs a warning. This catches the *cause* (a baked constant) at trace time.

### Single-NEFF assertion (compile-cache level)

A check in `parallel_compile` that asserts the *outcome*: every distinct traced graph must collapse to a single cache-hash directory shared by all ranks, rather than fanning out to one NEFF per rank.

It is **num-graphs-agnostic** — a model that legitimately compiles several graphs (prefill + multiple decode buckets + a vision tower) still passes, because each *distinct* graph is checked for rank collapse independently; it does not assert a single NEFF overall. Detection uses the **union of rank indices** that staged an HLO across the hash dirs being compiled: a correctly-collapsed graph's directory contains that entire set, and any directory missing ranks is a per-rank splinter. Under the strict flag a splinter raises with the offending hashes; otherwise it is tolerated.

The two checks are complementary: `RankConstantCheckPass` flags a baked constant in the graph, while the single-NEFF assertion catches any residual per-rank fan-out at the point NEFFs are compiled.

## Writing SPMD-clean model code

- Accept the runtime `rank` tensor in the forward and thread it to any helper that needs a per-rank value; default it to `None` and keep the legacy baked path for non-SPMD callers.
- Derive per-rank quantities with in-graph arithmetic on `rank` (`(rank % (ep * tp)) // tp`), not from `dist.get_rank()` / `group.rank_in_group` Python ints.
- Replace static slices that use a rank offset with `index_select` over `arange(n) + runtime_offset`.
- Weight-loading code (`_load_from_state_dict`) is **not** traced, so rank slicing there is fine and should not be converted.
- Some NKI kernels bake a rank offset into an `iota` at compile time; making those SPMD requires a kernel change (accept a runtime offset tensor), not just plumbing.
- Run with `NEURON_LIBTORCH_SPMD_STRICT=1` to have both guard rails fail hard on any regression.

## Related

- [FX passes design](fx_passes_design.md) — the FX pass pipeline, including `CollectiveReplicaGroupsPass` and `RankConstantCheckPass`.
- [Compilation cache](compilation_cache.md) — cache keying and the parallel-coordination flow that collapses SPMD compiles.
- [Expert parallelism](../parallelism/expert_parallelism.md) — EP process groups and rank layout.
