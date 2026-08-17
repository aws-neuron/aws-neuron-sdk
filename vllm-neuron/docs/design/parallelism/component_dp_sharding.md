# Component-Level DP Sharding

<!-- meta: description: Component DP sharding design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

Each model component (Attention, Embedding, MLP, LM Head) can independently shard its weights across `TP * component_dp_size` devices instead of just TP. This reduces per-device weight memory at the cost of additional collectives during decode.

**Decode-only.** Prefill uses standard TP sharding and is unaffected by these flags.

**DI (Disaggregated Inference) required.** Since prefill and decode use different effective TP sizes, weights must be loaded separately per graph. This is naturally supported in DI setups where prefill and decode run on separate compiled graphs.

## Configuration

```python
NeuronConfig(
    attention_dp_size=4,  # existing — Q/O (and optionally K/V) sharding
    embedding_dp_size=4,            # new — Embedding vocab sharding
    mlp_dp_size=4,                  # new — MLP intermediate sharding
    lm_head_dp_size=4,              # new — LM Head vocab sharding
)
```

Each parameter controls one component independently:

| Parameter | Component | What's sharded | Default |
|-----------|-----------|---------------|---------|
| `attention_dp_size` | Attention Q/O/KV | Q heads, O projection, optionally KV heads | 1 |
| `embedding_dp_size` | Embedding | Vocabulary dimension | 1 |
| `mlp_dp_size` | Dense MLP | Intermediate dimension (gate/up/down projections) | 1 |
| `lm_head_dp_size` | LM Head | Vocabulary dimension (output projection) | 1 |

**Constraints:**

- Each `*_dp_size` must be a positive integer that divides the overall `dp_size`
- `embedding_dp_size` and `lm_head_dp_size` are independent (can differ even with tied embeddings — weights are loaded separately)
- `mlp_dp_size > 1` is incompatible with `ep_degree > 1` (MoE models use EP for expert sharding instead)

## Process Groups

Two types of groups are created per component:

### Component TP group (for weight sharding and TP collectives)

A rectangular block of `TP * component_dp_size` ranks. Passed as `tp_group` to the module (`VocabDimShardedEmbedding`, `ColumnParallelLinear`, or used for MLP all-reduce). The module sees this as its TP group — weight sizing, weight loading rank, and collectives all use this group.

### DP Column Group (for batch all-gather and slice)

A column of `component_dp_size` ranks at the same TP position across DP groups. Used by the decoder layer's `_dp_transition` function to gather/scatter batches between modules with different DP sizes.

### Example: TP=8, DP=4, all dp_sizes=4

```text
         TP rank:  0   1   2   3   4   5   6   7
         ─────────────────────────────────────────
  DP0:           [ 0   1   2   3   4   5   6   7 ]
  DP1:           [ 8   9  10  11  12  13  14  15 ]
  DP2:           [16  17  18  19  20  21  22  23 ]
  DP3:           [24  25  26  27  28  29  30  31 ]

Component TP group (size 32): [0,1,...,31]  — all ranks
  Used by: embedding, MLP, LM head, attention for weight sharding + all-reduce

DP Column (size 4): [0,8,16,24], [1,9,17,25], ..., [7,15,23,31]
  Used by: _dp_transition for batch all-gather/slice between modules
```

### Group Aliasing

When multiple components share the same `dp_size`, their groups are aliased (same `GroupCoordinator` object, created once). A cache in `_create_neuron_groups` ensures no duplicate groups.

### Groups Always Created

All Component TP groups and DP column groups are created unconditionally, even when `dp_size=1`. With `dp_size=1`, the Component TP group equals the regular TP group, and the DP column group is a size-1 self-group. This ensures `_dp_transition` works without conditional logic.

## Batch State Machine (Decode)

During decode, the batch dimension may be "gathered" across multiple DP ranks. The **batch gathered size** tracks how many DP ranks' batches are present in the tensor. The `_dp_transition` function handles transitions between different gathered states.

### _dp_transition

```python
def _dp_transition(x, current_group, target_group, dim=0):
    current_dp = current_group.world_size
    target_dp = target_group.world_size
    if current_dp == target_dp:
        return x                          # no-op
    if current_dp > target_dp:
        # Down: slice to keep this rank's target sub-group batches
        per_dp = x.shape[dim] // current_dp
        start = (current_group.rank_in_group // target_dp) * target_dp * per_dp
        return x.narrow(dim, start, target_dp * per_dp)
    # Up: go through local to avoid duplicates, then all-gather to target
    per_dp = x.shape[dim] // current_dp
    x = x.narrow(dim, current_group.rank_in_group * per_dp, per_dp)
    return target_group.all_gather(x, dim=dim)
```

### Decoder Layer Flow

The decoder layer has fixed internal transitions (no state threading needed):

```text
Input arrives at mlp_dp (from previous layer or embedding transition)

  _dp_transition(mlp_dp → attn_dp)
  ┌─ Self Attention ──────────────────────────────────────┐
  │  Input: gathered at attn_dp                           │
  │  Megakernel: QKV + attention + O projection           │
  │  attn_tp_group.all_reduce: sums TP+DP weight partials │
  │  Output: gathered at attn_dp                          │
  └───────────────────────────────────────────────────────┘
  residual add (both at attn_dp)
  _dp_transition(attn_dp → mlp_dp)

  ┌─ MLP ────────────────────────────────────────────────┐
  │  Input: gathered at mlp_dp                            │
  │  gate/up/down projections                             │
  │  mlp_tp_group.all_reduce: sums TP+DP weight partials  │
  │  Output: gathered at mlp_dp                           │
  └──────────────────────────────────────────────────────┘
  residual add (both at mlp_dp)

Output at mlp_dp
```

### Full Model Flow (Decode)

```text
Embedding:
  all-gather input_ids across emb_dp column        [B_local] → [B_local * emb_dp]
  VocabDimShardedEmbedding (uses emb Component TP group)
  _dp_transition(emb_dp → mlp_dp)                 transition once before layer loop

Layer 0..N (each layer):
  _dp_transition(mlp_dp → attn_dp)                no-op if equal
  Attention (uses attn Component TP group all-reduce)
  _dp_transition(attn_dp → mlp_dp)                no-op if equal
  MLP (uses mlp Component TP group all-reduce)

Norm (elementwise, state-agnostic)

LM Head:
  Slice to local batch                             [B_local * mlp_dp] → [B_local]
  index_select(sampling_positions)                 [B_local] → [B_selected]
  all-gather across lm_head_dp column              [B_selected] → [B_selected * lm_head_dp]
  ColumnParallelLinear (uses lm_head Component TP group)
  Slice to local batch                             [B_selected * lm_head_dp] → [B_selected]
```

### All dp_sizes Equal

When `attn_dp == mlp_dp == emb_dp == lm_head_dp`:

- All `_dp_transition` calls are no-ops (same group sizes)
- Embedding gathers once, data stays gathered through all layers
- Only the LM head slices back to local at the end

**Collectives per layer: 3** (attn Component TP group all-reduce + MLP Component TP group all-reduce + residual add)
vs **current without optimization: 7** (AG + RS + TP-AR + AG + slice + TP-super-AR per layer)

### Mixed Case

When dp_sizes differ, `_dp_transition` inserts the minimal collective at each boundary. Each transition goes through local (slice → all-gather) to avoid duplicated data from overlapping gathered states.

## Weight Sharding and Loading

### Embedding and LM Head

The Component TP group is passed directly as `tp_group` to `VocabDimShardedEmbedding` and `ColumnParallelLinear`. These modules automatically size their weights based on the group's `world_size` and use `rank_in_group` for sharding.

Since `load_weights` passes `tp_rank` (not the component TP group rank) to the checkpoint loader, `with_rank_override` wraps the weight loader to substitute the component TP group rank:

```python
emb_loader = sharding_weight_loader(shard_dim=0, shard_size=..., num_shards=emb_tp_size)
emb_loader = with_rank_override(emb_loader, rank=emb_tp_group.rank_in_group)
set_weight_loader(embed_tokens.weight, emb_loader)
```

### MLP

Same pattern as Embedding/LM Head — `with_rank_override` using the MLP Component TP group rank:

```python
gate_up_loader = sharding_weight_loader(shard_dim=1, shard_size=..., num_shards=mlp_tp_size)
gate_up_loader = with_rank_override(gate_up_loader, rank=mlp_tp_group.rank_in_group)
```

### Attention

Attention weight sharding is unchanged from the existing `attention_dp_size` implementation (interleaved effective rank for Q/O, separate handling for K/V).

## Attention Changes

The attention decode path was simplified as part of this work:

**Before:** Internal all-gather + reduce-scatter + TP all-reduce (3 separate collectives)
**After:** Caller provides gathered input via `_dp_transition`, attention uses single `attn_tp_group.all_reduce` (1 collective)

The attention Component TP group (`_NEURON_ATTENTION_TP`, size `TP * attn_dp`) replaces the previous two-step collective. The `all_reduce` across the component TP group sums both TP partials and DP weight-shard partials in one operation.

Internal all-to-all for Q/KV head swapping is unchanged (handled by the megakernel).

## Implementation Files

| File | Change |
|------|--------|
| `model/neuron_config.py` | `embedding_dp_size`, `lm_head_dp_size`, `mlp_dp_size` config fields |
| `parallel/neuron_parallel_state.py` | Component TP groups (`_NEURON_*_TP`) and DP column groups (`_NEURON_*_DP`) for all components; `_build_component_tp_group_ranks()` helper; `_NEURON_ATTENTION_TP` component TP group |
| `model/llama3/model.py` | `_dp_transition()` helper; `LlamaDecoderLayer` batch state transitions; `LlamaMLP` uses MLP Component TP group; `LlamaModel` embedding DP bracket; `LlamaForCausalLM` LM head DP bracket; `LlamaAttention` uses attention Component TP group all-reduce |
| `utils/executor.py` | `embedding_dp_size`, `lm_head_dp_size`, `mlp_dp_size` params threaded through |
| `vllm/worker/neuron_worker.py` | Reads new config fields and passes to `init_neuron_distributed_environment` |
