# Attention DP

<!-- meta: description: Attention data parallelism design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-07-13 -->

## Overview

Attention DP shards Q and O projection weights across `TP * attention_dp` devices instead of just TP. When KV heads exceed TP, KV weights are also sharded across attention DP. The KV cache is always batch-sharded (each DP rank stores its own batches) with full per-TP KV heads. This eliminates redundant weight copies across DP groups, reducing HBM memory and bandwidth usage.

**Decode-only.** Prefill does not use attention DP.

The NKI attention API calls this same batch partition `KVDP`. It is an internal
name for attention DP, not a separate user setting. [DCP](dcp.md) instead
shards sequence positions within a batch.

**Case study**: GQA model with Q=64, KV=8 running on 64 devices with TP=8, DP=8.

With standard independent DP: each DP group replicates all Q/O weights (8 copies of the two largest attention matrices). With attention DP=8: Q/O weights are sharded across all 64 devices. Each device holds 1 Q head and 1 O column instead of 8. Zero Q/O replication.

## Problem Statement

For GQA models with few KV heads (e.g., Q=64, KV=8), Q and O projections dominate attention weight memory (64 heads vs 8 KV heads). Standard DP replicates these weights across all DP groups:

| Component | Standard TP=8 DP=8 | Q-only a2a (TP=8 DDP=8) | Q+K+V a2a (TP=2 DDP=4) |
|-----------|-------------------|------------------------|------------------------|
| Q weight per device | 8 Q heads | 1 Q head | 8 Q heads |
| O weight per device | 8 O columns | 1 O column | 8 O columns |
| KV weight per device | 1 KV head | 1 KV head (unchanged) | 1 KV head (sharded) |
| KV cache per device | own batch | own batch | own batch, full TP heads |
| Q/O copies across DP | 8x replicated | 1x (zero) | 1x (zero) |
| KV weight copies | 8x replicated | 8x replicated | 1x (zero) |

Attention DP eliminates this redundancy at the cost of additional collectives (all-gather, all-to-all, reduce-scatter) during decode.

## Configuration

```python
NeuronConfig(attention_dp_size=4)
```

- `attention_dp_size=1`: disabled (standard independent DP)
- `attention_dp_size=N`: Q/O sharded across `TP * N` devices
- Constraint: `attention_dp_size` must divide `dp_size`
- Constraint: `TP * attention_dp_size` must not exceed `num_attention_heads`

### Two Variants (Auto-Detected)

The code auto-detects which variant to use based on `num_kv_heads` vs `tp_size`:

| Condition | Variant | What's sharded across attention DP | KV flow |
|-----------|---------|-----------------------------------|---------|
| `num_kv_heads <= TP` | Q-only a2a | Q and O weights | K/V sliced to local batch |
| `num_kv_heads > TP` and `num_kv_heads % (TP * attention_dp) == 0` | Q+K+V a2a | Q, K, V, and O weights | K/V also all-to-all'd |

In the Q+K+V variant, KV weights are sharded to `num_kv_heads / (TP * attention_dp)` per rank. After all-to-all, each device has `num_kv_heads / TP` KV heads (the standard per-TP amount). The KV cache stores the gathered heads so attention runs locally without gathering prior context every step.

## Process Groups

### Full attention DP (`attention_dp_size == dp_size`): Zero Q/O Replication

```bash
# TP=8, DP=4, attention DP=4 — 32 ranks, one supergroup
```

```text
         TP rank:  0   1   2   3   4   5   6   7
         ─────────────────────────────────────────
  DP0:           [ 0   1   2   3   4   5   6   7 ]
  DP1:           [ 8   9  10  11  12  13  14  15 ]
  DP2:           [16  17  18  19  20  21  22  23 ]
  DP3:           [24  25  26  27  28  29  30  31 ]
                  │                           │
                 attention DP columns (8 groups, one per TP position)
                 [0,8,16,24]  ...  [7,15,23,31]
```

All 4 DP groups form one supergroup. Q/O weights are sharded across all `TP * attention DP = 32` devices with zero replication.

### Partial attention DP (`attention_dp_size < dp_size`): Multiple Supergroups

```bash
# TP=8, DP=8, attention DP=4 — 64 ranks, 2 independent supergroups of 32
```

```text
         TP rank:  0   1   2   3   4   5   6   7
         ─────────────────────────────────────────
Supergroup 0:
  DP0:           [ 0   1   2   3   4   5   6   7 ]
  DP1:           [ 8   9  10  11  12  13  14  15 ]
  DP2:           [16  17  18  19  20  21  22  23 ]
  DP3:           [24  25  26  27  28  29  30  31 ]
                  │                           │
                 [0,8,16,24]  ...  [7,15,23,31]

─ ─ ─ ─ ─ No communication across this boundary ─ ─ ─ ─ ─

Supergroup 1:
  DP4:           [32  33  34  35  36  37  38  39 ]
  DP5:           [40  41  42  43  44  45  46  47 ]
  DP6:           [48  49  50  51  52  53  54  55 ]
  DP7:           [56  57  58  59  60  61  62  63 ]
                  │                           │
                 [32,40,48,56]  ...  [39,47,55,63]
```

Q/O weights are sharded within each supergroup (`TP * attention DP = 32` devices) and replicated 2x across the two supergroups. Rank 0 and rank 32 load the same Q head.

### Summary

| Group | Type | Ranks (example for rank 0) | Collectives | Where |
|-------|------|----------------------------|-------------|-------|
| **Attention TP Supergroup** | `GroupCoordinator` | `[0,1,...,31]` (TP * attention_dp ranks) | `all_reduce` after O projection (sums both TP and DP weight partials) | Model code |
| **attention DP column** | `GroupCoordinator` | `[0,8,16,24]` (one per TP position, within supergroup) | Batch `all_gather` / `slice` via `_dp_transition` at decoder layer boundaries; `all_to_all` Q before/after attention | Model code + torch fallback |

> **Note:** The attention component TP group replaces the previous two-step collective (reduce-scatter across DP column + all-reduce across TP). A single `all_reduce` across the supergroup achieves the same result. The DP column group is now used only for batch transitions via `_dp_transition`, not for the O-projection collective. See [Component-Level DP Sharding](component_dp_sharding.md) for details on the batch state machine.

## Decode-Only Flow

attention DP is decode-only. Prefill uses standard TP with no attention DP awareness.

### Per-Layer Decode Flow

```text
Caller (_dp_transition) ensures input is gathered to attn_dp:
  X: [DDP*B_local, S_tkg, H]

Step 1 — Fused QKV projection on gathered X:
  [DDP*B_local, S_tkg, H] @ W_qkv → Q, K, V for all DDP batches
  Q: num_q_heads / (TP * DDP) heads per rank
  K, V: num_kv_heads / (TP * DDP) heads if kv_needs_a2a, else num_kv_heads / TP

Step 2 — All-to-all Q across attention DP column group:
  Swap "few heads x many batches" → "many heads x own batch"
  Q: [DDP*B_local, q_small, S, d] → [B_local, q_standard, S, d]

Step 2b — (if kv_needs_a2a) All-to-all K, V across attention DP column group:
  K: [DDP*B_local, kv_small, S, d] → [B_local, kv_standard, S, d]
  V: same as K

Step 2c — (if !kv_needs_a2a) Select local K, V:
  K = K[local_batch_slice]
  V = V[local_batch_slice]

Step 3 — RoPE on local Q and K:
  Uses local batch's cos/sin only. No cos/sin gathering needed.

Step 4 — Standard attention (completely standard, no attention DP awareness):
  Q (all heads, own batch) x KV cache (own batch) → attention output
  Block KV cache, GQA expansion, softmax — all unchanged.

Step 5 — Reverse all-to-all across attention DP column group:
  Swap "many heads x own batch" → "few heads x many batches"

Step 6 — O projection on all DDP batches:
  [DDP*B_local, S, q_small * d] @ W_o → [DDP*B_local, S, H]

Step 7 — Attention TP component TP group all-reduce:
  Sums both TP weight partials and DP weight-shard partials in one collective.
  [DDP*B_local, S, H] → [DDP*B_local, S, H]
  Output stays gathered at attn_dp.

Caller (_dp_transition) handles any transition to the next module's dp_size.
```

> **Change from previous design:** Steps 1 (all-gather) and 8-9 (reduce-scatter + TP all-reduce) were replaced. The caller now handles batch gathering via `_dp_transition`, and the two output collectives were merged into a single component TP group all-reduce. See [Component-Level DP Sharding](component_dp_sharding.md) for the batch state machine design.

## Weight Sharding

QKV weight stays fused as a single tensor. Weight loading uses effective rank:

```text
effective_rank = attention_dp_rank + tp_rank * attention_dp_size
```

### Q-only a2a variant (KV fits in TP)

```text
Example: Q=64, KV=8, TP=8, attention_dp=8
  q_size  = (64/64) * d = 1d   ← 1 Q head per rank (sharded across TP*DDP)
  kv_size = (8/8)   * d = 1d   ← 1 KV head per rank (sharded across TP only)
```

KV weights loaded by TP rank. Q/O loaded by effective rank.

### Q+K+V a2a variant (KV exceeds TP)

```text
Example: Q=32, KV=8, TP=2, attention_dp=4
  q_size  = (32/8) * d = 4d    ← 4 Q heads per rank (sharded across TP*DDP)
  kv_size = (8/8)  * d = 1d    ← 1 KV head per rank (sharded across TP*DDP)
```

Both Q and KV weights loaded by effective rank. The KV cache stores
`num_kv_heads / TP` heads (the gathered amount after a2a), not the
sharded per-rank amount.

O projection is always sharded across TP*DDP:

```text
o_proj_weight: [q_heads_per_rank * d, H]
  q_heads_per_rank = num_q_heads / (TP * attention_dp)
```

## Implementation Files

| File | Change |
|------|--------|
| `model/neuron_config.py` | `attention_dp_size` config field |
| `parallel/neuron_parallel_state.py` | `_NEURON_ATTENTION_DP` (DP column), `_NEURON_ATTENTION_TP` (component TP group), getters |
| `model/llama3/model.py` | `LlamaAttention` attention DP-aware `__init__` (weight shapes, loaders) and `forward_decode` (component TP group all-reduce); `LlamaDecoderLayer` batch transitions via `_dp_transition` |
| `functional/attention/attention_decode.py` | Torch fallback attention DP flow (all-to-all Q, local K/V selection, reverse all-to-all) |
| `utils/executor.py` | `attention_dp_size` param for test infrastructure |

## KV Cache

The KV cache layout is the same in both variants — each rank stores `num_kv_heads / TP` heads for its local batch (standard TP+DP layout). Attention DP does not change cache sharding:

```text
Cache per rank: [num_blocks, num_kv_heads / TP, block_len, d_head]
```

The two variants differ in how newly projected K/V tokens reach the cache:

```text
Q-only a2a (TP=8, DDP=8, KV=8):
  Weight projects: 1 KV head (= num_kv / TP)
  Cache stores:    1 KV head
  Flow: project → slice to local batch → write to cache (no KV collective)

Q+K+V a2a (TP=2, DDP=4, KV=8):
  Weight projects: 1 KV head (= num_kv / (TP * DDP))
  Cache stores:    4 KV heads (= num_kv / TP)
  Flow: project 1 head → a2a gathers to 4 heads → write 4 heads to cache
```

In the Q+K+V variant, the weight projects fewer heads than the cache stores. The a2a gathers newly projected tokens (S_tkg, typically 1) to the full per-TP head count before writing to cache. Prior context already in the cache reads locally — only new tokens go through the a2a each step.

## Constraints

- **Decode-only**: Prefill is not modified. With attention DP-sharded weights, prefill would need weight gathering or a separate flow.
- **Q head count**: `TP * attention_dp` must not exceed `num_attention_heads`.
- **KV divisibility**: For Q+K+V a2a, `num_kv_heads` must be divisible by `TP * attention_dp`. When it's not, KV falls back to TP-only sharding (Q-only a2a variant).
- **DP divisibility**: `attention_dp_size` must divide `dp_size` (for partial supergroups).
- **KV cache**: Always stores `num_kv_heads / TP` heads per rank (post-gather), regardless of variant. Weight sharding saves weight memory, not cache memory. Cache memory is reduced by batch-sharding (each DP rank stores fewer batches).

## When to Use Attention DP

- Models with large Q/O and small K/V (GQA with high Q:KV ratio) — eliminates redundant Q/O weight copies across DP groups.
- Models with large KV heads (small GQA ratio) where you want to shard KV weights while keeping the cache TP-sharded — the Q+K+V a2a variant shards KV weights across attention DP while the cache retains the standard TP head sharding.
- When KV optimizations like sliding window are used (Q/O weight loads dominate perf).
- Better option than DCP when concurrency is large (avoids additional DCP collectives).
- Required synchronization already exists for MoE models with EP across DP.
