# Encoder Parallelism

<!-- meta: description: Encoder parallelism design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

The encoder supports independent TP and DP configurations, decoupled from the text model's parallelism. Encoder TP shards encoder weights across ranks; Encoder DP scatters encoder blocks across ranks for independent processing, then all-gathers after the merger.

**Prefill-only.** The encoder runs during prefill. Decode steps do not invoke the encoder.

**Block-level DP.** Unlike text model DP which shards by batch, encoder DP shards by the `num_blocks` dimension of the block-packed encoder inputs. Each DP rank processes `num_blocks / dp_size` blocks independently, then results are gathered.

## Configuration

```python
EncoderNeuronConfig(
    tp_size=1,    # Encoder TP degree (default: 1, no weight sharding)
    dp_size=4,    # Encoder DP degree (default: inferred as world_size // tp_size)
)
```

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `tp_size` | Weight sharding across TP ranks | 1 |
| `dp_size` | Block scatter/gather across DP ranks | Inferred: `world_size // tp_size` |

**Constraints:**

- `tp_size` must be a positive integer that divides `world_size`
- `tp_size * dp_size == world_size` (all ranks must participate — no redundant compute)
- `num_blocks` is padded to be divisible by `dp_size` for even scatter

**Resolution rules** (`resolve_tp_dp`):

- Only `tp_size` set → `dp_size` inferred as `world_size // tp_size`
- Neither set (defaults `tp_size=1, dp_size=1`) → inferred to `tp_size=1, dp_size=world_size` (full DP)
- Both set explicitly → validated that `tp_size * dp_size == world_size`

`resolve_tp_dp` is called both at worker init (for process group creation) and at model runner init (for warmup and runtime block padding). It writes back the resolved `dp_size` to the config instance so downstream callers see the correct value.

## Modes

| User config | Resolved (world_size=16) | Behavior |
|-------------|--------------------------|----------|
| (default) | tp=1, dp=16 | Full DP — each rank holds full weights, processes `num_blocks / 16` blocks |
| `tp_size=4` | tp=4, dp=4 | Combined — weights sharded across 4 TP ranks, blocks scattered across 4 DP ranks |
| `tp_size=16` | tp=16, dp=1 | Full TP — weights sharded across 16 ranks, all ranks process all blocks |
| `tp_size=4, dp_size=4` | tp=4, dp=4 | Same as specifying only `tp_size=4` (explicit dp matches inferred) |

## Process Groups

### Encoder TP group

Created when `encoder_tp_size != text_tp_size`. Each group contains `encoder_tp_size` contiguous ranks.

```text
Example: world_size=16, encoder_tp_size=4

  Encoder TP groups (4 groups of 4):
    [0,1,2,3]  [4,5,6,7]  [8,9,10,11]  [12,13,14,15]
```

When `encoder_tp_size == text_tp_size`, the text TP group is reused (no new group created).

### Encoder DP group

Created when `encoder_dp_size > 1`. Column groups built via `_build_dp_column_group_ranks(world_size, encoder_tp_size, encoder_dp_size)` — ranks at the same TP position across DP replicas.

```text
Example: world_size=16, encoder_tp_size=4, encoder_dp_size=4

  Encoder TP groups:  [0,1,2,3]  [4,5,6,7]  [8,9,10,11]  [12,13,14,15]
                       │ │ │ │    │ │ │ │    │ │ │ │       │ │ │ │
  Encoder DP columns: [0,4,8,12] [1,5,9,13] [2,6,10,14]  [3,7,11,15]
```

```text
Example: world_size=8, encoder_tp_size=1, encoder_dp_size=4 (full DP, no weight sharding)

  Encoder TP groups:  [0] [1] [2] [3] [4] [5] [6] [7]   (each rank = own TP group)
  Encoder DP columns: [0,1,2,3]  [4,5,6,7]                (4 ranks per DP group)
```

### Constraint: tp_size * dp_size == world_size

Encoder TP and DP must fully utilize all ranks — no redundant compute. At runtime, `resolve_tp_dp` enforces `tp_size * dp_size == world_size`. If the user specifies only `tp_size`, `dp_size` is inferred as `world_size // tp_size`. If neither is specified, defaults to full DP (`tp_size=1, dp_size=world_size`).

## Forward Flow

```text
Input: pixel_values [num_blocks, block_size, patch_dim]
       (+ pos_emb_idx, pos_emb_weight, cos, sin, bound_min, bound_max, unpack_indices)

Step 0 — DP scatter (if dp_size > 1):
  blocks_per_rank = num_blocks // dp_size
  Each rank slices its portion: [start:end] along dim 0
  pixel_values: [num_blocks, ...] → [blocks_per_rank, ...]

Step 1 — Patch embedding + position embedding:
  pixel_values → hidden_states [blocks_per_rank, block_size, hidden]

Step 2 — Transformer blocks + deepstack extraction:
  For each layer: attention + MLP (TP collectives within encoder TP group)
  At deepstack layers: extract merged features

Step 3 — Final merger:
  hidden_states → main_merged [blocks_per_rank, merged_bs, D]

Step 4 — DP gather (if dp_size > 1):
  all_gather(main_merged, dim=0) across encoder DP group
  main_merged: [blocks_per_rank, ...] → [num_blocks, merged_bs, D]
  Same for deepstack features.

Step 5 — Unpack to per-image order:
  index_select with unpack_indices → [merged_seq_len, D]
  (Operates on full gathered tensor — same on all ranks)

Step 6 — Construct fat tensor (main + deepstack):
  Concatenate main and deepstack features → final output
```

## Block Padding

`num_blocks` is padded to be divisible by `dp_size` at three points:

1. **Config `__post_init__`**: `num_total_encoder_attention_blocks` padded for compiled graph shapes
2. **`select_vision_bucket`**: runtime bucket selection pads `num_blocks`
3. **Encoder warmup** (`neuron_worker.py`): warmup shapes padded

Padding blocks contain zeros and produce zero outputs. The `unpack_indices` tensor ensures only valid merged tokens are selected in the final output.

## Weight Loading

Encoder DP does not affect weight loading — each DP rank holds a **full copy** of the encoder weights (dp_size > 1 reduces activation memory, not weight memory).

Encoder TP shards weights across the encoder TP group using the same column/row parallel patterns as the text model, but scoped to the encoder TP group.

## When to Use Block-Level DP

- **Large items / many items per request**: High `num_blocks` makes the per-rank activation memory the bottleneck. DP reduces activation memory linearly with `dp_size`.
- **Small encoder**: When the encoder fits in a single device's weight memory, TP provides no benefit. Full DP (`tp_size=1`) avoids TP communication overhead entirely.
- **Combined TP+DP**: For very large encoders that don't fit on one device AND have high block counts, use both.

## Cross-request batching

The scheduler batches images from multiple requests into a single `embed_multimodal` call by using `group_and_batch_mm_kwargs`. Block-level DP therefore benefits from concurrent request load: more pending images produce more blocks and improve DP utilization.
