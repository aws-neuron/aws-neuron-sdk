# DCP (Decode Context Parallelism)

<!-- meta: description: Decode Context Parallelism design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-07-13 -->

## Overview

DCP shards the KV cache sequence dimension across ranks, reducing per-rank memory for long contexts. Two modes:

1. **DCP Prefill** (`apply_prefill_dcp=True` and DCP size greater than one): Replicates attention weights across DCP sub-groups and shards the input sequence during prefill. Each DCP rank processes different token chunks.
2. **DCP Decode** (DCP size greater than one): Shards the KV cache within the DCP replica set during decode. It gathers Q across DCP peers, computes attention against each local KV shard, and combines the partial results with LSE correction.

## DCP Prefill

### Configuration

```bash
vllm serve <model> \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 4 \
    --cp-kv-cache-interleave-size 16 \
    --additional-config '{"neuron_config": {"apply_prefill_dcp": true}}'
```

### Constraints

- `TP % DCP == 0`
- `apply_prefill_dcp` requires DI (kv_role=kv_producer) and DCP > 1
- `cp_kv_cache_interleave_size` must equal `block_size` for DI compatibility

### Process Groups (TP=8, DCP=4)

```text
Full TP group: [0, 1, 2, 3, 4, 5, 6, 7]  (size 8)

dcp_tp_group (size TP/DCP = 2):
  [0, 1], [2, 3], [4, 5], [6, 7]
  → SP gather/scatter, QKV/O weight sharding

cp_kv_group (size DCP = 4):
  [0, 2, 4, 6], [1, 3, 5, 7]
  → KV AllGather (same heads, different tokens)

cp_rank = tp_rank // (TP/DCP):
  Rank 0,1 → cp_rank 0 (tokens in blocks 0,4,8,...)
  Rank 2,3 → cp_rank 1 (tokens in blocks 1,5,9,...)
  Rank 4,5 → cp_rank 2 (tokens in blocks 2,6,10,...)
  Rank 6,7 → cp_rank 3 (tokens in blocks 3,7,11,...)
```

### Compute Pattern

```text
1. SP All-Gather (world_group): S/TP → S tokens
2. Interleave slice to owned positions: S → S/DCP tokens
3. QKV Projection (weights sharded across dcp_tp_group)
4. KV Cache Write (only local owned positions via slot_mapping)
5. Q AllGather (cp_kv_group): Q [Nh_q, S/DCP] → [Nh_q, S]
6. Unshuffle Q to global position order
7. Segmented attention: full_Q × local_KV (prior cache + current) with LSE correction
8. ReduceScatter (cp_kv_group): output [Nh_q, S] → [Nh_q, S/DCP]
9. O Projection + Reduce-Scatter (dcp_tp_group): S/DCP → S/TP
```

### Weight Loading

| Component | Sharding | Rank |
|-----------|----------|------|
| QKV, O | `dcp_tp_group.world_size` (TP/DCP) | `dcp_tp_group.rank_in_group` |
| MLP, Embedding, LM Head | Full TP | Full TP rank |

## DCP Decode

### Decode Configuration

```bash
vllm serve <model> \
    --tensor-parallel-size 16 \
    --decode-context-parallel-size 2 \
    --cp-kv-cache-interleave-size 16
```

### Decode Constraints

- `tp > num_kv_heads` (KV must be replicated across DCP group)
- `dcp <= tp // num_kv_heads`
- `(num_q_heads // num_kv_heads) % dcp == 0`
- `tp % dcp == 0`
- DCP cannot currently be combined with attention DP

### Process Groups (TP=16, DCP=2)

In this example, vLLM creates consecutive DCP rank pairs:
`[0,1], [2,3], ..., [14,15]`. Each pair shares the same replicated KV head and
splits the sequence.

```text
dcp_rank = tp_rank % dcp_size
  Rank 0 → dcp_rank 0 (even blocks)
  Rank 1 → dcp_rank 1 (odd blocks)
```

### Decode Compute Pattern

```text
1. QKV Projection (weights sharded by full TP)
2. AllGather Q across DCP group: q_heads_local → q_heads_local * dcp_size
3. Local attention: gathered Q against local KV shard (S/DCP tokens)
4. Extract partial LSE (logsumexp of local scores)
5. AllGather LSE across DCP group
6. Compute correction: weight_i = exp(local_lse - global_lse)
7. Apply correction to local attention output
8. ReduceScatter across DCP group: combine corrected outputs, scatter heads back
```

### local_filled Computation

The attention mask uses `local_filled_slots` to determine how many prior tokens
are in this rank's KV cache shard:

```python
vblock = block_size * dcp_size
stride = interleave_size * dcp_size
rank_start = dcp_rank * interleave_size

local_filled = floor(N / vblock) * block_size
remaining = N - floor(N / vblock) * vblock
local_filled += floor(remaining / stride) * interleave_size
leftover = remaining - floor(remaining / stride) * stride
local_filled += clamp(clamp(leftover - rank_start, min=0), max=interleave_size)
```

### Active Token Handling and dcp_active_mask

During decode, all DCP ranks project the same input hidden states and produce
the same K/V for the active token. The cache write uses `slot_mapping`:

- **Owning rank** (`slot_mapping >= 0`): writes to the correct cache slot.
- **Non-owning ranks** (`slot_mapping = -1`): writes to a garbage slot (last block).

After AllGather Q, each rank computes attention over its local KV shard.
Without masking, every rank would include the active token's K/V in its
local attention — the LSE correction would then count the active token
`dcp_size` times (once per rank).

`dcp_active_mask` prevents this. The attention mask has two parts:

- `prior_mask`: slots `[0..local_filled)` — prior tokens this rank owns
- `active_slots`: the current decode token's slot, gated by `dcp_active_mask`

Only the owning rank sets `dcp_active_mask = 1`; others set it to `0`.
This ensures the active token contributes exactly once to the combined output.

```python
pos_in_vblock = N - floor(N / vblock) * vblock
owner = floor(pos_in_vblock / interleave_size) % dcp_size
dcp_active_mask = (owner == dcp_rank).float()
```

## cp_rank: Prefill vs Decode

Both modes use the same `--decode-context-parallel-size` flag and the same
slot mapping function, but the **group topology differs**, producing different
`cp_rank` formulas:

### DCP Prefill: `cp_rank = tp_rank // (TP / DCP)`

DCP sub-divides TP into `DCP` token groups of `TP/DCP` consecutive ranks each.
Ranks within the same group share the same tokens but have different heads.

```text
TP=8, DCP=4:
  Group 0: ranks [0, 1]  → cp_rank 0 (token group 0)
  Group 1: ranks [2, 3]  → cp_rank 1 (token group 1)
  Group 2: ranks [4, 5]  → cp_rank 2 (token group 2)
  Group 3: ranks [6, 7]  → cp_rank 3 (token group 3)

Semantic: "which token chunk do I own?"
All ranks with the same cp_rank store identical token positions (different heads).
```

### DCP Decode: `cp_rank = tp_rank % DCP`

DCP forms `TP/DCP` independent replica sets of `DCP` consecutive ranks each.
Ranks within the same set share the same (replicated) KV heads but own different tokens.

```text
TP=16, DCP=2:
  Replica set 0: ranks [0, 1]   → cp_rank 0, 1
  Replica set 1: ranks [2, 3]   → cp_rank 0, 1
  ...
  Replica set 7: ranks [14, 15] → cp_rank 0, 1

Semantic: "which interleave position do I own within my replica set?"
cp_rank 0 owns even blocks, cp_rank 1 owns odd blocks.
```

### Why the Difference

In prefill, DCP replicates weights and shards sequence — the "group" is the set
of ranks that process different heads for the same tokens. The group index
(cp_rank) identifies the token chunk.

In decode, DCP replicates KV heads and shards sequence — the "group" is the
replica set where each rank owns a different sequence slice. The position within
the group (cp_rank) identifies which interleave slice.

Both formulas produce the correct `cp_rank` for `_compute_slot_mapping_cpu`:
the slot mapping assigns positions where
`(pos_in_vblock // interleave_size) % cp_world_size == cp_rank` to the local
rank, and `-1` to all others.

## Slot Mapping and KV Cache

Both DCP prefill and decode use CP-aware slot mapping:

```python
_compute_slot_mapping_cpu(
    block_table, slot_mapping, positions, req_indices,
    block_size,
    cp_world_size=dcp_size,
    cp_rank=cp_rank,
    cp_kv_cache_interleave_size=interleave_size,
)
```

With `interleave_size=block_size=16` and DCP=4:

- Block 0 (positions 0-15) → cp_rank 0
- Block 1 (positions 16-31) → cp_rank 1
- Block 2 (positions 32-47) → cp_rank 2
- Block 3 (positions 48-63) → cp_rank 3
- Block 4 (positions 64-79) → cp_rank 0 (wraps)

Non-local positions get `slot_mapping = -1`.

## Disaggregated Inference (NIXL Transfer)

### NeuronNixlConnector

The NIXL connector is `NeuronNixlConnector`, a standalone connector specified
in `--kv-transfer-config`:

```json
"kv_connector": "NeuronNixlConnector"
```

The platform auto-injects `kv_connector_module_path` — users only set the name.
Required when using DCP with DI (validated at startup).

Internally uses `NeuronNixlConnectorWorker` (subclass of `NixlConnectorWorker`)
with DCP-specific overrides:

- `register_kv_caches`: Stores local DCP rank/size; on prefill, re-encodes metadata with `NeuronNixlAgentMetadata` (includes `dcp_size` field); patches `tp_ratio` for match case
- `_validate_remote_agent_handshake`: Skips strict validation for DCP prefill engines
- `add_remote_agent`: Routes to split (head offset), merge (smaller descriptors), or match (passthrough)
- `_nixl_handshake`: Connects to ALL remote ranks when TP differs
- `_read_blocks_for_req`: Routes to DCP-specific block reading logic

### NeuronNixlAgentMetadata

Extends `NixlAgentMetadata` with a `dcp_size: int = 0` field. The prefill
advertises its DCP degree in the handshake metadata. The decode reads it during
detection to compute correct remote rank mappings.

### Transfer Topologies

#### DCP Prefill → Standard Decode

Each decode rank reads from `P_DCP` prefill ranks (all cp_ranks with matching head):

```text
P:TP=4/DCP=2 → D:TP=2:
  Decode rank 0 (heads 0-3) ← Prefill ranks [0, 2] (cp_ranks 0,1, heads 0-3)
  Decode rank 1 (heads 4-7) ← Prefill ranks [1, 3] (cp_ranks 0,1, heads 4-7)
```

Local blocks are interleaved across remote ranks by token group.

#### DCP Prefill → DCP Decode (Same DCP Degree)

1:1 cp_rank mapping. Each decode dcp_rank reads from exactly one prefill cp_rank:

```text
P:TP=32/DCP=2 → D:TP=16/DCP=2:
  Decode rank 0 (dcp_rank=0) ← Prefill rank 0 (cp_rank=0)
  Decode rank 1 (dcp_rank=1) ← Prefill rank 16 (cp_rank=1)
```

#### DCP Prefill → DCP Decode (Different DCP)

When P_DCP > D_DCP, each decode dcp_rank reads from P_DCP/D_DCP prefill cp_ranks:

```text
P:TP=32/DCP=4 → D:TP=16/DCP=2:
  Decode rank 0 (dcp_rank=0) ← Prefill ranks [0, 16] (cp_ranks 0,2)
  Decode rank 1 (dcp_rank=1) ← Prefill ranks [8, 24] (cp_ranks 1,3)
```

Filter formula: decode dcp_rank `r` reads from prefill cp_ranks `{r, r+D_DCP, r+2*D_DCP, ...}`.

#### Standard Prefill → DCP Decode

When prefill has no DCP (standard TP, same TP size), the decode filters
remote blocks to its interleaved subset:

```text
P:TP=16 → D:TP=16/DCP=2:
  Decode rank 0 (dcp_rank=0): keeps even-indexed remote blocks
  Decode rank 1 (dcp_rank=1): keeps odd-indexed remote blocks
```

### Head Splitting / Merging

| Topology | Condition | Action |
|----------|-----------|--------|
| Matching | `remote_block_len == local_block_len` | No offset, standard interleave |
| Splitting | `remote_block_len > local_block_len` | Head offset on remote descriptors |
| Merging | `remote_block_len < local_block_len` | Custom remote descriptors, split local handles |

### Remote Rank Computation

```python
# num_target_ranks = tp_ratio * head_ratio (accounts for all topologies)
tp_ratio = P_TP // D_TP
head_ratio = max(1, remote_block_len // local_block_len)
num_target_ranks = tp_ratio * head_ratio

if inverse_head_ratio > 1:
    # Merge: multiple head positions × cp_ranks
    p_dcp = num_target_ranks // inverse_head_ratio
    tp_pair_size = P_TP // p_dcp
    remote_ranks = [ep + cp * tp_pair_size
                    for hp in range(inverse_head_ratio)
                    for cp in range(p_dcp)
                    for ep in [decode_tp_rank * inverse_head_ratio + hp]]
else:
    # Split/Match: one head position × all cp_ranks
    stride = P_TP // num_target_ranks
    effective_pos = decode_tp_rank * (P_TP // num_target_ranks) // D_TP
    remote_ranks = [effective_pos + i * stride for i in range(num_target_ranks)]

# Filter for decode DCP alignment
if D_DCP > 1:
    remote_ranks = [r for i, r in enumerate(remote_ranks) if i % D_DCP == dcp_rank]
```

### Notification Handling

When a decode DCP rank has no blocks to transfer (short sequence), it still
sends a notification to the prefill rank via `_read_blocks(local_block_ids=[])`.
If no transfers are issued at all, an empty entry is added to `_recving_transfers`
so `get_finished()` can complete the request.

## Model Runner (neuron_model_runner.py)

### Slot Mapping for DCP Prefill

The model receives `S/DCP` owned tokens after the interleave slice in
`forward`. The slot_mapping is extracted from the full `S`-sized mapping
by selecting owned positions using the same interleave pattern:

```python
slot_mapping = slot_mapping.cpu().view(S // (W * I), W, I)[:, R, :].contiguous().reshape(-1).to(device)
```

This runs in `_build_attention_metadata` (runtime) and `_build_warmup_attention_metadata`
(warmup) to ensure consistent shapes. The output is always `S/DCP` entries.

## Segmented Attention CP (segmented_attention_cp)

### Purpose

Handles DCP prefill attention for both first-chunk and multi-chunk (chunked
prefill with prior). Each rank has:

- Full Q (AllGathered, unshuffled to global order)
- Local KV: prior from cache (`S_prior/DCP` tokens) + current from projection (`S_chunk/DCP` tokens)

### API

```python
segmented_attention_cp(
    q,              # [Nh_q, S_total, Dh] — AllGathered Q
    k_local,        # [Nh_kv, S_local, Dh] — current chunk's local KV
    v_local,
    k_cache,        # paged prior cache
    v_cache,
    block_tables,   # [1, max_blocks_per_seq]
    prior_tokens,   # [[local_prior_count]] (0 on first chunk)
    block_size,
    cp_rank, cp_world_size, cp_kv_cache_interleave_size,
    cp_group,       # for AllGather LSE + ReduceScatter
    scale, tp_q, tp_out,
)
```

### Algorithm

1. Read prior KV from cache (static padded shape, validity masked by slot index)
2. Concatenate prior + current local KV
3. Build causal mask using actual global positions:
   - Prior slot `s` → global pos `(s // I) * (W * I) + R * I + (s % I)`
   - Current token `j` → global pos `prior_global + (j // I) * (W * I) + R * I + (j % I)`
   - Q position `i` → global pos `prior_global + i`
4. Compute partial attention (Q × local_KV) + extract LSE
5. AllGather LSE across cp_group → global LSE
6. Weight local output by `exp(local_lse - global_lse)`
7. ReduceScatter weighted output (dim=1) → each rank gets its `S/DCP` output slice

When `prior_tokens = 0`, all prior slots are masked out and contribute nothing.
No dynamic branches — the mask handles it statically.
