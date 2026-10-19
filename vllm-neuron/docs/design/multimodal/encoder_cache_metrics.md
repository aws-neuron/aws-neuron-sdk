# Encoder Cache Metrics

## Overview

This document explains the three cache layers involved in multimodal (vision+text)
inference and when each Prometheus metric fires.

## Cache Layers

| Layer | Metrics | Location | What it caches |
|-------|---------|----------|----------------|
| 1. Input processor | `vllm:mm_cache_hits`, `vllm:mm_cache_queries` | CPU (renderer/P0) | Raw image → pixel tensors + metadata |
| 2a. Scheduler encoder cache | `vllm_neuron:scheduler_encoder_cache_queries`, `vllm_neuron:scheduler_encoder_cache_hits` | CPU (scheduler) | Logical "has this image been encoded?" (`EncoderCacheManager`) |
| 2b. Prefix cache (KV) | `vllm:prefix_cache_hits`, `vllm:prefix_cache_queries` | Device HBM | Computed KV states for all token positions |
| 3. On-device encoder cache | `vllm_neuron:device_encoder_cache_queries`, `..._misses`, `..._evictions`, + occupancy gauges (`..._active_blocks` / `..._free_blocks` / `..._held_blocks` / `..._active_real_tokens`) | Device HBM | Encoder output embeddings (post encoder) |

The encoder cache has **two observable boundaries**:

- **Scheduler (Layer 2a)** — `EncoderCacheManager.check_and_update_cache()` is
  the logical boundary. A **hit** here means an image was already encoded, so
  the scheduler skips re-scheduling the encoder. This is where genuine
  hits are visible — e.g. a multi-turn image re-sent in a *new* turn (outside
  the prefix-cached region) is recognized and not re-encoded. Instrumented via
  a wrapper on the inherited manager in `NeuronScheduler`.
- **Worker / device (Layer 3)** — `_execute_mm_encoder` follows the same
  `queries` / `misses` shape as `mm_cache` / `prefix_cache`, so **hits =
  queries − misses**. There is no explicit device-hit counter because, in
  normal operation, device hits are always 0: the scheduler (2a) only routes
  items it does NOT already have cached to the worker, so every item the worker
  queries is a miss (`queries == misses`). A device hit only occurs under
  concurrent-request races or scheduler/worker hold-time divergence.

Note: an image whose tokens fall entirely inside the prefix-cache (2b) hit
region is skipped even earlier — the scheduler neither queries 2a nor schedules
encoding, because the embedding is already baked into the cached KV.

## Data Flow

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT REQUEST                                 │
│                   (images + text via chat API)                          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: Input Processor (CPU, renderer)                               │
│                                                                         │
│  Processes ALL images in the message array regardless of prefix cache.  │
│  hash(image_bytes) → lookup in mm_processor_cache                       │
│  Every lookup → vllm:mm_cache_queries++ (counts hits AND misses)        │
│                                                                         │
│  ┌─── MM CACHE HIT ─────┐   ┌────────── MM CACHE MISS ───────────────┐  │
│  │ Return cached pixel  │   │ Decode PNG → Resize → Normalize →      │  │
│  │ tensors              │   │ Patch → Store cache                    │  │
│  │ vllm:mm_cache_hits++ │   │ (no hit increment)                     │  │
│  └──────────┬───────────┘   └────────────────────┬───────────────────┘  │
│             │                                    │                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ pixel tensors + token IDs
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: Scheduler (CPU, engine core)                                  │
│                                                                         │
│  For each token position in the request:                                │
│                                                                         │
│  Prefix-cache block hash = hash(parent_block_hash, block_token_ids,     │
│    extra_keys) where extra_keys folds in each image's (mm_hash,         │
│    intra-block offset). So a block only hits when the token IDs AND     │
│    the image identity AND its position all match — same-size but        │
│    different images (or same image at a different position) MISS.       │
│  Every block lookup → vllm:prefix_cache_queries++ (hits AND misses)     │
│                                                                         │
│  ┌─── PREFIX CACHE HIT ───────────┐  ┌─── PREFIX CACHE MISS ──────────┐ │
│  │ KV already computed. Skip ALL  │  │ Need to compute KV for these   │ │
│  │ compute (text + image). An     │  │ tokens. Schedule for prefill.  │ │
│  │ image inside the hit region    │  │                                │ │
│  │ needs NO encode — embed is     │  │ (no hit increment)             │ │
│  │ baked into the KV.             │  │                                │ │
│  │ vllm:prefix_cache_hits++       │  │                                │ │
│  └──────────────┬─────────────────┘  └────────────────┬───────────────┘ │
│                 │                                     │                 │
│                                                       │                 │
│  For image tokens in MISS region only (Layer 2a):     │                 │
│  check_and_update_cache → scheduler_encoder_cache_queries++             │
│                                                       │                 │
│  ┌─── CACHED (scheduler hit) ─────┐  ┌─── NOT CACHED ─────────────────┐ │
│  │ Scheduler knows the encoder    │  │ Schedule for encoding (add to  │ │
│  │ already ran. Don't add to      │  │ scheduled_encoder_inputs).     │ │
│  │ scheduled_encoder_inputs.      │  │                                │ │
│  │ scheduler_encoder_cache_hits++ │  │                                │ │
│  └───────────────┬────────────────┘  └────────────────┬───────────────┘ │
│                  │                                    │                 │
│          (embedding already      can_allocate(mm_hash): reserve a slot  │
│           on device — reused)    for the NEW item. If the token budget  │
│                             is full, evict LRU entries from `freeable`, │
│                             appending their mm_hash to `freed`.         │
│                                                       │                 │
│                             get_freed_mm_hashes() →                     │
│                             scheduler_output.free_encoder_mm_hashes     │
│                                                                         │
└──────────────┬───────────────────────────┬──────────────────────────────┘
               │ free_encoder_mm_hashes    │ scheduled_encoder_inputs
               │ (evicted)                 │ (to encode)
               ▼                           │
┌───────────────────────────────────┐      │
│  Model Runner (Neuron Worker)     │      │
│                                   │      │
│  For each evicted mm_hash:        │      │
│  n = EncoderCacheBlocks.free(hash)│      │
│  → pop from _slot_map, return     │      │
│    blocks to free queue; returns  │      │
│    the evicted block count n      │      │
│                                   │      │
│  vllm_neuron:device_encoder_      │      │
│  cache_evictions += sum(n)        │      │
└───────────────────────────────────┘      │
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: Model Runner — _execute_mm_encoder (Neuron Worker)            │
│                                                                         │
│  Only called for items in scheduled_encoder_inputs (NOT CACHED above).  │
│  Each item is a query → device_encoder_cache_queries++                  │
│  self.encoder_cache.contains(mm_hash)?                                  │
│                                                                         │
│  ┌─── ENCODER CACHE HIT ──────────┐  ┌─── ENCODER CACHE MISS ─────────┐ │
│  │ Embeddings already on HBM.     │  │ Run the Encoder NEFF on        │ │
│  │ Skip the encoder.              │  │ Neuron device.                 │ │
│  │                                │  │                                │ │
│  │ hits = queries - misses,       │  │ vllm_neuron:device_encoder_    │ │
│  │ ~always 0 (Layer 2 pre-empts)  │  │ cache_misses++                 │ │
│  └───────────────┬────────────────┘  └────────────────┬───────────────┘ │
│                  │                                    │                 │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Model Runner — _gather_mm_embeddings (Neuron Worker)                   │
│                                                                         │
│  Read cached embeddings from EncoderCacheBlocks buffer                  │
│  (whole HBM buffer + block-id index tensor, passed to prefill graph)    │
│                                                                         │
│  This always reads from the encoder cache for image tokens              │
│  in the scheduled region — regardless of which layer populated it.      │
└─────────────────────────────────────────────────────────────────────────┘

  After the allocate (Layer 3) and evict (free loop) batches each step, the
  runner snapshots EncoderCacheBlocks occupancy into gauges:

┌─────────────────────────────────────────────────────────────────────────┐
│  EncoderCacheBlocks occupancy (point-in-time gauges, per step)          │
│                                                                         │
│  active + free + held == num_blocks - 1   (last block = scratch)        │
│                                                                         │
│  vllm_neuron:device_encoder_cache_active_blocks        (allocated)      │
│  vllm_neuron:device_encoder_cache_free_blocks          (allocatable)    │
│  vllm_neuron:device_encoder_cache_held_blocks          (in hold window) │
│  vllm_neuron:device_encoder_cache_active_real_tokens   (non-pad tokens) │
│                                                                         │
│  active_blocks * block_size - active_real_tokens = padding waste        │
└─────────────────────────────────────────────────────────────────────────┘
```

## When Each Metric Fires

### Multi-turn conversation with prefix caching enabled

| Step | mm_cache_hits | prefix_cache_hits | device_encoder_cache_misses |
|------|:---:|:---:|:---:|
| Turn 1 (cold) | 0 | 0 | N (images encoded) |
| Turn 2 (text follow-up) | ✓ (images in msg array) | ✓ | 0 |
| Turn 3 (same img, new position) | ✓ (images in msg array) | ✓ | 0 |
| Replay (identical request) | ✓ | ✓ | 0 |
| Fresh (same img, different text) | ✓ | partial | 0 |

**`mm_cache_hits`**: The input processor (Layer 1) runs on ALL images in the
message array before any scheduler logic. In multi-turn chat, the full conversation
history (including Turn 1's images) is re-sent every turn. From Turn 2 onward, the
input processor finds these images already cached → `mm_cache_hits` increments.

**`prefix_cache_hits`**: Fires (Layer 2) when a token block from the
current request matches a previously-computed block — token IDs AND image
identity AND intra-block position must all match. In multi-turn chat the
accumulated prefix (prior turns' text + images) is unchanged each turn, so it
hits from Turn 2 onward. A same-images/different-text request still gets a
*partial* hit: the leading image-token blocks match (same images, same
positions) while the diverging text blocks miss.

**`device_encoder_cache_queries` / `device_encoder_cache_misses`**: A query is
counted per mm item the worker checks against the on-device cache in
`_execute_mm_encoder`; a miss is an item not present, which runs the vision
encoder. **Encoder-cache hits = queries − misses, and are ~always 0**: the
scheduler's `EncoderCacheManager` (Layer 2) only routes an item to the worker
when it is NOT already cached, so every item the worker queries is a miss
(`queries == misses`). A nonzero hit would require the scheduler to schedule an
item the worker still holds — only possible under concurrent-request races or
hold-time divergence — so we derive hits rather than emit a separate counter,
matching the `mm_cache` / `prefix_cache` `queries`/`hits` shape.

Both counters increment only on **cold starts** — the first time an image is
encoded. With prefix caching enabled and sufficient cache capacity, they stay
at 0 after Turn 1 because the scheduler never re-schedules already-encoded
images → Layer 3 is never reached for them. They increment again for the same
image only if the encoder cache evicts it (see below).

### When device_encoder_cache_evictions increments

When the working set of distinct images exceeds the encoder cache capacity.
The scheduler's `EncoderCacheManager.can_allocate()` (Layer 2) evicts LRU
entries to make room for a new image, emitting their `mm_hash` via
`scheduler_output.free_encoder_mm_hashes`. The model runner then calls
`EncoderCacheBlocks.free(mm_hash)` for each, which returns the blocks to the
free queue and reports the evicted block count; the runner sums those counts
and increments `device_encoder_cache_evictions` (once per freed block). The
counter is emitted from the runner — same as `device_encoder_cache_misses` —
while `EncoderCacheBlocks` stays free of metric coupling.

An evicted image that is later re-requested is NOT CACHED again, so it takes
the Layer 3 MISS path and re-increments `device_encoder_cache_misses`. So under
cache pressure the two device counters move together: evict-then-re-encode.

### On-device encoder cache occupancy gauges

Alongside the counters above, four **gauges** report the point-in-time state of
`EncoderCacheBlocks` (the physical device buffer). Unlike the counters (which
only move on encode/evict events), these reflect current occupancy:

| Gauge | Meaning |
| -- | -- |
| `vllm_neuron:device_encoder_cache_active_blocks` | Blocks currently allocated to cached mm items |
| `vllm_neuron:device_encoder_cache_free_blocks` | Blocks in the free queue (allocatable now) |
| `vllm_neuron:device_encoder_cache_held_blocks` | Blocks freed but still within the min-hold-time window (not yet reusable) |
| `vllm_neuron:device_encoder_cache_active_real_tokens` | Real (non-padding) merged tokens across active blocks |

Invariant per rank: `active + free + held == encoder_cache_num_blocks - 1`
(the last block is the scratch block, never allocated).

`active_real_tokens` vs. `active_blocks * block_size` exposes **padding waste** —
each item occupies whole block(s) regardless of token count, so many small
images leave most of each block empty. A low utilization ratio is the clearest
signal of the scheduler (token-budget) / device (block-budget) sizing mismatch.

`EncoderCacheBlocks` exposes these as read-only properties (`num_active_blocks`,
`num_free_blocks`, `num_held_blocks`, `num_active_real_tokens`); the model runner
reads them and sets the gauges once per step after the allocate and evict
batches, keeping metric emission centralized in the runner. Each of the 16
TP/DP ranks sets its own value in lockstep (SPMD), and the gauges use
`multiprocess_mode="sum"`, so the exported value is the fleet-wide total across
ranks (divide by the rank count for a per-rank figure).
