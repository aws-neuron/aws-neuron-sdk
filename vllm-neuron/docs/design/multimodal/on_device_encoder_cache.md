# On-Device Encoder Cache

<!-- meta: description: Block-based on-device encoder cache for multimodal embeddings -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-07-16 -->

## Overview

The on-device encoder cache stores encoder outputs in a pre-allocated HBM buffer so that a repeated reference to the same item skips the encoder entirely. Embeddings are organized in fixed-size blocks, written by the encoder graph via input-output aliasing, and read by the prefill graph, which is passed a pointer to the whole buffer plus a block-id index tensor and gathers the needed blocks in-graph — no CPU↔device data transfer in either direction.

### Background

Vision-language models (e.g., Qwen3-VL) encode each image/video into a sequence of embedding tokens before the language model's prefill. Without a cache, the same media would be re-encoded on every request that references it.

The cache read/write paths must handle two forms of dynamism that conflict with Neuron's static-graph execution model:

1. **Variable-length write path:** Each image/video produces a different number of tokens (256 for a 512×512 image, 7680 for a 30-frame video). The encoder output must be sliced into per-item embeddings for individual cache storage and eviction.
2. **Variable-position read path:** Each request places cached embeddings at different text-sequence offsets. Before prefill, cached items must be merged into the token embedding sequence at their respective positions.

On GPU, both operations happen eagerly outside the CUDA graph with negligible cost. On Neuron, eager-mode dynamic operations on device are expensive (291–503 ms for 30 512x512 images gather), and running them on CPU instead creates a 354 ms gap from data transfer round-trips. The on-device encoder cache eliminates this gap by keeping all data in HBM and using compile-compatible fixed-size structures.

## Architecture

The diagram below shows the full vision-language data flow and where the encoder cache sits. All heavy data stays on device — only lightweight metadata (block IDs, position maps) crosses the CPU↔device boundary.

``` text
PIL Images/Video (variable count, variable resolution)
                │
   CACHE HIT ------- CACHE MISS
       │                 │
       │                 ▼
       │  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
       │    HF Preprocessor (CPU)
       │  │   Resize, patch, normalize → pixel_values                     │
       │      Compute grid_thw per image → tokens_per_image (VARIABLE)
       │  │   Generate mm_hash per item, placeholder token positions      │
       │  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┬─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
       │                                    │ pixel_values + grid_thw
       │                                    ▼
       │  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
       │    Encoder (on device)
       │  │                                                               │
       │      ┌ ─ Shape fixing (CPU) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
       │  │     Bucket select: pick smallest bucket ≥ total_tokens        │
       │        Block-packing: pack items into fixed-size blocks
       │  │     Allocate cache blocks: allocate(mm_hash)                  │
       │      └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
       │  │                                                               │
       │      Encoder graph executes (cache buffer as I/O alias)
       │  │        → scatter-writes directly into encoder_cache.buffer    │
       │             at allocated block positions (no CPU copy)
       │  │                                                               │
       │      mark_written(mm_hash) — starts hold-time clock
       │  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
       │                                    │ data in cache buffer (HBM)
       ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Encoder Cache Buffer (HBM, pre-allocated at startup)                    │
│                                                                         │
│   [num_blocks, block_size, fat_dim]                                     │
│   slot_map: mm_hash → block_ids                                         │
│   On cache hit: skip the encoder entirely, read from buffer             │
│   On eviction: return blocks to free queue (no data movement)           │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ buffer pointer + block_ids index tensor
                                    │ + encoder_positions map
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Prefill (on device)                                                     │
│                                                                         │
│   _gather_mm_embeddings() builds:                                       │
│     buffer (whole) + encoder_block_ids = [max_blocks] index tensor       │
│     encoder_positions = [max_blocks, block_size] position map            │
│                                                                         │
│   Inside prefill graph (merge_multimodal_embeddings):                       │
│     index_select(buffer, block_ids) → flatten → index_put_ into hidden  │
│     Sentinel positions write to dummy row (discarded)                   │
│                                                                         │
│   Prefill graph executes → KV cache filled, first token generated       │
└─────────────────────────────────────────────────────────────────────────┘
```

**On cache hit** (same mm_hash seen before): the entire HF Preprocessor and Encoder box are skipped — the prefill graph reads directly from the cache buffer (the whole buffer plus a block-id index tensor is passed in, and the blocks are gathered in-graph).

### Buffer Layout

The cache is a pre-allocated contiguous buffer on device with shape `[num_blocks, block_size, fat_dim]`. Each block holds exactly one multimodal item. The last block is reserved as a scratch block that absorbs padding writes from encoder bucket padding and is never allocated to real items.

``` text
buffer: [num_blocks, block_size, fat_dim]

┌──────────────┬──────────────┬──────────────┬───┬──────────────┐
│ block 0      │ block 1      │ block 2      │...│ block N-1    │
│[block_size,D]│[block_size,D]│[block_size,D]│   │ (scratch)    │
└──────────────┴──────────────┴──────────────┴───┴──────────────┘

slot_map (mm_hash → block_ids):
  "img_hash_A" → [0]         (1 image = 1 block)
  "img_hash_B" → [2]         (1 image = 1 block)
  "video_hash" → [3, 4, 5]   (1 video = multiple blocks)
```

- `block_size` = `encoder_attention_block_size / merge_factor` (post-merger tokens per block). Currently tied to the encoder's attention block size for simplicity — encoder output maps directly to cache blocks with no remap. See [Block-Packing Attention](block_packing_attention.md) for how images are packed into blocks for vision attention.
- `fat_dim` = `out_hidden_size × (1 + num_deepstack_levels)`. For Qwen3-VL 32B: 5120 × 4 = 20,480.
- **One-item-per-block policy:** Each cache block holds exactly one mm_item. One image occupies one block; one video spans multiple blocks (frames packed within each). This simplifies eviction (free item = free its blocks), eliminates external fragmentation (any free block fits any allocation), and makes cross-request sharing trivial (mm_hash is the sole lookup key).
- **Trade-off: internal padding waste.** Images smaller than `block_size` waste the remainder.
- **Per-block content tag (`mm_hash_tags`).** A parallel `int64[num_blocks]` device tensor holds `mm_hash_fingerprint(mm_hash)` per block, to note which `mm_hash` a block holds (`0` = empty). In [EPD](../../tutorials/tutorial-epd-1e-1pd-xeypd.md) the tag is a NIXL-registered region. The consumer compares the pulled tag to the fingerprint it expects; a mismatch is a stale, reused, or wrong block, and the consumer reports a miss. The read order is deliberate. The producer stamps the tag in `allocate()`, before the encoder writes the embedding. The consumer reads the embedding first, then reads the tag on a second transfer. So a matching tag proves the embedding matches. The tag is unused in the monolithic path. `EncoderCacheBlocks` also carries a `model_id` constructor parameter, which the EPD handshake reads to reject a pool serving a different model.

### Write Path

When the scheduler dispatches new multimodal inputs for encoding:

1. `EncoderCacheBlocks.allocate()` reserves blocks from the free queue for each mm_item.
2. The encoder graph executes with the cache buffer as an input-output alias (same pattern as KV cache). It scatter-writes merged embeddings directly into the allocated block positions.
3. `mark_written()` records the write timestamp (used by the minimum-hold-time guard).

No device→host transfer occurs. The encoder graph writes directly into the cache buffer.

### Read Path

When prefill needs cached embeddings:

1. `_gather_mm_embeddings()` looks up block IDs for each mm_item in the current request.
2. Zero-copy views (`buffer[block_id]`) are gathered into a tuple — no data copy, just pointer arithmetic.
3. A **position map** (`encoder_positions`, shape `[max_num_encoder_blocks, block_size]`, int64) encodes where each token from each block should land in the batch sequence. Sentinel value (`num_tokens`) marks don't-care positions. The fixed shape is determined at warmup and stays constant across all steps — variable-position assembly is encoded as data (position values and sentinels) rather than control flow, satisfying Neuron's static-graph constraint.
4. Both are passed to the prefill graph as fixed-size inputs.

**Fail-loud contract.** A scheduled, in-window item whose slot is missing or whose stored token count disagrees with the sequence placeholder (`check_num_tokens()`, backed by a post-gather check that every scheduled encoder position was filled) is never silently skipped. It raises a detectable fault — a hard `EncoderCacheReadError` in monolithic mode, or a `ECLoadFailure` on an EPD consumer — instead of leaving placeholders unfilled and running the model as if no image were present.

### Merge Inside Compiled Graph

Inside the prefill graph, `merge_multimodal_embeddings()` assembles the embeddings:

1. `torch.stack(encoder_embedding_blocks)` → `[num_blocks, block_size, fat_dim]`
2. Flatten to `[num_blocks × block_size, fat_dim]`
3. `index_put_` scatters embeddings into `hidden_states` at positions from `encoder_positions`
4. Out-of-range positions (sentinel) write to a dummy row that is discarded

This approach handles sequence-parallel (SP) sharding via `global_to_local_positions()`, which remaps global batch positions to each rank's local shard. Positions outside the rank's range become sentinels.

## Configuration

The encoder cache is configured through `EncoderNeuronConfig` in `additional_config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder_cache_num_blocks` | Auto-derived | Override for cache block count. `None` = derived from scheduler budget. |
| `encoder_cache_min_hold_time_ms` | Auto (0 or 100) | Minimum time before freed blocks can be reused. 0 for monolithic; 100 ms for EPD (allows remote readers to finish pulling). |

### Auto-sizing

When `encoder_cache_num_blocks` is not set, the block count is derived from the scheduler's `encoder_cache_size` budget:

``` text
cache_block_size = encoder_attention_block_size / merge_factor
num_blocks = ceil(encoder_cache_size / cache_block_size) + 1  (scratch)
```

Where `cache_block_size` is derived from the encoder's attention block size divided by the token merge factor (e.g., 2048 / 4 = 512 post-merger tokens per block for Qwen3-VL), and `encoder_cache_size = max(max_num_batched_tokens, max_tokens_per_mm_item)` — the same token budget the scheduler's `EncoderCacheManager` uses for eviction decisions.

### Known Limitation: Padding Waste Mismatch

The scheduler's `EncoderCacheManager` controls cache sizing and eviction at the token level. `EncoderCacheBlocks` translates that into physical blocks. A mismatch occurs because the scheduler does not account for block-level padding: a small image (e.g., 80 tokens) in a larger block (e.g., 256 tokens) consumes one full block of physical capacity, but the scheduler only counts 80 tokens toward its budget. This means the scheduler may allocate without evicting when `EncoderCacheBlocks` does not have enough free blocks — causing inference stalls.

**Current mitigation:** Over-provision the block count via `encoder_cache_num_blocks`. This compensates for padding waste but padding percentage varies at runtime depending on image size distribution.

## Key Files

| File | Purpose |
|------|---------|
| `vllm_neuron/vllm/worker/encoder_cache_blocks.py` | `EncoderCacheBlocks` class — buffer allocation, free queue, hold-time guard, slot map |
| `vllm_neuron/functional/merge_multimodal_embeddings.py` | `merge_multimodal_embeddings()` — scatter encoder embeddings into hidden_states inside compiled graph (shared by vision + audio) |
| `vllm_neuron/vllm/worker/neuron_model_runner.py` | `_init_encoder_cache()`, `_execute_mm_encoder()`, `_gather_mm_embeddings()` — orchestrates write/read paths |
| `vllm_neuron/model/neuron_config.py` | `EncoderNeuronConfig` — `encoder_cache_num_blocks`, `encoder_cache_min_hold_time_ms` |
