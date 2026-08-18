# Prefill Processing

<!-- meta: description: Prefix caching design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

This document describes how prefill processing works in the NeuronScheduler, covering two key mechanisms that work together:

1. **Prefill Segmentation** - Breaking large prompts into smaller chunks for processing
2. **Prefix Caching** - Reusing KV cache from shared prompt prefixes

Both mechanisms rely on the same underlying concept: computing attention over **cached KV segments** while processing **active tokens**.

## Key Concepts

**Active Tokens**: The tokens being computed in the current scheduling step. These are the tokens for which we compute Q, K, V and write new KV to cache.

**Cached KV Segments**: Fixed-size chunks of previously computed KV cache that the active tokens must attend to. Each segment represents a portion of the sequence that has already been processed.

**kv_segment_size**: The fixed size of each KV cache segment. This is a kernel parameter that determines how the cached KV is chunked for iterative attention. Currently supported values: {512, 1024, 2048, 4096}.

**cached_seq_len**: The actual number of valid cached tokens (from prefix cache and/or previously computed segments). This is passed to the attention kernel as `prior_tokens`, and the kernel internally computes how many segments to iterate over.

**Block Alignment**: The KV cache is organized in blocks (e.g., 128 tokens per block). Both cached token counts and KV segment sizes are always multiples of the block size:

``` python
# Configuration constraints
block_size = 128              # Tokens per KV cache block
kv_segment_size = 1024        # Must be multiple of block_size (8 blocks)

# Cached tokens are always block-aligned
cached_seq_len = 4096         # Always a multiple of block_size (32 blocks)
```

## Prefill Segmentation

### What is Prefill Segmentation?

Prefill segmentation (also called chunked prefill or chunked context encoding) breaks large prompts into smaller segments that fit within hardware constraints. This is controlled by `max_num_batched_tokens`.

``` python
# Configuration
max_num_batched_tokens = 2048  # Process 2k tokens per iteration

# Example: 10,000 token prompt
# Iteration 1: Process tokens 0-2047     (num_computed_tokens: 0 → 2048)
# Iteration 2: Process tokens 2048-4095  (num_computed_tokens: 2048 → 4096)
# Iteration 3: Process tokens 4096-6143  (num_computed_tokens: 4096 → 6144)
# Iteration 4: Process tokens 6144-8191  (num_computed_tokens: 6144 → 8192)
# Iteration 5: Process tokens 8192-9999  (num_computed_tokens: 8192 → 10000)
```

### How Segmentation Works with KV Segments

Each iteration processes active tokens while attending to all previously computed KV cache. The kernel receives `cached_seq_len` (the total number of cached tokens) and `kv_segment_size`, and internally iterates over the cached KV in fixed-size segments:

``` python
# Configuration
max_num_batched_tokens = 2048  # Active tokens per iteration
kv_segment_size = 1024         # Size of each KV cache segment

# 10,000 token prompt, no prefix cache
# ================================================================

# Iteration 1: Process tokens 0-2047
# ---------------------------------
active_tokens = 2048           # Tokens being computed
cached_seq_len = 0             # No prior cache
# Kernel: no prior segments to attend to, self-attention only
# Write KV for positions 0-2047 to cache

# Iteration 2: Process tokens 2048-4095
# -------------------------------------
active_tokens = 2048           # Tokens being computed
cached_seq_len = 2048          # From iteration 1
# Kernel: iterates over 2 segments (2048 / 1024) of prior KV + self-attention
# Write KV for positions 2048-4095 to cache

# Iteration 3: Process tokens 4096-6143
# -------------------------------------
active_tokens = 2048           # Tokens being computed
cached_seq_len = 4096          # From iterations 1-2
# Kernel: iterates over 4 segments (4096 / 1024) of prior KV + self-attention
# Write KV for positions 4096-6143 to cache

# ... and so on
```

**Key Insight**: As prefill progresses, `cached_seq_len` increases because each iteration must attend to all previously computed tokens.

## Prefix Caching

### What is Prefix Caching?

Prefix caching (also called prompt caching) allows multiple requests that share a common prompt prefix to reuse the same KV cache blocks:

``` python
# Request 1 completes
prompt_1 = "You are a helpful assistant. User: "  # 10 tokens

# Request 2 arrives with same prefix
prompt_2 = "You are a helpful assistant. User: What is Python?"  # 15 tokens
# First 10 tokens cached, only compute last 5 tokens
```

From the scheduler's perspective, a request with cached prefix is treated identically to a segmented prefill - both have `num_computed_tokens > 0`.

### How Prefix Caching Works with KV Segments

The cached prefix tokens become prior KV that new tokens must attend to:

``` python
# Configuration
block_size = 128
max_num_batched_tokens = 2048
kv_segment_size = 1024         # 8 blocks

# Request with 4096 token prefix cache (32 blocks, block-aligned)
num_prompt_tokens = 6144       # Total prompt length
num_computed_tokens = 4096     # Prefix already cached (block-aligned)
tokens_to_compute = 2048       # New tokens

# Scheduling step
active_tokens = 2048           # Computing the new tokens
cached_seq_len = 4096          # From prefix cache
# Kernel: iterates over 4 segments (4096 / 1024) of prior KV + self-attention
# Write KV for new tokens to cache
```

#### Visual: Active Tokens Attending to Cached KV

``` text
KV Cache (4096 cached tokens = 32 blocks):
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                      CACHED PREFIX                                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Segment 0             Segment 1             Segment 2             Segment 3           │
│  (tokens 0-1023)       (tokens 1024-2047)    (tokens 2048-3071)    (tokens 3072-4095)  │
│ ┌─┬─┬─┬─┬─┬─┬─┬─┐     ┌─┬─┬─┬─┬─┬─┬─┬─┐     ┌─┬─┬─┬─┬─┬─┬─┬─┐     ┌─┬─┬─┬─┬─┬─┬─┬─┐    │
│ │0│1│2│3│4│5│6│7│     │8│9│A│B│C│D│E│F│     │G│H│I│J│K│L│M│N│     │O│P│Q│R│S│T│U│V│    │
│ └─┴─┴─┴─┴─┴─┴─┴─┘     └─┴─┴─┴─┴─┴─┴─┴─┘     └─┴─┴─┴─┴─┴─┴─┴─┘     └─┴─┴─┴─┴─┴─┴─┴─┘    │
│  └── 8 blocks ──┘      └── 8 blocks ──┘      └── 8 blocks ──┘      └── 8 blocks ──┘    │
└────────────────────────────────────────────────────────────────────────────────────────┘
        ▲                       ▲                       ▲                       ▲
        │                       │                       │                       │
        └───────────────────────┴───────────────────────┴───────────────────────┘
                                        │
                                ATTENTION (read K,V)
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        ACTIVE TOKENS (2048)                             │
    │                     Computing Q, K, V for this step                     │
    │                       (tokens 4096-6143)                                │
    │                                                                         │
    │  Query attends to:                                                      │
    │    • Prior KV from cache (cached_seq_len = 4096)                        │
    │    • Self-attention within active tokens                                │
    └─────────────────────────────────────────────────────────────────────────┘
```

## Combined: Prefix Caching + Segmentation

The most interesting case is when both mechanisms apply: a request has a cached prefix AND the remaining prompt exceeds `max_num_batched_tokens`.

### Example Walkthrough

``` python
# Configuration
block_size = 128               # Tokens per block
max_num_batched_tokens = 2048  # Segment size for active tokens
kv_segment_size = 1024         # Size of each KV cache segment (8 blocks)

# Request
num_prompt_tokens = 10240      # Total prompt length (80 blocks)
prefix_cached = 4096           # First 4k tokens in KV cache (32 blocks, block-aligned)
tokens_to_compute = 6144       # Remaining tokens: 10240 - 4096

# ================================================================
# Scheduling Step 1: Process tokens 4096-6143
# ================================================================
active_tokens = 2048           # First segment of new tokens
cached_seq_len = 4096          # From prefix cache (block-aligned)
# Kernel: iterates over 4 segments (4096 / 1024) of prior KV + self-attention

# After step:
# - KV cache now has positions 0-6143
# - num_computed_tokens = 6144 (block-aligned)

# ================================================================
# Scheduling Step 2: Process tokens 6144-8191
# ================================================================
active_tokens = 2048           # Second segment of new tokens
cached_seq_len = 6144          # 4k prefix + 2k from step 1 (block-aligned)
# Kernel: iterates over 6 segments (6144 / 1024) of prior KV + self-attention

# After step:
# - KV cache now has positions 0-8191
# - num_computed_tokens = 8192 (block-aligned)

# ================================================================
# Scheduling Step 3: Process tokens 8192-10239
# ================================================================
active_tokens = 2048           # Final segment of new tokens
cached_seq_len = 8192          # 4k prefix + 4k from steps 1-2 (block-aligned)
# Kernel: iterates over 8 segments (8192 / 1024) of prior KV + self-attention

# After step:
# - KV cache now has positions 0-10239
# - num_computed_tokens = 10240 (prefill complete!)
```

### Visual Representation

``` text
Token positions:  0        4096      6144      8192     10240
                  |         |         |         |         |
                  +---------+---------+---------+---------+
                  | PREFIX  |  SEG 1  |  SEG 2  |  SEG 3  |
                  | CACHE   | (active)| (active)| (active)|
                  | 4k tok  |  2k tok |  2k tok |  2k tok |
                  +---------+---------+---------+---------+
                  (32 blks)  (16 blks) (16 blks) (16 blks)

Step 1: Computing SEG 1 (tokens 4096-6143)
┌─────────────────────────────────────────────────────────────────────┐
│ cached_seq_len = 4096                                               │
│ Kernel iterates over 4 segments of 1024 tokens each                 │
│ Active tokens: 2048                                                 │
└─────────────────────────────────────────────────────────────────────┘

Step 2: Computing SEG 2 (tokens 6144-8191)
┌─────────────────────────────────────────────────────────────────────┐
│ cached_seq_len = 6144                                               │
│ Kernel iterates over 6 segments of 1024 tokens each                 │
│ Active tokens: 2048                                                 │
└─────────────────────────────────────────────────────────────────────┘

Step 3: Computing SEG 3 (tokens 8192-10239)
┌─────────────────────────────────────────────────────────────────────┐
│ cached_seq_len = 8192                                               │
│ Kernel iterates over 8 segments of 1024 tokens each                 │
│ Active tokens: 2048                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Kernel Interface

The segmented attention kernel (`segmented_attention` in `functional/attention/attention_segmented_cte.py`) receives two key parameters that control how it processes cached KV:

- **prior_tokens** (`cached_seq_len`): A tensor of shape `[B, 1]` containing the actual number of valid cached tokens. The kernel uses this to determine how many segments of prior KV to iterate over.
- **kv_segment_size**: The fixed segment size for iterating over prior KV. Must be one of {512, 1024, 2048, 4096}.

The kernel internally computes the number of segments as `prior_tokens / kv_segment_size` and iterates accordingly. There is no explicit `num_kv_segments` parameter — it is derived.

**Current Kernel Constraints:**

- `kv_segment_size` must be one of {512, 1024, 2048, 4096}
- `kv_segment_size` must be divisible by `block_size`
- Query sequence length must equal `kv_segment_size` (current constraint)
- `prior_tokens` must be a multiple of `block_size`

**kv_segment_size_buckets Configuration:**

The `kv_segment_size_buckets` parameter in `neuron_config` controls which segment sizes are compiled. Currently only a single segment size is supported:

``` python
neuron_config = {
    "kv_segment_size_buckets": [2048],  # Single segment size
}
```

When `kv_segment_size_buckets` is set, the current constraint requires `num_batched_tokens_buckets` to match, because the kernel requires the prefill bucket length to equal the segment size.

``` python
# Valid: buckets match
neuron_config = {
    "num_batched_tokens_buckets": [2048],
    "kv_segment_size_buckets": [2048],
}

# Invalid: buckets don't match
neuron_config = {
    "num_batched_tokens_buckets": [1024, 2048],
    "kv_segment_size_buckets": [2048],
}  # Error: num_batched_tokens_buckets must equal kv_segment_size_buckets
```

### Attention Metadata

The model runner builds attention metadata as a dict per layer, passing `cached_seq_len` and `kv_segment_size` to the model's attention module:

``` python
# In NeuronModelRunner._build_attention_metadata()
cached_seq_len = self._compute_cached_seq_len()  # max across batch

attn_metadata_i = {
    "block_table_tensor": blk_table_tensor,
    "slot_mapping": slot_mapping,
    "max_query_len": max_query_len,
    "block_size": block_size,
    "max_blocks_per_seq": max_num_blocks_per_req,
    "decode_token_threshold": 1 + max_num_draft_tokens,
    "cached_seq_len": torch.tensor([[cached_seq_len]], dtype=torch.int32),
    "kv_segment_size": kv_segment_size,  # 0 if segmented prefill not enabled
}
```

The model's attention layer then conditionally uses the segmented kernel:

``` python
# In model attention forward (e.g., Llama3Attention)
cached_seq_len = attn_metadata[layer_name]["cached_seq_len"]
kv_segment_size = attn_metadata[layer_name]["kv_segment_size"]

if kv_segment_size:
    # Use segmented attention kernel
    attn_score = NF.segmented_attention(
        q, k_cache=self.k_cache, v_cache=self.v_cache,
        block_tables=block_table,
        prior_tokens=cached_seq_len,
        block_size=block_size,
        kv_segment_size=kv_segment_size,
        scale=self.scaling,
        tp_q=True,
    )
else:
    # Use standard attention (no prior KV segments)
    attn_score = NF.attention(...)
```

### Partial KV Segment Handling

When `cached_seq_len` is not evenly divisible by `kv_segment_size`, there is a partial segment with valid cache that the kernel must handle:

``` python
# Example: Partial KV segment scenario
block_size = 128
kv_segment_size = 1024
cached_seq_len = 3584          # 28 blocks (not divisible by 8 blocks per segment)

# Full segments: 3584 // 1024 = 3
# Partial segment: 3584 % 1024 = 512 valid tokens (4 blocks)
# The kernel must mask the remaining 512 tokens in the last segment
```

#### Visual: Partial KV Segment with Masking

``` text
KV Cache Layout (cached_seq_len = 3584):
┌──────────────────────────────────────────────────────────────────────────────────┐
│  Segment 0             Segment 1             Segment 2             Segment 3     │
│  (tokens 0-1023)       (tokens 1024-2047)    (tokens 2048-3071)    (PARTIAL)     │
│                                                                                  │
│ ┌─┬─┬─┬─┬─┬─┬─┬─┐     ┌─┬─┬─┬─┬─┬─┬─┬─┐     ┌─┬─┬─┬─┬─┬─┬─┬─┐     ┌─┬─┬─┬─┐      │
│ │ │ │ │ │ │ │ │ │     │ │ │ │ │ │ │ │ │     │ │ │ │ │ │ │ │ │     │ │ │ │ │      │
│ └─┴─┴─┴─┴─┴─┴─┴─┘     └─┴─┴─┴─┴─┴─┴─┴─┘     └─┴─┴─┴─┴─┴─┴─┴─┘     └─┴─┴─┴─┘      │
│  └── 8 blocks ──┘      └── 8 blocks ──┘      └── 8 blocks ──┘      4 VALID       │
│      (FULL)                (FULL)                (FULL)            blocks        │
│                                                                    (512 tok)     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

The kernel receives `prior_tokens = 3584` and `kv_segment_size = 1024`. It computes:

- Number of full segments: `3584 // 1024 = 3`
- Valid tokens in partial segment: `3584 % 1024 = 512`
- Masks the remaining 512 tokens in the 4th segment internally

## Scheduler Integration

### Request State Tracking

The scheduler tracks prefill progress through `num_computed_tokens`:

``` python
class Request:
    num_prompt_tokens: int      # Total tokens in prompt
    num_computed_tokens: int    # Tokens already processed (cache + segments)

# Prefill detection
is_prefill = (num_computed_tokens < num_prompt_tokens)

# Tokens to schedule this iteration
remaining = num_prompt_tokens - num_computed_tokens
tokens_this_step = min(remaining, max_num_batched_tokens)
```

### Scheduler Output

The scheduler provides all information needed for the model runner:

``` python
SchedulerOutput:
    scheduled_new_reqs = [
        NewRequestData(
            req_id = "req_0",
            num_tokens = 2048,           # Active tokens this step
            num_computed_tokens = 4096,  # Cached tokens (block-aligned)
            block_table = [...]          # Blocks containing all KV
        )
    ]
    num_scheduled_tokens = {"req_0": 2048}         # Actual tokens
    num_scheduled_tokens_padded = {"req_0": 2048}  # Padded to bucket
```

The model runner computes `cached_seq_len` from the batch:

``` python
def _compute_cached_seq_len(self) -> int:
    """Max cached sequence length across all requests in the batch."""
    cached_seq_len = 0
    for req_index in range(self.input_batch.num_reqs):
        num_computed = self.input_batch.num_computed_tokens_cpu[req_index]
        if num_computed > cached_seq_len:
            cached_seq_len = num_computed
    return cached_seq_len
```

## Block Allocation

Blocks are allocated **incrementally** — only for tokens that need slots right now (cached + new this step), not the full sequence length upfront. The vLLM `KVCacheManager.allocate_slots()` computes `num_tokens_need_slot = total_computed_tokens + num_new_tokens` (including prefix cache hits) and allocates the delta between required blocks and already-held blocks:

``` python
# Request: 10240 tokens, 4096 cached, processing 2048 active this step
block_size = 128

# Step 1 (new request): allocate_slots(num_new_tokens=2048, num_new_computed_tokens=4096)
# The scheduler finds 32 prefix-cached blocks via get_computed_blocks()
# and passes them into allocate_slots().
#
# Inside allocate_slots:
num_tokens_need_slot = 4096 + 2048          # = 6144
blocks_needed = cdiv(6144, block_size)      # = 48 
# The 32 cached blocks are claimed via allocate_new_computed_blocks(),
# then 16 fresh blocks are allocated
# via allocate_new_blocks().

# Block table row after step 1 (max_model_len=10240 example):
# The tensor is pre-sized to cdiv(max_model_len, block_size) = 80 columns,
# zero-initialized.
# Only the first 48 entries contain valid block IDs.
# Unallocated columns contain 0 (the null block ID).
block_table = [1, 2, 3, ..., 48, 0, 0, ..., 0]
#              ^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^
#              48 valid block IDs      32 null entries (block_id=0)
#
# Blocks 1-32:  Contain cached KV (prefix, 4096 tokens)
# Blocks 33-48: Will store KV from this step (tokens 4096-6143)
# Columns 49-80 are 0 (null block) — blocks for tokens 6144-10239 are not yet allocated

# Step 2 (running request): allocate_slots(num_new_tokens=2048)
# request.num_computed_tokens is now 6144 (updated after step 1).
# No new cache hits for running requests (num_new_computed_tokens=0).
num_tokens_need_slot = 6144 + 2048          # = 8192
blocks_needed = cdiv(8192, block_size)      # = 64
# 48 blocks already held from step 1, so allocate 16 new blocks

# Step 3: allocate_slots(num_new_tokens=2048)
# ... and so on until prefill completes
```

## Padding Considerations

Padding is applied to **active tokens only**, not the entire sequence:

``` python
# Request: 10240 tokens, 4096 cached, 2048 active this step
active_tokens = 2048
bucket_size = 2048  # Smallest bucket >= 2048
padding = 0 tokens  # Exact bucket match

# Forward pass tensor shape: 2048 (active tokens only)
# NOT 10240 (total sequence)
```

The cached KV segments are read from cache, not recomputed, so they don't contribute to the padded tensor size.

## See Also

- `neuron-scheduler` - Core scheduler design and state machine
- `vllm-integration-kv-cache` - KV cache management
