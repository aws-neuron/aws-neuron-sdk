# Padding and Batching with Block KV

<!-- meta: description: Block KV batching and padding -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

This document explains how **Block KV Cache** works in vllm_neuron and how padding and batching interact with it. Block KV is a paged memory system that enables efficient KV cache management for variable-length sequences and dynamic batching.

## Block KV Cache Architecture

### What is Block KV?

Block KV divides the KV cache into fixed-size **blocks** rather than allocating contiguous memory per sequence. This enables:

1. **Memory Efficiency**: No wasted memory from over-allocation
2. **Dynamic Batching**: Sequences can grow without reallocation
3. **Memory Sharing**: Common prefixes can share cache blocks (prefix caching)

``` text
Traditional KV Cache (contiguous per sequence):
┌─────────────────────────────────┐
│ Seq 0: [pos 0-255 allocated]    │  ← Wastes memory if seq is shorter
├─────────────────────────────────┤
│ Seq 1: [pos 0-255 allocated]    │
└─────────────────────────────────┘

Block KV Cache (paged):
┌────────┬────────┬────────┬────────┬────────┐
│ Blk 0  │ Blk 1  │ Blk 2  │ Blk 3  │ Blk 4  │ ...
│(null)  │ Seq0   │ Seq1   │ Seq0   │ Seq1   │
│        │ p0-63  │ p0-63  │ p64-127│ p64-127│
└────────┴────────┴────────┴────────┴────────┘
Sequences allocate blocks on demand, non-contiguously
```

### Block Structure

Each block stores KV vectors for a fixed number of tokens:

``` python
# Block KV cache shape per layer
# (total_blocks, block_size, num_kv_heads, head_dim)
k_cache = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim)
v_cache = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim)

# Example configuration:
total_blocks = 1024      # Total blocks in the pool
block_size = 64          # Tokens per block
num_kv_heads = 8         # KV heads (after GQA)
head_dim = 128           # Dimension per head
```

### Block Table

Each request maintains a **block table** that maps logical block indices to physical block IDs:

``` python
# Block table shape: (num_requests, max_blocks_per_request)
# After the Neuron-side remap, unused slots are -1 (see "Block ID 0 vs.
# Sentinel -1" below). Upstream vLLM emits 0 for unused slots; the
# conversion happens in neuron_model_runner._remap_null_block_to_sentinel.
block_table = [
    [10, 15, 22, -1, -1, ...],  # Request 0: blocks 10, 15, 22 allocated
    [ 8, 12, -1, -1, -1, ...],  # Request 1: blocks 8, 12 allocated
]

# Reading the block table:
# Logical block 0 for request 0 → Physical block 10
# Logical block 1 for request 0 → Physical block 15
# Logical block 2 for request 0 → Physical block 22
```

## Slot Mapping: From Position to Cache Location

Slot mapping translates a token's sequence position to its physical location in the Block KV cache.

### The Core Formula

``` python
slot_id = block_number * block_size + (position % block_size)
```

Where:

- `position`: Token's position in the sequence (0-indexed)
- `block_number`: Physical block ID from `block_table[request][position // block_size]`
- `block_size`: Tokens per block (e.g., 64)

**Example**: Request 0 at position 100 with block_size=64:

``` text
block_index = 100 // 64 = 1            # Logical block 1
block_number = block_table[0][1] = 15  # Physical block 15
block_offset = 100 % 64 = 36           # Offset within block
slot_id = 15 * 64 + 36 = 996           # Final cache slot
```

### Slot Mapping Computation

The `compute_slot_mapping` function computes slots for all tokens in a batch:

``` python
def compute_slot_mapping(req_indices, positions):
    """
    Args:
        req_indices: Maps each token to its request index
                     [0, 0, 0, 1, 1, 1, 1] = 3 tokens req 0, 4 tokens req 1
        positions: Token positions within each sequence
                   [0, 1, 2, 0, 1, 2, 3]
    """
    # Which block in the table to look up
    block_table_indices = req_indices * max_blocks_per_req + positions // block_size

    # Get physical block numbers
    block_numbers = block_table.ravel()[block_table_indices]

    # Compute offset within each block
    block_offsets = positions % block_size

    # Final slot mapping
    slot_mapping = block_numbers * block_size + block_offsets
```

## Special Block and Slot Values

### Block ID 0 vs. Sentinel -1

Upstream vLLM reserves block 0 as the **null block** — popped from the free queue at init, so it is never allocated to a real request:

``` python
# From vLLM block_pool.py
self.null_block = self.free_block_queue.popleft()  # Gets block 0
self.null_block.is_null = True
```

Because of this invariant, **every 0 in a scheduler-produced block table is an unused slot**. The Neuron framework remaps 0 → `-1` at the attention boundary so the NKI attention kernel can elide DMA for inactive slots:

``` python
# vllm_neuron/vllm/worker/neuron_model_runner.py
def _remap_null_block_to_sentinel(block_table):
    return torch.where(block_table == 0, -1, block_table)
```

Kernel semantics:

- `dma_copy` pathway (`use_dma_transpose=False`): `oob_mode.skip` — the DMA engine skips transfers for `-1` entries, avoiding wasted HBM bandwidth on padded blocks. This is the main perf win at long context.
- `dma_transpose` pathway (`d_head=128`, non-FP8): the kernel casts `int32(-1) → uint32(0)` internally at zero cost, landing on block 0 (the null block). Hardware requires the number of non-OOB indices to be a multiple of 16 on this path, so `oob_mode.skip` can't be used here.

Either way, the attention mask — computed from `pos_ids` / `cache_len`, independent of block table values — zeroes the contribution of skipped/null blocks, so switching to the `-1` sentinel is bit-identical at the output.

This is the read-side mirror of the `slot_mapping = -1` convention used for KV writes (see {ref}`Slot ID -1: Padding Tokens <slot-id--1-padding-tokens>` below).

(slot-id--1-padding-tokens)=

### Slot ID -1: Padding Tokens

Slot ID `-1` marks **padding tokens** that should NOT write to the KV cache:

``` python
PAD_SLOT_ID = -1

# Padding tokens get slot_mapping = -1
slot_mapping = [
    slot_0, slot_1, ..., slot_99,  # Actual tokens
    -1, -1, -1, ..., -1            # Padding tokens
]
```

**How the kernel handles -1 slots:**

The attention kernel uses `oob_mode.skip` (out-of-bounds skip):

``` python
# From attention_decode_kernel.py
# With block KV, dummy batches' slot_mapping will be -1. So use oob_mode.skip.
nisa.dma_copy(
    dst=K_cache[slot_ids],
    src=K_new,
    oob_mode=nisa.oob_mode.skip  # Silently skip when slot is -1
)
```

When `slot_ids` contains -1, the DMA operation skips the write entirely, preventing padding tokens from corrupting the cache.

## Padding with Block KV

### Why Padding is Needed

The Neuron compiler does not yet fully support dynamic shapes. All compiled graphs have **fixed tensor dimensions**. This means:

- A model compiled for sequence length 128 cannot run sequences of length 100 or 200
- A model compiled for batch size 4 cannot run batches of 2 or 6

**Bucketing** is the solution: we pre-compile the model for a set of fixed sizes (buckets) and pad inputs to match the nearest bucket.

``` text
Without bucketing (not supported):
┌──────────────────────────────────────┐
│ Dynamic graph: accepts any shape     │  ← Not available on Neuron
└──────────────────────────────────────┘

With bucketing:
┌────────────┬────────────┬────────────┐
│ Graph @128 │ Graph @256 │ Graph @512 │  ← Pre-compiled fixed graphs
└────────────┴────────────┴────────────┘

Input: 100 tokens → Pad to 128 → Use Graph @128
```

### Trade-off: Compile Time vs. Padding Overhead

| More Buckets   | Fewer Buckets  |                  |
|----------------|----------------|------------------|
| Longer startup | Faster startup | Compile time     |
| Less padding   | More padding   | Runtime overhead |
| More memory    | Less memory    | Compiled graphs  |

Choose buckets based on your workload:

- **Development**: Single bucket for fastest iteration
- **Production**: Multiple buckets matching your traffic distribution

**Example - Prefill Padding:**

``` text
Actual tokens: 100
Bucket size:   128
Padding:       28 tokens

Token tensor: [tok_0, tok_1, ..., tok_99, PAD, PAD, ..., PAD]
              ├──── 100 actual ────┤├──── 28 padding ────┤
```

### Padding Does NOT Allocate Blocks

**Critical insight**: Block allocation happens for **actual tokens only**, not padding:

``` python
# Scheduler allocates blocks based on actual token count
actual_tokens = 100
blocks_needed = ceil(100 / 64) = 2  # Only 2 blocks allocated

# But tensor is padded to 128 tokens for kernel execution
padded_tensor_size = 128
```

The padding tokens exist in the input tensor but have no corresponding cache blocks.

### Slot Mapping for Padded Tokens

After computing slot_mapping for all positions, padding slots are overwritten with -1:

``` python
# From neuron_model_runner.py
PAD_SLOT_ID = -1

# First compute slots for all positions (including padding)
block_table.compute_slot_mapping(req_indices, positions)

# Then overwrite padding positions with -1
for req_id in req_ids:
    padding_count = padding_map.get(req_id, 0)
    if padding_count > 0:
        pad_start = current_idx + actual_tokens
        pad_end = current_idx + scheduled_tokens
        slot_mapping[pad_start:pad_end] = PAD_SLOT_ID
```

**Example**: 100 tokens padded to 128

``` text
Position    Slot Mapping
--------    ------------
0-99        Valid slots (computed from block table)
100-127     -1 (padding, skipped by kernel)
```

## Batching with Block KV

### Single-Token Decode (Standard)

Each request generates 1 token per decode step:

``` python
# Batch of 4 requests
batch_size = 4
tokens_per_request = 1

req_indices = [0, 1, 2, 3]
positions = [50, 75, 100, 125]

# 4 slot mappings computed, one per request
slot_mapping = [slot_0, slot_1, slot_2, slot_3]
```

### Multi-Token Decode (Speculative Decoding)

With EAGLE3, each request generates multiple tokens per step:

``` python
# Batch=4, speculation_length=4 (1 base + 3 draft)
batch_size = 4
tokens_per_request = 4
total_tokens = 16

req_indices = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
positions = [50,51,52,53, 75,76,77,78, 100,101,102,103, 125,126,127,128]

# 16 slot mappings computed
# Each request writes 4 consecutive positions to its cache blocks
```

### Block Boundary Crossing

When tokens span multiple blocks, the formula handles it automatically:

``` text
block_size=64, positions [62, 63, 64, 65]
block_table = [10, 15, ...]

Position  Block Index  Block Number  Offset  Slot
--------  -----------  ------------  ------  ----
62        0            10            62      10*64+62 = 702
63        0            10            63      10*64+63 = 703
64        1            15            0       15*64+0  = 960  ← New block!
65        1            15            1       15*64+1  = 961
```

No special handling needed - division and modulo automatically route to correct blocks.

### Decode Batch Padding

Just as prefill pads **sequence length** to bucket sizes, decode pads **batch size** to bucket sizes. This is configured via `num_seqs_buckets`.

The same compiler limitation applies (see [Why Padding is Needed](#why-padding-is-needed) above): the batch dimension must match a pre-compiled bucket. During decode, each request generates 1 token, so the batch size equals the number of active requests. If fewer requests are active than the bucket size, dummy requests are added to pad the batch.

**Example: num_seqs_buckets=\[8\] with 2 Active Requests**

``` text
Configuration:
- num_seqs_buckets: [8]
- Active requests: 2 (request 0 and request 1)
- Bucket size: 8

What the model receives:
┌─────────────────────────────────────────────────────────┐
│ input_ids:     [tok_0, tok_1, PAD, PAD, PAD, PAD, PAD, PAD]  │
│                 ├─ real ─┤  ├────── 6 padding tokens ──────┤ │
│                                                             │
│ positions:     [pos_0, pos_1, 0, 0, 0, 0, 0, 0]             │
│                                                             │
│ slot_mapping:  [slot_0, slot_1, -1, -1, -1, -1, -1, -1]     │
│                 ├─ real ─┤    ├─── padding (skipped) ───┤   │
└─────────────────────────────────────────────────────────┘
```

**How Padding is Handled:**

1. **Input IDs**: Padding tokens use a placeholder value (typically 0 or PAD token ID)
2. **Positions**: Padding positions are set to 0 (arbitrary, since they're skipped)
3. **Slot Mapping**: Padding slots are set to `-1`, triggering `oob_mode.skip` in the kernel
4. **Block Table**: Padding requests point to existing valid block table rows (no extra blocks allocated)
5. **Sampling**: Only real requests (indices 0-1) have their logits sampled

**Tensor Shapes:**

``` python
# With num_seqs_buckets=[8] and 2 active requests:
input_ids.shape      = (8,)      # Padded to bucket size
positions.shape      = (8,)      # Padded to bucket size
slot_mapping.shape   = (8,)      # Padded to bucket size
block_table.shape    = (8, max_blocks)  # Padded to bucket size
sampling_positions   = [0, 1]    # Only sample from real requests
```

**Key Points:**

- Padding requests do NOT allocate new KV cache blocks
- Padding requests do NOT write to the KV cache (slot_mapping = -1)
- Padding requests do NOT affect sampling (filtered out via sampling_positions)
- The model processes all 8 "requests" but only 2 produce meaningful output

**Bucket Selection During Inference:**

When the number of active requests changes, the scheduler selects the smallest bucket that fits:

``` text
num_seqs_buckets = [1, 2, 4, 8]

Active Requests    Bucket Used    Padding
---------------    -----------    -------
1                  1              0
2                  2              0
3                  4              1
4                  4              0
5                  8              3
6                  8              2
7                  8              1
8                  8              0
```

If only `num_seqs_buckets=[8]` is configured, all decode batches use bucket 8, even for 1 request (7 padding tokens).

## Cache Masking

The attention kernel uses masks to control which cache positions are valid:

### mask_cache: History Mask

Controls attention to cached KV (previously computed tokens):

``` python
# Shape: (B, num_heads, S_decode, S_ctx)
# Value 1: valid cache position (position < cache_length)
# Value 0: invalid (padding, null block, or future position)

mask_cache = gen_cache_mask(
    cache_len=cache_lens,  # [B, 1] cache length per batch
    num_heads=num_heads,
    S_tkg=S_decode,        # Tokens being generated
    S_ctx=S_ctx,           # Maximum cache capacity
)
```

**How mask_cache protects against invalid reads:**

1. Null block positions are beyond cache_length → masked to 0
2. Unwritten cache positions are beyond cache_length → masked to 0
3. Only positions \[0, cache_length) have mask = 1

### mask_active: Current Token Mask

Controls attention between tokens generated in the same step (multi-token decode):

``` python
# Shape: (B, num_heads, S_decode, S_decode)
# For speculative decoding verification, typically all 1s
mask_active = torch.ones(B, num_heads, S_decode, S_decode)
```

## End-to-End Examples

### Prefill with Padding

``` text
Request: 100 tokens, padded to 128
Block size: 64
Blocks allocated: 2 (block IDs 10, 15)

Block Table: [10, 15, 0, 0, ...]

Token    Position   Block Idx   Block Num   Slot
-----    --------   ---------   ---------   ----
0        0          0           10          640
1        1          0           10          641
...
63       63         0           10          703
64       64         1           15          960
...
99       99         1           15          995
100(pad) -          -           -           -1 (skipped)
...
127(pad) -          -           -           -1 (skipped)

KV Cache writes: 100 actual writes, 28 skipped (padding)
```

### Speculative Decode Crossing Block Boundary

``` text
Request at position 62, generating 4 tokens (1 base + 3 draft)
Block size: 64
Block table: [10, 15, ...]

Token    Position   Block Idx   Block Num   Slot
-----    --------   ---------   ---------   ----
base     62         0           10          702
draft1   63         0           10          703  ← Last slot in block 10
draft2   64         1           15          960  ← First slot in block 15
draft3   65         1           15          961

Automatic block boundary handling via modular arithmetic.
```

## Summary

| Concept | Description |
|---------|-------------|
| Block KV | Paged KV cache with fixed-size blocks |
| Block Table | Maps logical blocks to physical block IDs |
| Slot Mapping | `block_number * block_size + (position % block_size)` |
| Block ID 0 | Null block - placeholder, masked in attention |
| Slot ID -1 | Padding token - skipped by kernel (oob_mode.skip) |
| Prefill Padding | Pads sequence length to bucket, slot = -1 for pads |
| Decode Batch Padding | Pads batch size to bucket, slot = -1 for dummy reqs |
| mask_cache | Ensures only valid cache positions are read |

## See Also

- `neuron-scheduler` - Scheduler design and bucket padding strategy
- `vllm-integration-kv-cache` - KV cache memory management
- `additional-config` - Configuration options for num_batched_tokens_buckets and num_seqs_buckets
