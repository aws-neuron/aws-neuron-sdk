# KV Cache Analysis Design

<!-- meta: description: KV cache accuracy analysis for vLLM Neuron — compares per-layer, per-head key/value caches between a HuggingFace reference and Trainium/Inferentia target to debug attention-level accuracy issues; covers vLLM paged-attention layout, KVCacheSpec/KVCacheGroupSpec groups, TP-sharded reconstruction to contiguous format, sliding-window handling, and two-way vs three-way (FP32 baseline) comparison with relative L-inf/L2 and Bhattacharyya-coefficient metrics. -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-07-31 -->

## Overview

The KV Cache Analysis module compares KV caches between an expected reference model and a target model (vLLM Neuron) to debug accuracy issues at the attention layer level. It optionally includes a baseline (ground truth) model for three-way comparison, enabling separation of dtype-only errors from target-specific errors.

**Naming convention** (matches `logit_validation`):

- `expected`: reference to compare against (required)
- `actual`: target under test (vLLM Neuron)
- `baseline`: ground truth for three-way BC analysis (optional); when provided, all errors are relative to baseline since both expected and actual are teacher-forced against it

**Integration with Logit Validation**: KV cache extraction is integrated into the `logit_validation` API via the `kv_extract_fn(seq_len)` parameter. This ensures KV caches are captured during teacher-forced generation, where both models process identical token sequences. The `seq_len` argument tells the extraction function the current total sequence length in vLLM's KV cache.

**Metrics**:

- **Relative L-inf**: `max|diff| / max|ref|` — worst-case per-element error
- **Relative L2**: `||diff||_2 / ||ref||_2` — overall magnitude of error
- **Bhattacharyya Coefficient (BC)**: Overlap between expected and actual error distributions (three-way only). BC≈1.0 means target errors match dtype-only errors; BC→0 means the target has a fundamentally different error pattern.

In two-way mode, `ref` is the expected KV. In three-way mode, `ref` is the baseline KV (ground truth), and both expected and actual errors are measured relative to it.

## vLLM KV Cache Architecture

Understanding vLLM's KV cache design is essential for correct reconstruction.

### Abstraction Hierarchy

``` text
┌─────────────────────────────────────────────────────────────────────────┐
│                           MODEL LAYER LEVEL                            │
│  Each attention layer defines its own KVCacheSpec via                   │
│  get_kv_cache_spec()                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              KVCacheSpec                                │
│  Defines how ONE LAYER stores KV cache                                 │
│  Subclasses: FullAttentionSpec, SlidingWindowSpec, MLAAttentionSpec    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           KVCacheGroupSpec                              │
│  Groups layers that SHARE THE SAME BLOCK TABLE                         │
│  - layer_names: list[str]  (e.g., ["layers.0.self_attn", ...])         │
│  - kv_cache_spec: KVCacheSpec (merged spec for the group)              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             KVCacheConfig                               │
│  Complete KV cache configuration for the model                         │
│  - kv_cache_groups: list[KVCacheGroupSpec]                             │
│  - num_blocks: int                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

Key Insight: Layers with different attention patterns (e.g., sliding window vs full attention) are placed in **separate groups** with **separate block tables**.

### Model Examples

| Model | Layer Pattern | Groups | Block Tables |
| ---- | ---- | ---- | ---- |
| Llama 3.3 70B | 80 full attention | 1 | 1 shared |
| GPT-OSS 120B | 18 sliding + 18 full (alternating) | 2 | 2 separate |
| Gemma-3-27b | 52 sliding + 10 full | 7 (with padding) | 7 separate |

### Paged Attention Format

vLLM uses paged attention where KV cache is stored in fixed-size blocks:

``` python
# Paged format: [num_blocks, num_kv_heads, block_size, head_dim]
# Block table maps logical block index → physical block index

# To read token at position `pos`:
logical_block = pos // block_size
offset = pos % block_size
physical_block = block_table[logical_block]
k_value = k_cache[physical_block, :, offset, :]
```

### Sliding Window Handling

For models with sliding window attention (e.g., GPT-OSS alternating layers), there are two distinct vLLM code paths that affect KV cache layout:

**Hybrid allocator disabled** (current default for mixed full+sliding models):

vLLM's `unify_hybrid_kv_cache_specs()` converts `SlidingWindowSpec` → `FullAttentionSpec(sliding_window=N)`. All layers keep full block allocation. The `spec_type` in our config is `FullAttentionSpec` for both groups. Blocks are **not freed** — KV data is valid for all positions. The model still only *attends* to the window during computation, but the cache retains everything.

**Hybrid allocator enabled** (true `SlidingWindowSpec`):

Blocks outside the window are freed and replaced with `null_block`. KV data outside the window is garbage. `spec_type` is `SlidingWindowSpec`.

:::{note}
The `sliding_window` field in the model config (e.g., 128) counts the current query token. HF's `past_key_values` retains `window - 1` positions (e.g., 127). The KVCacheSpec also stores the config value (128).
:::

**Reconstruction** (`reconstruct_contiguous_kv`):

Only NaN-fills positions outside the window when `spec_type == 'SlidingWindowSpec'` (blocks actually freed). For `FullAttentionSpec` with `sliding_window` set, all positions are reconstructed normally.

**Comparison** (`compare_kv_caches`):

Requires all inputs to have the same sequence length per layer. Raises `ValueError` on shape mismatches — it does not silently align or pad.

For models with sliding window layers where HF golden KV is shorter than vLLM KV (HF trims `past_key_values` to the window), the caller must pad the shorter golden with NaN so both sides share the same per-layer sequence length before calling `compare_kv_caches`.

## TP Sharding

KV heads are sharded across tensor parallel ranks. Each rank's cache has shape `[num_blocks, heads_per_rank, block_size, head_dim]` where `heads_per_rank = max(1, total_num_kv_heads // tp_size)` (vLLM convention).

**Partitioned** (`total_num_kv_heads >= tp_size`):  
Each rank has unique heads. Concat all ranks on head dim.

**Replicated** (`total_num_kv_heads < tp_size`):  
Multiple ranks share the same head. Stride by `tp_size // total_num_kv_heads` to pick one rank per replica group, then concat.

``` python
# Example: total=8, tp=8 → stride=1 → concat all 8 ranks → 8 heads
# Example: total=2, tp=8 → stride=4 → ranks [0,4] → concat → 2 heads
# Example: total=16, tp=8 → partitioned → concat all → 16 heads
```

The mode is determined by comparing `kv_cache_config['total_num_kv_heads']` with `kv_cache_config['tp_size']`.

## Reconstruction Algorithm

Converting paged KV to contiguous format for comparison:

``` python
def reconstruct_contiguous_kv(paged_kv, kv_cache_config, block_tables, seq_len):
    # Build layer → group mapping from KVCacheConfig
    layer_to_group = {
        layer: group_idx
        for group_idx, group in enumerate(kv_cache_config['groups'])
        for layer in group['layer_names']
    }

    result = {}
    for layer_name, (k_paged, v_paged) in paged_kv.items():
        group_idx = layer_to_group[layer_name]
        block_size = kv_cache_config['groups'][group_idx]['block_size']
        block_table = block_tables[group_idx]

        # Gather physical blocks and reshape to contiguous
        num_blocks = ceil(seq_len / block_size)
        indices = block_table[:num_blocks]
        k_blocks = k_paged[indices]  # [num_blocks, heads, block_size, dim]
        k_cont = k_blocks.permute(1,0,2,3).reshape(heads, -1, dim)[:, :seq_len, :]

        # NaN-fill for SlidingWindowSpec (blocks freed)
        if spec_type == 'SlidingWindowSpec':
            k_cont[:, :valid_start, :] = float('nan')

        result[layer_name] = (k_cont, v_cont)

    return result
```

A batch variant `reconstruct_contiguous_kv_batch` reconstructs all sequences in a batch by calling `reconstruct_contiguous_kv` per sequence index.

## Two-Way vs Three-Way Comparison

**Two-way** (no baseline): compares expected vs actual directly.

``` python
result = compare_kv_caches(expected_kv, actual_kv)
# head.k_linf = |expected - actual| / |expected|
```

**Three-way** (with baseline): all errors are relative to baseline.

``` python
result, raw_errors = compare_kv_caches(
    expected_kv, actual_kv, baseline_kv=fp32_kv,
    return_raw_errors=True)

# head.k_linf     = |baseline - actual| / |baseline|    (target error)
# head.base_k_linf = |baseline - expected| / |baseline|  (dtype-only error)
# BC compares the two error distributions
```

If BC ≈ 1.0, the target's errors look like dtype conversion errors (expected). If BC → 0, the target has a fundamentally different error pattern (bug).

### Cross-Prompt Aggregation

BC can be aggregated across multiple prompts to get a more robust signal. Errors are aligned by **generation token index** (0 = first decode token), so prompts with different lengths are properly aligned.

``` python
all_raw_errors = []  # from compare_kv_caches(..., return_raw_errors=True)
prompt_lens = [8, 12, 10]

agg = aggregate_kv_bc_across_prompts(all_raw_errors, prompt_lens)
# agg[gen_token_idx][layer_name].k_bc → aggregated BC
```

A `compute_combined_bc_per_layer` function computes a single BC per layer across all prompts and all tokens.

## Integration with Logit Validation

The recommended workflow captures KV caches during teacher-forced generation.

:::{note}
`extract_vllm_block_tables()` automatically enables block table snapshotting on first call. Subsequent forward passes will preserve block tables even after requests are freed. There is zero overhead until the first call to `get_block_tables()`.
:::

``` python
from vllm_neuron.accuracy import logit_validation
from vllm_neuron.accuracy.kv_cache_analysis import (
    compare_kv_caches,
    extract_hf_kv_caches_teacher_forced,
    extract_vllm_kv_caches,
    extract_vllm_kv_cache_config,
    extract_vllm_block_tables,
    reconstruct_contiguous_kv,
)

kv_config = extract_vllm_kv_cache_config(llm)

def kv_extract_fn(seq_len):
    paged_kv = extract_vllm_kv_caches(llm, kv_config)
    block_tables = extract_vllm_block_tables(llm)
    return reconstruct_contiguous_kv(paged_kv, kv_config, block_tables, seq_len)

# logit_validation teacher-forces vLLM and captures KV at each step
passed, results, merged_kv = logit_validation(
    input_ids=input_ids.tolist(),
    generate_fn=vllm_generate_fn,
    expected_logits=expected_logits,
    baseline_logits=baseline_logits,  # None for two-way
    kv_extract_fn=kv_extract_fn,
)

# Compare with HF's KV caches
kv_result = compare_kv_caches(expected_kv, merged_kv, baseline_kv=baseline_kv)
```

Return types from `logit_validation`:

- **Two-way, no KV**: `passed` (bool)
- **Two-way + KV**: `(passed, results, merged_kv)`
- **Three-way, no KV**: `(passed, results)`
- **Three-way + KV**: `(passed, results, merged_kv)`

The `results` list contains per-token validation data including divergence information, which can be extracted and passed to the HTML visualization.

This ensures:

1. vLLM processes the exact same tokens as HF (teacher forcing)
2. KV caches are captured at the right points during generation
3. Multiple decode steps are merged into a single contiguous cache via `_update_kv_state` inside `logit_validation`

### Incremental KV Merging

`logit_validation` may call `generate_fn` multiple times (when divergence occurs, it re-feeds teacher tokens). `_update_kv_state` incrementally builds the merged KV cache:

- **First iteration**: copies prompt KV + valid decode KV
- **Subsequent iterations**: copies only newly validated decode KV
- Divergence index determines how many decode tokens are valid per iteration

### HF Golden Generation

Two functions for generating HF KV caches:

- `generate_hf_logits_and_kv(model, input_ids, num_tokens)`: Autoregressive generation. Returns logits and KV. Performs an extra forward pass after the last generated token to ensure `past_key_values` includes KV for all positions up to and including the last generated token.
- `extract_hf_kv_caches_teacher_forced(model, input_ids, teacher_tokens)`: Single forward pass with the full sequence (prompt + teacher tokens). Optionally returns logits for the generated positions via `return_logits=True`.

## Visualization

Visualization helpers live in `vllm_neuron.accuracy.kv_cache_visualize`
(the per-prompt report is `export_html_report`):

``` python
from vllm_neuron.accuracy.kv_cache_visualize import (
    export_html_report,
    export_aggregated_bc_html,
    launch_dashboard,
)
```

**Per-prompt HTML heatmaps** (Y=layer, X=token, color=max metric over heads):

``` python
export_html_report(result, "kv_prompt_0.html",
    prompt_len=prompt_len,
    divergence_indices=[8, 14])
```

Two-way mode shows:

- L-inf and L2 error heatmaps using `YlOrRd_r` colorscale (yellow=low, red=high)
- Default `zmax=0.2` (20% relative error)

Three-way mode additionally shows:

- Error ratio heatmaps (actual/expected, both vs baseline). Ratio ≈ 1.0x means actual error matches expected. Uses `RdYlGn_r` colorscale, zmax=3.0x.
- BC heatmaps with custom colorscale focused on 0.8–1.0 range: 0.0–0.8 flat red, 0.8–0.85 orange, 0.85–0.90 yellow, 0.90–0.95 light green, 0.95–1.0 green. Below 0.8 is considered irrelevant.

Annotations on all heatmaps:

- Red vertical line marks prefill/decode boundary
- Orange dashed lines mark divergence points (where vLLM would have sampled a different token)

**Aggregated BC HTML** (across prompts):

``` python
agg = aggregate_kv_bc_across_prompts(all_raw_errors, prompt_lens)
export_aggregated_bc_html(agg, "aggregated_bc.html", num_prompts=3)
```

Uses the same BC-focused colorscale. Includes divergence count annotations.

**Interactive dashboard** (optional, requires `dash`):

``` python
from vllm_neuron.accuracy.kv_cache_visualize import launch_dashboard

launch_dashboard(result, port=8050)
```

## API Reference

### Extraction

``` python
# HF autoregressive (generates tokens, returns logits + KV)
logits, kv = generate_hf_logits_and_kv(model, input_ids, num_tokens)

# HF teacher-forced (single forward pass, optionally with logits)
kv = extract_hf_kv_caches_teacher_forced(model, input_ids, teacher_tokens)
logits, kv = extract_hf_kv_caches_teacher_forced(
    model, input_ids, teacher_tokens, return_logits=True)

# vLLM (via collective_rpc)
kv_config = extract_vllm_kv_cache_config(llm)
paged_kv = extract_vllm_kv_caches(llm, kv_config)
block_tables = extract_vllm_block_tables(llm)
```

### Reconstruction

``` python
# Single sequence
contiguous_kv = reconstruct_contiguous_kv(
    paged_kv, kv_config, block_tables, seq_len)
# Shape: [1, num_kv_heads, seq_len, head_dim]

# Full batch
batch_kv = reconstruct_contiguous_kv_batch(
    paged_kv, kv_config, block_tables, seq_len, batch_size)
# Shape: [batch_size, num_kv_heads, seq_len, head_dim]
```

### Comparison

``` python
# Two-way
result = compare_kv_caches(expected_kv, actual_kv)

# Three-way with raw errors for aggregation
result, raw_errors = compare_kv_caches(
    expected_kv, actual_kv, baseline_kv=fp32_kv,
    return_raw_errors=True)

# Per-token, per-layer, per-head metrics:
head = result[token_idx][layer_name][head_idx]
head.k_linf       # actual vs ref (ref = baseline if three-way, else expected)
head.base_k_linf  # expected vs baseline (three-way only)

# Per-token, per-layer BC (three-way only):
bc = result[token_idx][f"{layer_name}._bc"]
bc.k_bc  # BC between expected K errors and actual K errors
```

### Data Structures

``` python
@dataclass
class HeadMetrics:
    k_cos: float       # Cosine similarity (1.0 = identical)
    v_cos: float
    k_linf: float      # Relative L-inf: max|diff| / max|ref|
    v_linf: float
    k_l2: float        # Relative L2: ||diff||_2 / ||ref||_2
    v_l2: float
    base_k_linf: float  # expected vs baseline (three-way only)
    base_v_linf: float
    base_k_l2: float
    base_v_l2: float

@dataclass
class TokenKVMetrics:
    k_bc: float  # BC between expected and actual K errors
    v_bc: float  # BC between expected and actual V errors

@dataclass
class TokenKVErrors:
    base_k: np.ndarray  # [num_heads * head_dim] expected K errors vs baseline
    base_v: np.ndarray
    tgt_k: np.ndarray   # [num_heads * head_dim] actual K errors vs baseline
    tgt_v: np.ndarray
```

### RPC Methods

Added to `NeuronModelRunner` and `NeuronWorker`:

``` python
def get_kv_caches(self) -> dict[str, bytes]:
    """Returns {layer_name: torch.save({'k': k, 'v': v})}"""

def get_kv_cache_config(self) -> dict:
    """Returns KVCacheConfig as dict with groups, TP info, CP info."""

def get_block_tables(self) -> list[bytes]:
    """Returns [torch.save(block_table) for each group]."""
```

## Dependencies

Required: `torch`, `numpy`

Optional (visualization): `plotly`

Optional (interactive dashboard): `dash`, `dash-bootstrap-components`

## Example Scripts

- `examples/vllm_neuron/accuracy/run_kv_cache_analysis.py` - Two-way and three-way KV cache comparison on Neuron
