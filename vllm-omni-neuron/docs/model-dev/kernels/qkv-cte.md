# Kernel reference: `qkv_cte` — Wan2.2 additions to the NKI Library QKV CTE MXFP8 projection

<!-- meta: description: What the vendored MXFP8 QKV context-encode projection adds
on top of the NKI Library's qkv kernel for Wan2.2: an across-heads
(Distributed) RMSNorm with an in-kernel cross-TP all-reduce, a bundled rotate-half
RoPE in the same apply pass, PSUM I-group tiling that decouples the
S-multibuffering degree from the output width, graduated MX weight prefetch, and
individual QKV projections — plus the interface Wan calls it through. -->
<!-- meta: keywords: NKI, QKV projection, qkv, qkv_cte, context encoding, MXFP8, ROW_MX,
FP8, DistributedRMSNorm, RMS_NORM_ACROSS_HEADS, qk-norm, rotate-half RoPE, PSUM
banks, I-group tiling, weight prefetch, multi-buffering, replica_groups, all_reduce,
Wan2.2, DiT, diffusion, Trainium, trn3, NeuronCore-v4 -->
<!-- meta: content_type: kernel-reference -->
<!-- meta: date_updated: 2026-09-22 -->

This document covers `qkv_cte`, the fused QKV projection kernel that the Wan2.2 DiT runs for every self- and
cross-attention. This kernel is built on top of the NKI Library `qkv` kernel, whose interface,
quantization modes, and layouts are documented in the NKI library's
[QKV Kernel API Reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/qkv.html).
This page describes the five Wan2.2-specific changes layered on top: across-heads (Distributed)
RMSNorm, bundled RoPE, PSUM I-group tiling, graduated MX weight prefetch, and individual QKV
projections. Read this page if you are changing the copy in vLLM-Omni-Neuron, or mapping the same
pattern onto another model.

## Applies to

- **Model/component:** Wan2.2 T2V/I2V A14B DiT — the ROW_MX FP8 attention projections in
  `WanSelfAttentionFP8` (fused Q\|K\|V) and `WanCrossAttentionFP8` (separate Q, K, V).
- **Pattern:** the kernel first multiplies the row-packed MXFP8 activation by the fused QKV
  weight matrix, tiling the output into 512-wide column groups that each accumulate over the whole
  contraction in PSUM before being dequantized, bias-added, and stored to HBM. A second pass
  re-reads Q and K to apply an RMS norm over all heads (all-reduced across ranks) and rotate-half
  RoPE.
- **Platforms:** Trainium3. LNC2 by default.
- **Source:**
  [`qkv_cte.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/core/qkv/qkv_cte.py)
  (the MX path is where all of this lives),
  [`qkv_cte_utils.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/core/qkv/qkv_cte_utils.py)
  (arg validation, `QKV_CTE_Config` / `QKV_CTE_Dims`),
  [`qkv.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/core/qkv/qkv.py)
  (the CTE-only dispatch closure),
  [`qkv_shim.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/qkv_shim.py)
  (the `@nki.jit` torch shim), and
  [`row_mx_kernels.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/quantization/row_mx_kernels.py)
  / [`row_mx_modules.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/quantization/row_mx_modules.py)
  (the framework bridge and the two Wan call sites)

## Upstream behavior

Everything in the list below is **upstream behavior**, unchanged here, and documented in the NKI
Library's [QKV Kernel API reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/qkv.html)
(source: [aws-neuron/nki-library](https://github.com/aws-neuron/nki-library), `nkilib/core/qkv/`):

- `O[S, I] = A[S, H] @ W[H, I]` (+ bias) as one HBM-in/HBM-out launch, where `I = (num_q_heads + 2 *
  num_kv_heads) * d_head`.
- Automatic TKG/CTE dispatch on `B * S` against `SEQLEN_THRESHOLD_FOR_QKV_CTE` (96).
- Quantization modes including MXFP8 and `ROW_MX`, with the `[H//4, I, 4]` FP8 weight layout and the
  packed `H + 4` activation layout (per-row FP8 codes plus a 4-byte fp32 dequant scale tail).
- Optional fused *per-head* qk-norm (`RMS_NORM` / `LAYER_NORM`) and fused rotate-half RoPE, applied
  as each finished matmul result is brought from PSUM to SBUF, along with the dequant scaling and
  the bias add — avoiding an HBM round-trip.
- `BSD` (`[B, S, I]`) and `NBSd` (`[num_heads, B, S, d_head]`) output layouts, in-kernel KV-cache
  writes, and per-segment sum-of-squares outputs.
- The S-multibuffering schedule: a degree is chosen at trace time against the SBUF and PSUM budgets,
  and MX weights are either fully prefetched into SBUF or chunk-reloaded through
  `NUM_MX_WEIGHT_BUFFERS` rotating buffers.

## Design

### 1) Across-heads QK RMSNorm with a single cross-rank AllReduce

Upstream qk-norm is *per head*: each head's `d_head` slice is normalized by its own RMS, so it fits
inside the PSUM→SBUF copy of that head's bank. Wan2.2 uses `DistributedRMSNorm` instead — one RMS
per token, computed over the **entire** Q (or K) vector, which under tensor parallelism spans heads
that live on other ranks:

```text
  PER-HEAD RMSNorm (upstream)          ACROSS-HEADS RMSNorm (Wan, RMS_NORM_ACROSS_HEADS)
  ┌──────┬──────┬──────┬──────┐        ┌──────┬──────┬──────┬──────┐
  │ h0   │ h1   │ h2   │ h3   │        │ h0   │ h1   │ h2   │ h3   │  rank 0
  └──┬───┴──┬───┴──┬───┴──┬───┘        └──────┴──────┴──────┴──────┘
     │      │      │      │             ┌──────┬──────┬──────┬──────┐
   rms0   rms1   rms2   rms3            │ h4   │ h5   │ h6   │ h7   │  rank 1
   (independent, fits in one bank)      └──────┴──────┴──────┴──────┘
                                          └──────── one rms ───────┘  ← needs a cross-rank reduce
```

The AllReduce can only happen after all ranks have finished the QKV projection matmul, so applying
the RMSNorm must be split into a separate pass for self-attention. In cross-attention every head is
replicated on every rank, so no AllReduce is needed.

- **a — accumulate.** Once a projected S-tile has been copied to SBUF, the kernel computes its own `sum(x²)`
  over the full Q segment (and K segment) of the un-normalized output into an SBUF-resident buffer
  that survives every S-block.
- **b — reduce.** One SBUF-to-SBUF `ncc.all_reduce` over that buffer across the TP replica group,
  then `inv_rms = rsqrt(sum / (n_heads * d_head * tp) + eps)` for Q and for K. The kernel deliberately does a
  single AllReduce on the full QK sequence, instead of multiple smaller-sized collectives per
  S-block, since one collective costs about the same latency regardless of this size difference.
- **c — apply.** Re-read each S-tile's Q/K from HBM, multiply by the segment-wide gamma (per
  channel), and write back in place. When RoPE is not fused, `inv_rms` (per token) is applied here
  too; with `fused_rope` set it is folded into the RoPE multiplies instead (see 2). V is never normed
  and is never re-read.

Because the norm is deferred, the PSUM→SBUF copy takes the **plain** (un-normalized, un-roped)
path even when `fused_rope` is set. The gamma also changes shape: `RMS_NORM_ACROSS_HEADS` takes a
segment-wide `[1, n_heads * d_head]` gamma rather than the per-head `[1, d_head]`, so the per-head
gamma broadcast must be skipped or its DMA width mismatches.

### 2) Bundled rotate-half RoPE in the same pass

Wan applies QK RMSNorm before RoPE. The apply pass already reads each S-tile's Q and K segments into
SBUF for the norm, so RoPE fuses there at no extra HBM cost to load QK. The kernel calculates RoPE on all
heads in four wide `[n_heads * d_head]` ops instead of ~`4 * n_heads` narrow `[d_head]` /
`[d_head/2]` ones, whose cost was dominated by fixed per-instruction Vector overhead:

```text
  out = (x * inv_rms) * cos  +  [-x_hi, x_lo] * inv_rms * sin
         └──── one wide op ─┘     └──── two wide ops ────┘  └─ one wide combine op ─┘
```

Two details carry the design:

- **`cos`/`sin` are position-only**, so they broadcast across the head dim through a stride-0
  `.broadcast()` view of the per-position `[s, d_head]` tile — no materialized `[s, n*d]` copy.
- **`inv_rms` is folded into the cos/sin multiplies** rather than applied as its own pass, so the
  norm costs no extra traversal of the segment. That is why step c applies only gamma when
  `fused_rope` is set, and the full `inv_rms * gamma` only when it is not.

The kernel implements **rotate-half** RoPE, pairing lanes `(i, i + d/2)`. Wan's checkpoint is
interleaved-lane `(2j, 2j+1)`; the caller resolves that offline by permuting the Q/K weights, bias,
gamma, and cos/sin caches, so the kernel never sees the interleaved form. The apply pass is also
layout-aware: under `NBSd` each head occupies its own block of HBM rows, so a segment's heads move
in a single strided DMA rather than one transfer per head — `d_head` stays contiguous and only the
head axis is strided, keeping it a coarse gather/scatter.

### 3) PSUM I-group tiling

Wan's cross-attention duplicates all heads on all ranks to remove collective cost, which makes `I`
5120 (40 heads × 128) — wider than the 8 PSUM banks can cover at 512 columns each. Two hardware
rules drive that constraint:

| Memory | Role | Trn3 size | Relevant shape |
|---|---|---|---|
| **SBUF** | matmul operands staged here | 32 MiB = 256 KiB × 128 partitions, of which **~240 KiB per partition is usable** (16 KiB is reserved) | the kernel budgets its allocations per partition |
| **PSUM** | matmul accumulator | **8 banks** per partition, each **512** fp32 (2 KiB) = 16 KiB per partition | `NUM_HW_PSUM_BANKS = 8`, `F_MAX = 512` |

1. A single `nc_matmul_mx` writes exactly **one** PSUM bank, so a 512-wide output slab is one bank
   and wider output needs more banks side by side.
2. A bank accumulates one `(S-tile, 512-wide I-tile)` block over the **full `H` reduction** before
   it is read out. Reading it closes the accumulation group.

The contraction `H` sits on the partition axis; `S` and `I` are the free dims. So keeping `degree`
S-tiles live, each spanning the full output width, needs `degree * ceil(I / 512) ≤ 8`. Upstream
satisfies that by shrinking the degree, which for any output width ≥ 3584 (7 tiles or more) pins it
to 1 — and a projection this wide then had to be split into whole-head chunked sub-calls, which
cannot fuse the across-heads qk-norm and RoPE.

The kernel instead tiles the **output width**: the matmul processes `banks_per_group` 512-wide banks
at a time, copying each group to SBUF before its banks are reused, so PSUM only needs

```text
degree * banks_per_group ≤ 8            with banks_per_group ≥ 1

banks_per_group = min(ceil(I / 512), 8 // degree)
num_i_groups    = ceil(ceil(I / 512) / banks_per_group)
```

This decouples the degree from the output width. Its ceiling is no longer a fixed constant but the 8
PSUM banks. SBUF capacity can still bind it lower.
The replicated Q/K/V can then run as single wide calls with the across-heads RMS norm fused
in-kernel.

### 4) Graduated MX weight prefetch on the `H` axis

The weight matrix is reused by every S-tile of every S-block, so ideally it stays SBUF-resident for
the whole invocation. Upstream makes that a binary choice: **full prefetch** (whole matrix resident)
or **fully chunked** (every `H_128` tile re-streamed every S-block ⇒ `H_128_tiles × num_blocks` HBM
weight loads). The cliff between them is roughly `num_blocks`×, and the fused norm+RoPE working set
is often what pushes full prefetch out of budget.

Graduated prefetch fills the gap: keep the first `N = free_SBUF // per_tile_bytes` weight tiles
resident and chunk-reload only the remaining `H_128_tiles - N`. `N == H_128_tiles` is full prefetch,
`N == 0` is the legacy chunked path. Selection happens once per `i_weight_block` at the loop head:

```python
if use_weight_prefetch_mx:                    buf_idx = 0; resident            # full
elif graduated and i_weight_block < N:        buf_idx = 0; resident            # first N tiles
elif graduated:  buf_idx = 1 + (blk - N) % NUM_MX_WEIGHT_BUFFERS; chunk-reload # overflow
else:            buf_idx = blk % NUM_MX_WEIGHT_BUFFERS;          chunk-reload  # legacy chunked
```

Buffer 0 is the resident `[P_MAX, N, I*4]` slab; buffers `1..` are the rotating chunk
double-buffers.

#### The two tilings are orthogonal, and both are required

The weight matrix `W[H, I]` is sliced along perpendicular axes:

```text
                 I  (output columns)  ── I-group tiling ──▶  PSUM banks
               ┌──────────────────────────┐
      H  rows  │   h_chunk  ×  i_group    │  one tile = 512 H-rows (128 partitions × 4) × 512 I-cols
   (contract)  │                          │
        ▲      └──────────────────────────┘
        └── graduated prefetch ──▶ SBUF residency
```

| | Graduated prefetch | I-group tiling |
|---|---|---|
| **Axis** | `H` (contraction) | `I` (output width) |
| **Resource** | SBUF residency | PSUM banks |
| **Unit** | one H-tile: 512 H-rows, 4 packed per partition | 512-wide PSUM bank |
| **Loop** | inner (`i_weight_block`) | outer (`i_group`) |
| **Cuts** | weight HBM re-streaming | PSUM bank pressure → keeps `degree` high |

Neither alone is sufficient. PSUM bank count caps the **output-width × degree** product; SBUF
capacity caps **weight residency × degree**. Prefetch reduces how *often* weights are re-streamed
but cannot manufacture PSUM banks — a full-width output still needs more than 8. Tiling removes the
width cap on `degree` but does not create SBUF for residency.

They do share the SBUF budget, and the degree function arbitrates: after tiling lets `degree` rise
to 8, a **prefetch back-off** *lowers* it again when a smaller degree is what keeps the full weight
matrix resident, because one resident weight pass beats any amount of S-multibuffering:

```text
extra = SBUF the resident weight tiles would need beyond the chunk buffers
degree_with_prefetch = (SBUF budget - fixed allocations - extra) // per-S-tile cost
if degree_with_prefetch >= 1:
    degree = min(degree, degree_with_prefetch)
```

### 5) Individual QKV projections for cross-attention

Wan's cross-attention projects Q, K, and V through three *separate* calls, each producing one
segment. Those calls pass `num_kv_heads = 0`, so `I = num_q_heads * d_head` and there is no K
segment. Whenever `num_kv_heads == 0`, the K `sum(x²)` accumulation and the K `rsqrt` are skipped —
the whole projection is normalized as Q.

### Best practices

- **Put a fused op where its inputs are already complete.** The PSUM→SBUF copy sees one bank at a
  time, but the across-heads norm needs every bank of the row plus the other ranks' heads. Deferring
  it to its own pass costs one HBM round-trip over Q/K and composes with I-group tiling.
- **Tile the axis the hardware actually limits, then re-derive the cost model.** Shrinking the
  degree "fixed" the PSUM constraint while making the dominant cost (weight re-streaming) worse. The
  tiling was only half the change; inverting which quantity gets maximized was the other half.
- **Prefer a graduated knob to a binary one.** Full-or-nothing prefetch turns a small SBUF shortfall
  into a `num_blocks`× HBM regression. Partial residency makes the degradation proportional.
- **Keep a collective inside the kernel when the data is already on-chip.** A framework-level
  all-reduce operates on HBM tensors; in-kernel, the sum never leaves SBUF, so the faster SBUF-to-SBUF
  reduce becomes possible. You can also iterate on the collectives fusion in-kernel, and microbenchmark before
  committing.

## Interface

Shown with the values Wan's self-attention passes; cross-attention differences are noted inline.

```python
projected = row_mx_qkv_proj(
    packed_hidden,                            # [B, S, H+4] FP8, from the fused AdaLN kernel
    qkv_proj_weight,                          # [H//4, I, 4] FP8
    qkv_proj_w_scale,                         # [1, I] fp32
    qkv_proj_bias.unsqueeze(0),               # [1, I]
    num_heads=num_heads,                      # per-rank TP slice; all 40 for cross-attention
    head_dim=128,
    num_kv_heads=num_heads,                   # 0 for a single-segment Q-only/K-only/V-only call
    fused_rope=True,                          # cross-attention has no RoPE
    cos_cache=cos_rotate_half,                # [B, S, 128], rotate-half convention
    sin_cache=sin_rotate_half,
    qk_norm_pre_rope=QKNormConfig(
        q_norm=NormType.RMS_NORM_ACROSS_HEADS,
        k_norm=NormType.RMS_NORM_ACROSS_HEADS,   # None when num_kv_heads == 0
        eps=norm_q.eps,
    ),
    qk_norm_pre_rope_q_gamma=norm_q.weight,   # [1, num_heads * 128], segment-wide
    qk_norm_pre_rope_k_gamma=norm_k.weight,
    norm_eps=norm_q.eps,
    replica_groups=qkv_tp_replica_groups,     # None when every head is already rank-local
    output_layout=QKVOutputLayout.BSD,        # NBSd for cross-attention
)
```

| Argument | Type / shape | Description |
|---|---|---|
| `packed_hidden` | `[B, S, H + 4]` FP8 | Row-packed activation: `H` FP8 codes plus a 4-byte fp32 dequant scale tail. The kernel does **not** quantize BF16 input — produce this with `adaln_modulate(quant="row")` or `row_quantize_packed`. |
| `weight` | `[H // 4, I, 4]` FP8 | MX-contiguous fused weight, `I = (num_heads + 2 * num_kv_heads) * head_dim`. |
| `weight_scales` | `[1, I]` fp32 | Per-output-channel weight dequant scales. |
| `bias` | `[1, I]` | Fused bias (the caller `unsqueeze(0)`s a `[I]` parameter). |
| `num_heads` | int | Q heads **in this call** — the per-rank TP slice for self-attention, the full count for replicated cross-attention. |
| `head_dim` | int | `d_head` (128 for Wan2.2). |
| `num_kv_heads` | int | K/V heads. **0** for a single-segment (Q-only, K-only, V-only) projection. |
| `fused_rope` | bool | Apply rotate-half RoPE to Q/K in the norm/RoPE pass. Requires `cos_cache` / `sin_cache`. |
| `cos_cache`, `sin_cache` | `[B, S, d_head]` | **Rotate-half** caches. Wan's interleaved freqs must be converted offline, alongside the matching Q/K weight/bias/gamma permutation. |
| `qk_norm_pre_rope` | `QKNormConfig` or `None` | `q_norm` / `k_norm` = `NormType.RMS_NORM_ACROSS_HEADS` (set `k_norm=None` when `num_kv_heads == 0`), plus `eps`. Decomposed into flat scalars before the jit boundary. |
| `qk_norm_pre_rope_q_gamma` | `[1, num_heads * head_dim]` | **Segment-wide** gamma (not per-head `[1, d_head]`) for the across-heads norm. |
| `qk_norm_pre_rope_k_gamma` | `[1, num_kv_heads * head_dim]` | Same for K. Omit when there is no K segment. |
| `norm_eps` | float | Norm epsilon; also the fallback when `qk_norm_pre_rope` is `None`. |
| `replica_groups` | tuple-of-tuples or `None` | TP ranks for the in-kernel across-heads `all_reduce`. `None` ⇒ rank-local RMS (correct only when this call already holds every head). Must be tuples: lists are rejected as `@nki.jit` shape params. |
| `output_layout` | `QKVOutputLayout` | `BSD` `[B, S, I]` (default) or `NBSd` `[num_heads, B, S, d_head]`. |

Returns the projected activation in the requested layout, BF16.

## Adapting this kernel

- **Check whether your model's qk-norm is per-head before reaching for any of this.** If it is,
  upstream already handles it while copying results from PSUM to SBUF, and you want the stock
  kernel — the entire norm/RoPE pass, and the `num_i_groups == 1` restriction it lifts, exist only
  because Wan's norm spans heads and ranks.
- **Recompute the PSUM arithmetic for your `I`.** I-group tiling matters when
  `num_512_tiles_per_I` is large relative to 8. For `I ≤ 4096` upstream's degree cap yields a valid
  (if slow) schedule, so the win is a better schedule, not feasibility.
- **Re-derive the degree cost model against your own dominant cost.** The "maximize degree, shrink
  `banks_per_group`" choice assumes weight re-streaming from HBM dominates. If your weights are tiny
  relative to activations, the trade can invert.
- **Keep the RoPE convention conversion outside the kernel.** Rotate-half is what the kernel
  implements; interleaved-lane models pay for it once, offline, in a weight/bias/gamma/cache
  permutation — not per token on device.
- **Watch the compatibility assertions when adding a new PSUM→SBUF copy variant.** Anything that
  reads a whole S-tile's output width inline must either force `num_i_groups == 1` when the degree
  is chosen, or be rewritten to work per I-group. Trace-time assertions in the PSUM→SBUF copy catch
  the mismatch.
- **Validation:** the ROW_MX device tests in
  [`test_wan22_dit_row_mx_fp8.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/test/neuron/test_wan22_dit_row_mx_fp8.py)
  cover both call shapes against torch references. For a change that should not alter numerics, A/B
  the generated video end to end — the fused norm/RoPE ordering and the FP8 rounding are not
  visible in a single-tile comparison.

## Related reference

- NKI Library [QKV Kernel API reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/qkv.html)
  and [aws-neuron/nki-library](https://github.com/aws-neuron/nki-library) `nkilib/core/qkv/` — the
  upstream kernel this page is a delta against
- [`qkv_cte.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/core/qkv/qkv_cte.py)
  — the MX path: I-group loop, graduated prefetch, the degree cost model, and the norm/RoPE pass
- [`row_mx_modules.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/quantization/row_mx_modules.py)
  — the projection wrapper and the two Wan attention classes that call it
- [`row_mx_kernels.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/quantization/row_mx_kernels.py)
  — `row_mx_qkv_proj`, plus the o-proj and MLP ROW_MX entry points
- [Kernel implementations](index.md) — other kernel reference pages
