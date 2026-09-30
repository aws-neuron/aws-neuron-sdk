# Kernel reference: `adaln_quant_kernel` fused adaptive-LayerNorm + modulation + FP8 quant

<!-- meta: description: Reference implementation of fused adaptive LayerNorm on
Trainium: one launch that fuses a gated residual add, a (Layer|RMS|no-)norm along
the hidden dimension, a per-hidden affine/AdaLN modulation, and an optional per-row FP8
quantization that emits the packed H+4 layout the ROW_MX projection consumes — why
each piece is fused, how the modulation vectors are broadcast once instead of per
tile, and what to reconsider when adapting the pattern. -->
<!-- meta: keywords: NKI, adaptive layer norm, AdaLN, LayerNorm, RMSNorm, modulation,
scale shift, FP8, row quantization, ROW_MX, two-pass variance, gated residual,
fused kernel, broadcast matmul, Wan2.2, DiT, diffusion, Trainium, trn3 -->
<!-- meta: content_type: kernel-reference -->
<!-- meta: date_updated: 2026-09-15 -->

This topic documents `adaln_quant_kernel`, the fused adaptive-LayerNorm kernel the Wan2.2 DiT runs
before every attention, FFN, and output projection. It is a reference implementation of one idea:
collapse the norm, the timestep-conditioned modulation, the gated residual add, and the FP8
activation quantization that a diffusion transformer block does around each sub-layer into a single
launch. Read it for the kernel's design and interface, or as a pattern to adapt.

## Applies to

- **Model/component:** Wan2.2 DiT transformer block — `norm1` (AdaLN before
  self-attention), `norm2` (affine LayerNorm before cross-attention), and `norm3` (AdaLN before the
  FFN)
- **Pattern:** fuse a normalization, a per-hidden affine/adaptive modulation, an optional gated
  residual add, and per-row FP8 quantization into one HBM-in/HBM-out kernel, and hoist every
  loop-invariant broadcast out of the token loop
- **Platforms:** Trainium2/Trainium3; LNC2 by default
- **Source:**
  [`adaln_quant.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/experimental/norm/adaln_quant.py)
  (the kernel and its `@nki.jit` entry point),
  [`adaln_quant_constants.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/experimental/norm/adaln_quant_constants.py)
  and
  [`adaln_quant_tile_info.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/experimental/norm/adaln_quant_tile_info.py)
  (config), and
  [`adaln_kernels.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/quantization/adaln_kernels.py)
  (the `adaln_modulate` framework bridge and its unfused fallback)

## What you should know before reading

Before you start, you must be familiar with the following:

- **The NKI programming model:** kernels are `@nki.jit`-traced functions over `nki.language`
  tensors, with explicit SBUF/PSUM tiles, per-engine instruction placement (PE / Vector / Scalar),
  and a partition axis of at most `pmax` (128).
- **Adaptive LayerNorm in a DiT:** a diffusion transformer conditions each block on the timestep by
  producing per-batch `scale` / `shift` / `gate` vectors from the timestep embedding. Around
  **self-attention and the FFN**, Wan2.2 applies adaptive modulation `norm(x) * (1 + scale) + shift`
  before the sub-layer and a gated residual `x = residual + gate * sublayer(x)` after it.
  **Cross-attention differs**: it is preceded by a plain affine LayerNorm (`gamma` / `beta`) — or no
  normalization at all when `cross_attn_norm` is off — not adaptive scale/shift, and its residual is
  un-gated (`x = residual + sublayer(x)`).
- **Per-row FP8 (ROW) quantization:** each row is scaled by `amax / FP8_MAX`, rounded to FP8 E4M3,
  and stored with its fp32 dequant scale appended as 4 FP8 bytes — the `H + 4` packed layout the
  native ROW_MX QKV/FFN kernels consume.
- **LayerNorm numerics:** the variance must be computed the stable two-pass way (mean first, then
  the mean of squared deviations), not via the one-pass `Var = E[x²] − (E[x])²` shortcut, which
  loses precision to catastrophic cancellation when the per-token mean is large.

## Overview

Around each sub-layer a DiT block does the same unfused sequence — add the previous sub-layer's gated
output back onto the residual, normalize, apply the timestep-conditioned modulation, and (when the
next projection is FP8) quantize the activation and pack its dequant scale. Per row of the collapsed
`[OD, H]` view that is:

```text
  x    = residual + gate * hidden       # fused gated residual add (identity when residual=None)
  xhat = LayerNorm(x) | RMSNorm(x) | x  # norm along the hidden dim
  y    = xhat * mul + add               # per-hidden modulation (mul=1+scale, add=shift for AdaLN)
  out  = Quantize(y)                    # NONE -> y (input dtype); ROW -> per-row FP8 + H+4 scale tail
```

Each separate op reads from and writes to HBM, so that is up to five HBM round-trips per norm site —
and Wan2.2 has three norm sites per block, across 40 blocks and 40 denoise steps. This
kernel runs the whole sequence in one launch: one HBM read of `hidden` (and optionally `residual`),
an on-chip pipeline, one HBM write.

```text
  UNFUSED  (each box is its own HBM read + write)
  ┌────────────┐   ┌──────┐   ┌───────────┐   ┌───────┐   ┌──────┐
  │ res+gate·h │ → │ norm │ → │ ·mul +add │ → │ quant │ → │ pack │   5 round-trips
  └────────────┘   └──────┘   └───────────┘   └───────┘   └──────┘

  FUSED  (one HBM read, on-chip pipeline, one HBM write)
  ┌─────────────────────────────────────────────────────────────┐
  │  res+gate·h  →  norm  →  ·mul +add  →  quant  →  pack       │   1 round-trip
  └─────────────────────────────────────────────────────────────┘
     HBM in                                                 HBM out
```

The layout is deliberately simple. The collapsed `[OD, H]` view puts the token (outer) dimension on
the partition axis in tiles of 128 and the hidden dimension on the free axis, so the norm reduction,
the modulation, and the row quantization all run along the free axis with no transpose:

```text
                     free axis  ── hidden H (≤ 16384) ──▶
                    ┌───────────────────────────────────┐
   partition axis   │ tok 0   x x x x x x x x x x x x x │  ← norm reduces →
   (128 tokens per  │ tok 1   x x x x x x x x x x x x x │  ← modulate  →
    tile, ragged    │  ...                              │  ← row amax  →
    last tile)      │ tok127  x x x x x x x x x x x x x │
                    └───────────────────────────────────┘
   mul/add/gate:      [1, H] broadcast ↓ across all 128 partitions, ONCE (one PE ones-matmul)
```

`mul`, `add`, and `gate` are per-hidden vectors shared across every row, so the only PE work —
broadcasting them across the partition axis — is loop-invariant and hoisted out of the token loop.

The kernel is reached from `adaln_modulate` in
[`adaln_kernels.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/quantization/adaln_kernels.py), which the
Wan2.2 block calls at `norm1` / `norm2` / `norm3` in
[`wan2_2_transformer.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/models/wan2_2/wan2_2_transformer.py).

## Design

Given the layout above (token on the partition axis, hidden on the free axis, modulation vectors
broadcast once), three decisions matter more than the plumbing: how the normalization is computed,
and the two optional fusions the kernel wraps around it — the activation quantization and the gated
residual add.

### 1) Two-pass vs. one-pass normalization

The LayerNorm variance is always accumulated in fp32, and by a **two-pass, mean-first** formula —
not the one-pass `Var = E[x²] − (E[x])²` shortcut.

Why the shortcut is unsafe: `E[x²]` and `(E[x])²` are each large positive numbers, and when a
token's mean is large relative to its spread the two are nearly equal, so subtracting them cancels
almost all the significant bits and leaves the variance dominated by rounding noise (*catastrophic
cancellation*):

```text
  ONE-PASS   Var = E[x²]  −  (E[x])²          e.g. mean ≫ spread
             = 1000000.05  −  1000000.00   →   0.05   ← only ~2 good bits survive
               └── large ──┘  └── large ──┘        the subtraction; noise dominates

  TWO-PASS   μ   = mean(x)                        center first
             Var = mean( (x − μ)² )           →   0.05   ← every (x−μ) is small,
                          └ small ┘                       no large-minus-large step
```

The error grows with the mean, so it is invisible on zero-centered test tensors and only shows up on
real activations — as a Neuron-specific drift from the fp32 reference that the end-to-end accuracy
test is sensitive to. The two-pass formula avoids the large-minus-large subtraction entirely:

1. **Pass 1 — mean.** Reduce `x` along the hidden dim to the per-token mean `μ`.
2. **Pass 2 — variance about the mean.** Compute `Var = mean((x − μ)²)` directly. Each `x − μ` is
   already centered and small, so squaring and averaging it never produces the two huge nearly-equal
   terms the shortcut subtracts — this avoids the catastrophic cancellation caused by subtracting
   nearly equal raw moments.

`xhat = (x − μ) · rsqrt(Var + eps)` follows. On this hardware the two passes are done by the
`bn_stats` / `bn_aggr` primitives (which chunk the row, emit per-chunk partial statistics, then fold
them into `[mean, var]`), but the point is the mean-first algorithm, not those specific ops. RMSNorm
needs no mean, so it skips all of this: one fp32 sum-of-squares into a per-token scalar, then
`rsqrt(sum_sq / H + eps)`. `NO_NORM` skips normalization entirely and feeds the (post-residual)
input straight into the modulation.

### 2) Fused quantization option

When the projection that consumes this activation is an FP8 (ROW_MX) leaf, the activation has to be
per-row FP8-quantized and packed with its dequant scale — work the unfused path does as a **separate**
`row_quantize_packed` pass, one more full HBM round-trip after the norm. `quant="row"` folds that
pass into the same launch, so it disappears:

- **`quant="none"`** — store the normalized/modulated tile in the input dtype (bf16), a norm-only
  fusion.
- **`quant="row"`** — per row, take `amax = max|y|`, set `dequant_scale = amax / FP8_RANGE` (240 for
  e4m3, 448 for OCP e4m3fn), clamp it to `1e-6` for reciprocal stability, and write
  `codes = round(y / dequant_scale)` as FP8. The fp32 dequant scale is reinterpreted as 4 FP8 bytes
  and appended to each row, producing the `[..., H + 4]` packed layout the ROW_MX QKV/FFN kernel
  reads directly. An optional non-negative `lower_bound` clips both the amax and the values before
  scaling.

  ```text
    one output row, quant="row":
    ◀────────────── H FP8 codes ──────────────▶◀─ 4 FP8 bytes ─▶
    ┌───┬───┬───┬───┬───┬─────┬───┬───┐        ┌────────────────┐
    │ q │ q │ q │ q │ q │ ... │ q │ q │        │  fp32 scale    │   → [..., H + 4]
    └───┴───┴───┴───┴───┴─────┴───┴───┘        └────────────────┘     read as-is by ROW_MX
      q = round(y / dequant_scale)               dequant_scale, reinterpreted as 4 FP8 bytes
  ```

Both modes run the norm and modulation in the bf16 compute dtype, because ROW rounds to FP8 and NONE
rounds to bf16 at the store, and the store dtype dominates the output precision. The *variance* still
accumulates in fp32 as above. `MX` is accepted by the argument
validator but **not implemented** in this kernel: the bridge rejects `quant="mx"` with
`NotImplementedError`; MX is not supported through `adaln_modulate` (the standalone torch reference
*does* implement it). The kernel asserts against it at trace time.

### 3) Fused gated residual add

When `residual` is given, the kernel first computes the gated residual add `t = residual + gate *
hidden`, stores that bf16 sum to its own HBM output (the block's carried residual), and then
normalizes `t` for the main output — folding the block's post-sub-layer add into the same launch as
the next sub-layer's norm. The two outputs go different places:

```text
   hidden (sub-layer output)                     the previous sub-layer's residual
        │                                                    │
        ▼                                                    ▼
   gate · hidden  ────────────────►  t = residual + gate·hidden
   (fp32 product)                          │        │
                                           │        └──────────────►  residual_out
                                           ▼                          (bf16 carried residual,
                                    norm → ·mul+add → quant             NEVER quantized;
                                           │                            next block adds onto it)
                                           ▼
                                    modulated output (bf16 or FP8 H+4)
```

The gate product accumulates in fp32 and rounds to bf16 only at the add's store:

> The product accumulates in fp32 and rounds to bf16 only at the add's store, so an fp32 gate
> matches the pre-fusion framework gated add (`attn_out.bf16 * gate.fp32` in fp32).

This is why the bridge hands `gate` over in fp32 while `mul` / `add` come over in the compute dtype:
the gate is the one modulation vector whose rounding order is observable against the pre-fusion
result. `gate = None` is a plain (un-gated) add. The un-normalized `t` is returned as `residual_out`
and is **never** quantized — it is the value the next block adds onto.

### Best practices

- **Fold a whole block's per-norm plumbing into one kernel, not one op per fusion.** The win is not
  any single fused multiply; it is eliminating the HBM round-trips between norm, modulation, quant,
  and the residual add — up to five per norm site, three sites per block, across every block and
  denoise step.
- **Keep the reduction dtype independent of the tile dtype.** Run the tile in bf16 for throughput,
  but accumulate the LayerNorm variance in fp32 and with the two-pass (mean-first) formula. The store
  dtype dominates the output precision, while the fp32 variance avoids the cancellation error the
  one-pass formula introduces.
- **Let the consumer define the quant contract.** The `H + 4` packed layout exists because the
  ROW_MX projection reads it; fuse the quantization the downstream kernel actually wants, not a
  generic one.
- **Match a fused op's accumulation order to the path it replaces.** The gated add stays fp32
  because the framework did it in fp32; changing the rounding order would make the fused kernel
  disagree with the unfused reference it is meant to reproduce.
- **Hoist every loop-invariant broadcast out of the tile loop, and choose the layout so the
  dominant reduction is free-axis.** The per-hidden modulation vectors cost one PE ones-matmul, not
  one per token tile; and because everything reduces along hidden, hidden is the free axis and there
  is no transpose in the kernel at all.

## Implementation

One `@nki.jit` entry point, `adaln_quant_kernel`. It validates inputs, allocates the HBM outputs,
then dispatches to a single-core or 1D-SPMD-sharded driver; both funnel into
`_adaln_quant_single_core_kernel`, which loads the modulation vectors, broadcasts them once, and
loops over token tiles.

### Entry point: shape collapse, output allocation, dispatch

```python
tensor_proc_shape = _collapse_shape_major_dimensions(hidden.shape)   # (prod(all but last), H)
tile_info = build_adaln_quant_tile_info(tensor_proc_shape)
constants = build_adaln_quant_constants(tile_info, kargs.eps, tensor_proc_shape, ...)
_validate_kernel_input(hidden, mul, add, constants, residual, gate)
```

All leading dimensions collapse into one outer (token) dimension, so `[B, S, H]` and `[T, H]` are
the same `[OD, H]` view. For ROW quant the output is `[..., H + 4]` FP8; otherwise it matches the
input shape and dtype. When `residual` is given, a second bf16 HBM output holds the carried
residual, and the kernel returns `(residual_out, out)`.

> **Note:** MX quantization is validated by the argument dataclass but **not implemented** in this
> kernel. The bridge rejects `quant="mx"` with `NotImplementedError`; MX is not supported through
> `adaln_modulate` (the standalone torch reference does implement it).

### Load and broadcast the modulation vectors once

```python
if mul_sbuf != None:
    bc_mul_sbuf = nl.ndarray((pmax, proc), dtype=constants.compute_data_type, buffer=nl.sbuf)
    _broadcast_vector_full(tile_info, constants, mul_sbuf, bc_mul_sbuf)
# ... add likewise; gate broadcast into an fp32 tile with a gate-dtype ones stationary
```

`mul` / `add` broadcast in the compute dtype (lossless from the exact ones-matmul PSUM); `gate`
broadcasts into an fp32 tile so the gated add keeps fp32 accumulation. `nc_matmul` requires
same-dtype operands, so the gate broadcast uses its own gate-dtype ones stationary rather than the
shared compute-dtype ones.

### Per-token-tile loop: fuse, normalize, modulate, quantize, store

```python
for outer_tile_num in range(tile_info.outer_tile_count):
    _load_input_tensor_tile(...)                       # HBM -> SBUF, ragged last tile handled
    if residual_hbm != None:
        _fuse_gated_residual_tile(...)                 # t = residual + gate * hidden, in place
        nisa.dma_copy(residual_out_hbm, in_tile_sbuf)  # store carried residual before norm
    if kargs.needs_normalization():
        _normalize_tile(...)                           # LayerNorm (two-pass, fp32 stats) or RMSNorm
    _modulate_tile(...)                                # x = x * bc_mul + bc_add, plain tensor_tensor
    if is_row:
        _row_quantize_tile(...)                        # per-row amax -> FP8 + fp32 dequant scale
        _store_tile(...)                               # writes H FP8 + 4-byte scale tail
    else:
        _store_tile(...)                               # copy to output dtype and DMA out
```

The norm writes into a tile that aliases the input, so it takes its own fp32 reduction scratch
rather than reusing the work tile. `NO_NORM` skips the norm and feeds the raw (post-residual) input
straight into the modulation.

### Row quantization and the packed scale tail

```python
nisa.tensor_scalar_reduce(dst=abs_tile, data=in_tile, op0=nl.abs, ...,
                          reduce_op=nl.maximum, reduce_res=dequant_scales)   # per-row amax
# dequant_scale = amax / FP8_RANGE, clamped to min_dequant_scale for reciprocal stability
nisa.reciprocal(dst=quant_scales, data=dequant_scales)
nisa.tensor_scalar(dst=out_tile, data=in_tile, op0=nl.multiply, operand0=quant_scales, ...)
```

The fp32 dequant scale is reinterpreted as 4 FP8 bytes and DMA'd into the `H : H + 4` tail of each
row, producing the packed `H + 4` layout the ROW_MX QKV/FFN kernel reads. An optional non-negative
`lower_bound` clips both the amax and the values before scaling.

## Interface

The kernel ships as a pair: `adaln_quant_kernel` (the NKI kernel) and `adaln_modulate` (the
framework bridge the model actually calls). The interface below describes the bridge; the
Implementation section above describes the kernel.

```python
adaln_modulate(
    hidden, mul=None, add=None, *,
    eps, norm_type="layer_norm", quant="none", ocp=True,
    residual=None, gate=None,
)
```

Fused `Quantize(Norm(residual + gate * hidden) * mul + add)` with an unfused fallback.

| Argument | Type/shape | Description |
| -------- | ---------- | ----------- |
| `hidden` | `[B, S, H]` or `[T, H]` | Input. When `residual` is given, the sub-layer output being gated and added. |
| `mul` | `[B, 1, H]`, `[H]`, `[1, H]`, or `None` | Multiplicative modulation: AdaLN `1 + scale` (per-batch) or affine `gamma` (shared). |
| `add` | same as `mul` | Additive modulation: AdaLN `shift` or affine `beta`. |
| `eps` | float | Norm epsilon. |
| `norm_type` | `"layer_norm"` / `"rms_norm"` / `"none"` | Normalization along the hidden dim. |
| `quant` | `"none"` / `"row"` | `row` emits the packed `H + 4` FP8 layout; `mx` raises `NotImplementedError`. |
| `ocp` | bool | FP8 range: `True` -> e4m3fn (448), `False` -> e4m3 (240). |
| `residual` | `[..., H]` or `None` | Residual added to `gate * hidden` before the norm; enables the two-output fused-add form. |
| `gate` | same layout as `mul`, or `None` | Multiplier on `hidden` before the residual add (`None` = plain add). Requires `residual`. |

Returns the modulated activation when `residual` is `None`, else `(residual_out, modulated)` where
`residual_out` is the bf16 carried residual. The bridge loops over the batch dimension because the
kernel processes one modulation "batch" at a time (AdaLN `scale` / `shift` differ per batch element,
but `mul` / `add` are per-hidden vectors shared across a batch's rows).

## Adapting this kernel

- **Confirm your norm sites really share one shape.** The payoff here comes from three sites
  (`norm1` / `norm2` / `norm3`) reducing to `Norm(x) * mul + add` with a per-hidden `mul` / `add`.
  If your model's modulation is not a per-hidden affine — e.g. it depends on the token — the
  loop-invariant broadcast no longer holds and the hoist is invalid.
- **Keep the reduction fp32, whatever the tile dtype.** Port the two-pass (mean-first) LayerNorm,
  not the one-pass variance, or you will introduce a mean-dependent error that a single-tile test
  may not catch but an end-to-end accuracy run will.
- **Match the fused residual add's accumulation to your framework.** If your unfused block did the
  gated add in fp32, keep the gate fp32 in the kernel; if it rounded earlier, match that. This is
  the one fusion whose rounding order is directly observable against the pre-fusion path.
- **Decide the quant contract with the consumer, not the norm.** The `H + 4` packed layout exists
  because the ROW_MX projection expects it. If your downstream kernel wants a different scale layout
  (per-block MX, a separate scale tensor), the quant stage changes even though the norm does not.
- **Validation:** compare against the dependency-light torch reference
  ([`adaln_quant_torch.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/experimental/norm/adaln_quant_torch.py)),
  which defines the exact fused semantics the kernel and the unfused fallback must both satisfy.
  For a refactor that should not change numerics, A/B the generated video end to end, not one tile.

## Related reference

- [`adaln_quant.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/experimental/norm/adaln_quant.py)
  — the kernel: entry point, tile loop, normalize, modulate, gated residual add, row quantize
- [`adaln_kernels.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/quantization/adaln_kernels.py)
  — `adaln_modulate`, the framework bridge: dispatch, batch loop, and the unfused fallback
- [`adaln_quant_torch.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/experimental/norm/adaln_quant_torch.py)
  — the dependency-light torch reference defining the fused semantics
- [`wan2_2_transformer.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/models/wan2_2/wan2_2_transformer.py)
  — the DiT block's `norm1` / `norm2` / `norm3` call sites and `_leaf_quant`
- [Kernel implementations](index.md) — other kernel reference pages
