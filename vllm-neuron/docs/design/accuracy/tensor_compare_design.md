# Tensor Compare Design

<!-- meta: description: Tensor compare design for accuracy validation -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-22 -->

## Overview

Tensor compare provides utilities for comparing intermediate tensors captured from different execution environments (HF CPU vs vLLM Neuron) to identify numerical divergence at the module level.

The core workflow is:

``` text
Capture (HF + Neuron) → Read → Align raw tensors → Reconstruct → Align reconstructed tensors → Compare → Report
```

Each step is pluggable: you provide model-specific reconstruction functions, and the framework handles the rest.

## End-to-End Workflow

``` python
from vllm_neuron.accuracy.tensor_io import read as tensor_io_read
from vllm_neuron.accuracy import (
    align_decode_captures,
    compare_captures_two_way,
    compare_captures_three_way,
    print_comparison_report,
    print_three_way_report,
)
from vllm_neuron.accuracy.tensor_alignment_utils import (
    align_and_truncate_hidden,
    hf_reference_reconstruction,
)

# 1. Read captures from disk
fp32 = tensor_io_read("/tmp/captures/hf_fp32")
bf16 = tensor_io_read("/tmp/captures/hf_bf16")
neuron = tensor_io_read("/tmp/captures/neuron")

# 2. Align decode steps by position (HF and Neuron may differ)
fp32 = align_decode_captures(fp32, neuron)
bf16 = align_decode_captures(bf16, neuron)

# 3. Two-way comparison (FP32 vs Neuron, static threshold)
results = compare_captures_two_way(
    fp32, neuron,
    ref_reconstruction_fn=hf_reference_reconstruction,
    test_reconstruction_fn=my_reconstruct,
    alignment_fn=align_and_truncate_hidden,
)
print_comparison_report(results, label1="HF FP32", label2="Neuron")

# 4. Three-way comparison (dynamic threshold via BF16 baseline)
results = compare_captures_three_way(
    fp32, bf16, neuron,
    reference_reconstruction_fn=hf_reference_reconstruction,
    target_reconstruction_fn=my_reconstruct,
    alignment_fn=align_and_truncate_hidden,
)
print_three_way_report(results)
```

See `examples/vllm_neuron/accuracy/compare_hf_vs_vllm_neuron_with_reconstruction.py` for a complete runnable example with Llama-3.2-1B-Instruct.

## Comparison Modes

### Two-Way Comparison

Compares two tensors directly with a static threshold:

``` python
from vllm_neuron.accuracy import compare_tensors

result = compare_tensors(cpu_tensor, neuron_tensor, name="layer_0")
# result.linf_rel — relative L-infinity error
# result.l2_rel   — relative L2 error
# result.max_abs  — maximum absolute difference
```

**When to use**: Quick check with a known threshold. Good for:

- CPU vs Neuron comparison (same dtype)
- Comparing two Neuron runs for determinism
- Module-level unit tests with known expected outputs

**Limitation**: Static thresholds don't adapt to different inputs or model architectures. Use three-way comparison for robust validation.

Example output:

``` text
TENSOR COMPARISON: HF FP32 vs Neuron
================================================================================
Threshold: L-inf > 0.01 highlighted in red

=== prompt_0 ===

--- Step 0 ---
Module                                               L-inf           L2      Max Abs
--------------------------------------------------------------------------------
model_embed_tokens/token0                         0.00e+00     0.00e+00     0.00e+00
model_layers_0_input_layernorm/token0             2.76e-03     1.95e-03     3.09e-02
model_layers_0_self_attn/token0                   3.52e-03     2.14e-03     5.21e-02
model_layers_0_mlp/token0                         4.18e-03     3.01e-03     7.83e-02
lm_head                                           3.78e-01     4.33e-01     4.86e+00

SUMMARY: 2 deviation(s) > 0.01, 0 shape mismatch(es)
```

### Three-Way Comparison

Compares both implementations against a common FP32 baseline:

``` text
Error₁ = |Neuron BF16 − FP32|     (target error)
Error₂ = |CPU BF16 − FP32|        (baseline error — expected precision loss)

Ratio = Error₁ / Error₂
```

- **Ratio ≈ 1.0** → Neuron matches expected BF16 quantization behavior ✓
- **Ratio \>\> 1.0** → Neuron introduces additional error beyond quantization ✗

``` python
from vllm_neuron.accuracy import compare_tensors_three_way

result = compare_tensors_three_way(
    baseline=fp32_tensor,      # Ground truth (FP32)
    expected=bf16_cpu_tensor,  # Reference quantized (BF16 CPU)
    actual=bf16_neuron_tensor, # Target (BF16 Neuron)
    name="layer_0"
)
# result.linf_ratio — ratio of L-inf errors
# result.l2_ratio   — ratio of L2 errors
# result.bc         — Bhattacharyya Coefficient (distribution similarity)
# result.passed     — True if ratios < multiplier threshold
```

**When to use**: Robust validation without manual threshold tuning.

**Default threshold**: Ratio \< 1.5x (configurable via `DynamicThresholdConfig`). For dense TP-only models (Llama), expect ratios close to 1.0 at the module level.

Example output:

``` text
THREE-WAY COMPARISON: HF BF16 & Neuron vs FP32
==========================================================================================
Ratio = Tgt/Base error. Red if ratio >= 1.5x

=== prompt_0 ===

--- Step 0 ---
Module                                    L-inf Ratio     L2 Ratio       BC
------------------------------------------------------------------------------------------
model_embed_tokens/token0                      0.0000       0.0000   1.0000
model_layers_0_input_layernorm/token0          1.0000       1.0000   1.0000
model_layers_0_self_attn/token0                1.0312       1.0099   0.9993
model_layers_0_mlp/token0                      0.8904       1.0694   0.9990
lm_head                                        1.1419       1.2111   0.9842

SUMMARY: 0 failure(s) (ratio >= 1.5x), 0 shape mismatch(es)
```

### Bhattacharyya Coefficient (BC)

For multi-token or multi-prompt validation, per-element error ratios can be noisy. BC compares the full error *distributions*:

- **BC = 1.0**: Identical error distributions
- **BC ≥ 0.99**: Excellent (production ready)
- **BC \< 0.95**: Significant divergence
- **BC = 0.0**: No overlap

BC is computed automatically in three-way comparison results.

### Error Metrics

**L-infinity (Relative)** — worst-case error, sensitive to outliers:

``` text
linf_rel = max|actual - reference| / max|reference|
```

**L2 (Relative)** — overall error magnitude, less sensitive to outliers:

``` text
l2_rel = norm(actual - reference) / norm(reference)
```

## Reconstruction Functions

### The `ReconstructionFn` Protocol

Reconstruction converts raw per-rank captured tensors into a single tensor comparable to the reference. The framework calls it as:

``` python
tensor = reconstruction_fn(rank_tensors, module_name, phase, positions)
```

**All reconstruction functions must accept these four parameters** (even if unused for a given model) so the compare framework can call them uniformly:

- `rank_tensors`: `List[Tensor]` — per-rank tensors sorted by rank index
- `module_name`: `str` — normalized module name (dots → underscores)
- `phase`: `str` — `"prefill"` or `"decode"`
- `positions`: `List[int]` — token positions from capture metadata

### Dense TP-only Models (Llama)

For models without sequence parallelism, all ranks produce identical outputs after the all-reduce in each TP layer. Reconstruction simply takes rank 0 and strips bucket padding:

``` python
def llama_reconstruct(rank_tensors, module_name, phase, positions):
    """All ranks identical after all-reduce. Use rank 0, strip padding."""
    tensor = rank_tensors[0]
    if positions:
        real = sum(1 for i, p in enumerate(positions)
                   if i == 0 or p > positions[i - 1])
        if tensor.shape[0] > real:
            return tensor[:real]
    return tensor
```

### HF Reference

HF runs full-sequence forward each decode step (recomputing all prior tokens). The hook captures `[1, seq_len, hidden]`. For decode, only the last token is the new result:

``` python
def hf_reference_reconstruction(rank_tensors, module_name, phase, positions):
    """Single rank; extract last token for decode."""
    tensor = rank_tensors[0]
    if phase != "prefill" and tensor.dim() >= 2:
        return tensor[:, -1:, :] if tensor.dim() == 3 else tensor[-1:, :]
    return tensor
```

### Dense SP Models (GPT-OSS on trn2)

GPT-OSS on trn2 uses sequence parallelism but outputs hidden states that are
already all-reduced by the time they are captured. The standard
`llama_reconstruct` function (rank 0, strip padding) works without modification.

### SP Models with Hidden-Dim Shuffling (GPT-OSS on trn3)

trn3 introduces a hidden-dimension shuffle in the attention and MLP kernels for
hardware efficiency. Captured tensors have their hidden dimension interleaved
and must be unshuffled before comparison with the HF reference. This requires a
custom `reconstruction_fn` that calls `_unshuffle_hidden_dim` after gathering
rank tensors. A custom `reconstruction_fn` passed to `TensorComparePlugin` is
required for trn3 GPT-OSS tensor comparison.

## Align Raw Tensors

Before reconstruction, raw captures from HF and Neuron must be aligned so that each decode step on the ref side corresponds to the same generation position on the target side.

HF and Neuron capture decode tensors differently:

| Source | Shape per decode step    | What reconstruction does |
|--------|--------------------------|--------------------------|
| HF     | `[1, seq_len, hidden]`   | Extract last token       |
| Neuron | `[batch_bucket, hidden]` | Strip batch padding      |

HF and Neuron may also produce **different numbers** of decode steps (HF captures every autoregressive step; Neuron may batch differently).

`align_decode_captures(ref, target)` matches them by token position:

1. Sort decode captures by `max(positions)`
2. Keep only ref steps whose position has a matching target step
3. Preserve prefill captures unchanged

``` python
from vllm_neuron.accuracy import align_decode_captures

fp32 = align_decode_captures(fp32, neuron)
bf16 = align_decode_captures(bf16, neuron)
```

## Align Reconstructed Tensors

After reconstruction, tensors from HF and Neuron may still differ in shape due to:

- **Batch dimension**: HF produces `[1, seq, hidden]` (3D) while Neuron produces `[seq, hidden]` (2D)
- **Hidden dimension padding**: Neuron pads hidden to a hardware-friendly size (e.g., 2048 → 2304)

`align_and_truncate_hidden(baseline, expected, actual)` normalizes these:

1. Promote 2D tensors to 3D (unsqueeze batch dim)
2. Truncate hidden dim to the minimum across all three tensors
3. Squeeze batch=1 back to 2D
4. Cast to float32

``` python
from vllm_neuron.accuracy.tensor_alignment_utils import align_and_truncate_hidden

# Pass as alignment_fn to compare functions:
results = compare_captures_three_way(
    fp32, bf16, neuron,
    alignment_fn=align_and_truncate_hidden,
    ...
)
```

The `alignment_fn` is called internally by the compare framework after reconstruction and before computing metrics. It returns `(t1, t2, t3, shapes_match)` where `shapes_match` is False if the tensors could not be aligned (e.g., sequence dimension mismatch).

## API Reference

``` python
# --- Tensor-level comparison ---
compare_tensors(t1, t2, name) -> ComparisonResult
compare_tensors_three_way(baseline, expected, actual, name) -> ThreeWayComparisonResult

# --- Capture-level comparison (operates on List[CapturedForwardPass]) ---
compare_captures_two_way(
    ref, test,
    ref_reconstruction_fn=None,
    test_reconstruction_fn=None,
    alignment_fn=None,          # defaults to align_and_truncate_hidden
    module_order=None,
    phase=None,                 # "prefill", "decode", or None for all
) -> Dict[prompt, Dict[step, List[ComparisonResult]]]

compare_captures_three_way(
    baseline, expected, actual,
    reference_reconstruction_fn=None,
    target_reconstruction_fn=None,
    alignment_fn=None,
    module_order=None,
    phase=None,
) -> Dict[prompt, Dict[step, List[ThreeWayComparisonResult]]]

# --- Alignment utilities ---
align_decode_captures(ref, target) -> List[CapturedForwardPass]
align_and_truncate_hidden(t1, t2, t3) -> (t1, t2, t3, shapes_match)

# --- Reporting ---
print_comparison_report(results, threshold=0.01, label1="Ref", label2="Test")
print_three_way_report(results, label_expected="BF16", label_actual="Neuron")
```

## Example Scripts

- `examples/vllm_neuron/accuracy/compare_hf_vs_vllm_neuron_with_reconstruction.py` — Full pipeline with Llama-3.2-1B-Instruct
- `examples/vllm_neuron/accuracy/compare_hf_vs_vllm_neuron.py` — Simpler rank-0 comparison (no reconstruction)

## Limitations

- **Same module names required**: Comparison matches tensors by module name. If HF and Neuron use different module hierarchies, the reconstruction function must normalize them.

- **Wildcard module capture causes dynamo recompile**: Using `modules: [".*"]` in `tensor_capture` config triggers a `torch.compile` recompile guard. Use explicit module patterns instead:

  ``` python
  # Works
  modules = ["model.layers.0-15.input_layernorm", "model.layers.0-15.self_attn", "lm_head"]

  # Fails
  modules = [".*"]
  ```

- **GPT-OSS on trn3 requires custom reconstruction**: trn3 shuffles the hidden
  dimension in attention/MLP kernels. Captured tensors must be unshuffled before
  comparison. Pass a custom `reconstruction_fn` to `TensorComparePlugin` that
  calls `_unshuffle_hidden_dim` after gathering rank tensors. GPT-OSS on trn2
  works with the default reconstruction (no unshuffle needed).

## Appendix: Statistical Background

### Dynamic Thresholds

Three-way comparison uses dynamic thresholds based on the baseline error:

``` text
pass if: |actual - baseline| < multiplier × |expected - baseline|
```

The default multiplier is 1.5x. Why \> 1.0?

Even when comparing the same precision (BF16 on both platforms), Neuron and CPU may not be numerically identical due to different accumulation orders in fused kernels. For dense TP-only models (Llama) at the module level, the ratio is typically ~1.0. Larger ratios (up to 1.5x) can appear at full-model scale due to error accumulation across layers.

### Bhattacharyya Coefficient Details

The BC measures overlap between two probability distributions. For tensor comparison, we bin the per-element errors into histograms and compute:

``` text
BC = Σ √(p_i × q_i)
```

where `p_i` and `q_i` are the normalized histogram bin values for the baseline and target error distributions respectively.

**Why BC over simpler metrics?**

- **Eliminates manual threshold tuning**: BC ≥ 0.99 works universally across models
- **Captures full error behavior**: Compares entire distributions, not just worst-case
- **Higher sensitivity**: Detects subtle systematic shifts that max-error misses
- **Robust to outliers**: A few high-error elements don't dominate

For multi-prompt validation, use `compute_aggregate_metrics()` to pool errors across prompts and compute aggregate BC.
