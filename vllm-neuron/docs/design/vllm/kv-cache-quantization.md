# KV Cache Quantization

<!-- meta: description: KV cache quantization design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

This document introduces KV cache quantization, outlines the mechanics that apply to any backend, and describes the Neuron-specific aspects of the current implementation in vLLM Neuron.

For background on how KV caches are managed in vLLM (block layout, specs, binding), see `vllm-integration-kv-cache`.

## Motivation

The KV cache is typically the dominant on-device memory consumer in high-throughput LLM serving. Its size scales as:

``` text
kv_cache_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_value
```

and scales linearly with both sequence length and the number of concurrent requests. Maximum sequence length is a model property, but concurrency is a knob we tune to balance latency and throughput. Reducing `bytes_per_value` (e.g., BF16 → FP8) shrinks the cache, freeing HBM for more concurrent requests and reducing the data movement required per token at the cost of quantization error and (depending on where the scales are applied) some extra compute.

As a concrete example, for GPT-OSS 120B (36 layers, 8 KV heads, head dim 64), a single 131k-token sequence takes ~9.66 GB in BF16 and ~4.83 GB in FP8. Across many concurrent requests this difference directly translates into additional throughput on a fixed HBM budget.

## Background: Quantization Mechanics

Quantization converts a tensor from a higher-precision dtype to a lower-precision dtype while introducing the smallest possible value-wise error. The typical scaled (`absmax`) approach is:

``` text
scale         = QUANT_DTYPE_MAX / absmax(X)     # per-tensor/channel/group
X_quant       = round(clamp(X * scale, -QUANT_DTYPE_MAX, QUANT_DTYPE_MAX))
X_dequantized = X_quant / scale
```

The same scale is used to dequantize back to the original range. Scales can be computed at different granularities (per-tensor, per-token, per-channel, per-group) trading off metadata size against quantization error.

For a matrix multiply or dot product of two quantized tensors with reciprocal scales `s_a` and `s_b`, the scale application commutes out of the reduction:

``` text
A @ B = (A_quant @ B_quant) / (s_a * s_b)
```

This is the property that lets us apply KV scales after attention's reduce steps (matmul + softmax + matmul) instead of dequantizing the full cache up-front.

## KV Cache + Attention Quantization

Two quantization variants are commonly discussed:

**FP8 KV cache, BF16 attention (memory reduction).** K and V are quantized to FP8 before being written to the cache and dequantized back to BF16 when read. The attention matmuls still run in BF16, so there is no compute speedup, but the cache footprint and its associated data movement are roughly halved.

``` text
┌── K(bf16) ─×s_k─→ K(fp8) ─→ KV cache ─→ K(fp8) ─×(1/s_k)─→ K(bf16) ──┐
hidden → W_QKV(bf16) ┤                                                                      ├→ Q@Kᵀ/√d → softmax → P@V → out(bf16)
├── V(bf16) ─×s_v─→ V(fp8) ─→ KV cache ─→ V(fp8) ─×(1/s_v)─→ V(bf16) ──┤
└── Q(bf16) ──────────────────────────────────────────────────────────-┘
```

**FP8 KV cache, FP8 attention (memory reduction + compute speedup).** Q is additionally quantized to FP8 and the two attention matmuls run in FP8 (with FP32 accumulation in hardware). The scales are fused into the matmul-level scalars so no per-element dequantization is needed inside attention:

``` text
bmm1_scale = q_scale * k_scale * (1 / √d)    # applied to Q @ Kᵀ output
bmm2_scale = v_scale                         # softmax output is already in [0, 1]
```

The current vLLM Neuron implementation targets the memory-reduction variant: K/V are stored as FP8 while the attention math is still performed in BF16, with the K and V scales folded into the attention kernel's `softmax_scale` and the attention output projection respectively (see `attention-integration` below).

## vLLM User Configuration

KV cache quantization is controlled through vLLM's standard cache config:

- `kv_cache_dtype`  
  `"auto"` (default) uses the model dtype. Set to `"fp8"` (alias `"fp8_e4m3"`) to enable FP8 KV cache quantization.

- `calculate_kv_scales`  
  When `True` (on GPU vLLM), estimates K/V scales from a single warmup batch of tokens. **Not supported on Neuron** today; see `scale-calibration` below.

- `kv_cache_scheme`  
  vLLM's quantization scheme descriptor for compressed-tensors checkpoints. On Neuron, only KV-cache-only schemes are accepted; see `quantization-config-validation`.

Scales themselves are expected to be provided by the model checkpoint (typically produced offline via `llm-compressor` calibration).

## Neuron Implementation

### Supported dtype

Supported `kv_cache_dtype` values on Neuron are defined in `vllm_neuron.utils.dtype_utils`:

``` python
QUANTIZED_KV_CACHE_DTYPES = ["fp8", "fp8_e4m3"]
_SUPPORTED_KV_CACHE_DTYPES = {
    "float32":   torch.float32,
    "float16":   torch.float16,
    "bfloat16":  torch.bfloat16,
    "fp8":       torch.float8_e4m3fn,
    "fp8_e4m3":  torch.float8_e4m3fn,
}
```

Unlike upstream vLLM (which maps FP8 types to `torch.uint8`), Neuron uses the native `torch.float8_e4m3fn` dtype so that ordinary PyTorch ops interpret the stored values correctly for fallback/simulation paths. Once all FP8 KV cache operations happen inside NKI kernels, this can be switched to the upstream behavior.

**TRN2 vs TRN3 clamp range.** TRN2 FP8 E4M3 has a max finite value of `240.0`, whereas TRN3-supported E4M3FN allows up to `448.0`. `FP8_CLAMP_MAX` is resolved once at import time based on platform and is used everywhere K/V are clamped before being cast into the cache. Because the PyTorch dtype is `float8_e4m3fn` but TRN2 only supports E4M3, the compiler is invoked with `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` whenever `cache_dtype in {"fp8", "fp8_e4m3"}` on TRN2 (see `neuron_model_runner.py`).

### Cache allocation

`NeuronModelRunner.get_kv_cache_spec()` resolves the user-facing dtype string into a torch dtype via `kv_cache_dtype_str_to_dtype` and propagates it through `FullAttentionSpec` / `SlidingWindowSpec` to vLLM, so that vLLM plans and allocates blocks with the correct per-element size. The actual cache tensors are then allocated in `NeuronModelRunner.initialize_kv_cache` with `dtype=kv_cache_dtype`.

### Attention integration (Llama3)

FP8 KV cache is currently wired into the Llama3 model (see `vllm_neuron/model/llama3/model.py`). Each `LlamaAttention` instance holds per-layer scale tensors, populated during weight loading:

``` python
self.k_scale       = None   # [1, 1] bf16 tensor (reciprocal of checkpoint scale)
self.v_scale       = None   # [1, 1] bf16 tensor
self.k_scale_float = 1.0    # Python scalar (avoid graph breaks from .item())
self.v_scale_float = 1.0
```

The scales stored on the module are **reciprocals** of the checkpoint scales. vLLM/llm-compressor store scales such that `X_quant = X / s`, but the kernel quantizes via `X * s`, so `_load_kv_cache_scales` inverts them once at load time:

``` python
val = 1.0 / checkpoint._get_slice(key)[:].to(dtype=torch.bfloat16, device=device)
```

If the checkpoint does not contain `k_scale`/`v_scale` tensors, the scales default to `1.0` (i.e., `absmax`-free FP8, which is rarely accurate enough).

**Quantize on write.** Before writing new K/V into the cache, attention multiplies by the scale, clamps into the FP8 range, and casts:

``` python
if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
    k_flat = (k.reshape(-1, self.head_dim) * self.k_scale) \
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX) \
                .to(self.k_cache.dtype)
    # same for V
```

**Dequantize on read via scale folding.** Inside the decode attention kernel, the K dequantization scale is folded into the softmax scale and the V dequantization scale is folded into the output projection weight, so no extra pass over the cached tensors is required:

``` python
NF.attention_decode(
    ...
    softmax_scale = self.scaling / self.k_scale_float,   # folds 1/k_scale
    W_out         = self.o_proj_weight / self.v_scale_float,  # folds 1/v_scale
    k_scale       = self.k_scale,   # used for quantizing decode tokens' K tensor
    v_scale       = self.v_scale,
    ...
)
```

The fallback Python path in `attention_decode.py` applies the same quantize/dequantize round-trip to active K/V so that its numerics match the kernel.

**Prefill path.** FP8 quantization is currently only applied in the decode path and when writing to the cache. The CTE (context encoding / prefill) kernels in `vllm_neuron/functional/attention/attention_cte.py` and `attention_segmented_cte.py` do not consume K/V scales today; K/V are quantized at cache-write time in the model attention module, but the prefill attention computation itself still runs in BF16 against freshly computed K/V.

### Scale calibration (offline only)

vLLM exposes three potential sources of K/V scales:

1. **No calibration** - all scales set to `1.0`.
2. **Warmup-based calibration** (`calculate_kv_scales=True`) - estimate scales from a single random-token batch during warmup, then fix them.
3. **Offline calibration** - use a tool like `llm-compressor` against a representative dataset to compute per-layer `k_scale`/`v_scale` and save them into the checkpoint.

Only (1) and (3) are supported on Neuron today:

- **Default** (checkpoint has no `*.k_scale`/`*.v_scale` tensors): scales fall back to `1.0`, which usually degrades accuracy substantially and is recommended only for debugging.
- **Offline llm-compressor calibration**: the expected production path. `_load_kv_cache_scales` reads `model.layers.{i}.self_attn.k_scale` and `...v_scale` out of the SafeTensors checkpoint and registers them on the attention modules.

Warmup-based calibration is **not** implemented. It would require executing the model end-to-end with a BF16 KV cache, inspecting K and V statistics per layer, computing scales, and re-binding them onto the attention modules before the compiled FP8 graph is used. Because vLLM Neuron compiles the model graph ahead of time with the cache dtype baked in, this would either require a separate pre-compile profiling pass or a parameterization of the scales that allows them to change without recompilation. Neither is currently plumbed through.

### Quantization config validation

`NeuronPlatform._validate_quantization_config` enforces that, when a checkpoint declares a `compressed-tensors` quantization config, only KV cache quantization is requested. Weight quantization, output-activation quantization, and input-activation quantization on non-attention targets are rejected up-front with a clear error message. This guards against loading a weight-quantized checkpoint that the Neuron plugin cannot yet serve correctly while still allowing FP8-KV-cache-only checkpoints produced by `llm-compressor`.

## Current Limitations

- FP8 KV cache is integrated only for Llama3; other models (e.g., GPT-OSS) currently ignore `kv_cache_dtype`.
- Scales must be provided by the checkpoint; there is no runtime / warmup scale calibration (`calculate_kv_scales` has no effect).
- FP8 is applied when reading/writing the KV cache only; the attention matmuls still run in BF16. FP8 attention (Q also quantized, matmuls in FP8) is not yet implemented.
