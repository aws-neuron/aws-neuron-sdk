# Kernel implementations

<!-- meta: description: Index of per-kernel design references for vLLM Omni Neuron
— what each vendored NKI kernel does, why it is shaped the way it is, and what to
reconsider when adapting the pattern to another model. -->
<!-- meta: keywords: NKI, kernel, reference implementation, vendored, Trainium,
attention, ring attention, adaptive layer norm, AdaLN, QKV projection, qkv_cte,
MXFP8, Wan2.2, diffusion -->
<!-- meta: content_type: index -->
<!-- meta: date_updated: 2026-09-17 -->

Some of the performance in this plugin comes from NKI kernels that are vendored into the repository
and modified for a specific model, rather than imported unchanged from the NKI Library. The pages in
this section document those kernels: what each one computes, the design decisions behind its shape,
its calling interface, and what changes when you map the same pattern onto a different model.

## Kernel reference pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Const-max ring attention
:link: ring-attention-const-max
:link-type: doc

Ring attention with a constant softmax max. Replaces softmax's online row-max with a provable per-row bound, which frees the score matmul to stay K-stationary so the ring's K/V rotation overlaps attention compute.
:::

:::{grid-item-card} Fused adaptive LayerNorm and FP8 quantization
:link: adaln-quant
:link-type: doc

Fused adaptive LayerNorm. Collapses a gated residual add, a (Layer|RMS|no-)norm, a per-hidden affine/AdaLN modulation, and optional per-row FP8 quantization into one launch, hoisting the loop-invariant modulation broadcast out of the token loop.
:::

:::{grid-item-card} QKV projection with QK Distributed RMSNorm and RoPE fusion
:link: qkv-cte
:link-type: doc

MXFP8 QKV projection kernel with an across-heads (Distributed) RMSNorm and in-kernel all-reduce, a bundled rotate-half RoPE, plus PSUM I-group tiling and graduated weight prefetch.
:::

::::

Wan2.2 vendors other kernels — the MLP variants and the distributed VAE
components — that are not documented here yet.

## When a kernel belongs here

These pages are for kernels you are expected to **read and adapt**. If you only want to call an
operation as it ships, the NKI Library's API reference is the right place, and adapting a vendored
copy is the wrong move — you would be forking away from upstream fixes for no benefit.

A page in this section exists because the kernel embodies a design decision worth reusing. It is
written to be useful even if you never run the kernel itself.

## Related information

- [Context parallelism](../../design/context_parallelism.md) — the parallelism strategy the attention kernel
  implements
- [Design: Engine, Worker, and Model Integration](../../design/vllm_omni_neuron_overview.md) — where kernels
  sit in the runtime
- [Onboard a model to vLLM-Omni Neuron](../onboarding-models.md) — the model-development
  workflow these kernels plug into

:::{toctree}
:maxdepth: 1
:hidden:

Const-max ring attention <ring-attention-const-max>
Fused adaptive LayerNorm and FP8 quantization <adaln-quant>
QKV projection with QK Distributed RMSNorm and RoPE fusion <qkv-cte>
:::
