# Kernel reference: `ring_attention_const_max_fwd` const-max ring attention

<!-- meta: description: Reference implementation of ring attention with a constant
softmax max on Trainium: why dropping the online-max pass buys the K-stationary
layout that lets the K/V rotation collective overlap attention compute, how the
per-row Cauchy-Schwarz bound keeps that safe on a cross-head-normalized model, and
what to reconsider when adapting the pattern. -->
<!-- meta: keywords: NKI, ring attention, context parallelism, const-max, constant
max, softmax bound, Cauchy-Schwarz, collective_permute, K-stationary, P-transpose,
collective overlap, Wan2.2, DiT, diffusion, Trainium, trn2, trn3 -->
<!-- meta: content_type: kernel-reference -->
<!-- meta: date_updated: 2026-09-15 -->

This topic documents `ring_attention_const_max_fwd`, the ring-attention kernel the Wan2.2 DiT runs
under context parallelism on Neuron. It is a reference implementation of one trade: give up
softmax's online maximum, and the score matmul keeps a layout that lets the ring's K/V collective
overlap attention compute. Read it for the kernel's design and interface, or as a pattern to adapt.

## Applies to

- **Model/component:** Wan2.2 T2V/I2V A14B DiT self-attention, inference, whenever context
  parallelism is enabled (`cp_size > 1`)
- **Pattern:** replace softmax's online row-max with a cheap provable upper bound, so the score
  matmul can stay K-stationary and the ring's K/V rotation hides behind attention compute
- **Platforms:** Trainium2 and Trainium3 (Trainium1 is rejected at trace time); LNC1 and LNC2
- **Source:**
  [`ring_attention_const_max_fwd.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/experimental/attention/ring_attention_const_max_fwd.py)
  (the ring driver and the only entry point) and
  [`attention_const_max.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/experimental/attention/attention_const_max.py)
  (the per-step attention it drives)

> **Note:** This copy is a fork. It was vendored from the NKI Library
> (KaenaNeuronKernelLibrary) and then modified in-tree for Wan2.2, so it is not byte-identical to
> any upstream revision.

## What you should know before reading

Before you start, you must be familiar with the following:

- **The NKI programming model:** kernels are `@nki.jit`-traced functions over `nki.language`
  tensors, with explicit SBUF/PSUM tiles and per-engine instruction placement.
- **Softmax shift invariance:** subtracting any constant `c` from every score in a row leaves the
  softmax result unchanged, because `exp(s - c)` scales numerator and denominator by the same
  `exp(-c)`. The maximum in a standard softmax exists for numerical range, not for correctness.
- **Context parallelism:** the sequence is sharded across a CP group and each rank holds only its
  own slice of Q, K, and V. See [Context parallelism](../../design/context_parallelism.md).
- **Collectives on Neuron:** replica groups, `collective_permute`, and `all_reduce`. A collective's
  only real cost is the latency you fail to hide behind compute.

## Overview

Under context parallelism each rank owns `local_S = S / cp_size` tokens, but attention is not local:
every query must see every key. This kernel rotates K/V around the CP group instead of gathering it.
Queries stay resident on their own rank. Over `num_workers = cp_size` steps each rank's K/V slice
visits every rank exactly once, and each step contributes a partial result that folds into an
accumulator. No rank ever materializes the full K/V, so K/V memory stays at `1/cp_size`.

```text
  rank 0        rank 1        rank 2        rank 3
 ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐
 │ Q0    │     │ Q1    │     │ Q2    │     │ Q3    │   Q is resident — never rotates
 │ acc0  │     │ acc1  │     │ acc2  │     │ acc3  │   accumulator, unnormalized
 └───────┘     └───────┘     └───────┘     └───────┘
   K0,V0  ──▶    K1,V1  ──▶    K2,V2  ──▶    K3,V3  ──┐   collective_permute, one hop per step
     ▲                                                │
     └────────────────────────────────────────────────┘

 step 0: acc_r  = attn(Q_r, K_r,   V_r)     ← initialize
 step 1: acc_r += attn(Q_r, K_r-1, V_r-1)   ← pure addition, no rescaling
 ...
 step 3: acc_r += attn(Q_r, K_r-3, V_r-3)   then one final o = acc_o / acc_sum
```

The kernel is reached from `WanSelfAttention.forward` through `wan_cp_self_attention` in
[`wan2_2_transformer.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/models/wan2_2/wan2_2_transformer.py).

## Design

### Dropping the online max buys a better layout

The Tensor engine computes `out = stationaryᵀ @ moving`, contracting over the partition axis of both
operands. Which operand you make stationary decides whether a transpose is needed afterwards.

Flash attention keeps a running row-max along the key axis. That reduction is cheap only when the
scores come out as `[Sq-partition, Sk-free]`, i.e. with Q stationary, because the reduction then
runs along the free axis. But the second matmul contracts over the key axis, so `P` must be
transposed from `[Sq, Sk]` to `[Sk, Sq]` before it can feed `PV`. That is the P-transpose.

With no online max to reduce, nothing prefers Q-stationary. This kernel makes K stationary: scores
come out as `[Sk-partition, Sq-free]` and `P` feeds the second matmul directly, so the P-transpose
never happens.

That matters because of how the P-transpose is implemented. `attention_cte`, which keeps the online
max, performs it as a `dma_transpose` inside its compute, and the hardware cannot run a
`dma_transpose` concurrently with a collective. While one is in flight the KV-rotation
`collective_permute` waits, and the ring loses its overlap.

The same layout makes the softmax denominator free. Augmenting `V` with a column of ones lets one
matmul produce both the weighted output and the row sum: `Pᵀ[V|1] = [Pᵀ V | Σ exp(s - c)]`.

### Picking a max that is safe without measuring it

Something still has to come off the scores before the exponential, or `exp` overflows. The kernel
uses a per-query-row bound computed at runtime from the vector norms:

```text
c_i = softmax_scale · ‖q_i‖ · max_j‖k_j‖
```

Because `q_i · k_j ≤ ‖q_i‖ · ‖k_j‖` (Cauchy–Schwarz), `c_i` is an upper bound on every score in row
`i`. The exponent is therefore always `≤ 0` and `exp` cannot overflow, for any input. This is a
bound, not an estimate.

The original const-max kernel subtracted a single hardcoded constant. That works when a model
applies QK-RMSNorm per head, because then every head's score scale is pinned and one number covers
all of them. Wan2.2 normalizes across the whole hidden dimension before splitting into 40 heads, so
only the total energy is pinned and individual heads float. Its learned RMSNorm gain is also not 1,
which shifts each head's scale again. A constant would have to guess both at once. Measuring the
norms at runtime skips the guess: the lengths already carry the floating per-head energy and the
learned gain, so nothing about the checkpoint has to be known and there is no retraining.

### A static bound makes the ring reduction pure addition

Ring attention normally needs online-softmax bookkeeping: each step's partial was normalized against
a running max that a later step may raise, so every partial has to be rescaled before it can merge.

Here the bound never changes. Query rows do not rotate, so `‖q_i‖` is local and fixed, and
`max_j‖k_j‖` is taken over the ring's *global* K once, with a single `all_reduce(max)` of a handful
of values per head. Both are identical at every ring step, so the merge is:

```text
sum_prev += sum_curr
o_prev   += o_curr        (both unnormalized)
```

One final normalize at the end. No correction factors, no running max, no per-step division. That is
also what lets the accumulator stay resident in SBUF for the whole ring instead of round-tripping
through HBM.

### Best practices

- **Compute side quantities inside a pass you already pay for.** The norms are squared and reduced
  off the tiles the transpose prologue has already loaded, on the Vector engine, while PE transposes
  and Scalar evicts. `O(S·d)` work against the score matmuls' `O(S²·d)`, on an otherwise idle
  engine, with no extra HBM read.
- **Give each query row its own bound.** A bound shared by many rows has to be large enough for the
  loudest of them. Applied to a quiet row, that same value subtracts far more than the row's own
  scores need, and the row's probabilities can underflow to zero. A per-row bound costs one
  length-S reduction and removes the problem.
- **Keep per-tile work off any in-order queue that also triggers collectives.** The compiler
  triggers this kernel's `collective_permute` from GpSimd, and that queue is in-order, so routing a
  per-tile copy stream there defers every permute trigger until the previous step's copies drain.
- **If a kernel is the only correct path, give it no selector.** Dispatch here is a capability
  check, so there is no configuration in which the wrong path can be chosen.

## Implementation

One `@nki.jit` entry point; everything else in both files is a private helper. The driver runs a
prologue, then the ring, then a normalize fused into the last ring step.

### Transpose prologue with fused norms

Q and K arrive token-major and are transposed to d-major for the score matmul. The transpose is a PE
`nc_transpose` with a Scalar-engine evict from PSUM. Each sub-tile goes to its own PSUM bank so the
transposes do not serialize on one accumulation group.

The norms ride along, off the same loaded tile, on the Vector engine:

```python
# Fused per-token L2² off the loaded group (Vector). Square the whole [_P, group_cols] group
nisa.tensor_tensor(
    dst=squared_buf[:_P, :group_cols],
    data1=loaded_tiles[buf][:_P, :group_cols],
    data2=loaded_tiles[buf][:_P, :group_cols],
    op=nl.multiply,
)
if q_sumsq_sb is not None:
    # Q: sum-over-head_dim for ALL sub-tiles in ONE reduce
    nisa.tensor_reduce(dst=q_sumsq_sb[...], op=nl.add, data=squared_view, axis=[2], keepdims=False)
else:
    for subtile_idx in range(subtiles_per_group):
        # K: fold per-token ‖k_j‖² into the running per-partition max.
        nisa.tensor_reduce(dst=per_token_sumsq, op=nl.add, data=squared_buf[...], axis=[1])
        nisa.tensor_tensor(dst=running_max_sumsq, data1=running_max_sumsq,
                           data2=per_token_sumsq, op=nl.maximum)
```

The two sides diverge because they need different things. Q keeps every row's norm, since the bound
is per row. K needs one maximum per head, so it folds into a running per-partition max and then
collapses to a scalar with one partition reduce.

> **Note:** Order matters. K is transposed **first** so its per-head max is ready as early as
> possible and the global-max `all_reduce` can be triggered before the Q transpose and the V copy,
> letting its launch latency hide behind them.

### Global key max

```python
ncc.all_reduce(dsts=[global_k_maxsq], srcs=[local_k_maxsq],
               op=nl.maximum, replica_group=replica_group)
```

One value per head, once per call. Q's norms need no reduce, because query rows never leave their
rank.

### Assembling the bound

Deferring the square roots to the end collapses the whole bound into one Scalar-engine instruction
per head, using `activation`'s scale operand to carry the broadcast key term:

```python
# c_i = sqrt(k_scale_sq · ‖q_i‖²) over all [128, num_grps] in one Scalar-engine op
nisa.activation(dst=c_tile, op=nl.sqrt, data=q_sumsq_head, scale=k_scale_sq)
```

> **Note:** `c` is stored in bf16 and the `scores - c` scratch in fp16, together keeping roughly
> 2 GB of SBUF traffic per 480p call off every engine. Rounding `c` is safe for the same reason the
> bound may be loose: it shifts a row's probabilities by one common factor that cancels. The shifted
> scores must be fp16 rather than bf16, because the exponential needs **absolute** precision in the
> exponent and bf16's 8-bit mantissa leaves about 4% error on a probability. Every partition must
> also hold the identical `c` for a given query column, or the common factor stops cancelling.

### The ring loop

K is transposed straight into the ring's first send buffer, so the collective rotates the
*already-transposed* K and no rank repeats the transpose. `kv_prefetch_depth + 1` rotating HBM
buffers hold the K/V in flight, and the first `prefetch_depth` permutes are hoisted so they overlap
step 0:

```python
if num_workers > 1:
    for prefetch_step in range(1, prefetch_depth + 1):
        _issue_kv_permute(
            k_bufs[(prefetch_step - 1) % num_kv_buf][chunk_start:chunk_end],
            k_bufs[prefetch_step % num_kv_buf][chunk_start:chunk_end],
            v_bufs[(prefetch_step - 1) % num_kv_buf][chunk_start:chunk_end],
            v_bufs[prefetch_step % num_kv_buf][chunk_start:chunk_end],
            replica_group,
        )
```

In the steady state each step launches the permute for the step `prefetch_depth` ahead *before*
running its own attention, so the hop is in flight while the Tensor engine works. Step 0 initializes
the accumulator. The last step folds and normalizes interleaved per head, so each head's output
write hides behind the next head's attention instead of running exposed as a tail.

### Accumulate, then normalize once

The whole cross-step reduction is one PSUM→SBUF operation per output tile, selecting initialize or
add:

```python
def _accumulate_partial(o_aug_psum, o_aug_acc_sb, aug_col, is_accumulate_step):
    sq_tile_size = o_aug_psum.shape[0]
    d = o_aug_psum.shape[1] - 1
    aug_dst = o_aug_acc_sb[:sq_tile_size, aug_col : aug_col + d + 1]
    if is_accumulate_step:
        nisa.tensor_tensor(dst=aug_dst, data1=aug_dst, data2=o_aug_psum, op=nl.add)
    else:
        nisa.tensor_copy(dst=aug_dst, src=o_aug_psum)
```

The accumulator is fp32. A bf16 accumulator would re-round the running sum at every ring step.
Output and sum share one tile, `d` output columns followed by the sum column, so the fold is a
single op.

The division happens once, at the end. Before it, the row sum is clamped to a floor near the
smallest fp32 normal (`_SUM_CLAMP = 1e-37`), so a row whose probabilities all underflowed divides by
the clamp instead of by zero. That floor also sets how far a bound may overshoot a row's true
maximum before it does damage: at roughly 85 nats of overshoot the whole row sum reaches the clamp
and the row is silently rescaled.

## Interface

```python
ring_attention_const_max_fwd(
    q, k, v,
    replica_groups=None, num_workers=1, softmax_scale=None,
    training=False, lse_dtype=nl.float32, kv_prefetch_depth=2,
)
```

Non-causal ring attention forward. Returns the attention output `o`, plus the log-sum-exp when
`training=True`.

| Argument | Type/shape | Description |
| -------- | ---------- | ----------- |
| `q` | `[b, seqlen, h, d]` | Query, token-major. Transposed to d-major internally. |
| `k` | `[b, seqlen, h, d]` | Key, token-major. Transposed into the ring send buffer, so the collective rotates already-transposed K. |
| `v` | `[b, seqlen, h, d]` | Value, token-major. |
| `replica_groups` | tuple of tuples of global ranks | The CP group for the ring's collectives. Must be hashable. |
| `num_workers` | int | Ring size, i.e. the CP degree. |
| `softmax_scale` | float | Defaults to `1/sqrt(d)`. |
| `training` | bool | Emit `lse = c_i + log(sum_exp)`. The Wan2.2 path passes `False`. |
| `lse_dtype` | dtype | LSE output dtype, default fp32. |
| `kv_prefetch_depth` | int | Ring steps ahead to launch the K/V permutes. Clamped to `[1, num_workers - 1]`. |

Outputs are `o` as `[b, h, seqlen, d]` and, when requested, `lse` as
`[b, h, 128, ceil(seqlen/128)]`. `seqlen` is the *per-rank* sequence length and may be any value; a
non-128-multiple tail is peeled and handled. Three constraints are asserted at trace time:
`d == 128`, MHA only (`q_h == k_h`, broadcast beforehand), and Trainium2 or later. Attention is
non-causal because the kernel takes no mask argument, so there is nothing to assert.

**Dispatch and fallback.** The caller is `wan_cp_self_attention`, and its predicate is a capability
check only:

```python
if cp_size > 1 and can_run_kernel(value):
    ring_out = _nf_ring_attend(query, key, value, replica_groups=cp_replica_groups,
                               num_workers=cp_size, scale=scale)
    return ring_out.transpose(2, 3).contiguous()
```

Where the kernel cannot run (CPU mode, fake-tensor tracing, NKI kernels disabled) the fallback
all-gathers K/V across the CP group and runs local flash attention.

> **Warning:** The ring branch has no **shape** gate, only the capability check. A head dimension
> other than 128, or GQA/MQA, fails a `kernel_assert` at trace time rather than quietly falling
> back. Causal attention is a different case: the kernel accepts no mask argument, so a causal mask
> cannot be expressed at all and nothing rejects it. Callers that need one must use the standard
> attention path.

The replica groups come from `get_cp_replica_groups` in
[`parallel_state.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/distributed/parallel_state.py), which
returns every ring in the world rather than just the caller's, because SPMD compilation needs the
full partition.

## Performance notes

- **Wins:** the K/V rotation runs while the Tensor engine works, so the collective is largely off
  the critical path, and K/V memory stays at `1/cp_size` because the full sequence is never
  materialized. On Wan2.2, self-attention MFU goes from roughly 44% on the online-max path to
  roughly 75%. Quality holds: on [VBench 1.0](https://arxiv.org/abs/2311.17982) (248 prompts, matched seed and settings) the per-row
  bound shows no difference against a measured GPU baseline.
- **Costs:** the bound is only tight enough on models whose normalization keeps per-token q/k norm
  spread bounded. That is the real precondition, and [Adapting this kernel](#adapting-this-kernel)
  covers how to check it. The kernel is also narrower than a generic attention implementation:
  `d == 128`, MHA, non-causal, Trainium2+.

## Adapting this kernel

- **Check the precondition first: how far apart are your per-token q/k norms?** The never-overflow
  guarantee is universal. Usefulness depends on the bound not overshooting a typical row, and the
  bound is set by the loudest key norm in the sequence, so the overshoot on an ordinary row scales
  with the ratio of loudest to typical:

  | Model normalization | Per-token norm spread | Verdict |
  |---|---|---|
  | Per-head QK-RMSNorm | Fixed | Safe; a hardcoded constant would also work |
  | Norm across all heads (Wan2.2, LTX, OLMo) | Bounded by `sqrt(D/d)` | Safe with the per-row bound; a hardcoded constant is not |
  | No QK norm, with massive-activation key sinks | Unbounded (100-1000x) | A typical row underflows. Do not port as-is |

- **Attention sinks are the one real failure mode, and the fix is local.** Decoder LLMs commonly
  develop a handful of tokens whose norms dwarf everything else, which inflates `max_j‖k_j‖` for
  every row. Excluding or clipping those few keys in the max reduction recovers the bound without
  giving up the K-stationary layout. Bidirectional DiTs have no privileged token, so this does not
  arise for Wan2.2.
- **Granularity is not a tuning knob.** Per row is the answer unless you can re-run the overshoot
  argument for your model and show the coarser span's loudest and quietest rows are close. We tried
  coarser on Wan2.2 and it cost visible sharpness.
- **Validation:** compare against exact attention on CPU, never against another Neuron path, because
  a second approximate path cannot tell you whether the bound is sound. Run the reference
  implementation over the full sequence in both fp32 and bf16, then compare the two error
  distributions against the kernel's.
  Check quality end to end as well. A bound that is loose but survives will pass a single-step test
  and still degrade over dozens of denoise steps, which is exactly how the per-block bound got
  through review the first time. For a refactor that should not change numerics, the strongest check
  is an A/B on the generated output.

## Common issues

### Output is all zeros, or the generated video is black

- **Possible solution**: A query row's probabilities all underflowed, so the row divided by the
  clamp and came out meaningless. The bound overshot that row's true max by more than ~85 nats,
  which means it was set by something the row does not share. Check the per-token norm spread and
  look for key sinks (see [Adapting this kernel](#adapting-this-kernel)). The bound cannot overflow,
  so underflow is the only numerical direction worth investigating.

### Output is subtly washed out, but every single-step test passes

- **Possible solution**: A small per-step approximation compounds over the denoise loop. A loose
  bound that never reaches a floor is harmless in relative terms, but a genuinely bad bound, or a
  clamp set near the working range instead of at the fp32 floor, will rescale peaked rows. Compare
  end-to-end output over the whole loop, and do not rely on a single step.

### `kernel_assert` failure on head dimension, head count, or platform

- **Possible solution**: These three are asserted: `d == 128`, MHA only, and Trainium2 or later.
  There is no fallback for a violation. Broadcast GQA/MQA K/V to full heads before calling, and keep
  the standard attention path for anything else, including any workload that needs a causal mask.

## Related reference

- [`ring_attention_const_max_fwd.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/experimental/attention/ring_attention_const_max_fwd.py)
  — the ring driver: prologue, norms, bound assembly, ring loop, normalize
- [`attention_const_max.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/kernels/nkilib/experimental/attention/attention_const_max.py)
  — the per-step K-stationary attention the driver folds into its accumulator
- [`wan2_2_transformer.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/models/wan2_2/wan2_2_transformer.py)
  — `wan_cp_self_attention` (dispatch) and `_nf_ring_attend` (the launch wrapper)
- [`parallel_state.py`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/vllm_omni_neuron/diffusion/distributed/parallel_state.py)
  — `get_cp_replica_groups`, and how CP groups are registered for SPMD compilation
- [Context parallelism](../../design/context_parallelism.md) — the CP design this kernel implements
- [Kernel implementations](index.md) — other kernel reference pages
