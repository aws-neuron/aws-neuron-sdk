# Optimizing Vision-Language Model

<!-- meta: description: How to reason about vision-language model performance on
vLLM Neuron — the three-stage cost model, roofline latency targets, sharding
trade-offs across the vision and text towers, reading host/system/device
profiles, and the features and kernels that close the gap. -->
<!-- meta: keywords: vLLM, Neuron, VLM, vision-language model, multimodal,
performance, roofline, MFU, MBU, vision encoder, sharding, tensor parallelism,
data parallelism, context parallelism, decode context parallelism, block packing,
encoder cache, Qwen3-VL, Trainium -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-07-31 -->
<!-- Jira: NMI-325 -->

## Overview

A vision-language model (VLM) runs in three on-device stages, and each is bound by
a different resource — so each is analyzed and optimized differently:

| Stage | What it does | Bound by | Metric to maximize |
| --- | --- | --- | --- |
| **Vision encoder (VE)** | Encodes images/video into embeddings | Compute | MFU (model FLOPs utilization) |
| **Prefill** | Encodes the text + merged vision prompt | Compute | MFU (model FLOPs utilization) |
| **Decode** | Generates output tokens | Memory bandwidth (weights + KV load) | MBU (memory bandwidth utilization) |

On image- and video-heavy workloads the vision encoder and prefill latency, measured by
time-to-first-token (TTFT), dominates the e2e latency. On long-output workloads decode dominates, measured by time-per-output-token (TPOT). Because the
vision tower compresses a long vision sequence (typically 4× fewer tokens into the
text model), the encoder could potentially be a large share of end-to-end latency even though it
has far fewer parameters than the text decoder.

Performance analysis on a VLM requires reasoning from a roofline model. This document elaborates this process: ground the analysis in a **representative workload**, compute a **roofline**
latency target per stage, use it to choose a **sharding scheme**, then **profile**
the running model to see the gap between the roofline and reality — and close that
gap with **framework features and specialized kernels**. This document assumes you already have the model deployed (see the
[Qwen3-VL deployment tutorial](../tutorials/tutorial-qwen3-vl-32b.md)), know the
configuration interfaces (see
[How to onboard a Vision-Language Model](onboarding-vlm-models.md#5-benchmark-and-tune-performance)),
and can capture profiles (see
[how to profile workloads](../guides/how-to-profile-workloads.md)). The design docs
referenced throughout are the reference for *why* each configuration behaves as it does:
[vision encoder parallelism](../design/parallelism/vision_encoder_parallelism.md),
[block packing vision attention](../design/multimodal/block_packing_attention.md),
[M-RoPE](../design/multimodal/mrope.md), and the
[on-device encoder cache](../design/multimodal/on_device_encoder_cache.md).

**Qwen3-VL-32B** on a `trn2.48xlarge` is the running example, but the reasoning
applies to any VLM on any Trainium instance. Throughout, `world_size` is the total
number of NeuronCores (ranks) exposed by the instance — check `neuron-ls` on your
host, since it depends on the instance type and the logical-NeuronCore (LNC)
configuration.

## Grounding in a representative workload

Analysis is only meaningful against traffic that looks like production. Two
properties drive vision cost — **how many** media items per request and **at what
resolution** (resolution sets the vision token count per image). From those, the
context length each stage sees follows directly:

```text
vision_seq_len_per_image     = (H / patch_size) * (W / patch_size)      # e.g. 512x512, patch 16 -> 1024
vision_seq_length            = num_images * vision_seq_len_per_image     # VE attention length
compressed_vision_seq_length = vision_seq_length / spatial_merge_area    # e.g. /4, tokens into text model
total_context_length         = compressed_vision_seq_length + text_context_length   # prefill/decode length
```

For example, a 1-minute video at 1 frame per 2 seconds = 30 frames × 512×512 =
30,720 vision tokens, compressed 4× to 7,680, plus 128 text tokens → a 7,808-token
text context. The `random-mm` dataset in `vllm bench serve` reproduces such a
workload synthetically — set `--random-mm-bucket-config` to your measured
resolution mix (`(H, W, T)` → probability; `T=1` is an image, `T>1` a video with
that many frames) and `--random-mm-base-items-per-request` to your typical media
count:

```bash
vllm bench serve \
    --model /path/to/Qwen3-VL-32B-Instruct \
    --backend openai-chat --endpoint /v1/chat/completions \
    --dataset-name random-mm \
    --random-input-len 128 --random-output-len 256 \
    --random-mm-base-items-per-request 30 \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --random-mm-limit-mm-per-prompt '{"image": 30}' \
    --random-mm-bucket-config '{(512, 512, 1): 1.0}' \
    --num-prompts 10 --max-concurrency 1 --ignore-eos --temperature 0
```

Two properties of the workload matter for interpretation. First, the Neuron
scheduler admits **one request into prefill per step** (batch size 1), and a
request's vision-encoder inputs are scheduled together with that prefill — so the
vision encoder processes **one request's** images (which may still be many
images or video frames) per forward pass, not images batched across concurrent
requests. Size `num_vision_tokens_buckets` for the most images a single request
carries. Concurrency still affects decode batching and overall throughput, so
measure at your target concurrency. Second, keep the workload **fixed** while
comparing options — a latency change only means something when the input shape
didn't also change.

## Roofline analysis: setting a target

Roofline analysis computes each component's *minimum* latency per rank, assuming a
target device utilization (MFU/MBU). It is a lower bound —
real device time is always higher — so the gap between profile and roofline is what
optimization chases, and its size ranks which components are most worth optimizing.

Target device utilization measured by MFU (compute-bound stages) and MBU
(memory-bound stage) is **as close to 100% as possible**, but this should be analyzed
**component by component** — a dense matmul can get much closer to the ceiling than
core attention. Utilization drops when one engine idles waiting for another engine
or a collective to finish. For example, attention chains across engines: `QKᵀ`
runs on the Tensor Engine, then **softmax** (row-max on the Vector engine, exp and
exp-sum on the Scalar engine) runs over that result, then `softmax·V` runs back on
the Tensor Engine. A naive kernel would leave the Tensor Engine idle during the
softmax; the shipped prefill attention kernel avoids most of that by software
pipelining — while one Q-group's `softmax·V` matmul runs, the next group's exp runs
on the Scalar engine and a later group's `QKᵀ` is already issued (see
[`attention_cte`](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_cte.py)).
The overlap is close but not perfect, so measured attention MFU still trails the
dense-matmul ceiling. Collectives are the other common stall — the all-reduce /
all-gather / reduce-scatter a given TP/DCP/SP degree adds rarely overlaps compute
fully. Benchmark those collectives with
[`nccom-test`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/nccom-test.html)
and fold their cost into the roofline, then use the roofline to compare sharding
options and choose a scheme
([Choosing a sharding scheme](#choosing-a-sharding-scheme)).

Two hardware parameters set the ceilings; look them up for your Trainium
generation from the Neuron hardware specs:

| Parameter | Applies to | Sets the ceiling for |
| --- | --- | --- |
| BF16 Tensor Engine throughput (FLOP/s per core) | Compute-bound stages (VE, prefill) | matmul latency |
| HBM bandwidth (bytes/s) | Memory-bound stage (decode) | KV-cache and weight load latency |

### Compute-bound stages (VE and prefill): latency from FLOPs

For a matmul-heavy stage, count the FLOPs per rank, divide by achievable
throughput (peak × MFU), and convert to milliseconds:

```text
latency = total_FLOPs_per_rank / (MFU * peak_FLOPs_per_core)
```

The FLOPs per attention block and MLP (all `2×` because a multiply-add is two
FLOPs), sharded by the parallelism under consideration — `TP` (tensor
parallelism, shards heads/hidden), `DP` (data parallelism, shards the
batch/block dimension), and `CP` (context parallelism, shards the sequence). These
formulas assume a **dense** model; a mixture-of-experts (MoE) decoder replaces the
MLP term with per-expert FLOPs times the number of active experts (and adds
expert-parallel collectives):

```text
# Attention (per rank), sequence length S, batch B, TP, DP, CP
QKV proj      = 2 * ceil(B/DP) * (S/CP) * hidden * head_dim * (n_q + 2*n_kv) / TP
QK^T          = 2 * ceil(B/DP) * S * S * (n_q/TP) * head_dim / CP
softmax @ V   = 2 * ceil(B/DP) * S * S * (n_q/TP) * head_dim / CP
O proj        = 2 * ceil(B/DP) * (S/CP) * (n_q/TP) * head_dim * hidden

# Gated MLP (per rank)  — the vision encoder MLP has no separate up projection
up + gate     = 2 * 2 * ceil(B/DP) * S * hidden * (intermediate/TP)
down          = 2 * ceil(B/DP) * S * (intermediate/TP) * hidden
```

`B` and `DP` mean different things per stage. In the **vision encoder**, `B` is the
`num_blocks` dimension (block-packed images), and vision DP scatters those blocks
across ranks, so `ceil(B/DP)` is the blocks each rank processes. In **prefill**, `B`
is the number of requests — always 1 on Neuron (the scheduler runs prefill at batch
size 1) — and there is no prefill DP, so `ceil(B/DP) = 1` and the `DP` term drops
out. The applicable prefill split today is `TP` (the `CP` term is shown for
completeness; prefill context parallelism is not yet available on Neuron — see
[Choosing a sharding scheme](#choosing-a-sharding-scheme)).

The goal is MFU as close to 100% as the operation allows. Dense matmuls (QKV / O
projection / MLP) can approach it; core attention is inherently lower, and the
vision encoder's core attention is lower still because its small head dimension
(e.g. 72) is below the **128-wide systolic array** of the NeuronCore
[Tensor Engine](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html)
and leaves part of the array idle — the engine needs matmul dimensions of at least
128 to run efficiently (see the
[NKI performance guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki_perf_guide.html)).
The achievable ceiling scales roughly with `head_dim / 128`, so a small vision
head dimension caps attention MFU well below the dense-matmul ceiling. Core
attention within one image scales with its `S²`, so raising an image's resolution
grows its attention cost quadratically. Adding *more* images at fixed resolution is
different: because each image only attends within itself, per-image cost stays
constant and the total grows linearly with the image count — but only if attention
is computed per image rather than over the concatenated sequence. This is exactly
what [block-packed vision attention](../design/multimodal/block_packing_attention.md)
guarantees: instead of one flat sequence over all images (whose `S²` would make
many-image cost quadratic), it packs images into fixed-size blocks and runs
attention per block — so `S` in the formulas above is the block size (not the full
vision sequence) and `B` becomes the number of blocks, keeping many-image cost
linear (how those blocks are then sharded across ranks is a sharding choice — see
[Choosing a sharding scheme](#choosing-a-sharding-scheme)).

### Memory-bound stage (decode): latency from bytes moved

Decode generates one token at a time, so it does very little compute per token and
is dominated by *moving bytes* from HBM, not by matmul throughput. Two things must
be read from HBM per layer, per rank, and **both** count toward MBU — the goal is
MBU as close to 100% as possible:

1. **Model weights** — the QKV, O, and MLP parameters. Decode processes only one
   token per sequence, so even across a batch there is little compute to amortize
   the weight load against, and these projections stay memory-bound on their own
   weights (unlike in prefill, where the long sequence makes them compute-bound).
   This only flips toward compute-bound at large batch sizes.
2. **KV cache** — the K and V tensors for every prior token in the context.

Their per-layer, per-rank byte counts:

```text
# QKV / O / MLP: bound by loading their weight tensors (per layer, per rank)
qkv_o_mlp bytes = (QKV + O + MLP parameter count on this rank) * bytes_per_elem
qkv_o_mlp latency = qkv_o_mlp bytes / (MBU * HBM_bandwidth)

# Attention: bound by loading the KV cache for the whole context (per layer, per rank)
kv bytes = 2 * ceil(B/DP) * (n_kv/TP) * seq * head_dim * bytes_per_elem
attention latency = kv bytes / (MBU * HBM_bandwidth)

# Per-layer decode latency, summed over layers
latency = sum over layers of (qkv_o_mlp latency + attention latency)
```

Which term dominates depends on context length. At short context, weight bytes
dominate; at long context (long video, long output), the KV term — whose decisive
factor is `seq` — dominates and grows unbounded, and if the KV cache spills,
effective bandwidth utilization collapses. That balance points to the decode
sharding choice — see [Choosing a sharding scheme](#choosing-a-sharding-scheme).

Summing the per-component targets gives each stage's floor. The stage with the
largest share of the end-to-end target is where optimization matters most, and the
per-component breakdown says *which* component (attention vs. MLP vs. a projection)
to scrutinize first in a profile.

## Choosing a sharding scheme

The vision and text towers shard independently (`tp_size * dp_size == world_size`
each). Re-running the roofline with different splits and picking the best per stage
is the core of the sharding decision. The three stages pull in different
directions.

### Vision encoder — higher DP for more images

Each image attends only within itself, so the encoder can process different images
(blocks) independently and gather once at the end.
**Block-level [data parallelism](../design/parallelism/vision_encoder_parallelism.md)**
scatters blocks across ranks, cutting the per-rank sequence length — which helps
the `S²` attention term most — and avoiding
[tensor-parallel](../design/parallelism/tensor_parallelism.md) collectives.

- **Full DP** (`tp_size=1`, the default) is best for multi-image / video
  throughput.
- **Higher TP** shards encoder weights; reserve it for a single very large image
  that must fit, or when the encoder doesn't fit in one core's weight memory.

The trade-off is that DP pads all images in a call to the largest image's length
(data skew hurts) and replicates encoder weights per rank. For the block
scatter/gather mechanics and the DP-vs-TP roofline, see
[vision encoder parallelism](../design/parallelism/vision_encoder_parallelism.md).
The split lives in `vision_neuron_config`:

```python
"vision_neuron_config": {
    "num_vision_tokens_buckets": [30720],
    "vision_attention_block_size": 1024,
    "tp_size": 1,        # no weight sharding (recommended: full DP)
    "dp_size": 16,       # scatter blocks across 16 replicas (tp_size * dp_size == world_size)
}
```

You only need to set one of `tp_size` / `dp_size`; the other is inferred so their
product equals `world_size`. Set only `tp_size` → `dp_size = world_size / tp_size`;
set only `dp_size` → `tp_size = world_size / dp_size`; set neither → full DP
(`tp_size=1`, `dp_size=world_size`); set both → they must multiply to `world_size`.

These values are model- and workload-specific — the numbers above are for the
30-image example, and a different model (e.g. Qwen3-VL-4B, whose vision config and
`patch_size` differ) or a different resolution needs different numbers. Compute
them from your workload:

- **`num_vision_tokens_buckets`** — the vision token count one request produces
  (prefill is batch size 1, so the encoder processes one request's images per
  step), from the [formula above](#grounding-in-a-representative-workload)
  (`vision_seq_len_per_image × num_images`). Add more entries if request sizes
  vary. Omit it entirely to let the buckets auto-generate from the serving config.
- **`vision_attention_block_size`** — must be **≥ the largest single image's token
  count** (`vision_seq_len_per_image`), rounded to a multiple of 128.

See the
[configuration reference](../guides/reference-configuration.md#vision-encoder-options-vision_neuron_config)
for the auto-generation rules and a capacity table.

### Prefill (text) — tensor parallelism, plus sequence parallelism

Prefill is compute-bound and its cost scales with the text sequence length. The
primary lever is **[tensor parallelism (TP)](../design/parallelism/tensor_parallelism.md)**,
which shards the attention heads and the MLP hidden dimension across ranks so each
rank does `1/TP` of the matmul work. Higher TP lowers prefill latency until the
collective overhead (an all-reduce per layer) or KV-head padding (`TP` capped at
one KV head per rank without further sequence splitting) stops paying off.

**Sequence parallelism (SP)** stacks on top of TP: it shards the sequence across
the layer's normalization and residual regions (outside attention), so those
elementwise ops run on `S/TP` tokens per rank. Its main win on Neuron is removing
transposes in the MLP that a non-SP layout would otherwise insert around the
all-gather / reduce-scatter boundaries.

Splitting the *sequence* itself during prefill (context parallelism) is a further
sharding optimization opportunity — the perf analyses project ~6–10% at CP 2–8 — but prefill
context parallelism is **not yet available** in vLLM Neuron.

### Decode (text) — TP + DCP for latency, DP for throughput

Decode reads two byte streams (weights and KV cache; see the
[decode roofline](#memory-bound-stage-decode-latency-from-bytes-moved)), and the
right split depends on which dominates and on whether you are optimizing latency
or throughput:

- **Low batch / short context** favors higher **TP** for the best per-token
  latency: TP shards the (dominant) weight bytes across ranks. Combine it with
  **[decode context parallelism (DCP)](../design/parallelism/dcp.md)** — set
  `--decode-context-parallel-size` — which distributes the KV cache across each GQA
  group (at the cost of two extra collectives) to also shrink the KV read.
- **High batch / long context** favors higher
  **[DP attention](../design/parallelism/attention_dp.md)** for the best
  throughput (OTPS/QPS): it shrinks the (now dominant) KV load per rank with no
  extra collectives.

### Tying the text tower together

In a **monolithic** setup (prefill and decode on the same instance), prefill and
decode should share one copy of the text-decoder weights to save memory, which
requires them to use the **same TP degree**. Giving them different TP degrees is
possible but doubles weight memory, since the same weights must be loaded twice
under two shardings.

With **[disaggregated inference (DI)](../design/vllm/disaggregated-inference.md)**,
prefill and decode run as separate instances with independent weights, which
decouples their sharding — you can give prefill and decode **different TP degrees**
and scale each independently (e.g. more decode ranks for a long-output workload
without over-sharding prefill).

Either way, the vision encoder is free to use a different split. Fold the collective
cost into the roofline for each candidate split (benchmark it with `nccom-test`, as
above) to rank them, then confirm the final choice against real profiles.

## Reading profiles: locating the gap

With sharding chosen, a profile of the representative workload (captured per
[how to profile workloads](../guides/how-to-profile-workloads.md)) reveals where
the gap to the roofline actually comes from. Three levels each answer a different
question. For an end-to-end walkthrough of capturing and reading these profiles on
a vLLM workload, see
[Profiling a vLLM Inference Workload on AWS Trainium](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/tutorials/performance-profiling-vllm.html).

(torch-profile--framework-overhead)=

### Torch profile — framework overhead

The torch profile captures CPU-side work — both the API-server host process and
the Neuron worker processes (which run on CPU and drive the device):

- **HF input pre-processing** on the **host** process — image resize,
  normalization, video frame sampling — often hundreds of milliseconds and easy to
  overlook.
- **Worker-side pre-processing** on the **Neuron worker** process — position
  embeddings, block packing, and attention-bound construction — plus the host work
  in `embed_multimodal()` around the encoder graph and the prefill-side read path
  (`_gather_mm_embeddings`); repeated `gather_mm_embeddings` calls between encoder
  and prefill are a classic removable cost.
- **Non-device framework/runtime overhead** — a small fixed amount (order ~20 ms)
  per step is normal; much larger is worth chasing. With async scheduling this
  host work is meant to overlap the previous step's on-device graph execution, so
  its wall-clock cost should be hidden — cross-reference the
  {ref}`system profile <system-nrt-profile--bubbles-and-data-movement>` to confirm it
  overlaps rather than showing up as a system profile bubble.

When host time rivals device time, it is the first thing to optimize — a faster
kernel doesn't help a request that is waiting on CPU pre-processing.

(system-nrt-profile--bubbles-and-data-movement)=

### System (NRT) profile — bubbles and data movement

The system profile shows the timeline across engines and the gaps between graphs.
With async scheduling working, the VE, prefill, and decode graph executions should
run back-to-back — the host prepares the next step's inputs while the device runs
the current graph — so the timeline is a near-continuous run of graphs with no
idle device time between them. Gaps mean the device is waiting on the host.

- **Bubbles between graph executions** — idle device time where nothing is
  scheduled, i.e. async scheduling failed to hide the host work; the
  {ref}`torch profile <torch-profile--framework-overhead>` shows what the device is
  waiting on.
- **Device↔CPU data transfer** (`nrt_tensor_write` / `nrt_tensor_read`) — most
  importantly the gap between the vision encoder and prefill, where embeddings are
  written out and read back. This is exactly what the
  [on-device encoder cache](../design/multimodal/on_device_encoder_cache.md) removes.

### Device profile — the gap to roofline

The device profile shows on-device compute vs. idle per NeuronCore, compared
against the roofline per component. View it in
[Neuron Explorer](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/index.html)
(its Device Trace Viewer gives the per-engine timeline, operator table, and
utilization). Read it per component:

- **MFU per matmul** — dense matmuls (QKV / O / MLP) should be highest; core
  attention is inherently lower, and vision-encoder attention lower still
  (head-dim-limited). A component far below what its type can reach is an
  optimization candidate — e.g. RoPE/M-RoPE regions often show very low MFU, which
  is usually recovered when the rotation is fused into the QKV projection.
- **MBU for decode** — a low value often signals KV spill (points back to the
  decode sharding choice), but can also mean the decode kernel isn't running: if the
  attention mega-kernel falls back to the flat torch path, or modular flow prevents
  fusion, MBU drops. vLLM Neuron decides kernel-vs-flat per call via
  `_can_use_attention_block_kernel` (in
  [`attention_decode.py`](https://github.com/vllm-project/vllm-neuron/blob/HEAD/vllm_neuron/functional/attention/attention_decode.py));
  when you see low decode MBU, first confirm the mega-kernel is actually enabled for
  your config before chasing sharding.

The per-component delta between profile and roofline is a ranked to-do list: the
largest absolute gap is the biggest opportunity.

## Closing the gap

On top of the sharding choice, three categories of optimization close the
remaining gap: multimodal-specific features, the general text-decoder features
that apply to any decoder, and kernels.

### Multimodal-specific features

- **Block-packed vision attention** packs variable-size images into fixed blocks
  and computes attention per block with no cross-block work, giving 2×–2.5× encoder
  efficiency on heterogeneous images; `vision_attention_block_size` tunes it to the
  resolution mix. See
  [block packing vision attention](../design/multimodal/block_packing_attention.md#choosing-the-right-block-size).
- **On-device encoder cache** keeps vision embeddings in an HBM buffer so repeated
  media skips re-encoding, and removes the encoder↔prefill CPU round-trip (the
  `nrt_tensor_read`/`write` gap and the `gather_mm_embeddings` cost seen in the
  profile); `encoder_cache_num_blocks` sizes it against the KV budget. See the
  [on-device encoder cache](../design/multimodal/on_device_encoder_cache.md).

### General text-decoder features

A VLM's text tower is an ordinary decoder, so the general vLLM Neuron features
apply to it unchanged — most relevant here are **prefix caching (APC)**, which
reuses the KV cache across requests sharing a prompt prefix (e.g. the same video or
system prompt across turns), and **segmented prefill**, which processes a long
prompt in `max_num_batched_tokens` segments to avoid compiling one NEFF for the
full context (worth it only for long-sequence workloads). These are not
multimodal-specific; see the [features guide](../guides/features-guide.md) for the
full set and how to enable each.

### Kernels

Where the device profile shows a low-MFU region, a specialized kernel often
recovers it. The usual optimization is **kernel fusion** — merging adjacent ops
(e.g. layer-norm, the QKV matmul, and RoPE) into one kernel to cut transposes and
intermediate reads/writes and keep the Tensor Engine fed. The reference
implementations live in the [NKI library](https://github.com/aws-neuron/nki-library); the kernels that apply per stage:

- **Vision encoder / prefill (compute-bound):** the QKV kernel
  (`core/qkv/qkv.py`), which can fuse the QK-norm and RoPE (M-RoPE for text prefill) into the projection; the prefill
  attention kernel (`core/attention/attention_cte.py`) — plus its KV-parallel
  segmented variant (`attention_kv_parallel_segmented_cte.py`) for long context;
  the standalone RoPE kernel (`core/embeddings/rope.py`) when not fused into QKV;
  the output-projection kernel (`core/output_projection/output_projection_cte.py`);
  and the MLP kernel (`core/mlp/mlp.py`). Block-packed vision attention builds on
  `attention_cte`.
- **Decode (memory-bound):** Qwen3-VL's decode uses the fused attention-block
  mega-kernel (`experimental/transformer/attention_block_tkg.py` — QK-norm + QKV +
  RoPE + attention + output projection in one call, via `NF.attention_decode`) plus
  the MLP kernel (`core/mlp/mlp.py`). The standalone decode attention and output
  projection kernels (`core/attention/attention_tkg.py`,
  `core/output_projection/output_projection_tkg.py`) are the unfused alternative
  when a model doesn't use the mega-kernel.

Which kernels exist (and their supported configurations) is documented in the
[NKI library](https://github.com/aws-neuron/nki-library); which ones a given model
already uses is visible in its modeling code (the `NF.*` calls). Before wiring a kernel into the model code, check its documentation to
confirm it fits the model architecture — e.g. the
norm type, pre- vs. post-norm placement, and RoPE variant it assumes — since a
kernel written for a different layout will produce wrong results. Then
validate it bottom-up: first at the **module level** against the HF reference, then
at the **model level**, following the
[accuracy debugging guide](accuracy-debugging-guide.md) and the
[onboarding accuracy validation](onboarding-vlm-models.md#4-validate-accuracy)
workflow.

## Iterative optimization loop

The preceding sections are the steps of a loop, not a one-shot procedure. Compute
the roofline once up front ([Roofline analysis](#roofline-analysis-setting-a-target)):
it is the fixed per-stage target for your workload and doesn't change as you tune.
The loop then repeatedly profiles the running model and compares against that
target. Optimizing one stage usually shifts the bottleneck — faster encoding
exposes a KV-bound decode, a smaller decode exposes prefill — so you repeat until
the largest gap to the roofline isn't worth closing:

1. **Profile** the fixed representative workload
   ([Reading profiles](#reading-profiles-locating-the-gap)). Hold the workload
   constant across iterations — a latency change only means something when the
   input shape didn't also change.
2. **Analyze the bottleneck** — compare the profile against the (already-computed)
   roofline and find the stage and component with the largest absolute gap. That
   gap is the ranked to-do list; the top item is the biggest opportunity.
3. **Optimize that opportunity** — apply the sharding, feature, or kernel that
   targets it ([Choosing a sharding scheme](#choosing-a-sharding-scheme),
   [Closing the gap](#closing-the-gap)). Change one thing at a time so the next
   profile attributes the effect cleanly.
4. **Validate bottom-up** — confirm accuracy first at the module level, then the
   model level, and confirm the performance gain on the
   *same* workload.
5. **Re-profile and re-analyze** — the bottleneck has likely moved; return to
   step 1.

## Related information

- [Deploy Qwen3-VL-32B](../tutorials/tutorial-qwen3-vl-32b.md) — Get the model
  serving first; this analysis optimizes that deployment.
- [How to onboard a Vision-Language Model](onboarding-vlm-models.md) — Full VLM-specific
  configuration API and summary.
- [Vision encoder parallelism](../design/parallelism/vision_encoder_parallelism.md)
  — TP/DP design, block scatter, and the DP-vs-TP roofline.
- [DCP (Decode Context Parallelism)](../design/parallelism/dcp.md) — KV-cache
  sequence sharding at decode; shrinks the per-rank KV read for the text decoder.
- [Parallelism design docs](../design/parallelism/index.md) — Mechanics of every
  strategy: tensor, data, attention DP, DCP, vision encoder, expert, and
  component DP sharding.
- [Block packing vision attention](../design/multimodal/block_packing_attention.md)
  — Packing algorithm, block-size tuning, efficiency analysis.
- [On-device encoder cache](../design/multimodal/on_device_encoder_cache.md) — Buffer layout,
  sizing, and eviction.
- [M-RoPE](../design/multimodal/mrope.md) — Multimodal position embeddings, relevant
  to the low-MFU RoPE regions and the fused attention kernel.
- [NKI library](https://github.com/aws-neuron/nki-library) — Reference NKI kernels
  (QKV, attention CTE/TKG, RoPE, output projection, MLP, transformer mega-kernels).
- [How to profile workloads](../guides/how-to-profile-workloads.md) — Capturing and
  reading profiles.
- [Profiling a vLLM Inference Workload on AWS Trainium](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/tutorials/performance-profiling-vllm.html)
  — AWS Neuron tutorial: capture and read system- and device-level profiles end to end.
- [Neuron Explorer](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/index.html)
  — The profile-viewer UI (Device Trace Viewer, System Trace Viewer) for reading device profiles.
- [Features guide](../guides/features-guide.md) — Prefix caching, segmented prefill,
  quantization, and the text-decoder configurations that apply to a VLM's language model.
