<!-- meta: description: Design of the GPT-OSS-120B giga-kernel (GK) decode integration in vllm-neuron: architecture, kernel contract, per-buffer KV-cache layout, and disaggregated-inference (DI) wiring. -->

# GPT-OSS-120B giga-kernel decode

The **giga-kernel (GK)** fuses the entire GPT-OSS-120B decoder stack (all 36
layers: attention + MoE) into a **single decode NEFF**, replacing the per-layer
decode loop. It is **decode-only**, runs at **logical-NC config 1 (lnc1)**, and
uses a **packed-FP8 KV cache**. The target deployment is disaggregated inference
(DI): a native per-layer **prefill** engine produces KV, which is transferred
over NIXL to a GK **decode** engine (`TP8 × DP16 × EP128`).

This document describes the integration as it stands: the model, runner, and
kernel seams, the per-buffer KV-cache layout, and the DI wiring.

## Why decode-only + DI

The kernel is a single lnc1 NEFF (grid[1]). A per-layer **prefill** graph uses
lnc2 and lowers TP/EP `SendRecv` collectives that do not lower at lnc1
(`NCC_IBTN243`). A fused single-engine server (prefill + decode in one `vllm
serve`, lnc1) therefore cannot compile. GK is architecturally a **DI decode**
component:

- **Prefill role** — native per-layer `model_mxfp4` (`TP4/EP2/lnc2`),
  `kv_role=kv_producer`.
- **Decode role** — GK `model_mxfp4_perf` (`TP8/DP16/EP128/lnc1`, packed-FP8),
  `kv_role=kv_consumer`.

The two roles run as independent single-node engines wired by IP over
NIXL/LIBFABRIC.

## Component map

| Component | File | Responsibility |
|---|---|---|
| Model factory gate | `model/gpt_oss/factory.py` | selects the GK perf model when `neuron_config.gk_decode` |
| GK decode model | `model/gpt_oss/model_mxfp4_perf.py` | decode-only weight holders; `get_kv_spec`, `bind_kv_cache`, `set_kv_cache_buffers`, `use_stacked_kv_cache` |
| Kernel assembly | `model/gpt_oss/mega_decode.py` | stacks per-layer weights once; builds attention-control tensors; passes the per-buffer KV; calls the kernel |
| Kernel entry | `functional/gpt_oss_mega_decode.py` | wraps `gpt_oss_decode_mega_kernel` |
| KV allocation | `vllm/worker/neuron_model_runner.py` | GK-gated per-buffer `initialize_kv_cache`; attn-metadata build |
| NIXL connector | `vllm/kv_connector/neuron_nixl_connector.py` | per-buffer registration; DCP read logic; SWA DI truncation |
| Greedy sampler | `functional/argmax.py`, `functional/sampling.py` | lnc-matched distributed argmax |

The GK path is confined to the perf model + its runner overrides; **non-GK
models are untouched** by the model/kernel changes.

## Kernel contract

The kernel (`gpt_oss_decode_mega_kernel`, an nkilib `c128_giga_kernel` fork)
consumes, per decode step:

- **Weights**, stacked once onto a leading `[num_layers, …]` axis in the exact
  per-layer shapes the kernel indexes as `W[i]` (see the layout list in
  `mega_decode.py`). Each stack is built **once** on the host and loaded to the
  device directly by its own weight loader (`build_stacked_weight_specs`), so the
  per-layer copies never reach the device. Stacking is not inside the traced
  `decode()`; restacking in-graph captured an ~810 MB `concatenate` per step.
- **KV cache**: `group_size` per-group base-0 buffers, passed as separate K and V
  args (see "KV-cache layout"). The kernel also accepts the stacked
  `K_cache`/`V_cache` and asserts exactly one form is present; it reads the
  packed-FP8 path from the K buffer rank.
- **Attention control** (layer-invariant, built from per-layer `attn_metadata`):
  RoPE cos/sin, a **full** block table + a **trimmed-SWA** block table, active
  mask, absolute + window-frame positions, and **two** KV-update indices
  (`kv_cache_update_idx` + `kv_cache_update_idx_swa`).
- **TP and EP rank meshes** as tuples-of-tuples (two `ReplicaGroup` args).

GPT-OSS alternates **even = sliding-window (SWA, window 128)** and **odd = full
attention** layers. The kernel bakes this parity: it selects the block table and
write index per layer by `i % 2`, and indexes the KV cache buffer by `i // 2`
(paired layers `2j`, `2j+1` share buffer `j`). Weights stay indexed by `i`; only
the KV-cache axis is `group_size`.

## KV-cache layout (per-buffer, HMA-preserving)

vLLM's Hybrid KV-cache Manager (HMA) reports GPT-OSS as **2 groups** (SWA + full)
and packs the 36 per-layer specs into **`group_size = 18` shared buffers** (each
buffer shared by one SWA + one full layer, `shared_by = [2i, 2i+1]`). `num_blocks`
is a single global scalar; all buffers are identical size.

`get_kv_spec` stays **honest and per-layer** (SWA layers report their window;
full layers report full attention) so HMA stays enabled and SWA keeps its reduced
per-request block-ID consumption. The GK-gated `initialize_kv_cache` branch
allocates **one separate tensor per HMA group**:

```text
group_bufs[i] : [2, num_blocks, kv, block, d]   # K half, then V half, per group i
K_i = group_bufs[i][0]   # kernel per-buffer K arg (packed-FP8: a trailing .view)
V_i = group_bufs[i][1]   # kernel per-buffer V arg
```

Each group is its own allocation the kernel indexes base-0, so the block-KV DMA
offset spans one group and stays under uint32 with no cap. Packed-FP8 K is a
trailing `.view([…, block//2, d, 2])` on `group_bufs[i][0]`, same bytes and same
`num_blocks`. V is never packed.

The buffers are handed to the kernel via a **GK-only** `set_kv_cache_buffers`,
which stores each as an individual model attribute (`K_cache_buf_i` /
`V_cache_buf_i`) rather than one list. A list would not survive the fork-child
meta-swap, which walks tensor attributes but does not recurse into a list, so the
buffers would stay on the real device and fail the meta-only compile. Per-layer
`bind_kv_cache({layer: [k, v]})` keeps the generic signature every model shares;
packed-ness is inferred from `k.dim() == v.dim() + 1`.

Each group buffer is `[2, num_blocks, …]` in its own storage, so the per-layer DI
region is that whole buffer and the NIXL connector registers each group's K and V
directly.

## DI wiring (NIXL)

**The problem.** The transfer must land each prefill KV block on the group buffer
block the kernel reads, matching what the native per-layer prefill registers.
vLLM's stock `_register_packed_kv_cache` assumes a block-major,
all-layers-interleaved ("DSv4 packed") layout and would address the wrong bytes.

**The fix.** For the GK decode (`gk_decode` set, `kv_role != kv_producer`),
`NeuronNixlConnectorWorker._register_packed_kv_cache` registers **per-buffer K/V
regions**, one per group buffer's K and one per group buffer's V, each
`block_len = kv·block·d` (one layer-pair block), de-duplicating buffers shared by
paired layers via base address. This reproduces what a native per-layer prefill
registers, so the handshake sees `head_ratio = 1` (match path) and a NIXL
descriptor `base_region + b·block_len` lands on that buffer's block `b`. A non-GK
cache defers to `super()`.

The prefill role runs the native per-layer model (`gk_decode=False`), which
registers the same per-buffer region set natively — so P and D region counts and
block-lens match with no producer-side change.

The connector also carries a **sliding-window DI truncation** fix
(`NeuronNixlConnectorScheduler`): on a hybrid SWA model the decode node admits
the prompt at length `N` but recomputes the last token at `N-1`, so the SWA skip
evicts the window's oldest block one step early at the `N % block_size ==
block_size-1` boundary and that block is never transferred. Both nodes are made
to operate at `N-1`. This is independent of the GK KV layout.

## KV budget (no cap)

Each group buffer is indexed base-0, so the block-KV DMA `HWAPOffset` spans one
group's `num_blocks · kv_heads · block · head_size`, which stays under uint32
(`NCC_IBTN127`) at any pool size the HBM holds. No GK-specific KV-budget cap is
needed; `num_blocks` is the standard profiled budget. `neuron_worker.py` no longer
adjusts it.

For the 120B GK config (`group_size=18`, `kv_heads=1`, `block=128`,
`head_size=64`) the profiled budget is **`num_blocks ≈ 36,175` per engine
(≈9.94 GiB KV, ~2.54M KV tokens)**, about 2x the count the earlier stacked uint32
cap allowed. The KV budget is now bounded by `gpu_memory_utilization`, not by any
GK offset limit.
