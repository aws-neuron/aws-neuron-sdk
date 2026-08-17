# Decode Context-Length Bucketing

<!-- meta: description: Context-length bucketing design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Problem

vLLM Neuron compiles each decode NEFF for a fixed shape: one NEFF per batch-size bucket, with the attention block-table sized to `max_model_len` blocks. Every decode step reads `ceil(max_model_len / block_size)` KV blocks per request from HBM regardless of the request's actual position in the sequence. The attention mask zeros out the unused range so the *result* is correct, but the DMA traffic is real and grows with `max_model_len`.

In disaggregated inference, `max_model_len` is typically configured to the largest request the cluster expects to serve (e.g. 131072), but the median in-flight effective KV is much smaller. A request whose real KV is 4K still pays for `131072 / block_size` blocks of HBM traffic on every decode step. That's a ~32× DMA overhead amplifier on the dominant operation in the dominant phase of inference.

The same effect exists in non-DI deployments where `max_model_len` exceeds the typical workload, but DI is where it's loudest: prefill and decode are on different workers, decode is the sustained bottleneck, and `max_model_len` is set conservatively.

## Idea

Let the user opt in to a second bucketing dimension on decode: in addition to the existing `num_seqs_buckets` (one NEFF per batch size), accept a list of *decode context-length buckets*. The decode worker compiles one NEFF per `(batch, seq)` pair, plus an implicit `max_model_len` fallback that is always available.

**Bucket selection is per-forward-pass and adapts to the live batch.** On every decode step the runtime inspects the requests currently in the batch, computes the largest KV position any of them will touch on this step, and dispatches the smallest configured bucket that covers it; if no configured bucket fits, it falls back to the `max_model_len` NEFF. As a request grows past a bucket boundary, the next step automatically transitions to the next-larger NEFF — no recompilation, no Python-level dispatch logic, just a shape-specialized cache hit.

The feature is **opt-in**. With no `decode_context_length_buckets` configured, behavior is bit-identical to today.

## Examples

Disaggregated inference, single decode context-length bucket:

``` python
additional_config = {
    "neuron_config": {
        "num_seqs_buckets": [1, 2, 4, 8],
        "decode_context_length_buckets": [16384],
    }
}
# max_model_len typically 131072 in this deployment.
# Decode steps with max(ctx) < 16384 run the small (16K-shaped)
# NEFF. Steps that grow past 16K transparently fall back to the
# max_model_len NEFF.
```

Multi-bucket grid for varied effective-KV distributions:

``` python
additional_config = {
    "neuron_config": {
        "num_seqs_buckets": [1, 2, 4],
        "decode_context_length_buckets": [4096, 8192, 16384],
    }
}
# Picks 4096 at low ctx, 8192 in the middle, 16384 for longer
# contexts, max_model_len for the tail.
```

## Design

### Per-step bucket selection

The runtime needs to answer one question every decode step: "what is the smallest configured bucket that still leaves the attention kernel seeing every KV the request needs?"

The answer is a function of the largest *future* KV position any request in the batch will reference *during this step*:

```{math}
\text{needed_seq} = \max(\text{num_computed_tokens}) + 1 + \text{num_speculative_tokens}
```

The `+1` covers the new token being generated this step (non-spec decode) or the bonus token (spec decode). The `+num_speculative_tokens` covers the draft tokens the target model will verify in this same step. Picking the smallest bucket `>= needed_seq` from `decode_context_length_buckets ∪ {max_model_len}` guarantees correctness.

Per-step selection means a request that grows past one bucket boundary smoothly transitions to the next-larger NEFF on the subsequent step. Same mechanism that handles `num_seqs_buckets` today: `torch.compile` shape-specializes on the block-table tensor's shape and dispatches to the correct cached NEFF.

### Cross-DP synchronization

Per-step bucket selection is a *local* decision — each rank picks the smallest bucket that fits its own batch. With `dp_size > 1` this isn't enough on its own. The decode worker may dispatch `execute_dummy_batch` on idle DP ranks while one rank does real decode work; if those ranks pick different buckets they dispatch to *different compiled NEFFs* in the same step. The MoE all-to-all and the attention-DP all-gather/all-reduce then operate against structurally divergent graphs and corrupt the busy rank's hidden state from the first decode token.

The fix is to synchronize the bucket choice across the DP group. The only rank-varying input to bucket selection is `max_ctx = num_computed_tokens_cpu[:padded_num_reqs].max()` — every other input (`block_size`, `decode_context_length_buckets`, `max_model_len`) is config-deterministic and identical on every rank. So instead of reducing the chosen `bucket_blocks`, we reduce `max_ctx` itself; `get_bucket_for_count` is monotonic, so each rank independently picks the same bucket from the synced `max_ctx`. Idle dummy ranks contribute `0`, which the MAX-reduce promotes to the busy rank's value.

The reduce is piggybacked onto the existing `_get_dp_padding` all-reduce: a single 2-element host-side gloo MAX-reduce over `[padded_num_reqs, local_max_ctx]` synchronizes both inputs in one round-trip. This keeps the host-side coordination cost flat even as decode_context_length_buckets is added — important for cross-node DP where each RTT is non-trivial.

Why CPU and not device: both synced values are host-side dispatch decisions — `padded_num_reqs` selects the input batch shape; the bucket pick selects which compiled NEFF to call. Both must materialize as Python `int` before any device op is issued. Reading them back from a device reduce would force a device→host sync that defeats the purpose. The payload is one 2-element `int32` tensor per step on the gloo group, on the host orchestration thread, parallel to the device pipeline.

Why `MAX`: correctness requires every participating rank to dispatch to the same NEFF. `MAX` resolves to whichever rank's actual ctx demands the largest bucket; idle ranks (contributing `0`) and ranks at small ctx are promoted up. The chosen bucket still covers every rank's KV ⇒ the trim is never narrower than each rank actually needs.

The reduce is a no-op when `dp_size == 1` or when no cross-DP collective fires this step (no EP, no component DP) — non-DP and TP-only setups pay nothing. `decode_context_length_buckets` being unset short-circuits the per-step bucket math (every rank returns `full_blocks` directly); the padding reduce still fires in that case, exactly as it did before this feature landed. The DP collective gating depends only on `parallel_config`, which vLLM propagates identically to all DP workers, so the gating predicate is the same on every rank by construction.

### Block-table head-trim vs. SWA tail-trim

vLLM Neuron already trims the block table for sliding-window attention layers — the SWA path keeps only the last `window/block_size` blocks (a *tail* trim) and ships a `swa_kv_pos_offset` tensor so the kernel's causal mask can compute positions in the trimmed frame.

For non-SWA (full-attention) layers the right trim is different: keep the *first* `ceil(ctx_bucket/block_size)` blocks (a *head* trim), starting at logical block 0. There is no offset to pass — positions stay absolute, and the attention mask works without any translation.

The two trims compose cleanly because they apply to different `kv_cache_group` instances. SWA layers route to the existing SWA trim; non-SWA layers with `decode_context_length_buckets` configured route to the new head trim. The metadata schema is uniform: every group emits a `swa_kv_pos_offset` (zeros for non-SWA so the FX graph stays consistent across paths).

This is why `decode_context_length_buckets` does not weaken or replace SWA — for an SWA layer, the SWA window is the tighter, more correct bound; `decode_context_length_buckets` simply doesn't apply.

### EAGLE3 speculative decoding

EAGLE3 multiplies the decode NEFF count by a factor of three because the target model needs two compiled families (with-spec and without-spec) plus the draft model has its own family. All three families participate in the new sequence dimension.

**Why two target families?** The target model decodes a different `max_query_len` depending on whether it has draft tokens to verify:

- **target-with-spec** (`max_query_len = 1 + num_speculative_tokens`): the happy path. The proposer emits `num_speculative_tokens` drafts, the target verifies them in one decode forward.
- **target-without-spec** (`max_query_len = 1`): used when the proposer is skipped. Two situations:
  1. *Near max_model_len.* If the post-step position would exceed `max_model_len - 1`, the proposer is skipped to avoid overflowing the KV slot mapping. The target falls back to a plain 1-token decode.
  2. *DI first decode step.* In disaggregated inference, prefill runs on a different worker. The decode worker receives the KV cache via the connector but does not receive the auxiliary hidden states EAGLE3 needs for proposing drafts. The first decode step therefore runs without-spec; once the target's own forward populates the aux hidden state buffer, subsequent steps use the draft.

Both target families share the same bucket grid and the same runtime trim (the draft tokens are accounted for in `needed_seq`). The draft model warmup runs separately and inherits the same per-pair shape grid, so each draft NEFF is also parameterized by `ctx_bucket`.

### NEFF count

- No spec decode: `|num_seqs_buckets| × (|decode_context_length_buckets| + 1)`.
- With EAGLE3: above × 3 (target-with-spec + target-without-spec + draft).

Compile time scales linearly with the bucket count, so the user implicitly controls startup cost by choosing how many context-length buckets to configure. Every additional bucket adds one more set of NEFFs across all batch buckets and (with EAGLE3) all three families.

### Validation

The `decode_context_length_buckets` list, when set, must satisfy:

1. Non-empty list of positive integers in strictly ascending order.
2. Every value strictly less than `max_model_len`. Equality is redundant — the `max_model_len` fallback NEFF is always compiled.
3. Every value divisible by `P_MAX = 128`. This is the NKI attention kernel's tile-size constraint; the same constant the SWA helper rounds to.

## User-facing knob

See `additional-config` for the full `decode_context_length_buckets` option reference and configuration examples.

## Backwards compatibility

With no `decode_context_length_buckets` configured, all paths are bit-identical to before this feature. The compile-target enumeration returns the same `(batch, max_model_len)` set, the runtime attention-metadata builder skips the new head-trim branch, and the metric labels are unchanged. The runtime `neff_execution_count` Prometheus label is intentionally left unchanged even when the feature is on, so existing dashboards continue to work; only the compile-time `COMPILATION_TIME` label gains a `_ctx{S}` suffix to let operators see per-pair compile cost.

## Limitations

- **Compile time.** Each `(batch, ctx)` pair is a separate NEFF compile. Startup cost grows linearly with the number of configured buckets; this is the cost of opting in.
