# DFlash Speculative Decoding in the vLLM Neuron Framework

<!-- meta: description: DFlash block-parallel speculative decoding design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-08-04 -->

## What is DFlash Speculative Decoding?

This document focuses on how DFlash differs from the
[EAGLE3 design](speculative_decoding_design.md). It assumes familiarity with
the shared speculative decoding flow, including target verification, rejection
sampling, and scheduler integration.

DFlash is a **block-parallel** draft method. Like EAGLE3, it consumes auxiliary
hidden states from intermediate target layers and uses the shared target
verification and rejection-sampling flow. EAGLE3 feeds those states into its
draft head and generates drafts through dependent autoregressive iterations.
DFlash instead projects the states directly into each draft layer's KV cache,
then evaluates a fixed `[bonus, mask, ..., mask]` query block in one non-causal
forward. Each query position attends to the injected context and the rest of
the block, allowing the borrowed target LM head to produce all `K` draft tokens
in parallel.

Net trade-off: DFlash swaps EAGLE3's recurrent decoder iterations for one wider,
heavier block evaluation plus a KV-injection precompute. The balance is
workload-dependent.

## API Usage and Configuration

### Enabling DFlash Speculative Decoding

DFlash is selected through `speculative_config`:

``` python
speculative_config = {
    "method": "dflash",
    "model": "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
    "num_speculative_tokens": 9,
}
```

Pass this to `LLM(..., speculative_config=speculative_config)` or
`vllm serve --speculative-config`. For serving commands, benchmarking, and
acceptance metrics, see the
[DFlash tutorial](../../tutorials/tutorial-dflash-speculative-decoding.md).

### Configuration Parameters

`speculative_config` fields:

| Parameter | Required | Description |
|----|----|----|
| `method` | Yes | Must be `"dflash"`. |
| `model` | Yes | Path or HuggingFace ID for the DFlash draft checkpoint. The checkpoint must be paired with the target model — it defines the target-layer mapping and relies on the target's vocabulary, token embedding, and LM head. |
| `num_speculative_tokens` | Yes | Number of draft tokens to propose per step. Must equal the checkpoint's draft width, `block_size - 1`, because one block slot is reserved for the bonus token. For the reference checkpoint, `block_size = 10`, so this must be `9`. |

## Architecture

This section describes how DFlash speculative decoding is architected within
the vLLM Neuron plugin and how the components connect.

### Draft Model

In contrast to EAGLE3's single-decoder-layer drafter, the DFlash draft model
contains:

- A small Qwen3 decode layer stack (five layers for the reference
  checkpoint) regardless of target model family
- An FC layer that projects the concatenated auxiliary hidden states into one hidden-size context representation per token before KV projection
- Token embedding and LM head aliased from the target model

### Component Overview

The scheduler, target verification path, and rejection sampler are shared with
EAGLE3. DFlash changes the proposal path after target verification:

``` text
Target forward
  ├── concatenated auxiliary hidden states ───────────────────┐
  └── target token predictions                                │
               │                                              │
               ▼                                              │
       Rejection sampler                                      │
               │                                              │
               └── accepted-token output ─────────────────────┤
                                                              ▼
                                                       DFlashProposer
                                                              │
                                                              ▼
                                                       Compiled proposal graph
                                                         1. Correct positions and sequence lengths
                                                         2. Build paged slots and [bonus, mask, ..., mask]
                                                         3. Project auxiliary states into draft-layer KV
                                                         4. Run the non-causal block decode over the injected target KV context
                                                         5. Apply the target LM head
                                                              │
                                                              ▼
                                                       K drafts for the next target step
```

Key source files:

| File | Role |
|----|----|
| `vllm_neuron/vllm/spec_decode/dflash.py` | `DFlashProposer`: loads and compiles the draft model, prepares optional async inputs, and invokes the compiled graph |
| `vllm_neuron/model/qwen3/dflash_model.py` | `DFlashQwen3ForCausalLM`: prepares proposal tensors on device, injects context KV, and runs the block forward |
| `vllm_neuron/model/qwen3/config.py` | `DFlashConfig`: block size, mask token id, target layer ids |
| `vllm_neuron/nn/rejection_sampler.py` | On-device greedy rejection sampler (shared with EAGLE3) |
| `vllm_neuron/vllm/worker/neuron_model_runner.py` | Integration: orchestrates target verification, rejection, and draft proposal; aliases target weights; supplies and remaps async correction inputs |
| `vllm_neuron/model/registry.py` | Model registry (maps `"DFlashDraftModel"` to `DFlashQwen3ForCausalLM`) |

## Implementation Details

### DFlashProposer

The `DFlashProposer` class (`dflash.py`) uses the same load, compile, graph
capture, and warmup lifecycle as `EagleProposer`.

On the execution path, `propose()` moves inputs to the draft device, supplies
shape-stable no-op tensors when prior async state is unavailable, and dispatches
one compiled `DFlashQwen3ForCausalLM.forward()` call.

### DFlash Draft Model Architecture

The draft model (`dflash_model.py`) is a compact Qwen3-based drafter.

**`DFlashQwen3DecoderLayer`**:

A standard Qwen3 decoder layer (attention + MLP with pre-norms), but its
attention runs in a **non-causal decode** mode: `self_attn.forward_decode()` is
called with an explicit non-causal `attention_mask`.

**`DFlashQwen3Model`** (backbone):

- A small, checkpoint-configured number of `DFlashQwen3DecoderLayer`s form the decoder stack.
- `precompute_and_store_context_kv()` combines the concatenated target
  auxiliary states and projects them into the paged KV cache for every draft
  layer.
- `forward()` embeds the `[bonus, mask, ..., mask]` query block with the
  target's embedding table and evaluates it against the injected context using
  non-causal attention.

**`DFlashQwen3ForCausalLM`** (wrapper):

Its `forward()` captures four DFlash-specific stages in one compiled call:

- **Async state correction:** Correct optimistic positions and sequence lengths
  using previous- and current-step rejection counts. See
  [Async Rejection and Position Correction](#async-rejection-and-position-correction).
- **Context and block placement:** Map each target auxiliary-state row to its
  paged KV position and place the `K + 1` query-block positions after each
  post-verification sequence.
- **Query block construction:** Use the last valid rejection-sampler output as
  the bonus token and append `K` mask tokens. See
  [Draft Block Construction](#draft-block-construction).
- **Block proposal:** Inject the target auxiliary states into each draft
  layer's KV cache, evaluate the query block with non-causal attention, apply
  the target LM head, and assemble the bonus-prefixed and drafts-only outputs.

``` python
def forward(...):
    # Correct async positions and sequence lengths, then build proposal inputs.
    (input_ids, positions, draft_attn_metadata, keep_idx,
     context_positions, context_slot_mapping,
     bonus_token_ids) = self._prepare_proposal_inputs(...)

    # Inject target auxiliary states into every draft layer's KV cache.
    self.model.precompute_and_store_context_kv(
        target_hidden_states, context_positions, context_slot_mapping)

    # Evaluate and sample all draft positions.
    hidden_states = self.model(input_ids, positions, draft_attn_metadata)
    drafts_hidden = torch.index_select(hidden_states, 0, keep_idx)
    drafts_only = self.compute_draft_tokens(drafts_hidden).view(
        bonus_token_ids.shape[0], -1)
    draft_token_ids = torch.cat([bonus_token_ids[:, None], drafts_only], dim=1)
    return draft_token_ids, drafts_only
```

`_prepare_proposal_inputs()` implements the first three stages as device tensor
operations that execute as part of the drafter forward NEFF.

`compute_draft_tokens()` applies the borrowed LM head (`ColumnParallelLinear`)
and greedy sampling. When `debug_logits_dir` is configured, `forward()` also
returns raw logits for capture.

### Target Auxiliary Hidden State Extraction

DFlash consumes auxiliary hidden states from several intermediate target
layers. The layer ids come from the draft checkpoint config
(`DFlashConfig.target_layer_ids`, e.g. `[1, 8, 15, 22, 29]`), read either
directly or from a nested `dflash_config`.

There is a **crucial off-by-one** between the checkpoint's convention and
Neuron's capture point:

- DFlash checkpoints store HuggingFace **output** hidden-state layer ids — the
  state *after* layer `i`.
- The Neuron target captures **pre-layer** states — the state *before* a layer.

So the runner must request each stored id **plus one**. This is handled in
`DFlashProposer.get_target_aux_hidden_state_layers()`:

``` python
def get_target_aux_hidden_state_layers(self) -> tuple[int, ...]:
    target_layer_ids = self._target_layer_ids_from_config(...)  # e.g. [1,8,15,22,29]
    return tuple(t + 1 for t in target_layer_ids)               # -> (2,9,16,23,30)
```

For a reduced target used only for profiling, if those configured capture
layers do not fit, DFlash instead captures the same number of evenly spaced
pre-layer states. The target must still have at least that many layers, but note that performance will regress due to differences with the config's original intended target layer indices.

The runner passes these layers to the target model's aux-capture interface (the
same interface EAGLE3 uses, `neuron_model_runner.py:1330-1348`). During the
target forward, the captured states are concatenated inside the target NEFF and
returned as a single `[num_ctx, n_target * hidden_size]` device tensor named
`aux_hidden_states`.

### Context KV Injection (Precompute)

`precompute_and_store_context_kv()` is the KV Injection step that improves the quality of DFlash block-parallel draft tokens by enriching the draft model's context using target aux hidden states:

1. **Combine and normalize** the concatenated auxiliary states:
   `ctx = hidden_norm(fc(context_states))` — `[num_ctx, hidden_size]`.
2. **Project to K/V for all layers at once.** At `bind_kv_cache` time the
   drafter stacks every layer's K and V projection weights (this rank's KV
   heads only) into one `_fused_kv_weight`. A single matmul
   (`linear(ctx, _fused_kv_weight)`) produces the K/V for every draft layer,
   then each layer's keys are normalized and RoPE-encoded for their context
   positions.
3. **Scatter into the paged cache.** For each layer, `index_put_` writes the
   rotated keys and values into `k_cache` / `v_cache` at the block/pos indices
   derived from `context_slot_mapping`.

After precompute, the draft layers' KV caches hold a projected representation of
the target's context at the correct paged positions, ready for the block query
pass to attend to.

### Draft Block Construction

Each request's query block has the fixed form `[bonus, mask, mask, …, mask]` of
length `block_size`:

- **Slot 0 = the bonus token** — the most recently accepted target token,
  extracted from `raw_sampled_token_ids`. This anchors the block on real
  context.
- **Slots 1 … block_size-1 = the mask token** (`mask_token_id`, `128002` for the
  reference checkpoint) — placeholders whose output hidden states become the
  draft predictions.

`_prepare_proposal_inputs()` builds this flattened
`[num_reqs * block_size]` tensor. The backbone embeds it with the borrowed
target embedding and computes RoPE from the block positions
(`seq_lens + [0 … block_size-1]`).

After the block forward, `keep_idx` selects positions `1 … block_size-1` of
each request (dropping the bonus slot), yielding `block_size - 1` draft hidden
states per request. This `block_size - 1` is the draft width; the reference
checkpoint's `block_size = 10` gives nine draft positions, hence
`num_speculative_tokens = 9`.

### Paged KV Integration

DFlash uses the same paged vLLM KV cache paradigm as the target model. Both
report their specs via `get_kv_spec()`, vLLM allocates block tables spanning
target + draft layers, and `bind_kv_cache()` binds the pre-allocated tensors to
each draft layer's attention (and builds the fused KV weight used by precompute).

The model-side `_paged_slots()` helper maps absolute positions to cache slots
for both the injected context and the query block:

``` python
block_numbers = clamp(positions // block_size, max=max_blocks_per_seq - 1)
block_ids     = block_table_flat[req_ids * max_blocks_per_seq + block_numbers]
slot          = block_ids * block_size + (positions % block_size)
```

`_prepare_proposal_inputs()` derives each target row's request id from
`last_token_indices`, handles trailing padded rows, and maps the context rows
used by KV injection. It uses the same helper for the fixed query block and
packages those slots into per-layer attention metadata.

### Async Rejection and Position Correction

With async scheduling, the scheduler prepares the next step assuming full
acceptance while the previous rejection output may still be an on-device
future. DFlash passes the resulting optimistic positions and rejection tensors
to `_prepare_proposal_inputs()`, which applies two corrections:

- **`_correct_target_positions_for_async()`** counts how many of the previous
  step's drafts were actually valid (`prev_sampled_token_ids != -1` and
  `< vocab_size`), computes `prev_num_rejected = (1 + prev_num_draft_tokens) -
  prev_valid_count`, and subtracts that from `target_positions` per request.
- **`_current_seq_lens()`** derives each request's post-verify sequence length
  by removing rejected positions from the current step, using
  `cur_num_draft_tokens` when available.

The runner supplies the previous/current-step tensors through
`_build_async_spec_kwargs()`, which returns real data when a prior step exists
and no-op identity tensors otherwise. At batch-composition changes, the runner
remaps the previous step's drafts by request id. DFlash applies the correction
and sequence-length calculation inside the compiled proposal graph. See the
[async scheduling design](../vllm/async-scheduling-and-async-execution.md) for
the broader mechanism.

## Adding DFlash Support for a New Target Model

DFlash's Qwen3-based drafter architecture is target-agnostic, but expanding DFlash support to other target models requires the following:

1. **Auxiliary hidden state capture on the target.** The target model must
   support capturing pre-layer hidden states at requested layer indices and
   returning them concatenated (the same capability EAGLE3 relies on). See the
   [EAGLE3 design](speculative_decoding_design.md#auxiliary-hidden-state-extraction)
   for the target-side interface.

2. **A trained DFlash checkpoint.** The checkpoint must:
   - Provide `target_layer_ids` within the valid range of
     target's number of layers.
   - Share the target's vocabulary, so the borrowed embedding and LM head line
     up.

Because training a draft model is the largest part of this work, start with the
[checkpoints published by the DFlash authors](https://huggingface.co/collections/z-lab/dflash).
