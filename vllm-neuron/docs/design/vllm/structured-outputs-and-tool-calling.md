# Structured Outputs and Tool Calling

<!-- meta: description: Structured outputs and tool calling -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

Structured outputs and tool calling enable constrained generation on the vLLM Neuron backend using JSON schemas, regex patterns, choice enums, and BNF grammars. The model output is guaranteed to conform to the specified constraint by applying a grammar-derived bitmask to logits before sampling. Tool calling (function calling) builds directly on this same infrastructure vLLM internally converts tool schemas into structured output constraints, so no additional code changes were needed.

This feature is implemented across three layers:

- **Scheduler** (`scheduler.py`): Computes the grammar bitmask on CPU via vLLM's `StructuredOutputManager` and attaches it to `SchedulerOutput`
- **Model Runner** (`neuron_model_runner.py`): Unpacks the packed int32 bitmask into a boolean tensor, reorders it to match the batch layout, and passes it into the model forward call
- **vLLM Neuron Sampler** (on-device): Applies `logits.masked_fill(~logit_mask, -inf)` before argmax so only grammar-valid tokens can be selected

For scheduler background, see `neuron-scheduler`. For overall vLLM integration, see `vllm-integration-design-reference`.

## Problem Statement

With on-device sampling enabled, the model forward pass is compiled into a NEFF (Neuron Executable File Format). In this mode, logits are never returned to the CPU. Instead, the device runs the full pipeline from logits through argmax and returns only the sampled token IDs. (Without on-device sampling, logits are returned to CPU for sampling there.)

In standard vLLM (v1, GPU), the bitmask is NOT passed into the model forward call. The model executes inside a CUDA graph and returns logits; then, in a separate step outside the graph, a Triton kernel applies the grammar bitmask on-GPU (`logits.masked_fill → -inf`) before the sampler runs. The key distinction is that logits are accessible between the model forward and sampling steps.

In our Neuron implementation, structured outputs use the on-device sampling path. Since logits are never materialized outside the compiled NEFF, we cannot intercept them between the forward pass and sampling. The mask must be applied ON device before sampling, so we thread it through `model.forward()` to get it to the sampler:

``` text
Scheduler: attach bitmask to SchedulerOutput
                    ↓
Model Runner: extract bitmask, pass as logit_mask parameter
                    ↓
Model.forward(logit_mask=...) → On-device sampler applies mask
```

We added the `logit_mask` parameter because vLLM Neuron does on-device sampling and the mask must be applied ON device before sampling. We had to thread it through `model.forward()` to get it to the sampler.

## Solution

### Data Flow

The end-to-end flow spans the CPU host and the Neuron device, crossing the NEFF boundary once per decode step:

``` text
┌─────────────────────────────────────────────────────────────────┐
│  CPU (Host)                                                     │
├─────────────────────────────────────────────────────────────────┤
│  SCHEDULER (scheduler.py)                                       │
│  1. Request arrives with structured_output (JSON/regex/etc.)    │
│  2. grammar_init(request)  initialize FSM from schema          │
│  3. Each decode step: grammar_bitmask() → packed int32 tensor   │
│  4. Attach bitmask to scheduler_output._grammar_bitmask         │
├─────────────────────────────────────────────────────────────────┤
│  MODEL RUNNER (neuron_model_runner.py)                           │
│  5. _get_grammar_bitmask() extracts from scheduler_output       │
│  6. Unpack: [batch, packed_vocab] int32 → [batch, vocab] bool   │
│  7. Pass logit_mask to model forward                            │
└────────────────────────────┬────────────────────────────────────┘
                             │  Inputs transferred to device:
                             │  - input_ids, positions
                             │  - logit_mask [batch, vocab_size]
                             ▼
═══════════════════════════════════════════════════════════════════
CPU EXECUTION ↑  |  ↓ ON-DEVICE EXECUTION
═══════════════════════════════════════════════════════════════════
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Neuron Device (NEFF)                                           │
├─────────────────────────────────────────────────────────────────┤
│  MODEL (e.g. llama3/model.py)                                   │
│  8. Forward pass computes logits                                │
│  9. Pass logit_mask to sampler                                  │
├─────────────────────────────────────────────────────────────────┤
│  SAMPLER (nn/sampler.py)                                        │
│  10. logits.masked_fill(~logit_mask, -inf)                      │
│  11. argmax → only grammar-valid tokens can be selected         │
└────────────────────────────┬────────────────────────────────────┘
                             │  Output: token_ids [batch]
                             ▼
═══════════════════════════════════════════════════════════════════
ON-DEVICE EXECUTION ↑  |  ↓ CPU EXECUTION
═══════════════════════════════════════════════════════════════════
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  CPU (Host)                                                     │
│  12. Update grammar FSM state with selected token               │
│  13. Next iteration: goto step 3 (new bitmask for new state)    │
└─────────────────────────────────────────────────────────────────┘
```

Per decode step the cycle is: forward pass → get bitmask → apply mask → sample → update FSM state → repeat.

### Bitmask Format

vLLM constructs a packed bitmask to minimize memory transfer between CPU and device.

- **Shape:** `[batch_size, packed_vocab_size]` where `packed_vocab_size = ceil(vocab_size / 32)`
- **Dtype:** `int32`
- Each `int32` element encodes 32 vocabulary tokens. Bit = 1 means token is ALLOWED, bit = 0 means DISALLOWED (little-endian bit order).

For Llama-3 with `vocab_size = 128,256`, this yields `packed_vocab_size = 4,008` int32 values per request roughly 16 KB per row.

### Scheduler Integration

In `ContinuousBatchingNeuronScheduler.schedule()`, after padding is applied (Step 6), the grammar bitmask is computed and attached:

``` python
# Step 7: Attach grammar bitmask for structured outputs
grammar_output = self.get_grammar_bitmask(scheduler_output)
if grammar_output is not None and grammar_output.grammar_bitmask is not None:
    scheduler_output._grammar_bitmask = grammar_output.grammar_bitmask
    scheduler_output._structured_output_request_ids = (
        grammar_output.structured_output_request_ids
    )
```

`get_grammar_bitmask()` is inherited from vLLM's base `Scheduler` class. It internally calls `StructuredOutputManager` to compute the FSM-based bitmask. The bitmask is kept as a numpy array at this stage conversion to torch happens in the model runner.

### Model Runner Integration

`_get_grammar_bitmask()` in `neuron_model_runner.py` performs three operations:

**1. Reordering** The compact bitmask from the scheduler only contains rows for requests with structured output constraints. The model runner expands this to match the full batch layout (including non-SO requests and speculative decode token offsets). Non-SO rows are filled with `-1` (all bits set = all tokens allowed).

``` python
# Build mapping: req_id -> logit_index (accounting for spec tokens)
for batch_index, req_id in enumerate(self.input_batch.req_ids):
    logit_index = batch_index + cumulative_offset
    cumulative_offset += len(spec_tokens.get(req_id, ()))
    if req_id in struct_out_req_ids_set:
        struct_out_req_batch_indices[req_id] = logit_index

# Full bitmask: -1 (all allowed) for non-SO rows
sorted_bitmask = torch.full(
    (num_logit_rows, packed_vocab_size), fill_value=-1, dtype=torch.int32
)
```

**2. Unpacking** The packed int32 bitmask is unpacked to a boolean tensor of shape `[num_logit_rows, vocab_size]`:

``` python
packed_uint = sorted_bitmask.view(num_logit_rows, packed_vocab_size, 1)
bit_positions = torch.arange(32, dtype=torch.int32)
unpacked = ((packed_uint >> bit_positions) & 1).view(num_logit_rows, -1)
unpacked = unpacked[:, :self.vocab_size]
bitmask = unpacked.bool()
```

**3. Device Transfer** The boolean mask is moved to the Neuron device and passed as `logit_mask` to the model forward call:

``` python
bitmask = bitmask.to(device=self.device)

# In execute_model():
model_kwargs["logit_mask"] = logit_mask
model_output = self.model(**model_kwargs)
```

Note: TP sharding of the mask is handled inside the vLLM Neuron sampler, matching the `lm_head` sharding strategy. The model runner passes the full-vocab mask.

## Benchmarking

Benchmarked on Llama-3-8B with TP8, `max_model_len=256`, on-device greedy sampling (`all_greedy=true`), using the `xgrammar` backend. Results averaged over 10 runs.

### Per-Token Overhead

| Configuration    | Per-Token Latency | vs Baseline |
|------------------|-------------------|-------------|
| Simple JSON SO   | 759.7 ms/token    | +1.8%       |
| Complex JSON SO  | 739.0 ms/token    | −1.0%       |
| No SO (baseline) | 746.4 ms/token    |             |

The ~1-2% variation is within measurement noise. Structured output adds virtually no per-token overhead.

### Component Breakdown (Warm, Steady-State)

| Component                                  | Latency                 |
|--------------------------------------------|-------------------------|
| Scheduler: grammar bitmask computation     | ~0.12 ms                |
| Model runner: bitmask unpacking + transfer | ~0.24 ms                |
| Forward pass delta (with vs without mask)  | ~0 ms (within variance) |
| **Total per-token SO overhead**            | **~0.4 ms**             |

### Forward Pass Execution

| Phase              | With Mask | Without Mask | Delta   |
|--------------------|-----------|--------------|---------|
| 1st (NEFF compile) | 8,395 ms  | 8,117 ms     | +278 ms |
| 2nd (warmup)       | 5,372 ms  | 5,361 ms     | +11 ms  |
| 3rd+ (steady)      | ~750 ms   | ~762 ms      | −12 ms  |

The first structured output request triggers a NEFF recompilation (~8.4s one-time cost) because the model graph changes to accept the `logit_mask` input tensor. After warmup, the forward pass with mask is within noise of the baseline.

## Tool Calling

Tool calling (function calling) works on top of the structured outputs infrastructure. No additional code changes were needed vLLM internally converts tool schemas into structured output constraints.

### How It Works

Tool calling in vLLM has two distinct paths depending on `tool_choice`:

``` text
tool_choice
    |
+------------+------------+
|                         |
auto / none              required / named
|                         |
No structured outputs       adjust_request() injects
Model generates free text   JSON schema into request
|                         |
Llama3JsonToolParser        xgrammar compiles grammar FSM
extracts tool calls         logit_mask constrains each token
via regex post-gen                    |
|               vLLM Neuron sampler applies:
|               logits.masked_fill(~mask, -inf)
|                         |
+------------+------------+
    |
tool_calls returned
```

- **auto/none**: No structured outputs involved. The model generates free text, and `Llama3JsonToolParser` extracts tool calls post-generation using regex + JSON parsing.
- **required/named**: vLLM's `adjust_request()` converts the tool schema into `StructuredOutputsParams(json=tool_array_schema)`. From that point, it flows through the same grammar bitmask pipeline as any structured output request no modification needed.

This works because vLLM internally treats `tool_choice=required` identically to a structured output request. `adjust_request()` sets `request.structured_outputs = StructuredOutputsParams(json=tool_array_schema)`.

### Why On-Device Sampling Is Required for required/named

The vLLM Neuron model's `forward()` has two code paths:

``` python
# Path A: No on-device sampling → returns raw logits
if self.on_device_sampling_config is None:
    return logits  # logit_mask is IGNORED here

# Path B: On-device sampling → sampler applies mask
sampled_tokens = self.sampler(logits, sampling_params, logit_mask=logit_mask)
```

Without on-device sampling, the model returns raw logits before reaching the sampler. The `logit_mask` is passed into `forward()` but never applied unconstrained tokens violate the grammar FSM and cause `Failed to advance FSM` errors.

| Mode | CPU Sampling | On-Device Sampling | Why |
|----|----|----|----|
| `tool_choice="auto"` | ✅ | ✅ | No SO needed |
| `tool_choice="none"` | ✅ | ✅ | No SO needed |
| `tool_choice="required"` | ❌ | ✅ | Needs `logit_mask` applied in sampler |
| Named tool choice | ❌ | ✅ | Needs `logit_mask` applied in sampler |

### Avoiding Recompilation in SO-Enabled Mode

When structured-output support is enabled, `logit_mask` is always passed as a tensor to the model forward call, never `None`. This ensures `torch.compile` traces a single code path during warmup, and no recompilation is needed when the first structured output request arrives at serving time.

When structured-output support is enabled, warmup passes an all-True mask (all tokens allowed):

``` python
# In warmup_prefill() and warmup_decode():
dummy_logit_mask = torch.ones(
    num_reqs, self.vocab_size, dtype=torch.bool, device=self.device
)
_ = self.model(..., logit_mask=dummy_logit_mask)
```

By default, vLLM Neuron optimizes for SO-off traffic and does not create a dummy logit mask. This avoids the no-op mask allocation / transfer / sampling overhead for deployments that do not serve structured-output requests.

For servers that need to accept structured-output requests, enable SO support explicitly:

``` json
{
  "neuron_config": {
    "enable_structured_outputs": true
  }
}
```

In that mode, if no structured output requests are active, an all-True mask is created as a no-op:

``` python
# In execute_model():
logit_mask = self._get_grammar_bitmask(scheduler_output)
if logit_mask is None:
    logit_mask = torch.ones(
        num_logit_rows, self.vocab_size, dtype=torch.bool, device=self.device
    )
```

An all-True mask is mathematically a no-op: `logits.masked_fill(~True, -inf)` fills nothing. This avoids any `if logit_mask is not None` branches that would cause `torch.compile` to trace a different graph.

In SO-enabled mixed-traffic mode, the vLLM Neuron sampler always receives a mask tensor, so `masked_fill` stays on the traced path:

``` python
# In Sampler.forward():
logits = logits.masked_fill(~logit_mask, float('-inf'))
```

This "always-pass, never-None" behavior keeps a mixed structured-output / non-structured-output server on one stable tensor path and avoids recompilation when the first structured-output request arrives.

For SO-off-only deployments, this no-op mask is skipped by default, or can be made explicit by setting:

``` json
{
  "neuron_config": {
    "enable_structured_outputs": false
  }
}
```

This is a performance mode for servers that will not accept structured-output requests. If a request does include `structured_outputs` while this mode is enabled, vLLM Neuron rejects it with an actionable error asking the user to restart the server with `enable_structured_outputs=true`. It does not silently ignore the schema request, because returning an unconstrained answer would violate the client's requested contract.

## Limitations

1. **On-device sampling required** Structured outputs are only supported with on-device sampling enabled. Attempting to use them with CPU sampling raises `NotImplementedError`.

## FAQ

**Q: How does vLLM pass the bitmask? Is it through logit_mask?**

In standard vLLM, the bitmask is NOT passed into the model. It's applied after the model returns logits, during CPU sampling. The grammar/bitmask logic lives in the sampler layer, not the model layer. We added the `logit_mask` parameter because vLLM Neuron does on-device sampling and the mask must be applied ON device before sampling. We had to thread it through `model.forward()` to get it to the sampler.

**Q: Why is the scheduler attaching the bitmask?**

The scheduler has access to `StructuredOutputManager`, which stores the grammar and computes the bitmask. We attach `_grammar_bitmask` to `SchedulerOutput` because that's the data structure that flows from scheduler to model_runner. The model runner does not have access to `StructuredOutputManager`.

**Q: Why not use SamplingMetadata or create a NeuronSamplingMetadata instead?**

`SamplingMetadata` is created inside vLLM's `InputBatch`, which we don't subclass. To add `grammar_bitmask` there, we'd need to subclass `InputBatch` or patch it after creation. `SchedulerOutput` is simpler since we already override the scheduler and it's the known data channel to model_runner. Even if we defined `NeuronSamplingMetadata`, we'd still need all the places that read metadata to understand it (or at least to carry an extra field through). That's a core vLLM change, not just plugin code.

**Q: But runner and scheduler are the same process why serialize through SchedulerOutput?**

They are in the same process, but the scheduler and model runner are separate components with a well-defined interface (`SchedulerOutput`). Using this existing data channel keeps the design clean and avoids coupling the model runner to the `StructuredOutputManager` internals.

## See Also

- `neuron-scheduler` Scheduler design including holdback queue and padding
- `vllm-integration-design-reference` Overall vLLM integration architecture
- [PR \#754](https://github.com/aws-neuron/vllm-neuron/pull/754) Structured outputs implementation
- [PR \#786](https://github.com/aws-neuron/vllm-neuron/pull/786) Tool calling implementation (built on structured outputs)
