# Scheduler Design

<!-- meta: description: Neuron scheduler design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

The Neuron scheduler implementation provides custom scheduling optimized for Neuron hardware. It extends vLLM's `Scheduler` using a **holdback queue pattern** for cleaner request management and **bucket-aware admission control**.

The implementation consists of three classes:

- `NeuronScheduler`: Production scheduler with holdback queue, prefill/decode separation, and state machine
- `NeuronAsyncScheduler`: Async variant combining `NeuronScheduler` with vLLM's `AsyncScheduler`

**Scheduler Hierarchy:**

``` text
Scheduler (vLLM base)                    AsyncScheduler (vLLM async scheduling)
  ↓                                        ↓
NeuronScheduler                             │
  (holdback queue, state machine,          │
   prefill/decode separation, padding)     │
  ↓                                        ↓
NeuronAsyncScheduler ◄─────────────────────┘
  (multiple inheritance: NeuronScheduler + AsyncScheduler)
  (overrides _call_base_schedule to use AsyncScheduler.schedule)
```

`NeuronScheduler` extends `Scheduler` directly and uses a `_call_base_schedule()` virtual method to delegate to the parent. `NeuronAsyncScheduler` overrides this to call `AsyncScheduler.schedule()` instead of `Scheduler.schedule()`, enabling async output placeholders.

For comprehensive vLLM integration background, see `vllm-integration-design-reference`.

## Scheduler Configuration

The scheduler behavior can be controlled through several key configuration parameters:

**max_num_seqs** - Decode Batch Size Limit

This parameter controls the maximum number of requests that can be scheduled in a decode batch. It sets the upper bound on concurrent decode operations.

``` python
# In vLLM engine initialization
LLM(
    model="meta-llama/Llama-3-8b",
    max_num_seqs=8,  # Max 8 concurrent decode requests
    ...
)

# Scheduler uses this as:
# max_num_seqs = 8
```

When the running queue reaches `max_num_seqs`, new prefill requests are blocked until decode requests complete and free capacity.

**max_num_batched_tokens** - Prefill Chunking Size (vLLM parameter)

This is a **vLLM-level parameter** (not Neuron-specific). vLLM uses it to chunk prefill requests: when a prompt exceeds this limit, vLLM breaks it into multiple scheduling iterations, each processing at most `max_num_batched_tokens` tokens. vLLM always provides a default value for this parameter.

``` python
# In vLLM engine initialization
LLM(
    model="meta-llama/Llama-3-8b",
    max_num_batched_tokens=8192,  # Process up to 8192 tokens per iteration
    ...
)

# Example: Request with 32,000 tokens and max_num_batched_tokens=8192
# Iteration 1: Process tokens 0-8191      (num_computed_tokens: 0 → 8192)
# Iteration 2: Process tokens 8192-16383   (num_computed_tokens: 8192 → 16384)
# Iteration 3: Process tokens 16384-24575  (num_computed_tokens: 16384 → 24576)
# Iteration 4: Process tokens 24576-31999  (num_computed_tokens: 24576 → 32000)
```

**Segmentation Notes:**

- Each chunk is padded independently to the nearest `num_batched_tokens_buckets` size
- The request remains in `ACTIVE_PREFILL` state across all chunks
- `num_computed_tokens` tracks progress through the prompt
- The segmented attention kernel (`kv_segment_size`) is a separate concept — the kernel internally iterates over prior KV in fixed-size segments, independent of the chunk size

For detailed documentation on prefill segmentation and prefix caching, see `prefix-caching`.

**num_batched_tokens_buckets** - Prefill Token Buckets (Optional)

This optional Neuron-specific parameter defines the compiled bucket sizes for prefill operations. Each prefill chunk (as determined by `max_num_batched_tokens`) is padded to the nearest bucket in this list. The model is compiled once per bucket size, avoiding recompilation for every unique token count.

``` python
# In Neuron configuration
neuron_config = {
    "num_batched_tokens_buckets": [512, 1024, 2048, 4096, 8192],
}

# Example: With max_num_batched_tokens=8192 and these buckets:
# - A prefill chunk of 3000 tokens → padded to 4096 bucket
# - A prefill chunk of 500 tokens → padded to 512 bucket
# - A prefill chunk of 8192 tokens → padded to 8192 bucket (exact match)
```

If not provided, default power-of-2 buckets are generated from 128 up to `min(max_num_batched_tokens, max_model_len)`.

**Constraints:**

- Bucket values must be sorted in ascending order
- The last (largest) bucket must equal `min(max_num_batched_tokens, max_model_len)`

``` python
# Valid: last bucket matches effective max
# (assuming max_num_batched_tokens=8192, max_model_len=131072)
num_batched_tokens_buckets = [512, 1024, 2048, 4096, 8192]  # ✓

# Invalid: last bucket doesn't match
num_batched_tokens_buckets = [512, 1024, 2048]  # ✗ Error
```

**num_seqs_buckets** - Decode Batch Sizes (Optional)

This optional Neuron-specific parameter defines the compiled bucket sizes for decode batch operations.

``` python
# In Neuron configuration
neuron_config = {
    "num_seqs_buckets": [1, 2, 4, 8],
}

# Example: With max_num_seqs=8 and num_seqs_buckets=[1, 2, 4, 8]
# - Decode batches are padded to nearest bucket size
# - 3 decode requests → padded to batch size 4
```

**Constraints:**

- Bucket values must be sorted in ascending order
- The last (largest) bucket must equal `max_num_seqs`

``` python
# Valid: last bucket matches max_num_seqs
max_num_seqs = 8
num_seqs_buckets = [1, 2, 4, 8]  # ✓

# Invalid: last bucket doesn't match
max_num_seqs = 8
num_seqs_buckets = [1, 2, 4]  # ✗ Error
```

### Current Limitation: Only 1 Prefill Per Batch

The scheduler currently processes only one prefill request per iteration. This is hardcoded internally and not configurable.

``` python
# Hardcoded in NeuronScheduler
self.max_prefills_per_batch = 1
```

This means:

- Prefill requests are processed sequentially (FIFO order)
- Multiple waiting prefill requests each get their own iteration
- Decode requests are hidden during each prefill iteration

**Configuration Summary:**

``` python
# Typical configuration
LLM(
    model="meta-llama/Llama-3-8b",
    max_num_seqs=8,              # Max 8 concurrent decode requests
    max_num_batched_tokens=8192, # vLLM chunks prefills at 8192 tokens
    max_model_len=131072,        # Max sequence length (128k context)
    ...
)

# Behavior:
# • Up to 8 requests can be in decode phase simultaneously
# • Prefill requests > 8192 tokens are chunked by vLLM
# • Each chunk is padded to num_batched_tokens_buckets (e.g., [512, 1024, 2048, 4096, 8192])
# • Each prefill processes 1 request at a time (max_prefills_per_batch=1)
# • Decode batches are padded to num_seqs_buckets (e.g., [1, 2, 4, 8])
```

## Scheduler Architecture

### Scheduling Decision Flow

The scheduler takes requests from queues, makes admission and separation decisions, and produces a SchedulerOutput with padded token counts.

**Scheduler State Machine:**

The scheduler operates as a state machine with three states:

``` text
┌──────────────────────┐
┌───────────────────────────▶  │                      │
│                              │        IDLE          │
│                              │                      │
│                              └──────────────────────┘
│                                       │
│   All decode complete                 │ Prefill in waiting queue
│   AND                                 │
│   No prefill waiting                  ▼
│                              ┌──────────────────────┐
│                              │                      │
│                              │                      │◀─────┐
│                              │                      │      │  Prefill has more segments
│                     ┌───────▶│    ACTIVE_PREFILL    │      │          OR
│                     │        │                      │      │   (Decode has capacity AND
│                     │        │                      │      │    New prefill is waiting)
│                     │        │                      │──────┘
│                     │        └──────────────────────┘
│                     │                   │
│ Decode has capacity │                   │  Prefill completes
│               AND   │                   │
│ Prefill is waiting  │                   ▼
│                     │         ┌──────────────────────┐
│                     │         │                      │◀─────┐
│                     │         │  ACTIVE_DECODE       │      │ Decode not finished
│                     └─────────┤                      │──────┘
│                               └──────────────────────┘
│                                         │
└─────────────────────────────────────────┘
```

**State Determination:**

After each `schedule()` call, the `_update_state()` method determines the new state from the `SchedulerOutput`. It uses `num_computed_tokens` from the SchedulerOutput (pre-increment values) rather than from the request objects (which are already incremented by the base scheduler's `_update_after_schedule`):

``` python
def _update_state(self, scheduler_output):
    if len(self.waiting) == 0 and len(self.running) == 0:
        self._state = SchedulerState.IDLE
    elif scheduler_output.num_scheduled_tokens:
        has_prefill = False

        # Check new requests
        for new_req in scheduler_output.scheduled_new_reqs:
            request = self.requests.get(new_req.req_id)
            if request is not None:
                if new_req.num_computed_tokens < request.num_prompt_tokens:
                    has_prefill = True
                    break

        # Check cached requests
        if not has_prefill and scheduler_output.scheduled_cached_reqs is not None:
            cached_reqs = scheduler_output.scheduled_cached_reqs
            for i, req_id in enumerate(cached_reqs.req_ids):
                request = self.requests.get(req_id)
                if request is not None:
                    num_computed = cached_reqs.num_computed_tokens[i]
                    if num_computed < request.num_prompt_tokens:
                        has_prefill = True
                        break

        if has_prefill:
            self._state = SchedulerState.ACTIVE_PREFILL
        else:
            self._state = SchedulerState.ACTIVE_DECODE
```

**Important Notes:**

- **Segmented Prefill**: ACTIVE_PREFILL can span multiple iterations when processing prefill in segments (e.g., 10k tokens in 2k segments = 5 iterations). The state remains ACTIVE_PREFILL until `num_computed_tokens >= num_prompt_tokens`.
- **Capacity**: The `at_capacity` property checks if the running queue has reached `max_num_seqs`. The running queue may contain either prefill or decode requests. When at capacity, new requests from the holdback queue are blocked until running requests complete and free up slots.
- **Prefill Priority**: When decode frees capacity and prefills are waiting, the scheduler transitions to ACTIVE_PREFILL in the next iteration (prefill priority).

**End-to-end Flow:**

``` text
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUTS                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Queues:                                                            │
│  • waiting: [req_0, req_1, ...]                                     │
│  • running: [req_2, req_3, ...]                                     │
│                                                                     │
│  Configuration:                                                     │
│  • max_prefills_per_batch: 1                                        │
│  • num_batched_tokens_buckets: [512, 1024, 2048]                    │
│                                                                     │
│  Request State:                                                     │
│  • num_prompt_tokens: 200                                           │
│  • num_computed_tokens: 0 (prefill) or ≥ num_prompt_tokens (decode) │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SCHEDULE() FLOW                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Move waiting → holdback queue                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ while self.waiting:                                          │   │
│  │     self.holdback_queue.append(self.waiting.popleft())       │   │
│  │                                                              │   │
│  │ Clears the parent scheduler's view of waiting requests       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 2: Selectively restore via can_schedule()                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ while holdback_queue:                                        │   │
│  │     if can_schedule(holdback_queue[0]):                      │   │
│  │         waiting.append(holdback_queue.popleft())             │   │
│  │     else:                                                    │   │
│  │         break  # Preserve FIFO order                         │   │
│  │                                                              │   │
│  │ can_schedule checks:                                         │   │
│  │   1. at_capacity? (len(running) >= max_num_seqs)             │   │
│  │      └─ YES → REJECT                                         │   │
│  │   2. has_prefill_in_running?                                 │   │
│  │      └─ YES → REJECT                                         │   │
│  │   3. len(waiting) < max_prefills_per_batch?                  │   │
│  │      └─ NO  → REJECT                                         │   │
│  │      └─ YES → ACCEPT                                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 3: Prefill/Decode Separation (three branches)                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ IF has_prefill_in_running:                                   │   │
│  │   • Filter running: keep only prefill requests               │   │
│  │   • Move decode requests to running_holdback                 │   │
│  │                                                              │   │
│  │ ELIF waiting has requests (new prefill):                     │   │
│  │   • Move ALL running → running_holdback (hide decodes)       │   │
│  │   • running = []                                             │   │
│  │                                                              │   │
│  │ ELSE (no prefill, decode only):                              │   │
│  │   • Keep running as-is                                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 4: Delegate to Base Scheduler                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ _call_base_schedule() → Scheduler.schedule()                 │   │
│  │   (or AsyncScheduler.schedule() for NeuronAsyncScheduler)    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 5: Restore Queues                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ • running = running + running_holdback                       │   │
│  │ • Drain holdback_queue back to waiting                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 6: Prefill Sequence Padding                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ For each scheduled prefill request:                          │   │
│  │   1. Find smallest bucket ≥ num_tokens                       │   │
│  │   2. Store in num_scheduled_tokens_padded                    │   │
│  │   3. Keep num_scheduled_tokens as actual (unpadded)          │   │
│  │                                                              │   │
│  │ * Decode requests: No sequence padding (1 token per request) │   │
│  │ * Decode batch padding (along batch dimension via            │   │
│  │   num_seqs_buckets) is handled by the model runner,          │   │
│  │   not the scheduler.                                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 7: Verification                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ ASSERT: No mixed prefill and decode in same batch            │   │
│  │ IF (prefill_count > 0 AND decode_count > 0):                 │   │
│  │   → RuntimeError (hardware constraint violation)             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 8: Update State                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ _update_state(scheduler_output) → IDLE/ACTIVE_PREFILL/       │   │
│  │                                    ACTIVE_DECODE             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  OUTPUTS & STATE                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  State 1: IDLE                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ waiting = []                                                 │   │
│  │ running = []                                                 │   │
│  │                                                              │   │
│  │ SchedulerOutput:                                             │   │
│  │   scheduled_new_reqs = []                                    │   │
│  │   scheduled_cached_reqs = []                                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  State 2: ACTIVE_PREFILL                                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ SchedulerOutput:                                             │   │
│  │   scheduled_new_reqs = [                                     │   │
│  │     NewRequestData(                                          │   │
│  │       req_id="req_0",                                        │   │
│  │       num_tokens=200,              # Actual tokens           │   │
│  │       num_computed_tokens=0,       # Cache position          │   │
│  │       block_table=[1,2,...,13]     # 13 blocks for 200 tokens│   │
│  │     )                                                        │   │
│  │   ]                                                          │   │
│  │   num_scheduled_tokens = {"req_0": 200}  # ACTUAL tokens     │   │
│  │   num_scheduled_tokens_padded = {"req_0": 256}  # PADDED     │   │
│  │   scheduled_cached_reqs = []             # No decode         │   │
│  │                                                              │   │
│  │ Note: For segmented prefill, same request may appear in      │   │
│  │       multiple iterations with increasing num_computed_tokens│   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  State 3: ACTIVE_DECODE                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ SchedulerOutput:                                             │   │
│  │   scheduled_new_reqs = []                  # No prefill      │   │
│  │   scheduled_cached_reqs = CachedRequestsData(                │   │
│  │     req_ids=["req_2", "req_3", "req_4"],                     │   │
│  │     num_tokens={"req_2": 1, "req_3": 1, "req_4": 1}          │   │
│  │   )                                                          │   │
│  │   num_scheduled_tokens_padded = {                            │   │
│  │     "req_2": 1, "req_3": 1, "req_4": 1  # No seq padding     │   │
│  │   }                                                          │   │
│  │                                                              │   │
│  │ Property:                                                    │   │
│  │   at_capacity = (len(running) >= max_num_seqs)               │   │
│  │   • If True: Prefills in waiting queue are blocked           │   │
│  │   • If False: Prefills can be scheduled in next iteration    │   │
│  │                                                              │   │
│  │ Note: Decode batch SIZE may be padded (e.g., 3 → 4) but      │   │
│  │       individual tokens are NOT padded (each generates 1)    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MODEL RUNNER INTEGRATION                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  The SchedulerOutput flows to NeuronModelRunner which creates       │
│  padded input tensors:                                              │
│                                                                     │
│  For ACTIVE_PREFILL (example: 200 tokens → 256 bucket):             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Input from scheduler:                                        │   │
│  │   num_scheduled_tokens["req_0"] = 200        # Actual count  │   │
│  │   num_scheduled_tokens_padded["req_0"] = 256 # Padded count  │   │
│  │                                                              │   │
│  │ Model runner creates:                                        │   │
│  │   input_ids = [token_0, ..., token_199,                      │   │
│  │                PAD, PAD, ..., PAD]  # 200 → 256              │   │
│  │   positions = [0, 1, ..., 199,                               │   │
│  │                199, 199, ..., 199]  # Padding uses last pos  │   │
│  │   slot_mapping = [16, 17, ..., 215,                          │   │
│  │                   -1, -1, ..., -1]  # Padding: no KV write   │   │
│  │                                                              │   │
│  │ Result:                                                      │   │
│  │   • Tensor shape matches compiled model (256 tokens)         │   │
│  │   • Attention masks out padding positions                    │   │
│  │   • Only 200 tokens write to KV cache (slot_mapping != -1)   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  For ACTIVE_DECODE:                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Input from scheduler:                                        │   │
│  │   req_ids = ["req_0", "req_1", "req_2"]  # 3 requests        │   │
│  │   Each request generates exactly 1 token                     │   │
│  │                                                              │   │
│  │ Model runner creates:                                        │   │
│  │   input_ids = [token_req0, token_req1, token_req2]           │   │
│  │   No token-level padding (each request = 1 token)            │   │
│  │                                                              │   │
│  │ Note: Batch SIZE may be padded (3 → 4 requests) but          │   │
│  │       individual tokens are NOT padded                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Holdback Queue Pattern

### Influencing the vLLM Scheduler

The Neuron scheduler extends vLLM's base `Scheduler` class. We want to leverage the parent scheduler's proven scheduling logic (block allocation, request tracking, completion handling) while adding Neuron-specific constraints (prefill/decode separation, bucket-based admission control).

**The Challenge:**

The parent scheduler operates on `waiting` and `running` queues directly. We need to:

1. Control which requests the parent scheduler sees (admission control)
2. Hide decode requests when scheduling prefill (prefill/decode separation)
3. Restore all requests after the parent finishes (preserve state)

#### The Solution: Holdback Queue

The holdback queue is a temporary staging area that allows us to selectively control what the parent scheduler processes:

``` python
class NeuronScheduler(Scheduler):
    def __init__(self, ...):
        super().__init__(...)
        self.holdback_queue: deque[Request] = deque()

def schedule(self) -> SchedulerOutput:
    # Step 1: Move waiting → holdback (clear parent's view)
    while self.waiting:
        self.holdback_queue.append(self.waiting.popleft())

    # Step 2: Selectively restore based on can_schedule()
    while self.holdback_queue:
        if self.can_schedule(self.holdback_queue[0]):
            self.waiting.append(self.holdback_queue.popleft())
        else:
            break  # Preserve FIFO order

    # Step 3: Separate prefill/decode (three branches)
    running_holdback = []
    if self.has_prefill_in_running:
        # Ongoing segmented prefill: filter out decode from running
        running_holdback = [r for r in self.running if not self._is_prefill_request(r)]
        self.running = [r for r in self.running if self._is_prefill_request(r)]
    elif len(self.waiting) > 0:
        # New prefill: hide all running decodes
        running_holdback = self.running
        self.running = []
    # else: decode only, no hiding needed

    # Step 4: Let parent scheduler do its work
    scheduler_output = self._call_base_schedule()

    # Step 5: Restore everything
    self.running = self.running + running_holdback
    while self.holdback_queue:
        self.waiting.append(self.holdback_queue.popleft())

    # Step 6: Apply padding
    scheduler_output = self._apply_padding_and_log_stats(scheduler_output)

    # Step 7: Update state machine
    self._update_state(scheduler_output)

    return scheduler_output
```

**How It Works:**

1. **Admission Control**: By moving all waiting requests to holdback, then selectively restoring only those that pass `can_schedule()`, we filter what the parent sees.
2. **Prefill/Decode Separation**: Three branches handle different scenarios:
    - **Ongoing segmented prefill**: When a multi-segment prefill is in the running queue, we filter out decode requests but keep the prefill request in running.
    - **New prefill**: When new prefill requests are in waiting, we hide all running decode requests.
    - **Decode only**: No manipulation needed.
3. **State Preservation**: After the parent finishes, we restore all requests. The holdback queue ensures nothing is lost.
4. **FIFO Order**: We stop restoration when a request fails `can_schedule()`, preserving priority order.

**Key Insight:**

The holdback queue lets the parent scheduler operate normally on its queues while we manipulate those queues before and after. The parent is unaware of our interventions - it just sees the filtered view we provide.

## Admission Control

The `can_schedule()` method provides admission control for new requests. Notably, it does **not** check bucket fit because chunked prefill allows long prompts to be processed in multiple iterations — each chunk will fit in a bucket. The `max_model_len` limit is enforced by vLLM itself.

``` python
def can_schedule(self, request: Request) -> bool:
    """Admission control for scheduling requests."""

    # Scheduler is already running at capacity
    if self.at_capacity:
        return False

    # There is already an active prefill request in the running queue
    if self.has_prefill_in_running:
        return False

    # Check prefill batch capacity (only 1 prefill at a time)
    has_prefill_capacity = len(self.waiting) < self.max_prefills_per_batch

    return has_prefill_capacity
```

**Admission Criteria:**

1. **Capacity**: Running queue must not be at `max_num_seqs`
2. **No ongoing prefill**: No request in the running queue can be in prefill phase (segmented prefill in progress)
3. **Prefill batch capacity**: Only one prefill request at a time (configurable via `max_prefills_per_batch`)

**Properties used:**

``` python
@property
def at_capacity(self) -> bool:
    """True if running requests >= max_num_seqs."""
    return len(self.running) >= self.max_num_seqs

@property
def has_prefill_in_running(self) -> bool:
    """True if any request in running queue is in prefill phase."""
    return any(self._is_prefill_request(req) for req in self.running)
```

## Prefill/Decode Separation

**Critical Requirement**: Neuron requires prefill and decode operations to run in **separate batches**. Mixing them in the same batch is not supported.

**How Separation is Enforced:**

The holdback queue pattern (described in the previous section) implements prefill/decode separation with three branches:

1. **Ongoing segmented prefill** (`has_prefill_in_running`): A multi-segment prefill is still in the running queue. Filter out decode requests from running, keeping only the prefill request(s).
2. **New prefill** (`len(self.waiting) > 0`): New prefill requests are in the waiting queue. Hide all running decode requests before calling the parent scheduler.
3. **Decode only**: No prefill requests anywhere. Let the parent scheduler handle decode normally.

**Verification:**

After scheduling, we verify no mixed batches occurred:

``` python
if prefill_count > 0 and decode_count > 0:
    raise RuntimeError(
        f"Neuron constraint violated: Mixed prefill ({prefill_count}) "
        f"and decode ({decode_count}) in the same batch."
    )
```

**Request Classification:**

A request's phase is determined by comparing computed vs total prompt tokens:

``` python
def _is_prefill_request(self, request: Request) -> bool:
    return request.num_computed_tokens < request.num_prompt_tokens
```

- **Prefill Phase**: `num_computed_tokens` \< `num_prompt_tokens`
- **Decode Phase**: `num_computed_tokens` \>= `num_prompt_tokens`

## Bucket Configuration

Buckets can be configured in two ways:

**1. Default Buckets (Automatic):**

``` python
def get_default_num_batched_tokens_buckets(max_num_batched_tokens: int) -> list[int]:
    """Generate default power-of-2 buckets from 128 to max_num_batched_tokens."""
    buckets = []
    current = 128
    while current <= max_num_batched_tokens:
        buckets.append(current)
        current *= 2
    return buckets

# For max_num_batched_tokens=8192: [128, 256, 512, 1024, 2048, 4096, 8192]
```

Note: `max_num_batched_tokens` is computed as `min(scheduler_config.max_num_batched_tokens, max_model_len)`.

**2. Custom Buckets (Configuration):**

``` python
neuron_config = {
    "num_batched_tokens_buckets": [256, 512, 1024],  # Custom prefill buckets
    "num_seqs_buckets": [1, 2, 4, 8],                # Custom decode buckets
}
```

Custom buckets are validated to ensure they're in ascending order and the last bucket matches the corresponding max parameter.

## Padding Algorithm

### Bucket Selection Logic

The scheduler uses a simple algorithm to select the appropriate bucket:

``` python
def _calculate_padded_count(self, token_count: int) -> int:
    """Find the smallest bucket >= token_count."""

    if token_count == 0:
        return 0

    return get_bucket_for_count(token_count, self.num_batched_tokens_buckets)
```

**Examples:**

- 50 tokens → 128 bucket (78 padding tokens)
- 127 tokens → 128 bucket (1 padding token)
- 128 tokens → 128 bucket (0 padding tokens)
- 129 tokens → 256 bucket (127 padding tokens)
- 1500 tokens → 2048 bucket (548 padding tokens)

### Padding Storage and Usage

**Critical Design Decision:** Padding affects tensor sizes but not semantic token counts.

The scheduler stores padding information separately to maintain correct semantics:

``` python
@dataclass
class SchedulerOutput:
    # Actual tokens scheduled (semantic count, unpadded)
    num_scheduled_tokens: dict[str, int]

    # Total tensor size including padding (Neuron-specific)
    # Only populated for prefill requests that need padding
    num_scheduled_tokens_padded: dict[str, int] | None = None
```

**Why Separate Fields?**

1. **Semantic correctness**: `num_scheduled_tokens` represents actual tokens to compute
2. **Token accounting**: `_update_after_schedule()` advances `num_computed_tokens` by actual count
3. **Hardware requirements**: Model runner needs padded size for tensor allocation
4. **Single source of truth**: Scheduler computes both values, model runner uses them directly

### Padding Application

Sequence padding is applied only to **prefill requests**. Decode is only padded along the batch dimension to match the smallest `num_seqs_buckets` that fits.

``` python
def _apply_padding_and_log_stats(self, scheduler_output):
    """Apply padding to prefill requests in scheduler output."""

    scheduler_output.num_scheduled_tokens_padded = {}

    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        request = self.requests.get(req_id)
        if request is None:
            continue

        num_computed = req_num_computed.get(req_id, request.num_computed_tokens)
        is_prefill = num_computed < request.num_prompt_tokens

        if is_prefill:
            padded_tokens = self._calculate_padded_count(num_tokens)
            scheduler_output.num_scheduled_tokens_padded[req_id] = padded_tokens
            self.total_padding_tokens += padded_tokens - num_tokens
        else:
            scheduler_output.num_scheduled_tokens_padded[req_id] = num_tokens

    # Safety check: no mixed batches
    if prefill_count > 0 and decode_count > 0:
        raise RuntimeError(...)
```

### Why Separate Fields Matter

The separation of actual and padded token counts solves a critical token accounting bug. The problem occurs when the scheduler modifies `num_scheduled_tokens` to include padding, then the base scheduler's `_update_after_schedule()` method uses this padded value to advance `num_computed_tokens`.

**The Bug:**

When a 5000-token prefill request is padded to 8192 tokens, if the scheduler stores the padded count (8192) in `num_scheduled_tokens`, then `num_computed_tokens` incorrectly advances by 8192 instead of 5000. In the next iteration, when the scheduler tries to calculate remaining work (`num_tokens - num_computed_tokens`), it gets a negative number (5000 - 8192 = -3192), breaking the scheduler's logic.

**The Solution:**

By storing actual tokens (5000) in `num_scheduled_tokens` and padded tokens (8192) in a separate `num_scheduled_tokens_padded` field, the scheduler maintains correct token accounting. The base scheduler's `_update_after_schedule()` advances `num_computed_tokens` by the actual count (5000), so the next iteration correctly transitions to decode phase when `num_computed_tokens` equals `num_prompt_tokens`.

**With Prefix Caching:**

This separation also handles prefix caching correctly. When a 5000-token request has 1000 tokens already cached, the scheduler computes that 4000 new tokens need processing. It stores 4000 in `num_scheduled_tokens` (actual work) and 4096 in `num_scheduled_tokens_padded` (padded to bucket). The model runner extracts 4000 tokens from the prompt and adds 96 padding tokens, creating a 4096-token tensor. Token accounting remains correct because `num_computed_tokens` advances by 4000, not 4096.

**Tensor Size Summary:**

| Request State | num_scheduled_tokens | num_scheduled_tokens_padded | Tensor Size |
|----|----|----|----|
| Prefill (5000 tokens) | 5000 (actual) | 8192 (padded) | 8192 |
| Decode (1 token) | 1 (actual) | 1 | 1 |
| Prefill with prefix cache (4000 new tokens) | 4000 (actual) | 4096 (padded) | 4096 |
| Prefill exact bucket match (2048 tokens) | 2048 (actual) | 2048 (no padding) | 2048 |

### Model Runner Integration

The model runner uses the scheduler's token counts directly without recalculation:

``` python
# In NeuronModelRunner._prepare_model_input_impl()

if use_new_scheduler:
    tokens = [
        scheduler_output.num_scheduled_tokens_padded[req_id]
        for req_id in req_ids
    ]
# ...
actual_tokens_list = [
    scheduler_output.num_scheduled_tokens[req_id] for req_id in req_ids
]

# Padding is the difference - no complex calculation needed
# _create_padded_inputs uses actual tokens for extraction,
# padded count for tensor sizing
```

**Benefits of Simplified Approach:**

1. **No request state lookup** - All information comes from scheduler output
2. **No prefill/decode detection** - Scheduler already determined this
3. **No recalculation** - Simple subtraction: `padded - actual`
4. **Single source of truth** - Scheduler decides, model runner uses

## Deadlock Prevention: KV Cache Admission Control

### Problem

When `max_num_seqs` exceeds the number of requests that can concurrently fit in the KV cache (at worst-case `max_model_len` per request), the scheduler can deadlock due to the holdback mechanism in Step 3 of `schedule()`:

1. A new prefill request is in the waiting queue.
2. Step 3.2 hides all running decode requests into `running_holdback` so the parent scheduler only sees the prefill.
3. The parent scheduler tries to schedule the prefill but there are not enough free KV cache blocks — it schedules **nothing**.
4. Decode requests are restored, but on the next iteration the same prefill is still waiting, so Step 3.2 hides decodes again.
5. **Deadlock**: decode requests never get scheduled (always hidden behind the pending prefill), so they never generate tokens and never complete. The prefill can never proceed because no KV space is freed.

``` text
Deadlock cycle:

schedule() → prefill waiting → hide decodes → no KV space → schedule nothing
    ↑                                                              ↓
    └──────────────────────────────────────────────────────────────┘
(decodes never run, KV never freed, prefill never fits)
```

### Solution

The `can_schedule()` method in `NeuronScheduler` now checks KV cache capacity before admitting a new prefill. If the prefill would not fit, it stays in the holdback queue, the decode requests are **not hidden**, and decodes continue running until they complete and free KV space.

During `__init__`, the scheduler computes `_max_kv_concurrent` — the maximum number of requests that can coexist in the KV cache assuming each uses `max_model_len` tokens:

``` python
# From NeuronScheduler.__init__:
if kv_cache_config.kv_cache_groups:
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    max_concurrent = int(max_concurrency)
else:
    total_kv_tokens = kv_cache_config.num_blocks * block_size
    max_concurrent = total_kv_tokens // self.max_model_len

self._max_kv_concurrent = max_concurrent if max_concurrent > 0 else self.max_num_seqs
```

Then in `can_schedule()`, new prefills are deferred when running requests already saturate this capacity:

``` python
if len(self.running) >= self._max_kv_concurrent:
    return False
```

### Why the Conservative Bound

The capacity check uses a worst-case assumption: every running request may grow to `max_model_len` tokens. This is deliberately conservative because decode requests continuously generate tokens, consuming more KV cache blocks over time. If we admitted prefills based on *current* KV usage, running decode requests could still exhaust KV cache mid-generation, which would trigger request preemption. **Preemption is not supported on Neuron**, so this must be prevented at admission time rather than handled at runtime.

The tradeoff is reduced concurrency — some KV capacity may go unused when requests finish well before `max_model_len`. To increase concurrency, raise `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` or reduce `max_model_len`.

### Admission Control Flow

The full `can_schedule()` check order in Step 2 of `schedule()`:

1. **At capacity** — `len(running) >= max_num_seqs` → reject
2. **Prefill in progress** — another request is mid-prefill in running → reject
3. **Prefill batch full** — `len(waiting) >= max_prefills_per_batch` (currently 1) → reject
4. **KV cache saturated** — `len(running) >= _max_kv_concurrent` → reject (deadlock prevention)
5. Otherwise → admit

## See Also

- `prefix-caching` - Prefill segmentation and prefix caching
- `vllm-integration-design-reference` - Overall vLLM integration architecture
- `vllm-integration-kv-cache` - KV cache management and memory allocation
