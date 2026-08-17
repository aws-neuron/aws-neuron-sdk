# Tensor Replacement — Design

<!-- meta: description: Tensor replacement design for accuracy debugging -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-22 -->

## Purpose

Tensor replacement injects reference tensors into a model's forward pass at runtime. This lets you swap any named intermediate tensor (e.g., MoE router logits) with a reference value, isolating specific components from hardware numerics during accuracy debugging.

## How It Works

Reference tensors (e.g., from an HF model) are captured once and held in memory (or loaded from disk via `reference_captures_path`). On each forward pass, the model runner builds replacement tensors by indexing into the reference data using the scheduler's positions and request IDs. The result is a `[bucket_size, feature_dim]` tensor that matches Neuron's runtime layout — bucketed, padded, and ordered by the scheduler.

Replacement tensors are injected via global context (`set_active_context` / `get_replacement_tensor`). The model runner sets the context before each forward pass and clears it after.

Each DP replica builds its own replacement context from the same reference captures. The cross-DP all-gather in the MoE kernel combines injected values downstream.

``` text
Phase 1: Reference Capture
──────────────────────────
  Source model (e.g., HF)
  ───────────────────────
  Capture target tensors per prompt
  (e.g. router logits via monkey-patching or TensorCaptureModel)
          │
          ▼
  reference_captures: List[Dict[module_name, List[step_tensor]]]
  - One entry per prompt
  - Step 0 = prefill [1, prompt_len, E]
  - Steps 1+ = decode [1, seq_so_far, E]

Phase 2: Init (TensorReplacer)
───────────────────────────────
  Flatten step-wise captures → position-indexed tensors
  - Prefill: keep all positions from step 0
  - Decode: take last token from each step (new position only)
  - Result: {prompt_idx: {module: Tensor[total_positions, E]}}

Phase 3: Runtime Injection (per forward pass)
──────────────────────────────────────────────
  Model runner                     Model code
  ────────────                     ──────────
  build_context()                  get_replacement_tensor("...router")
  → index flat tensor at            → returns tensor or None
    scheduler positions            if not None:
  → zero-pad to bucket size         use as router logits override
  set_active_context(ctx)
  set_active_context(None)

Phase 4: Validation
───────────────────
  logit_validation() with replacement active
  - Token divergence retries work automatically (build_context()
    indexes the same flat tensor at new scheduler positions)
```

### Data Flow

1. **Reference capture**: Capture reference tensors from the source model (e.g., monkey-patch HF router forward to record pre-top-k logits). Result: `List[Dict[module_name, List[step_tensor]]]` — one entry per prompt, step 0 = prefill, steps 1+ = decode.
2. **Flatten at init**: `TensorReplacer.__init__` flattens step-wise captures into a single `[total_positions, feature_dim]` tensor per (prompt, module). Prefill positions come from step 0; each decode step contributes its last token (HF recomputes the full sequence each step, so only the last row is new).
3. **Build context per forward pass**: `build_context()` takes the scheduler's `req_ids`, `positions`, and `is_prefill` flag, then indexes into the flat reference tensor at each position. Zero-pads padded slots. Returns `{module_name: Tensor[bucket_size, E]}`.
4. **Inject via global context**: `get_replacement_tensor()` retrieves the tensor from the active context. When present, the replacement logits directly override the router's computed logits.

### Prompt Index Resolution

Each prompt's token IDs are hashed at init. When a new request arrives, the model runner calls `register_request(req_id, prompt_token_ids)` which hashes the request's tokens at each known prompt length and selects the **longest verified match**:

``` text
Init (from prompt_token_ids):
  prompt 0: [10, 20, 30]       → hash([10,20,30]) → idx 0
  prompt 1: [40, 50]           → hash([40,50])    → idx 1

New request arrives with tokens [10, 20, 30, 77, 88]:
  try [:3] → hash([10,20,30]) → matches prompt 0, best_len=3
  try [:2] → hash([10,20])    → no match
  → longest match is prompt 0 ✓
  → store "req-abc" → 0

Decode batch ["req-abc", "req-def"]:
  _resolve_prompt_idx("req-abc") → 0
  _resolve_prompt_idx("req-def") → 1
```

#### Longest-Match-Wins Strategy

Prefix matching is required because `logit_validation`'s teacher-forcing loop re-prompts vLLM with extended tokens (original prompt + generated tokens) as a new request. However, naively returning on the first hash match can cause silent wrong-matches when one stored prompt is a prefix of another.

The algorithm tries all stored prompt lengths, verifies the actual token content (not just the hash), and picks the longest match:

``` text
Stored prompts:
  prompt 0: [1, 2, 3]          (length 3)
  prompt 1: [1, 2, 3, 4, 5]    (length 5)

Request with [1, 2, 3, 4, 5]:
  try [:3] → [1,2,3] matches prompt 0, best_len=3
  try [:5] → [1,2,3,4,5] matches prompt 1, best_len=5 (wins)
  → maps to prompt 1 ✓

Retry of prompt 0 with [1, 2, 3, 77, 88]:
  try [:3] → [1,2,3] matches prompt 0, best_len=3
  try [:5] → [1,2,3,77,88] → no match (hash differs from prompt 1)
  → maps to prompt 0 ✓

Request with [1, 2, 3, 4] (shorter than prompt 1):
  try [:3] → [1,2,3] matches prompt 0, best_len=3
  plen=5 > len([1,2,3,4])=4 → skipped
  → maps to prompt 0 ✓
```

#### Known Corner Case

The only remaining ambiguity occurs when a retry of a shorter prompt generates tokens that exactly reproduce a longer stored prompt:

``` text
Stored prompts:
  prompt 0: [1, 2, 3]
  prompt 1: [1, 2, 3, 4, 5]

Retry of prompt 0 where model generated [4, 5]:
  incoming = [1, 2, 3, 4, 5]
  → longest match picks prompt 1 (length 5) instead of prompt 0
```

This requires the model to coincidentally generate the exact tokens that complete another stored prompt — unlikely in practice.

## Core Components

### `TensorReplacer` — Context Builder

Holds flattened reference tensors in memory. Called per forward pass to build replacement tensors from the scheduler's metadata.

``` python
from vllm_neuron.accuracy.tensor_replacement import TensorReplacer

replacer = TensorReplacer(reference_captures, prompt_token_ids=prompt_token_ids)

# Warmup: zero tensors with correct shape for torch.compile tracing
ctx = replacer.warmup_context(num_tokens=256, device=device)

# Inference: build from scheduler metadata
ctx = replacer.build_context(req_ids, positions, is_prefill, device)
```

`build_context()` internals:

- **Prefill**: single request per forward pass. Iterate positions in order; each position indexes into the flat reference tensor. First duplicate position marks padding start — remaining slots stay zero.
- **Decode**: multiple requests batched. For each request, look up prompt index via `_resolve_prompt_idx(req_id)` (populated by `register_request` at prefill time), then index that prompt's flat reference tensor at the request's position. Slots beyond real requests stay zero (batch padding).

### `set_active_context` / `get_replacement_tensor` — Global Context

The model runner sets the context before each forward pass and clears it after. Model code calls `get_replacement_tensor()` to retrieve tensors by module name. Returns `None` when replacement is not configured.

``` python
# Model runner — before each forward
set_active_context(replacer.build_context(req_ids, positions, is_prefill, device))
model(**model_kwargs)
set_active_context(None)

# Inside model — during forward
replacement = get_replacement_tensor("model.layers.0.mlp.router")
# Returns None when not configured → replacement is skipped
```

## Model Integration

Tensor replacement requires model-specific integration to inject replacement tensors at the correct point in the computation. This can be done by adding `get_replacement_tensor()` calls in the model's forward methods.

### Sharding at Injection Point

Replacement tensors enter the compiled graph unsharded (full `[T, feature_dim]`). If the injection point operates on sharded tensors (e.g., after SP all-gather/reduce-scatter), the replacement must be sliced inline to match the local rank's portion.

For example, with sequence parallelism (SP) the model operates on `[T/world_size, E]` per rank:

``` python
if replacement_logits is not None and tp_group.world_size > 1:
    shard_size = replacement_logits.shape[0] // tp_group.world_size
    start = self.rank * shard_size
    replacement_logits = replacement_logits[start : start + shard_size]
```

The same principle applies to any parallelism dimension (EP, DP) where the injection point expects a subset of the full tensor.

### Integration Pattern 1: No 1:1 Mapping (Fused Downstream)

When the downstream expects a different format than raw logits (e.g., the CTE prefill kernel expects pre-computed expert affinities), the injection point must replicate the post-logits pipeline to convert replacement logits into the expected format:

``` python
# Downstream expects expert_affinities [T, E], not raw logits
if replacement_logits is not None:
    # Replicate the routing pipeline: TopK → Softmax → Scatter
    top_values, top_indices = torch.topk(
        replacement_logits.to(torch.float32),
        self.num_experts_per_token, dim=-1,
    )
    top_values = torch.nn.functional.softmax(top_values, dim=-1)
    expert_affinities = torch.zeros(
        replacement_logits.shape[0], self.total_num_experts,
        device=hidden_states.device, dtype=torch.float32,
    ).scatter_(1, top_indices, top_values)
else:
    expert_affinities = NF.router(hidden_states=hidden_states, ...)
```

### Integration Pattern 2: Direct Interception Point

When the downstream function already accepts raw logits as a parameter, the replacement tensor can be passed directly without replicating any pipeline logic:

``` python
# Add replacement tensor as an optional argument
def _run_moe_block_tkg(
    self,
    hidden_states: torch.Tensor,
    rank: torch.Tensor,
    router_logits_override: torch.Tensor | None = None,
):
    ...

# At the call site:
replacement_logits = get_replacement_tensor(f"model.layers.{self.layer_idx}.mlp.router")
output = self._run_moe_block_tkg(
    hidden_states, rank,
    router_logits_override=replacement_logits,
)
```

### Layer Index Requirement

Each module that uses `get_replacement_tensor()` must know its layer index to construct the correct tensor name. This is typically stored at init:

``` python
class GptOssExperts(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        ...

    def forward_decode(self, hidden_states, rank):
        # layer_idx identifies which layer's replacement to fetch
        replacement = get_replacement_tensor(
            f"model.layers.{self.layer_idx}.mlp.router"
        )
```

## Token Divergence Retry

During logit validation with teacher forcing, Neuron may pick a different token than the reference. `logit_validation()` handles this by extending `input_ids` with the reference token and retrying.

`build_context()` is called each forward pass with the scheduler's current positions, and the flat reference tensor covers all positions up to `prompt_len + max_tokens`, so the extended positions map directly to existing rows in the flat tensor.

### Why Prefix Matching Is Needed

Each teacher-forcing retry calls `llm.generate()` with extended tokens, which creates a brand new vLLM request with a new `req_id`. The scheduler calls `register_request(new_req_id, extended_tokens)` where `extended_tokens = original_prompt + generated_tokens`. Since the incoming tokens are longer than what was stored at init, exact matching would fail with `KeyError`. Prefix matching solves this by slicing the incoming tokens to each known prompt length.

``` text
Iteration 1:
  llm.generate([{"prompt_token_ids": [1, 2, 3]}])
  → req_id="abc-123"
  → register_request("abc-123", [1, 2, 3]) → exact match prompt 0

Model diverges at token 2, teacher forcing extends:
  input_ids = [1, 2, 3, 77, 88]

Iteration 2:
  llm.generate([{"prompt_token_ids": [1, 2, 3, 77, 88]}])
  → req_id="def-456" (new request, new ID)
  → register_request("def-456", [1, 2, 3, 77, 88])
  → prefix [:3] = [1,2,3] matches prompt 0 ✓
```

Note: This is distinct from vLLM scheduler preemption/resume, where the `req_id` stays the same and `register_request` is never called again (preempted requests resume via `scheduled_cached_reqs.resumed_req_ids`, not `scheduled_new_reqs`).

## vLLM Configuration

Offline (in-memory captures):

``` python
from vllm_neuron.model.neuron_config import TensorReplacementConfig

llm = LLM(
    model="openai/gpt-oss-20b",
    additional_config={
        "neuron_config": {
            "tensor_replacement": TensorReplacementConfig(
                tensors=reference_captures,
                prompt_token_ids=all_input_ids,
            ),
        }
    },
)
```

Online (captures and token IDs loaded from disk):

``` python
# Save captures and token IDs to disk
torch.save(reference_captures, "/tmp/tensors.pt")
torch.save(all_input_ids, "/tmp/prompt_token_ids.pt")

# Server CLI: pass paths in JSON config
server = start_server(f"""
    vllm serve {model_checkpoint}
        --additional-config '{{
            "neuron_config": {{
                "tensor_replacement": {{
                    "tensors_path": "/tmp/tensors.pt",
                    "prompt_token_ids_path": "/tmp/prompt_token_ids.pt"
                }}
            }}
        }}'
""")
```

## Speculative Decoding Support

Tensor replacement works with speculative decoding. During spec decode verification, the target model processes `1 + num_spec_tokens` positions per request in a single decode forward pass. `build_context()` handles this via the `tokens_per_req` parameter.

### Slot-to-Request Mapping

In standard decode, each slot maps 1:1 to a request. With speculative decoding, each request occupies `tokens_per_req` consecutive slots (1 verified token + N speculative tokens). The mapping is:

``` text
req_idx = slot_idx // tokens_per_req
```

This allows `build_context()` to determine which request owns each slot and index into the correct prompt's flat reference tensor:

``` text
Standard decode (tokens_per_req=1, 2 requests, 4 slots with padding):
  positions = [5, 3, 0, 0]
  slot 0 → req_idx 0//1=0 → position 5 → prompt_0_tensor[5]
  slot 1 → req_idx 1//1=1 → position 3 → prompt_1_tensor[3]
  slot 2 → req_idx 2//1=2 → 2 >= len(req_ids)=2 → padding (zero)
  slot 3 → req_idx 3//1=3 → 3 >= len(req_ids)=2 → padding (zero)

Decode with spec decode (tokens_per_req=4, 1 request):
  positions = [8, 9, 10, 11]   (verifying 3 draft tokens + 1 bonus)
  slot 0 → req_idx 0//4=0 → position 8  → prompt_0_tensor[8]
  slot 1 → req_idx 1//4=0 → position 9  → prompt_0_tensor[9]
  slot 2 → req_idx 2//4=0 → position 10 → prompt_0_tensor[10]
  slot 3 → req_idx 3//4=0 → position 11 → prompt_0_tensor[11]

Decode with spec decode (tokens_per_req=4, 2 requests):
  positions = [8, 9, 10, 11, 5, 6, 7, 8]
  slot 0 → req_idx 0//4=0 → position 8  → prompt_0_tensor[8]
  slot 1 → req_idx 1//4=0 → position 9  → prompt_0_tensor[9]
  slot 2 → req_idx 2//4=0 → position 10 → prompt_0_tensor[10]
  slot 3 → req_idx 3//4=0 → position 11 → prompt_0_tensor[11]
  slot 4 → req_idx 4//4=1 → position 5  → prompt_1_tensor[5]
  slot 5 → req_idx 5//4=1 → position 6  → prompt_1_tensor[6]
  slot 6 → req_idx 6//4=1 → position 7  → prompt_1_tensor[7]
  slot 7 → req_idx 7//4=1 → position 8  → prompt_1_tensor[8]
```

Slots beyond `len(req_ids) * tokens_per_req` are padding and stay zero.

The model runner passes `tokens_per_req` from `attn_metadata["decode_token_threshold"]` (which equals `1 + num_speculative_tokens` when spec decode is active). Without spec decode, `tokens_per_req` defaults to 1 and the mapping reduces to the standard 1:1 slot-to-request behavior.
