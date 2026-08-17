# Prompt embeddings on Neuron

<!-- meta: description: Prompt embeddings support -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## What this feature does

vLLM supports passing precomputed prompt embedding tensors instead of prompt IDs. Set `enable_prompt_embeds=True` and send `{"prompt_embeds": tensor}` with shape `[seq_len, hidden_size]`.

This is useful when another system already produced embeddings (for example, a multimodal encoder or retrieval pipeline) and you want to feed them directly into the model.

## How vLLM handles prompt_embeds

1. The request includes `prompt_embeds`.
2. vLLM stores the tensor in `CachedRequestState.prompt_embeds` and keeps a per-batch reference in `InputBatch.req_prompt_embeds`.
3. Scheduling still works in token-length space, so batching and padding logic stay unchanged.
4. The model runner builds batch-aligned `inputs_embeds` and `is_token_ids` tensors and passes them to the model.

## Request flow

``` text
User request
    |
    v
vLLM Engine
    stores prompt_embeds in CachedRequestState
    |
    v
Neuron scheduler
    treats request like any other request
    |
    v
NeuronModelRunner._prepare_model_input_impl()
    checks whether any scheduled request still has embed tokens
    |
    +-- if yes: _build_prompt_embeds_tensors()
    |       inputs_embeds [T, H]  (embed values where needed)
    |       is_token_ids  [T]     (True => token path, False => embed path)
    |
    v
NeuronModelRunner._execute_model_forward()
    always attaches inputs_embeds/is_token_ids when feature is enabled
    (real tensors or dummy tensors)
    |
    v
Model backbone forward (LlamaModel / GptOssModel)
    embed_tokens(input_ids)
    SP slice for inputs_embeds/is_token_ids when needed
    optional GPT-OSS dim pad
    merge_prompt_embeds(hidden_states, inputs_embeds, is_token_ids)
    |
    v
Transformer layers consume merged hidden states
```

## Key components

### `NF.merge_prompt_embeds`

Defined in `vllm_neuron/functional/prompt_embeds.py`. It is intentionally small: one guard clause and one `torch.where`. It assumes tensors are already aligned in sequence space.

``` python
def merge_prompt_embeds(hidden_states, inputs_embeds, is_token_ids):
    if inputs_embeds is None or is_token_ids is None:
        return hidden_states
    mask = is_token_ids.unsqueeze(-1)
    return torch.where(mask, hidden_states, inputs_embeds.to(hidden_states.dtype))
```

### SP slicing in model backbones

Each rank slices prompt-embed tensors to match local SP layout. `hidden_states.shape[0]` gives local token count after `embed_tokens`.

``` python
if is_prefill and world_size > 1 and inputs_embeds is not None and is_token_ids is not None:
    local_len = hidden_states.shape[0]
    start = rank * local_len
    inputs_embeds = inputs_embeds[start : start + local_len]
    is_token_ids = is_token_ids[start : start + local_len]
```

### Model-runner activation check

In `neuron_model_runner.py`, `_prepare_model_input_impl()` scans scheduled requests and checks `InputBatch.req_prompt_embeds` for remaining embed positions. It activates the prompt-embed path only when needed, then builds batch tensors via `_build_prompt_embeds_tensors()`.

### Warmup behavior

Prefill and decode warmup both pass dummy `inputs_embeds` (zeros) and `is_token_ids` (all True). This keeps compiled signatures stable and avoids first-use recompilation when prompt embeds appear.

## Behavior by phase

**Prefill (prompt-embed request):** prompt portion typically has `is_token_ids=False` and uses user embeddings.

**Decode:** newly generated tokens use token IDs, so `is_token_ids` is `True` for decode positions and merge becomes a pass-through.
