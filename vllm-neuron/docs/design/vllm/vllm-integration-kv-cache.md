# KV Caching with vLLM

<!-- meta: description: KV cache integration points with vLLM -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-22 -->

## Overview

For a comprehensive overview of how vLLM works and its integration architecture, see `vllm-integration-design-reference`.

## vLLM Integration

vLLM's KV caching system relies on four key integration points that work together to manage memory-efficient inference:

**Memory Planning Phase**: vLLM first calls `determine_available_memory` to understand how much Neuron device memory can be allocated for KV caching. Then it calls `get_kv_cache_spec` to determine layer-specific cache requirements. The CacheEngine uses both pieces of information to calculate optimal block allocation across layers.

**Cache Initialization Phase**: Once memory limits and layer specifications are established, vLLM calls `initialize_kv_cache` to physically allocate the KV cache tensors on Neuron devices. This sets up the block-based memory layout that vLLM uses for efficient sequence management.

**Runtime Execution Phase**: During each inference step, vLLM calls `_build_attention_metadata` to construct the metadata that maps sequences to their allocated cache blocks. This metadata enables the attention implementation to access the correct KV data for each sequence in the batch.

These four integration points allow vLLM to seamlessly manage KV caching on Neuron hardware while maintaining its high-level scheduling and batching logic.

### Memory Allocation Decision Making

#### Layer-specific cache specification

`NeuronModelRunner::get_kv_cache_spec()` analyzes the model and determines cache requirements for each layer:

- **Attention Type**: Standard attention needs full KV cache, while sliding window attention (SWA) only caches recent tokens within the window size
- **Head Configuration**: Number of key-value heads and head dimensions vary per layer
- **Block Size**: Configurable block size affects memory granularity and allocation efficiency

#### Block allocation and memory distribution

The CacheEngine uses the specifications from `get_kv_cache_spec()` along with available memory to allocate cache blocks:

- **Global Attention Layers**: Get full block allocation for complete sequence caching
- **Sliding Window Layers**: Get reduced allocation matching window size requirements
- **Shared Layers**: Some architectures share KV cache between layers to reduce memory usage
- **Memory Budget**: Distributes available memory across all layers while staying within hardware limits

vLLM handles the complexity of mixed attention types automatically - the Neuron integration only needs to provide the total available memory via `determine_available_memory`.

### How does vLLM initialize KV cache?

#### KV cache workflow

vLLM follows a three-step process to set up KV caching:

1. **Specification Phase**: vLLM calls `get_kv_spec()` on the model to determine KV caching requirements (number of heads, head dimensions, data types) for each layer.
2. **Initialization Phase**: vLLM uses this specification to initialize block tables and allocate the appropriate KV cache tensors.
3. **Binding Phase**: vLLM calls `bind_kv_cache()` to bind the pre-allocated tensors to the model's attention layers, enabling efficient KV caching during inference.

First NeuronModelRunner.get_kv_cache_spec() builds the per rank KVCacheSpec. The modeling code defines get_kv_spec() function which returns the sharded config for the NeuronModelRunner to build the vLLM's KVCacheSpec.

``` python
def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:

    vllm_neuron_kv_spec = self.model.get_kv_spec() # Defined in each Model

    vllm_kv_cache_specs = {}
    for layer in vllm_neuron_kv_spec.layers:
        layer_name = layer.name
        vllm_kv_cache_specs[layer_name] = FullAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=layer.num_kv_heads,
            head_size=layer.head_size,
            dtype=layer.dtype,
            sliding_window=layer.sliding_window_size,
            attention_chunk_size=layer.chunk_size,
        )

    return vllm_kv_cache_specs
```

Here is an example of the KVCacheSpec for TP degree 2 sharding of a model with 8 kv heads. You can see the kv heads is sharded.

``` python
# Tensor Parallel Size = 2 
KVCacheConfig(
    num_blocks=16384, 
    kv_cache_tensors=[
        KVCacheTensor(
            size=536870912, 
            shared_by=['layers.0.self_attn']
        ), 
        KVCacheTensor(
            size=536870912, 
            shared_by=['layers.1.self_attn'])], 
    kv_cache_groups=[
        KVCacheGroupSpec(
            layer_names=['layers.0.self_attn', 'layers.1.self_attn'], 
            kv_cache_spec=FullAttentionSpec(
                block_size=16, 
                num_kv_heads=4, 
                head_size=128, 
                dtype=torch.bfloat16, 
                sliding_window=None, 
                attention_chunk_size=None)
            )
    ]
)
```

Then vLLM Engine uses get_kv_cache_configs() function to process specs from all workers and creates unified configuration.

``` python
def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]], 
    available_memory: list[int],
) -> list[KVCacheConfig]:
    # Merge KV cache specs from all workers
    # Generate KV cache groups based on layer compatibility  
    # Calculate number of blocks based on available memory
    # Create KVCacheConfig with sharded specifications
```

This KVCacheConfig is then used to initialize the KV cache of every worker. The NeuronModelRunner.initialize_kv_cache(kv_cache_config) function is invoked and it initializes the caches.

### How does vLLM schedule requests?

vLLM Scheduler sets the schedule. Here is an example output of the scheduler.

``` python
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData
```

``` python
SchedulerOutput(
    scheduled_new_reqs=[
        NewRequestData(
            req_id=0,
            prompt_token_ids=[128000, 15546, 1051, 279, 386, 7595, 1147, 30],
            mm_features=[],
            sampling_params=SamplingParams(
                n=1, 
                presence_penalty=0.0, 
                frequency_penalty=0.0, 
                repetition_penalty=1.0, 
                temperature=0.0, 
                top_p=1.0, 
                top_k=0, 
                min_p=0.0, 
                seed=None, 
                stop=[], 
                stop_token_ids=[128001], 
                bad_words=[], 
                include_stop_str_in_output=False, 
                ignore_eos=False, 
                max_tokens=2, 
                min_tokens=0, 
                logprobs=None, 
                prompt_logprobs=None, 
                skip_special_tokens=True, 
                spaces_between_special_tokens=True, 
                truncate_prompt_tokens=None, 
                structured_outputs=None, 
                extra_args=None),
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
            prompt_embeds_shape=None)
    ], 
    scheduled_cached_reqs=CachedRequestData(
        req_ids=[], 
        resumed_from_preemption=[], 
        new_token_ids=[], 
        new_block_ids=[], 
        num_computed_tokens=[]), 
        num_scheduled_tokens={'0': 8}, 
        total_num_scheduled_tokens=8, 
        scheduled_spec_decode_tokens={}, 
        scheduled_encoder_inputs={}, 
        num_common_prefix_blocks=[1], 
        finished_req_ids=set(), 
        free_encoder_mm_hashes=[], 
        structured_output_request_ids={}, 
        grammar_bitmask=None, 
        kv_connector_metadata=None)
```
