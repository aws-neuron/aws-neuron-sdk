# How to onboard a model to vLLM Neuron

<!-- meta: description: Onboard a new model architecture to vLLM Neuron and
make it available for online and offline serving on AWS Trainium and
Inferentia. -->
<!-- meta: keywords: vLLM, Neuron, model onboarding, custom models, Trainium,
Inferentia, tensor parallelism, KV cache -->
<!-- meta: date_updated: 2026-06-11 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: NDOC-186 -->

## Task overview

This topic discusses how to onboard a new model architecture so it can be served
through vLLM Neuron. You will implement the model using the vLLM Neuron
plugin's building blocks, register it with vLLM's model registry, compile,
validate accuracy, and benchmark performance.

For annotated code patterns and per-component implementation details, refer to
the canonical reference implementation in `vllm_neuron/model/gpt_oss/`.

:::{note}
The vLLM Neuron plugin implements models directly using `torch_neuronx`, NKI
(Neuron Kernel Interface) kernels, and its own parallel infrastructure. The
plugin provides its own functional operators (`vllm_neuron.functional`), neural
network modules (`vllm_neuron.nn`), weight loading utilities, and parallel state
management built on top of `torch.distributed` and vLLM's distributed primitives.
:::

:::{note}
For models deployed through NxD Inference directly (without vLLM), see the legacy
[NxDI model onboarding guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/onboarding-models.html).
:::

## Prerequisites

- **vLLM Neuron environment**: A working vLLM Neuron setup on a Trainium or
  Inferentia instance. See [setup guide](../getting-started/setup-guide.md).
- **A model you want to onboard**: You may be porting from an existing
  implementation (e.g., Hugging Face Transformers), writing from scratch, or
  adapting from another framework. Any approach works as long as the result
  satisfies the runner interface described below.
- **Familiarity with the plugin codebase**: The
  [vLLM Neuron plugin source](https://github.com/vllm-project/vllm-neuron) contains
  annotated reference implementations (Llama, GPT-OSS, Qwen3 MoE) that demonstrate
  common patterns. These are useful as examples, not mandatory templates.

## End-to-end onboarding flow

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Implement model    2. Register with   3. Compile &   4. Validate      │
│     (config, layers,      vLLM registry      smoke test     accuracy      │
│      factory, weights)                                                    │
│                                                          5. Benchmark     │
└─────────────────────────────────────────────────────────────────────────┘
```

The reference implementations are heavily annotated with
`# <-- MODEL-SPECIFIC` and `# >>> PARALLELISM <<<` comments to distinguish
model-specific code from reusable parallelism infrastructure. Use the Llama
implementation (`vllm_neuron/model/llama3/`) as your primary porting template
for dense models, or the GPT-OSS implementation (`vllm_neuron/model/gpt_oss/`)
for MoE models.

## Instructions

### 1. Implement the model

Create a model directory under `src/model/` with the following structure:

```text
src/model/your_model/
├── __init__.py
├── config.py          # Model-specific dataclass config
├── factory.py         # Factory class for vLLM ModelRegistry
└── model.py           # Full model implementation
```

#### 1a. Define the model config (`config.py`)

Create a dataclass that holds the model's architecture parameters. Implement a
`from_configs` classmethod that parses a HuggingFace `PretrainedConfig` into your
config.

```python
from dataclasses import dataclass
import torch
from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig

@dataclass
class YourModelConfig:
    # Architecture parameters (model-specific)
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    num_hidden_layers: int = 32
    intermediate_size: int = 14336
    head_dim: int = 128
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling: dict | None = None
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None):
        config_dict = hf_config.to_dict()
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config
        return cls(**filtered_dict)
```

The `NeuronConfig` dataclass (provided by the plugin at
`vllm_neuron.model.neuron_config`) carries Neuron-specific settings: parallelism
degrees (`attention_dp_size`, `mlp_dp_size`, `embedding_dp_size`,
`lm_head_dp_size`, `ep_degree`), on-device sampling config, bucket
configurations, and quantization settings.

#### 1b. Implement model components (`model.py`)

Your model implementation can use any valid PyTorch code that the Neuron compiler
can trace. The plugin provides optimized building blocks (functional operators
backed by NKI kernels, parallel modules, weight loading utilities) that you can
use if they fit your architecture. These are conveniences, not requirements — you
are free to write your own operators, use different parallelism strategies, or
structure your model however you like, as long as the result satisfies the runner
interface contract described below.

##### Available building blocks

The following utilities are provided by the plugin and used by the existing model
implementations. They handle common patterns efficiently on Neuron:

| Component | Available utilities |
| --- | --- |
| Attention | `vllm_neuron.functional.qkv_proj`, `flash_attention`, `segmented_attention`, `attention_decode`, `o_proj`, `gen_attention_decode_mask` — NKI-backed operators with PyTorch CPU fallbacks for tracing |
| MLP | `vllm_neuron.functional.mlp` — fused gate/up/down projection with NKI kernel |
| Embedding | `vllm_neuron.nn.VocabDimShardedEmbedding` — vocab-dimension-sharded embedding with built-in scatter for SP |
| LM Head | `vllm_neuron.nn.ColumnParallelLinear` — column-parallel linear with optional output gathering |
| Normalization | Standard `nn.Module` implementations (RMSNorm, LayerNorm) — no special Neuron module needed |
| Weight loading | `vllm_neuron.utils.weight_loader.sharding_weight_loader`, `fused_qkv_weight_loader`, `with_rank_override` — closure-based loaders for checkpoint → parameter transforms |
| Collectives | `vllm.distributed.parallel_state.get_tp_group()` — provides `all_gather`, `reduce_scatter`, `all_reduce`; plugin also provides custom group coordinators in `vllm_neuron.parallel.neuron_parallel_state` |
| KV cache | `vllm_neuron.model.kv_cache.KVSpec`, `LayerSpec` — dataclasses for declaring cache requirements |

:::{note}
The functional operators in `vllm_neuron.functional` (imported as `NF`) dispatch
to NKI kernels when running on Neuron hardware and fall back to equivalent PyTorch
operations on CPU. This dual-path design allows `torch.compile` to trace the graph
on CPU while the compiled NEFF uses optimized kernels at runtime. You can use
these operators, write your own, or mix both.
:::

##### Understanding the runner → model interface

The `NeuronModelRunner` orchestrates execution between vLLM's scheduler and your
model. Understanding what it passes to `forward()` is essential for implementing a
correct model.

*Forward signature:*

Your top-level model's `forward()` receives these keyword arguments (all are
always present once torch.compile traces the graph — you cannot conditionally
accept/reject kwargs between calls):

```python
def forward(
    self,
    input_ids: torch.LongTensor,         # [total_tokens] — token IDs, padded to bucket size
    positions: torch.Tensor,             # [total_tokens] — position IDs (int32)
    attn_metadata: dict,                 # Per-layer attention metadata (see below)
    sampling_positions: torch.Tensor,    # [num_reqs] — indices into the token dim for LM head
    sampling_params: torch.Tensor,       # [num_reqs, 3] — [top_k, top_p, temperature] (None if CPU sampling)
    spec_decode_metadata=None,           # SpecDecodeMetadata for rejection sampling
    logit_mask: torch.Tensor,            # [num_logit_rows, vocab_size] — grammar/structured output mask
    rank: torch.Tensor,                  # scalar — TP rank
    inputs_embeds: torch.Tensor | None = None,   # [total_tokens, hidden_size] (prompt-embeds path)
    is_token_ids: torch.Tensor | None = None,    # [total_tokens] bool (prompt-embeds path)
) -> torch.Tensor | tuple:
```

*Attention metadata structure:*

`attn_metadata` is a **dict keyed by layer name** (`"layers.0.self_attn"`,
`"layers.1.self_attn"`, etc.). Each attention layer indexes into it by its own
name. The per-layer dict contains:

| Key | Description |
| --- | --- |
| `block_table_tensor` | `[padded_num_reqs, max_blocks_per_seq]` int32 — maps (request, block_index) → physical block ID |
| `slot_mapping` | `[total_tokens]` int64 — maps each token position → KV cache slot. Padding tokens have value `-1`. |
| `max_query_len` | int — maximum tokens scheduled per request in this batch |
| `block_size` | int — number of positions per KV cache block |
| `max_blocks_per_seq` | int — max blocks allocated per sequence (equals `block_table_tensor.shape[1]`) |
| `decode_token_threshold` | int — if `max_query_len <= decode_token_threshold`, use the decode path |
| `cached_seq_len` | `[[int]]` tensor — number of previously cached tokens (for segmented prefill) |
| `kv_segment_size` | int — 0 = full prefill; >0 = segmented prefill chunk size |

*Prefill vs. decode dispatch:*

Each attention layer dispatches between its prefill and decode paths using:

```python
layer_name = f"layers.{self.layer_idx}.self_attn"
max_query_len = attn_metadata[layer_name]["max_query_len"]
decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

if max_query_len <= decode_token_threshold:
    return self.forward_decode(...)
else:
    return self.forward_prefill(...)
```

*Return values:*

What `forward()` returns depends on on-device sampling (ODS) configuration:

| Mode | Return value |
| --- | --- |
| CPU sampling (no ODS) | `logits` — `[num_reqs, vocab_size]` float tensor |
| CPU sampling + Eagle3 | `(logits, aux_hidden_states)` |
| On-device sampling | `(sampled_token_ids, gathered_logits)` |
| On-device sampling + Eagle3 | `(sampled_token_ids, aux_hidden_states, gathered_logits)` |

*Things to be aware of:*

- **Padding is real.** `input_ids` is padded to compiled bucket sizes. The model
  processes all tokens including padding. Padding tokens have `slot_mapping = -1` —
  if your model writes to the KV cache, skip slots with value `-1`.
- **`sampling_positions`** tells the model which token positions to produce logits
  for. The existing models use `torch.index_select` on the final hidden states to
  select only these positions before the LM head projection, avoiding unnecessary
  computation.
- **The model owns KV cache writes.** The runner allocates cache blocks and binds
  them via `bind_kv_cache()`, but it's up to the model's attention implementation
  to actually write K/V into paged blocks using `slot_mapping`, `block_size`, and
  computed block/position indices.
- **Forward signature is fixed at compile time.** `torch.compile` traces a static
  graph. All kwargs that appear in any call must always be present — you cannot
  conditionally accept different kwargs between calls.
- **Conditional paths create separate compiled graphs.** Python-level branching
  (e.g., prefill vs. decode dispatch based on `attn_metadata` values) is normal —
  each branch is traced into its own NEFF during warmup. After warmup,
  `torch.compiler.set_stance("fail_on_recompile")` prevents new graph variants.
  Data-dependent control flow (branching on tensor *values*) will break
  compilation because tensor values aren't known at trace time.

---

##### Attention class pattern (example)

The existing models implement attention with these patterns. Your implementation
can differ, but these illustrate common choices:

- **TP head sharding:** Q/K/V heads divided across TP ranks. GQA replication when
  KV heads < TP size.
- **Two execution paths:** Prefill (flash attention with SP all-gather/reduce-
  scatter) and decode (fused megakernel via `NF.attention_decode`). These are
  separate code paths dispatched based on `attn_metadata`.
- **KV cache writes:** The model writes new K/V into the paged cache using
  `slot_mapping` from `attn_metadata`. This is the model's responsibility — the
  runner does not write to the cache.
- **Weight loaders:** Closures attached to parameters that handle checkpoint →
  parameter transforms (sharding, fusion).

```python
import vllm_neuron.functional as NF
from vllm.distributed.parallel_state import get_tp_group
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader, sharding_weight_loader, set_weight_loader
)

class YourModelAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        # Shard Q/K/V heads across TP ranks
        self.num_attention_heads_per_rank = config.num_attention_heads // self.world_size
        self.num_key_value_heads_per_rank = max(1, config.num_key_value_heads // self.world_size)
        # ... define qkv_proj_weight, o_proj_weight parameters ...
        # Attach weight loaders for TP sharding
        set_weight_loader(self.qkv_proj_weight, fused_qkv_weight_loader(...))

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata):
        # Dispatch to prefill or decode based on attn_metadata
        ...
```

:::{warning}
**Match the RoPE rotation style to the checkpoint.** Using the wrong RoPE
rotation style produces garbage attention outputs that are hard to debug because
the model still runs without errors — just with wrong results. Two common styles:

- **Interleaved (rotate_half):** Used by Llama, Mistral. Splits into first/second
  half, rotates as `(-x2, x1)`.
- **Non-interleaved (split in half):** Used by GPT-OSS. Applies cos/sin to each
  half independently.

Also pass `None` for bias when the model has no bias (GPT-OSS has attention bias;
Llama does not). Passing a zero tensor is **not** the same — it may trigger a
different kernel path.
:::

##### MLP class pattern

```python
class YourModelMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tp_group = get_tp_group()
        self.intermediate_size_per_rank = config.intermediate_size // self.tp_group.world_size
        # Define gate_proj_weight, up_proj_weight, down_proj_weight
        # Attach sharding weight loaders
        ...

    def forward(self, hidden_states, is_prefill):
        if is_prefill and self.tp_group.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        output = NF.mlp(hidden_states, self.gate_proj_weight, self.up_proj_weight, self.down_proj_weight)
        if is_prefill:
            output = self.tp_group.reduce_scatter(output, dim=0)
        else:
            self.tp_group.all_reduce(output)
        return output
```

(top-level-model-class-yourmodelforcausallm)=

##### Top-level model class (`YourModelForCausalLM`)

This class implements the interface methods that the vLLM Neuron runtime calls.
These are the hard requirements — the runner expects them to exist:

```python
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec

class YourModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = YourModelBackbone(config)
        self.lm_head = neuron_nn.ColumnParallelLinear(...)
        ...

    def forward(self, input_ids, positions, attn_metadata, sampling_positions, sampling_params, ...):
        """Run the full model: embedding → layers → norm → LM head → sampling."""
        ...

    def get_kv_spec(self) -> KVSpec:
        """Declare the KV cache shape for each layer."""
        layers = []
        for i, layer in enumerate(self.model.layers):
            layers.append(LayerSpec(
                name=f"layers.{i}.self_attn",
                num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                head_size=layer.self_attn.head_dim,
                dtype=layer.self_attn.dtype,
            ))
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        """Bind externally-allocated KV cache tensors to attention layers."""
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    def load_weights(self, checkpoint_path, device, cache_dir):
        """Load weights from a safetensors checkpoint with TP sharding."""
        ...

    @classmethod
    def from_configs(cls, hf_config, neuron_config):
        config = YourModelConfig.from_configs(hf_config, neuron_config)
        return cls(config)
```

##### KV cache lifecycle

The KV cache is managed through a three-step contract between your model and the
runner:

1. **Declaration (`get_kv_spec`)** — At startup, the runner calls your model's
   `get_kv_spec()` to learn the cache requirements for each layer: number of KV
   heads (per rank), head size, dtype, and optionally sliding window size. The
   runner uses this to compute how much memory to allocate and how to organize
   cache groups.

2. **Allocation and binding (`bind_kv_cache`)** — The runner allocates paged KV
   cache tensors on device and calls `bind_kv_cache(kv_caches)` on your model.
   `kv_caches` is a dict mapping layer names (e.g., `"layers.0.self_attn"`) to
   `[k_cache, v_cache]` pairs. Your model stores references to these tensors
   (typically on the attention layer). Each cache tensor has shape
   `[num_blocks, num_kv_heads, block_size, head_size]` (HND paged layout).

3. **Writing at runtime** — During forward passes, the model writes new K/V values
   into the cache using `slot_mapping` from `attn_metadata`. The slot mapping
   converts token positions into (block_index, position_within_block) pairs.
   Padding tokens have `slot_mapping = -1` and should not be written.

The layer names used in `get_kv_spec` must match the keys that the model reads
from `attn_metadata` — the runner uses the same names for both. By convention,
existing models use `f"layers.{i}.self_attn"`.

#### 1c. Define the factory (`factory.py`)

The factory class satisfies vLLM's ModelRegistry interface and selects the correct
implementation variant (e.g., BF16 vs. quantized).

```python
import torch.nn as nn
from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig

class YourModelForCausalLM(nn.Module):
    """Factory that validates config and selects implementation."""

    def __init__(self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(cls, hf_config, neuron_config):
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(cls, hf_config, neuron_config):
        cls._validate_config(hf_config, neuron_config)
        from .model import YourModelForCausalLM as Model
        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(cls, hf_config, neuron_config):
        # Add validation rules (unsupported features, incompatible settings)
        pass
```

(text-model-implement-weight-loading)=

#### 1d. Implement weight loading

Weight loading connects checkpoint tensors (stored in safetensors format) to your
model's parameters, applying any necessary transforms along the way (sharding for
TP, fusing separate weights, transposing). The plugin provides utilities for
this, but you can also implement `load_weights` however you prefer.

##### How the weight loader system works

The flow has three parts: a **mapping** (checkpoint key → parameter name),
**weight loaders** (transform closures attached to parameters), and a **checkpoint
reader** that ties them together.

1. **Mappings** — A dict that maps your model's parameter names (as they appear in
   `state_dict()`) to one or more checkpoint keys. If a parameter comes from
   multiple checkpoint tensors (e.g., fused QKV from separate Q, K, V), map it to a
   list.

2. **Weight loaders** — A `SafetensorsWeightLoader` attached to each parameter via
   `set_weight_loader()`. It contains a `transform` function:
   `(slices: list[PySafeSlice], rank: int) -> torch.Tensor`. The `slices` are lazy
   safetensors references (one per checkpoint key from the mapping), and `rank` is
   the current TP rank. The transform extracts the rank's shard.

3. **SafetensorsCheckpoint** — Reads safetensors files without loading everything
   into memory. `load_sharded_pipelined()` iterates over your mappings, finds each
   checkpoint key, passes slices to the parameter's weight loader, and collects the
   results into a state dict.

##### Example: simple sharding weight loader

`sharding_weight_loader` creates a transform that slices a single checkpoint
tensor along a dimension:

```python
from vllm_neuron.utils.weight_loader import sharding_weight_loader, set_weight_loader

# Shard gate_proj along dim 1 (output features) across TP ranks
loader = sharding_weight_loader(
    shard_dim=1,                          # Which dim to slice
    shard_size=intermediate_size // tp,   # Size of each rank's slice
    num_shards=tp,                        # Total number of shards
    is_storage_transposed=True,           # Checkpoint stores [in, out], not [out, in]
)
set_weight_loader(self.gate_proj_weight, loader)
```

##### Example: fused QKV weight loader

`fused_qkv_weight_loader` handles the case where Q, K, V are stored as separate
checkpoint tensors but your model uses a single fused parameter:

```python
from vllm_neuron.utils.weight_loader import fused_qkv_weight_loader, set_weight_loader

loader = fused_qkv_weight_loader(
    q_size=num_q_heads_per_rank * head_dim,    # Q shard size for this rank
    kv_size=num_kv_heads_per_rank * head_dim,  # KV shard size for this rank
    shard_dim=1,
    num_shards=tp_size,
    is_storage_transposed=True,
    num_kv_replicas=num_kv_replicas,           # >1 when KV heads < TP size (GQA)
)
set_weight_loader(self.qkv_proj_weight, loader)
```

##### Example: rank override for component DP

When a weight should be sharded across a larger group (e.g., TP × attention_dp),
use `with_rank_override` to change the effective rank the loader uses:

```python
from vllm_neuron.utils.weight_loader import with_rank_override

effective_rank = attention_dp_rank + tp_rank * attention_dp_size
loader = with_rank_override(sharding_weight_loader(...), rank=effective_rank)
set_weight_loader(self.o_proj_weight, loader)
```

##### Example: expert-parallel weight loaders (MoE)

For Mixture-of-Experts models with expert parallelism (EP), each rank owns a
contiguous subset of experts. The EP loaders **wrap an inner loader** and restrict
its input to the local expert range *before* the inner loader runs — so per-expert
work (e.g. MXFP4 dequant, TP sharding on another dim) only touches the local
experts instead of loading all experts and discarding the rest.

All three take `local_expert_indices` (this rank's contiguous expert range) and an
`original_loader` constructed for `num_local_experts`. They differ only in how
experts are laid out in the incoming `slices`. Pick the one that matches how your
checkpoint mapping enumerates the expert tensors:

```python
from vllm_neuron.utils.weight_loader import (
    expert_parallel_tensor_dim_loader,
    expert_parallel_grouped_loader,
    expert_parallel_interleaved_loader,
    sharding_weight_loader,
    set_weight_loader,
)

# This rank owns experts [ep_rank*L, (ep_rank+1)*L); indices must be contiguous.
local_expert_indices = list(range(ep_rank * L, (ep_rank + 1) * L))

# (a) Single tensor with experts on a dim — e.g. GPT-OSS [E, ...] stacked weight.
#     Wraps each input in a SliceView over [lo:hi+1] on expert_dim, so the inner
#     TP shard on another dim composes into a single read of the final shape.
loader = expert_parallel_tensor_dim_loader(
    local_expert_indices,
    original_loader=sharding_weight_loader(shard_dim=..., shard_size=..., num_shards=tp),
    expert_dim=0,
)
set_weight_loader(self.experts_weight, loader)

# (b) Flat list grouped by item across experts — e.g. Qwen3 fused gate_up
#     [gate_0..gate_{E-1}, up_0..up_{E-1}] (2 groups), or a single per-expert
#     weight [down_0..down_{E-1}] (1 group). Selects [lo:hi+1] within each group.
loader = expert_parallel_grouped_loader(
    local_expert_indices,
    original_loader=sharding_weight_loader(...),
    total_num_experts=E,        # TOTAL experts the mapping enumerated, not local L
)
set_weight_loader(self.gate_up_proj_weight, loader)

# (c) Flat list interleaved per expert — e.g. DeepSeek gate_up bf16
#     [gate_0, up_0, gate_1, up_1, ...] (stride 2) or fp8
#     [gate_w_0, gate_s_0, up_w_0, up_s_0, ...] (stride 4). Per-expert stride is
#     derived from total_num_experts; selects the contiguous local-expert block.
loader = expert_parallel_interleaved_loader(
    local_expert_indices,
    original_loader=sharding_weight_loader(...),
    total_num_experts=E,
)
set_weight_loader(self.gate_up_proj_weight, loader)
```

Only contiguous expert ranges are supported (non-contiguous
`local_expert_indices` raise). The `original_loader` must define a `transform` —
these wrappers call `original_loader.transform` directly, so a bare
`SafetensorsWeightLoader()` with no transform is not supported.

##### Writing a custom weight loader

If the provided utilities don't fit your needs, write a custom
`SafetensorsWeightLoader`:

```python
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

def my_custom_loader(slices, rank):
    # slices: list of PySafeSlice objects (lazy references to checkpoint tensors)
    # rank: current TP rank
    # Return: the final tensor for this parameter on this rank
    full_tensor = slices[0][:]  # Load the full tensor
    # ... do whatever transform you need ...
    return full_tensor[rank * shard_size : (rank + 1) * shard_size]

set_weight_loader(self.some_param, SafetensorsWeightLoader(transform=my_custom_loader))
```

##### Putting it all together in `load_weights`

```python
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

def load_weights(self, checkpoint_path, device, cache_dir):
    # Define checkpoint key → parameter name mappings
    mappings = {}
    for layer_id in range(len(self.model.layers)):
        prefix = f"model.layers.{layer_id}"
        # Fused QKV: 3 checkpoint keys → 1 parameter
        mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
        ]
        # 1:1 mappings
        mappings[f"{prefix}.self_attn.o_proj_weight"] = f"{prefix}.self_attn.o_proj.weight"
        mappings[f"{prefix}.mlp.gate_proj_weight"] = f"{prefix}.mlp.gate_proj.weight"
        mappings[f"{prefix}.mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
        mappings[f"{prefix}.mlp.down_proj_weight"] = f"{prefix}.mlp.down_proj.weight"
        # Norms: no sharding (weight loaders not set → loaded as-is)
        mappings[f"{prefix}.input_layernorm.weight"] = f"{prefix}.input_layernorm.weight"
        mappings[f"{prefix}.post_attention_layernorm.weight"] = f"{prefix}.post_attention_layernorm.weight"

    # Open checkpoint and load with per-parameter transforms
    checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
    rank_sharded = checkpoint.load_sharded_pipelined(
        self.rank, self.world_size, self, mappings, device
    ).state_dict

    self.load_state_dict(rank_sharded, strict=False, assign=True)
```

Parameters without a weight loader attached are loaded from the checkpoint as-is
(no transform). This is appropriate for unsharded weights like norm scales.

:::{note}
**Tied embeddings (`tie_word_embeddings=True`).** Models like Llama-3.2-1B share
the embedding weight with the LM head. The checkpoint has no `lm_head.weight` key,
so add an explicit mapping
(`mappings["lm_head.weight"] = "model.embed_tokens.weight"`) and load both
independently from the same tensor. Do **not** tie the Python references before
`load_state_dict(assign=True)` — vLLM creates models on the `meta` device, so an
early assignment leaves the LM head pointing at a stale meta tensor ("cannot copy
out of meta tensor").
:::

#### 1e. (Optional) Add EAGLE3 speculative decoding support

Check whether a public EAGLE3 draft model checkpoint exists for the target model.
The EAGLE3 drafter is Llama-architecture-based regardless of the target model, so
no new draft-model code is needed.

If a draft checkpoint exists:

1. Inherit `SupportsEagle3` on the `ForCausalLM` class.
2. Add an `aux_hidden_state_layers` list to the backbone model.
3. Collect hidden states at the specified layer indices during forward.
4. Thread `aux_hidden_states` through all return paths (see the return-value table
   in Step 1b).
5. Implement `set_aux_hidden_state_layers()` and
   `get_eagle3_aux_hidden_state_layers()`.
6. Add an `examples/.../run_eagle3.py` script.

If no draft checkpoint exists, skip this and mark EAGLE3 as not supported. For
deployment steps, see the
EAGLE3 speculative decoding tutorials for
[Llama 3.1](../tutorials/tutorial-eagle3-speculative-decoding-llama-3-1.md) and
[GPT-OSS](../tutorials/tutorial-eagle3-speculative-decoding-gpt-oss.md).

#### 1f. (Optional) Write the model README

Add a `README.md` in `src/model/<model_name>/` documenting the
architecture and feature status:

```markdown
# <Model Name>

<One-line description.>

## Architecture

| Parameter           | Value |
| ------------------- | ----- |
| hidden_size         |       |
| num_attention_heads |       |
| num_key_value_heads |       |
| head_dim            |       |
| num_hidden_layers   |       |
| intermediate_size   |       |
| vocab_size          |       |
| RoPE                |       |
| Activation          |       |
| Normalization       |       |
| tie_word_embeddings |       |

## Key Differences from Reference

- ...

## Feature Status

| Feature            | Status | Notes |
| ------------------ | ------ | ----- |
| TP (head sharding) |        |       |
| SP (seq parallel)  |        |       |
| DP (data parallel) |        |       |
| EP                 |        |       |
| Cross-DP EP        |        |       |
| Eagle3 spec decode |        |       |
| FP8 KV cache       |        |       |
| On-device sampling |        |       |
```

### 2. Register the model with vLLM

Register your model directly with vLLM's `ModelRegistry` so vLLM discovers it at
worker initialization. Using the factory pattern, no changes to vLLM Neuron core
code are needed:

```python
from vllm import ModelRegistry
from .factory import MyCustomModelForCausalLM

ModelRegistry.register_model(
    "MyCustomModelForCausalLM", MyCustomModelForCausalLM
)
```

The string key (`"MyCustomModelForCausalLM"`) must match the `architectures` field
in the model's `config.json` on HuggingFace.

### 3. Compile and run a smoke test

Run a first inference to trigger Neuron compilation and verify the model loads.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/model",
    max_num_seqs=4,
    max_model_len=2048,
    max_num_batched_tokens=1024,
    tensor_parallel_size=8,
)

output = llm.generate(
    ["Hello, my name is"],
    SamplingParams(max_tokens=32, temperature=0.0),
)
print(output[0].outputs[0].text)
```

#### Understanding warmup and compilation

Before the model can serve real requests, the runner executes a **warmup phase**.
During warmup, it calls your model's `forward()` with synthetic inputs for every
bucket size — each prefill sequence length bucket and each decode batch size
bucket. Each call triggers `torch.compile` to trace the graph for that input shape
and compile it into a NEFF (Neuron Executable File Format). This means:

- A separate NEFF is compiled for each unique combination of (prefill bucket size,
  kv_segment_size) and each (decode batch size, decode context length).
- Conditional paths (prefill vs. decode) are traced separately per bucket — each
  bucket's warmup call exercises one branch.
- After warmup completes, `fail_on_recompile` is set — any input shape not covered
  by the warmup buckets will fail rather than silently recompiling.
- Compiled NEFFs are cached on disk. Subsequent server starts with the same model,
  bucket config, and TP degree will skip compilation and load from cache.

Bucket sizes are controlled via NeuronConfig (`num_batched_tokens_buckets` for
prefill, `num_seqs_buckets` for decode). If not specified, defaults are computed
from `max_model_len` and `max_num_seqs`. More buckets means more compilations
during warmup but better padding efficiency at runtime.

#### Runtime padding to buckets

At runtime, actual inputs are padded up to the nearest compiled bucket size before
being passed to the model. For prefill, the token sequence is padded to the next
`num_batched_tokens_buckets` entry. For decode, the batch is padded to the next
`num_seqs_buckets` entry. This means:

- Your model always receives inputs at one of the pre-compiled bucket sizes —
  never the raw unpadded size.
- Padding tokens use `pad_token_id` for `input_ids` and have `slot_mapping = -1`
  so they don't write to the KV cache.
- `sampling_positions` only points at real (non-padding) token positions, so
  padding doesn't affect output correctness.
- Fewer buckets means more padding waste (the gap between actual size and the next
  bucket can be large). More buckets means less waste but longer warmup.

### 4. Validate accuracy

Verify that the model produces correct outputs. The most common approach is
comparing against a reference (e.g., the same model running on GPU via Hugging
Face Transformers), but you can use whatever validation method makes sense for
your model — evaluation benchmarks, known-answer prompts, perplexity checks, etc.

If comparing against a reference:

1. Run the same prompt through vLLM Neuron and the reference, both with greedy
   sampling (`temperature=0.0`).
2. Compare generated tokens. For large models, token-level divergence at later
   positions is expected — use logit-level comparison to confirm outputs are
   within tolerance.
3. If accuracy issues appear, use the systematic workflow in
   [accuracy debugging guide](accuracy-debugging-guide.md) and the
   [accuracy debugger tool](how-to-use-accuracy-debugger.md) to isolate the cause.

#### Running on CPU for debugging

You can run models on CPU by setting the environment variable
`VLLM_NEURON_CPU_MODE=1`. No Neuron hardware or compilation is needed. This is
useful for:

- Rapid iteration during development — no compile wait.
- Isolating whether an accuracy issue is in your model logic vs. the Neuron
  compilation. If outputs differ between CPU mode and on-device, the issue is
  likely in compilation or a kernel; if they match, the issue is in your model
  code.
- Running module tests locally without a Neuron instance
  (`VLLM_NEURON_CPU_MODE=1 pytest ...`).

By default in CPU mode, the functional operators (`vllm_neuron.functional`) use
their PyTorch fallback implementations. You can also enable the NKI CPU simulator
by setting `NKI_SIMULATOR=1`, which runs the actual NKI kernel logic on CPU —
useful for validating kernel correctness without hardware.

#### Existing test patterns to follow

Two levels of accuracy tests give you confidence that the implementation is
correct and provide regression coverage going forward:

- **Module tests** — Test individual components (attention, MLP, RoPE) in
  isolation against a HuggingFace reference. These load real checkpoint weights
  into a single module, run it with known inputs, and compare outputs. Useful for
  catching issues early without running the full model. Can run on CPU
  (`VLLM_NEURON_CPU_MODE=1`) or on device.

- **End-to-end logit tests** — Run the full model through vLLM (offline or online
  serving) and validate output logits against golden references across a grid of
  configurations (TP sizes, sequence lengths, batch sizes). These use
  `vllm_neuron.accuracy.logit_validation.multi_prompt_logit_validation` to compare
  multi-prompt logit outputs with tolerances.

When onboarding a new model, write module tests for each component (attention,
MLP, RoPE, etc.) and an end-to-end logit test.

*Three-way comparison pattern (module tests):* a useful accuracy gate compares
three implementations at each module boundary — FP32 HF (gold standard), BF16 HF
(precision floor from dtype alone), and BF16 vLLM Neuron (the target):

```python
from vllm_neuron.accuracy.testing import assert_close_three_way

assert_close_three_way(
    target=neuron_output,       # what we're testing
    expected=hf_fp32_output,    # gold standard
    baseline=hf_bf16_output,    # precision floor
    rtol=0.01,
    name="attn_prefill",
)
```

*Tiny end-to-end test:* a minimal test that validates the full pipeline
without real weights or Neuron hardware keeps regressions cheap to catch:

```python
# test/model/<name>/tiny/test_tiny_<name>_e2e.py
import tempfile
import pytest
import torch
from transformers import ModelConfig, ModelForCausalLM
from vllm import LLM, SamplingParams

pytestmark = [pytest.mark.fast, pytest.mark.forked]

TINY_CONFIG = ModelConfig(
    vocab_size=256,
    hidden_size=512,       # Must satisfy H % 256 == 0
    intermediate_size=1024,
    num_hidden_layers=1,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=128,
    tie_word_embeddings=False,
)

def _run_inference(tp_size):
    model_dir = tempfile.mkdtemp()
    torch.manual_seed(42)
    ModelForCausalLM(TINY_CONFIG).to(torch.bfloat16).save_pretrained(model_dir)
    llm = LLM(
        model=model_dir,
        max_num_seqs=1,
        max_model_len=128,
        block_size=128,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        num_gpu_blocks_override=4,
        additional_config={"neuron_config": {"num_batched_tokens_buckets": [16, 128]}},
    )
    outputs = llm.generate(
        [{"prompt_token_ids": list(range(1, 11))}],
        SamplingParams(temperature=0.0, max_tokens=3),
    )
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0

def test_tp2():
    _run_inference(tp_size=2)

def test_tp1():
    _run_inference(tp_size=1)
```

:::{note}
The plugin provides a `tensor_capture` NeuronConfig option for capturing
intermediate activations and a `tensor_replacement` option for injecting reference
tensors layer-by-layer. These are valuable for isolating where drift is
introduced.
:::

### 5. Benchmark and tune performance

Once accuracy is validated, measure latency and throughput with the
[`vllm bench`](https://docs.vllm.ai/en/latest/cli/bench/serve.html) CLI.

```bash
# Start the server
vllm serve /path/to/model \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32

# Benchmark (in a separate terminal) using randomly generated input data
vllm bench serve \
    --model /path/to/model \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 100
```

Key tuning parameters (passed via `--additional-config '{"neuron_config": {...}}'`):

- `tensor_parallel_size` — Number of NeuronCores for TP.
- `max_num_seqs` — Controls continuous batching concurrency.
- `max_model_len` — Affects compilation bucket sizes and memory usage.
- `attention_dp_size` — Shards Q/O across DP ranks during decode for better batch
  scaling.
- `mlp_dp_size` — Shards MLP intermediate across DP ranks.
- `on_device_sampling_config` — Enables on-device sampling to overlap sampling with
  next-token compute.
- `num_batched_tokens_buckets` / `num_seqs_buckets` — Control compilation bucket
  sizes.

For more features you can enable after onboarding, see
[features guide](../guides/features-guide.md).

## Confirm your work

To confirm the model is correctly onboarded:

1. **Compilation:** The model compiles without errors and loads from cache on
   subsequent runs.
2. **Accuracy:** Generated output is within tolerance of the reference
   implementation under greedy sampling.
3. **Performance:** Latency and throughput meet targets on the intended instance
   type.
4. **Serving:** The model serves requests correctly through both offline
   (`LLM.generate()`) and online (`vllm serve`) modes.

## Common issues

### Unsupported or non-compilable operations

- **Possible solution**: The plugin uses `torch.compile` to trace your model into a
  static graph that the Neuron compiler can lower to hardware. Everything in your
  `forward()` path must be compilable — no operations that break out of the graph.
  Common graph breaks include: calling `.item()` or `.tolist()` on a tensor
  (materializes data to Python), data-dependent control flow
  (`if tensor.sum() > 0`), in-place ops on views in some contexts, and Python-side
  print/logging that references tensor values. If you hit a graph break, the error
  message from `torch._dynamo` will point at the offending line. Restructure to use
  only tensor operations that can be traced statically.

  **On conditional paths:** Python-level branching based on metadata values (e.g.,
  `if max_query_len <= decode_token_threshold`) does create separate compiled
  graphs — one per branch taken. This is expected and handled by the system: during
  warmup, each bucket size is traced and compiled into its own NEFF. Once warmup is
  complete, `torch.compiler.set_stance("fail_on_recompile")` is active, meaning no
  *new* graph variants are allowed at runtime. So conditional paths are fine as long
  as they are exercised during warmup (the runner handles this automatically for
  standard prefill/decode dispatch). What you cannot do is introduce conditionals
  that produce an unbounded or unexpected number of graph variants — each unique
  path is a separate compilation.

### `KeyError` or missing key during weight loading

- **Possible solution**: Your checkpoint-to-parameter name mapping is incomplete.
  Check that your `mappings` dict in `load_weights` covers all parameters. For
  fused weights (e.g., separate Q/K/V → fused QKV), ensure the mapping is a list of
  checkpoint keys, not a single key. If a parameter name in your model doesn't
  match the checkpoint's naming convention, add the appropriate mapping entry.
  Parameters without a weight loader attached are loaded as-is — if the checkpoint
  key name differs from the parameter name, you'll get a `KeyError`.

### Accuracy drift

- **Possible solution**: Start with the
  [accuracy debugging guide](accuracy-debugging-guide.md) to isolate whether drift
  comes from attention, KV cache, weight loading, or sampling. Common culprits:
  incorrect RoPE variant (interleaved vs. split-half), wrong head dimension after
  sharding, dtype mismatches, and KV cache quantization scale errors. Use
  `tensor_capture` / `tensor_replacement` in NeuronConfig to inject reference
  tensors at specific module boundaries and narrow down which layer introduces the
  divergence.

## Related information

- [Onboard a vision-language model](onboarding-vlm-models.md) — VLM-specific
  companion to this guide: adds a vision encoder tower, block-packed attention,
  M-RoPE, and the on-device encoder cache.
- [Features guide](../guides/features-guide.md) — Features you can enable once the model is
  onboarded.
- [Accuracy debugging guide](accuracy-debugging-guide.md) — Diagnose accuracy
  issues discovered during onboarding.
- [Accuracy debugger tool](how-to-use-accuracy-debugger.md) — Step-by-step
  procedure for the accuracy debugger tool.
- For supported models and features, see the [README](https://github.com/vllm-project/vllm-neuron#supported-models)
  and [model cards](../model-recipes/index.md).
- [Deep dive: vLLM integration](../design/vllm/vllm-integration-design-reference.md) —
  Architecture reference for the Neuron backend.
- [NxDI model onboarding guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/onboarding-models.html)
  — Legacy onboarding guide for NxDI-only deployments (without vLLM).
- [vLLM Neuron plugin source](https://github.com/vllm-project/vllm-neuron) — Plugin
  source code with annotated reference implementations.
