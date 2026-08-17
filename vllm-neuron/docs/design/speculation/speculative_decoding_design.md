# Speculative Decoding (EAGLE3) in the vLLM Neuron Framework

<!-- meta: description: Speculative decoding (EAGLE3) design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

### What is Speculative Decoding?

Speculative decoding is a technique for accelerating autoregressive LLM inference without changing the output distribution. The core idea: use a fast, lightweight **draft model** to predict several tokens ahead, then **verify** those predictions in parallel using the full **target model**.

In standard autoregressive decoding, the target model generates one token per forward pass. Each pass is memory-bandwidth-bound on accelerators — the model weights must be loaded from HBM for every single token. Speculative decoding amortizes this cost by verifying multiple draft tokens in a single target forward pass.

The algorithm proceeds as follows:

1. The target model generates a token and produces hidden states.
2. The draft model consumes those hidden states and proposes `K` speculative tokens (`K = num_speculative_tokens`).
3. On the next step, the target model processes all `K` draft tokens plus the original token in a single forward pass, producing logits for each position.
4. A **rejection sampler** compares draft predictions against target logits. Tokens are accepted sequentially until the first mismatch. The corrected token at the mismatch position (or a bonus token if all are accepted) is emitted.
5. The process repeats from step 1.

This guarantees the output distribution is identical to the target model alone (for greedy decoding, outputs are bit-exact). The speedup depends on the draft model's **acceptance rate** — how often its predictions match the target.

### What is EAGLE3?

EAGLE3 (Extrapolation Algorithm for Greater Language-model Efficiency, version 3) is a speculative decoding method that uses **auxiliary hidden states** from intermediate layers of the target model as input to the draft model. This is a key improvement over earlier approaches:

- **EAGLE (v1)**: Uses the target model's final hidden state to predict draft tokens.
- **EAGLE3**: Extracts hidden states from **three** intermediate target layers (configurable via `eagle_aux_hidden_state_layer_ids`), concatenates them, and fuses them through a linear projection. This gives the draft model richer context from different depths of the target model, significantly improving acceptance rates.

The EAGLE3 draft model architecture is minimal:

- A **single decoder layer** (not a full copy of the target model)
- An **FC layer** that combines the 3 auxiliary hidden states into one
- An **embedding layer** for token inputs
- An **LM head** for producing logits

This makes the draft model extremely lightweight compared to the target — it adds minimal latency per speculative token while achieving high acceptance rates.

## API Usage and Configuration

### Enabling EAGLE3 Speculative Decoding

EAGLE3 is enabled through the `speculative_config` parameter when creating a vLLM `LLM` instance. On-device sampling should always be used in production deployments for optimal performance — CPU sampling is only useful when access to raw logits is needed for debugging or accuracy testing.

``` python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=8,
    max_model_len=256,
    max_num_seqs=4,
    speculative_config={
        "method": "eagle3",
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 5,
    },
    additional_config={
        "neuron_config": {
            "on_device_sampling_config": {"all_greedy": "true"},
        }
    },
)

sampling_params = SamplingParams(temperature=0, max_tokens=128)
outputs = llm.generate(["What is deep learning?"], sampling_params)
```

### Configuration Parameters

`speculative_config` fields:

| Parameter | Required | Description |
|----|----|----|
| `method` | Yes | Must be `"eagle3"`. Currently the only supported speculation method on Neuron. |
| `model` | Yes | Path or HuggingFace ID for the EAGLE3 draft model checkpoint. |
| `num_speculative_tokens` | Yes | Number of draft tokens to propose per step. See {ref}`Choosing num_speculative_tokens <choosing-num_speculative_tokens>` for guidance. |

`additional_config.neuron_config` fields relevant to speculation:

| Parameter | Description |
|----|----|
| `on_device_sampling_config` | Enables on-device sampling. When enabled, both sampling and rejection sampling run on the Neuron device, avoiding CPU round-trips. This is the recommended production configuration. CPU sampling (when this is not set) is only useful when raw logits are needed for debugging or accuracy testing. |

Note on draft model sampling: regardless of what sampling configuration is used for the target model (greedy, temperature, top-k, etc.), the draft model **always uses greedy sampling** internally. This is a design choice — the draft model's role is to predict the most likely continuation, and greedy decoding is the most efficient strategy for that purpose. The rejection sampler ensures the final output distribution matches the target model's sampling configuration.

### Design Choices

**Greedy draft sampling**: The draft model always uses greedy (argmax) decoding regardless of the target model's sampling parameters. The draft model's purpose is to produce the best-guess continuation as quickly as possible. Whether those guesses are accepted is determined by the rejection sampler, which respects the target model's full sampling configuration (temperature, top-k, top-p).

**Position-aware speculation cutoff**: Speculation is automatically skipped when a sequence's position reaches `max_model_len - num_speculative_tokens`. The draft model generates tokens beyond the current position, so this cutoff prevents KV cache overflow. When a sequence is near its length limit, the system falls back to standard single-token decoding for the remaining tokens.

## Architecture

This section describes how EAGLE3 speculative decoding is architected within the vLLM Neuron vLLM plugin and how the components connect.

### Component Overview

``` text
┌─────────────────────────────────────────────────────────────────────┐
│                        vLLM Engine (Scheduler)                      │
│  Manages requests, assigns draft tokens to next step's input        │
└───────────────┬───────────────────────────────┬─────────────────────┘
                │                               │
                │ SchedulerOutput               │ take_draft_token_ids()
                ▼                               ▲
┌───────────────────────────────────────────────┴─────────────────────┐
│                        NeuronWorker                                 │
│  Delegates to NeuronModelRunner                                     │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     NeuronModelRunner                               │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │ Target Model │───▶│ EagleProposer    │───▶│ RejectionSampler │   │
│  │ (compiled)   │    │ (draft, compiled)│    │ (on-device/CPU)  │   │
│  └──────┬───────┘    └──────────────────┘    └──────────────────┘   │
│         │                                                           │
│         │ aux_hidden_states (3 layers)                              │
│         │ hidden_states                                             │
│         │ logits / sampled_tokens                                   │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────┐                │
│  │            Shared KV Cache (Block Tables)       │                │
│  │  Target layers: layers.0 ... layers.N-1         │                │
│  │  Draft layers:  layers.N ... layers.N+D-1       │                │
│  └─────────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

Key source files:

| File | Role |
|----|----|
| `vllm_neuron/vllm/spec_decode/eagle.py` | `EagleProposer`: loads, compiles, and runs the draft model |
| `vllm_neuron/model/llama3/eagle3_model.py` | `Eagle3LlamaForCausalLM`: EAGLE3 draft model architecture |
| `vllm_neuron/nn/rejection_sampler.py` | On-device greedy rejection sampler (tensor operations) |
| `vllm_neuron/vllm/sample/rejection_sampler.py` | CPU rejection sampler (greedy + probabilistic) |
| `vllm_neuron/vllm/worker/neuron_model_runner.py` | Integration: orchestrates target → draft → rejection flow |
| `vllm_neuron/vllm/worker/neuron_worker.py` | Exposes `take_draft_token_ids()` to vLLM scheduler |
| `vllm_neuron/model/llama3/factory.py` | Factory pattern for `Eagle3LlamaForCausalLM` |
| `vllm_neuron/model/registry.py` | Model registry (exports `Eagle3LlamaForCausalLM`) |

### End-to-End Execution Flow

The following describes one complete step of speculative decoding:

#### Step 0: Initialization

1. `NeuronModelRunner.__init__` detects `speculative_config.method == "eagle3"` and creates an `EagleProposer` and `RejectionSampler`. (`neuron_model_runner.py:286-291`)
2. `load_model()` loads and compiles the target model, configures EAGLE3 auxiliary layers via `SupportsEagle3` interface, then loads and compiles the draft model. (`neuron_model_runner.py:549-659`)
3. During warmup, both target and draft models are warmed up with synthetic inputs to trigger Neuron compilation. See [Warmup with Speculation Enabled](#warmup-with-speculation-enabled) for details.

#### Step N: Verification + Proposal (steady state)

``` text
┌──────────────────────────────────────────────────────────────────┐
│ 1. Prepare Input                                                 │
│    - Scheduler provides input_ids with draft tokens interleaved  │
│    - _build_spec_decode_metadata() computes indices:             │
│      target_logits_indices, bonus_logits_indices, logits_indices │
│      draft_token_ids (for rejection comparison)                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. Target Model Forward Pass                                     │
│    - Processes original + draft tokens in one pass               │
│    - Returns: (sampled_tokens/logits, hidden_states,             │
│               aux_hidden_states, [gathered_logits])              │
│    - On-device path: rejection_sampler() runs inside the model   │
│      compiled graph, returns already-rejected tokens             │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. Sampling / Rejection                                          │
│    On-device: parse rejection_sampler output → accepted tokens   │
│    CPU: bonus_logits → sample bonus token                        │
│         target_logits → RejectionSampler.forward()               │
│         → accepted + recovered + bonus tokens                    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. Draft Proposal (_propose_draft_token_ids)                     │
│    - Concatenate 3 aux_hidden_states → [T, hidden_size*3]        │
│    - Call EagleProposer.propose():                               │
│      a) Shift input_ids, inject next_token_ids at last positions │
│      b) Prefill pass: draft model with initial_target_hidden_    │
│         states, full attention metadata → 1st draft token        │
│      c) Recurrent loop (K-1 times):                              │
│         - Increment positions, compute slot mapping              │
│         - Draft model with recurrent_target_hidden_states        │
│         - Greedy argmax → next draft token                       │
│    - Return [batch_size, num_speculative_tokens]                 │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. Output + Scheduling                                           │
│    - NeuronWorker.take_draft_token_ids() returns draft tokens    │
│    - vLLM scheduler interleaves them in next step's input        │
└──────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### EagleProposer: Draft Model Management

The `EagleProposer` class (`eagle.py`) is the central orchestrator for draft token generation. It handles:

**Model Loading and Compilation** (`compile_and_load_draft_model`):

- Prepends `"Eagle3"` to the draft architecture name if not already present (e.g., `LlamaForCausalLM` → `Eagle3LlamaForCausalLM`).
- Resolves the model class from vLLM Neuron's model registry.
- Handles **hidden dimension padding**: if the target model's padded hidden size differs from the draft config's hidden size, sets `draft_hf_config.unpadded_hidden_size` and updates `hidden_size` to match.
- Applies the same TP group sharding as the target model.
- Forces greedy sampling for the draft model (`OnDeviceSamplingConfig(all_greedy=True)`).
- Compiles with `torch.compile(model, backend="vllm_neuron", fullgraph=True)`.

**Layer Indexing**:

Draft model layers are assigned indices starting after the target model's layers. For a target with `N` layers and a draft with `D` layers (typically `D=1`):

``` text
Target layers:  layers.0, layers.1, ..., layers.N-1
Draft layers:   layers.N, layers.N+1, ..., layers.N+D-1
```

This is set via `start_layer_idx=target_num_layers` during `from_configs()`. The naming scheme (`layers.{N}.self_attn`) is used for KV cache binding and attention metadata routing.

**The \`\`propose()\`\` Method**:

Two-phase draft token generation:

1. **Prefill phase**: Processes all target tokens through the draft model to update the draft KV cache. Uses full attention metadata from the target's forward pass. Input IDs are shifted by one and patched with `next_token_ids` at last-token positions via `scatter()`. Returns the first draft token and a recurrent hidden state.

2. **Recurrent phase**: Generates remaining `K-1` tokens autoregressively. Each iteration:

    - Increments positions by 1
    - Computes block IDs and slot mapping from block tables: `slot = block_table[pos // block_size] * block_size + pos % block_size`
    - Constructs minimal decode attention metadata
    - Runs draft forward with `recurrent_target_hidden_states`
    - Applies greedy argmax (or on-device sampler) to get next token

    Returns `[batch_size, num_speculative_tokens]` tensor.

**Rank Tensor Workaround**:

The TP rank is passed as an input tensor (`self.rank_tensor`) rather than retrieved inside the model. This prevents XLA from treating `dist.get_rank()` as a compile-time constant during Neuron compilation (`eagle.py:43-49`).

### Eagle3 Draft Model Architecture

The draft model (`eagle3_model.py`) has three layers:

**Eagle3LlamaDecoderLayer**:

A modified decoder layer that concatenates token embeddings with target hidden states before attention:

``` text
embeds = input_layernorm(embeds)               # [T, H]
hidden = hidden_norm(target_hidden_states)     # [T, H]
concat = cat([embeds, hidden], dim=-1)         # [T, 2H]
out = self_attn(concat)                        # QKV sized for 2H input → H output
out = residual + out
out = mlp(post_attention_layernorm(out)) + out
```

The QKV projections are initialized with `qkv_input_size_override=2*hidden_size` to handle the doubled input dimension.

**Eagle3LlamaModel**:

- `embed_tokens`: `VocabDimShardedEmbedding` — vocabulary-sharded for TP
- `fc`: `Linear(hidden_size * 3, hidden_size)` — combines 3 auxiliary hidden states
- Single `Eagle3LlamaDecoderLayer`
- `norm`: Final RMSNorm
- `rotary_emb`: Shared rotary embeddings

Returns `(hidden_states, hidden_prenorm)` where `hidden_prenorm` is the pre-norm state used as recurrent input for subsequent passes.

**Eagle3LlamaForCausalLM**:

Top-level wrapper with two forward modes:

- **Initial (prefill)**: Receives `initial_target_hidden_states` of shape `[T, hidden_size * 3]` → calls `combine_hidden_states()` (FC projection) → forward through model → LM head → sample.
- **Recurrent (decode)**: Receives `recurrent_target_hidden_states` of shape `[T, hidden_size]` → forward through model → LM head → sample.

Returns `(sampled_token_ids_or_logits, recurrent_state)`.

### Auxiliary Hidden State Extraction

The target model implements the `SupportsEagle3` interface (`model/llama3/model.py:1105`):

``` python
class LlamaForCausalLM(nn.Module, SupportsEagle3):
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)
```

During the target model's forward pass, hidden states are captured at the specified layer indices:

``` python
# In LlamaModel.forward():
aux_hidden_states = []
for idx, decoder_layer in enumerate(self.layers):
    if idx in self.aux_hidden_state_layers:
        aux_hidden_states.append(hidden_states)
    hidden_states = decoder_layer(hidden_states, ...)
return hidden_states, aux_hidden_states
```

The layer indices can be:

1. Specified in the draft model config via `eagle_aux_hidden_state_layer_ids`
2. Defaulted by the target model's `get_eagle3_aux_hidden_state_layers()` (typically layers 2, N/2, N-3 for an N-layer model)

Configuration is done in `NeuronModelRunner._get_eagle3_aux_layers_from_config()` (`neuron_model_runner.py:662-684`).

### KV Cache Sharing

Target and draft models share the same vLLM KV cache infrastructure:

- Both models report their KV specs via `get_kv_spec()` (`neuron_model_runner.py:3101-3105` combines target + drafter specs).
- vLLM allocates block tables spanning all layers (target + draft).
- `bind_kv_cache()` binds the pre-allocated KV tensors to each model's attention layers.
- `validate_same_kv_cache_group()` ensures all draft layers belong to the same KV cache group (required so they can share a single `AttentionMetadata`).

During the recurrent phase of `propose()`, the draft model computes its own slot mappings from the shared block tables to update its KV cache entries:

``` python
block_numbers = positions // block_size
block_ids = block_table_tensor.gather(dim=1, index=block_numbers.view(-1, 1)).view(-1)
slot_mapping = block_ids * block_size + (positions % block_size)
```

### Rejection Sampling

Two implementations exist, selected based on whether on-device sampling is enabled. On-device sampling should always be used in production. CPU sampling exists for cases where raw logits are needed (debugging, accuracy testing).

**On-Device Rejection Sampler** (`nn/rejection_sampler.py`):

Runs inside the target model's compiled graph. Pure tensor operations compatible with `torch.compile`:

1. Extracts bonus tokens from interleaved target output
2. Reshapes flat draft/target tokens to `[batch_size, max_spec_len]`
3. Computes match mask: `drafts_2d == targets_2d`
4. Finds first mismatch via `argmax` on inverted match mask with sentinel
5. Accepts up to and including the mismatch position (emitting the corrected target token)
6. Appends bonus token if all drafts were accepted

Returns `[batch_size, max_spec_len + 1]` with `-1` for rejected positions.

**CPU Rejection Sampler** (`vllm/sample/rejection_sampler.py`):

Full probabilistic rejection sampling per [arXiv:2211.17192](https://arxiv.org/abs/2211.17192). Handles mixed greedy/random batches:

- **Greedy requests**: Accept if `draft_token == argmax(target_logits)`
- **Random requests**: Accept if `uniform_sample <= target_prob(draft_token) / draft_prob` (where `draft_prob = 1` since draft probabilities are not available). Rejected positions use **recovered tokens** sampled via Gumbel-max trick: `argmax(adjusted_prob / exponential_noise)` where `adjusted_prob = max(target_prob - draft_prob, 0)` with the draft token zeroed out.
- **Bonus tokens**: Sampled from target distribution at the last draft position when all drafts are accepted.

Supports per-request temperature, top-k, and top-p via `SamplingMetadata`.

### SpecDecodeMetadata: Index Bookkeeping

`SpecDecodeMetadata` (from `vllm.v1.spec_decode.metadata`) tracks the interleaved structure of verified inputs:

``` text
Given 5 requests with cu_num_scheduled_tokens = [4, 104, 107, 207, 209]
and num_draft_tokens = [3, 0, 2, 0, 1]:

cu_num_draft_tokens:   [3, 3, 5, 5, 6]
logits_indices:        [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
target_logits_indices: [0, 1, 2, 5, 6, 9]     (draft positions for verification)
bonus_logits_indices:  [3, 4, 7, 8, 10]        (bonus positions, one per request)
```

- `logits_indices`: Positions in the padded input where logits should be computed (both draft verification and bonus positions).
- `target_logits_indices`: Subset of `logits_indices` pointing to draft positions — used for comparing draft tokens against target logits.
- `bonus_logits_indices`: Subset pointing to the last position per request — used for sampling the bonus token.
- `draft_token_ids`: Extracted from `input_ids` at the draft positions for comparison during rejection sampling.

Built in `_build_spec_decode_metadata()` (`neuron_model_runner.py:1140-1207`).

### Decode Token Threshold

The attention metadata includes a `decode_token_threshold` that controls prefill/decode behavior classification. During speculation:

``` text
decode_token_threshold = 1 + max_num_draft_tokens
```

This ensures that when the target model processes `1 + K` tokens per request (1 original + K draft tokens), attention correctly treats this as a decode operation (not prefill). Without speculation, the threshold is 1.

Set in `_build_attn_metadata()` (`neuron_model_runner.py:1477-1484`).

### Weight Loading and Padding

EAGLE3 weight loading handles two challenges:

**1. TP Sharding**: Same tensor-parallel sharding as the target model. All linear layers use rank-specific slices. Embeddings are sharded along the vocab dimension.

**2. Hidden Dimension Padding**: When the target model's hidden size is padded for alignment (e.g., 2880 → 3072), the draft model must match. Two custom weight loaders handle this:

- `fc_interleaved_padding_weight_loader`: Pads the FC layer's weight from `[unpadded_H, unpadded_H * 3]` to `[padded_H, padded_H * 3]`. The input dimension has 3 concatenated aux hidden states, each padded separately.
- `embedding_sharding_padding_weight_loader`: Shards embedding along vocab dim for TP and pads along hidden dim. Handles last-rank padding when vocab doesn't divide evenly.

### Warmup with Speculation Enabled

When speculative decoding is enabled, warmup must compile **additional graph variants** beyond what standard (non-speculative) warmup produces. The warmup process ensures that both the target model and the draft model are compiled for all bucket sizes before serving begins. This is critical because `torch.compile` with the vLLM Neuron backend traces and compiles a new NEFF artifact for each distinct input shape — warmup triggers these compilations upfront so they don't happen during serving.

**Prefill warmup** (`warmup_prefill`, per prefill bucket size):

1. **Target model**: Compiled with `spec_decode_metadata=None` (no draft tokens exist during the first prefill of a request). Synthetic inputs simulate a single request of `bucket_size` tokens.

2. **Draft model — initial pass**: The draft model is warmed up with the same bucket size. `EagleProposer.warmup()` creates synthetic inputs:

    - `target_token_ids`: `[num_tokens]` of ones
    - `target_positions`: zeros (safe values to prevent KV cache overflow during recurrent passes)
    - `target_hidden_states`: `[num_tokens, hidden_size * 3]` (simulating 3 concatenated auxiliary hidden states)
    - `next_token_ids` and `last_token_indices`: per-request synthetic values

    This calls `propose()` end-to-end, compiling both the prefill graph (initial pass with `initial_target_hidden_states`) and `K-1` recurrent decode graphs (with `recurrent_target_hidden_states`).

**Decode warmup** (`warmup_decode`, per decode batch size bucket):

1. **Target model — verification pass**: With speculation, the target model processes `batch_size * (1 + num_speculative_tokens)` tokens per decode step (1 original + K draft tokens per request). Warmup creates synthetic `SpecDecodeMetadata` (via `_create_warmup_spec_decode_metadata`) with:

    - `num_draft_tokens = [K] * batch_size`
    - Properly computed `cu_num_draft_tokens`, `target_logits_indices`, `bonus_logits_indices`, and `logits_indices`
    - Moved to device to match the inference code path

    This compiles the target model graph that includes spec decode metadata handling and on-device rejection sampling.

2. **Draft model — decode pass**: The draft model is warmed up with `batch_size * (1 + num_speculative_tokens)` tokens and `batch_size` requests. Fresh attention metadata is built for the draft layers. This compiles the draft model's decode-mode graph variants.

**Why separate warmup graphs are needed**:

Without speculation, the target model has two graph variants: prefill (many tokens, 1 request) and decode (1 token per request). With speculation, an additional decode variant is needed: the verification graph where the target processes `1 + K` tokens per request with `SpecDecodeMetadata`. The draft model adds its own graphs: the initial pass (prefill-like, with `initial_target_hidden_states`) and recurrent passes (decode-like, with `recurrent_target_hidden_states`). All must be compiled during warmup to avoid compilation stalls during serving.

## Adding EAGLE3 Support for a New Model

To add EAGLE3 speculative decoding support for a new model architecture, follow these steps:

### Step 1: Implement the `SupportsEagle3` Interface on the Target Model

The target model must implement the `SupportsEagle3` interface from vLLM:

``` python
from vllm.model_executor.models.interfaces import SupportsEagle3

class MyModelForCausalLM(nn.Module, SupportsEagle3):

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Store which layer indices should capture auxiliary hidden states."""
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Return default auxiliary layer indices for this architecture.

        Convention: early layer, middle layer, late layer.
        """
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)
```

In the backbone model's `forward()`, capture hidden states at the specified layers:

``` python
def forward(self, input_ids, positions, attn_metadata, ...):
    hidden_states = self.embed_tokens(input_ids)
    aux_hidden_states = []

    for idx, layer in enumerate(self.layers):
        if idx in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states)
        hidden_states = layer(hidden_states, ...)

    hidden_states = self.norm(hidden_states)
    return hidden_states, aux_hidden_states
```

The top-level `ForCausalLM` class must return auxiliary hidden states in its output tuple when they are non-empty. The expected return signatures are:

- **Without on-device sampling**: `(logits, hidden_states, aux_hidden_states)`
- **With on-device sampling, no spec decode metadata**: `(sampled_tokens, hidden_states, aux_hidden_states, gathered_logits)`
- **With on-device sampling + spec decode metadata**: `(rejection_sampled_tokens, hidden_states, aux_hidden_states)`

See `LlamaForCausalLM.forward()` in `model/llama3/model.py:1140-1197` for the reference implementation.

### Step 2: Implement the EAGLE3 Draft Model

Create an `Eagle3<ModelName>ForCausalLM` class following the pattern in `model/llama3/eagle3_model.py`. The key components:

1. **Decoder layer** with concatenated input (`2 * hidden_size`):
    - Set `qkv_input_size_override=2 * config.hidden_size` on the attention layer
    - Add `hidden_norm` for normalizing target hidden states
    - Concatenate `[embeds, hidden_states]` before attention
2. **Backbone model** with:
    - `embed_tokens`: `VocabDimShardedEmbedding`
    - `fc`: `Linear(hidden_size * 3, hidden_size)` for combining aux states
    - Single decoder layer (`assert config.num_hidden_layers == 1`)
    - Return `(hidden_states, hidden_prenorm)` for recurrent state
3. **ForCausalLM wrapper** with:
    - Two forward modes: `initial_target_hidden_states` vs. `recurrent_target_hidden_states`
    - LM head (`ColumnParallelLinear`)
    - Optional on-device sampler
    - `from_configs()` class method accepting `start_layer_idx` and `batch_size`
    - `load_weights()` with checkpoint mapping
    - `get_kv_spec()` and `bind_kv_cache()` for KV cache integration

### Step 3: Create a Factory and Register the Model

Add a factory class in your model's `factory.py`:

``` python
class Eagle3MyModelForCausalLM:
    @staticmethod
    def from_configs(config, start_layer_idx, batch_size, neuron_config=None):
        model_config = MyModelConfig.from_configs(hf_config=config, neuron_config=neuron_config)
        return Eagle3MyModelImpl(model_config, batch_size, start_layer_idx)
```

Register in `model/registry.py`:

``` python
from vllm_neuron.model.mymodel.factory import Eagle3MyModelForCausalLM

def get_models():
    return [
        ("MyModelForCausalLM", MyModelForCausalLM),
        ("Eagle3MyModelForCausalLM", Eagle3MyModelForCausalLM),
        # ...
    ]
```

The `EagleProposer` will automatically prepend `"Eagle3"` to the target architecture name and look it up in the registry.

### Step 4: Configure Auxiliary Hidden State Layer IDs

Two options:

1. **In the draft model's HuggingFace config** (preferred for portability):

    ``` json
    {
      "eagle_aux_hidden_state_layer_ids": [2, 16, 29]
    }
    ```

2. **In the target model's \`\`get_eagle3_aux_hidden_state_layers()\`\`** as a default fallback.

Choose layer indices that provide diverse representations — typically an early layer, a middle layer, and a late layer.

### Step 5: Test

1. Create a run example in `examples/vllm_neuron/models/<model>/run_eagle3.py`
2. Verify output matches non-speculative decoding (bit-exact for greedy)

Checklist summary:

|  |  |
|----|----|
| \[ \] | Target model implements `SupportsEagle3` and returns `aux_hidden_states` |
| \[ \] | `Eagle3<Model>ForCausalLM` draft model with single layer, FC combiner, two forward modes |
| \[ \] | Factory + registry registration with `"Eagle3<Model>ForCausalLM"` key |
| \[ \] | Weight loaders handle TP sharding + padding (if applicable) |
| \[ \] | `get_kv_spec()` and `bind_kv_cache()` on draft model |
| \[ \] | Auxiliary layer IDs configured (config or default) |
| \[ \] | Integration test verifies bit-exact greedy output |

## Performance Considerations

### Acceptance Rate and Speedup

The theoretical speedup from speculative decoding is:

``` text
Speedup ≈ (1 + K * α) / (1 + K * draft_overhead)
```

Where `K` is `num_speculative_tokens`, `α` is the mean acceptance rate, and `draft_overhead` is the ratio of draft model latency to target model decode latency.

EAGLE3's auxiliary hidden states from multiple target layers improve `α` compared to earlier approaches that only use the final hidden state.

(choosing-num_speculative_tokens)=

### Choosing `num_speculative_tokens`

`num_speculative_tokens` is the most important tuning knob for speculative decoding performance. The optimal value depends on the model, task, and workload:

- **Higher values** increase the potential speedup per step when acceptance rates are high, but also increase the cost of each step (more draft model forward passes, more tokens for the target to verify).
- **Lower values** reduce overhead per step but limit the maximum tokens gained per step.
- Acceptance probability **compounds** — the probability of accepting all K tokens is roughly `α^K`. This means there are diminishing returns as K increases. At some point, the extra draft overhead exceeds the value of the rarely-accepted last token.

General guidelines:

- Start with `num_speculative_tokens=3` and benchmark.
- If acceptance rates are consistently high (\>80%), try increasing to 5.
- If acceptance rates are low (\<60%), reduce to 2 or investigate draft model quality.
- Profile the actual wall-clock time — acceptance rate alone doesn't capture the overhead trade-off. The goal is to minimize end-to-end latency (TPOT), not to maximize acceptance rate.

### On-Device vs. CPU Sampling

**On-device sampling** is the recommended production configuration. Both sampling and rejection happen on the Neuron device, avoiding CPU round-trips. The on-device rejection sampler uses pure tensor operations compatible with `torch.compile`.

**CPU sampling** should only be used when access to raw logits is needed — for example, during accuracy testing, logit validation, or debugging. In this mode, the model returns full logits to CPU, and both bonus token sampling and rejection sampling happen on CPU. The CPU rejection sampler supports the full range of sampling parameters (temperature, top-k, top-p, probabilistic rejection with Gumbel-max recovery).
