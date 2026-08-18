# Canonical Model — Design & Porting Guide

<!-- meta: description: Model bringup workflow -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

> This is the detailed code-pattern reference for implementing models. For the
> end-to-end onboarding workflow (compile, validate, benchmark), see
> [Onboard a new model](../../model-dev/onboarding-models.md).

## Purpose

The GPT-OSS BF16 model (`vllm_neuron/model/gpt_oss/model_bf16.py`)
serves as the **canonical reference implementation** for bringing up models on
the Neuron backend with full parallelism support (TP, SP, DP, EP).

Every section in `model_bf16.py` is annotated:

- `# >>> PARALLELISM: ... <<<` — Infrastructure code. **Keep as-is** when porting.
- `# <-- MODEL-SPECIFIC: ...` — Architecture-specific code. **Change** when porting.

An AI (or human) porting a new model should copy this code, change the
`MODEL-SPECIFIC` sections to match the new architecture, and keep all
`PARALLELISM` sections unchanged.

### Critical principles for porting

**The canonical code is a structural template, not a spec for the target
model.** Everything marked `MODEL-SPECIFIC` in `model_bf16.py` is specific
to GPT-OSS — do not copy it blindly into the target model. Always derive
model-specific decisions (head counts, activation functions, normalization
type, RoPE variant, expert routing, sliding window, etc.) from the user-provided
reference code and model config for the target architecture. Read the target
model's HuggingFace `modeling_*.py` and `config.json` as the source of truth.
The canonical code shows *where* model-specific logic goes and *what kind* of
decisions need to be made — it does not tell you what the right answers are
for a different model.

**When unsure, ask — do not guess.** If you are not confident about a
dimension, a sharding strategy, a weight mapping, or how a HuggingFace
config field maps to the canonical code, stop and ask rather than inserting
something plausible. A wrong shard dimension or an incorrect transpose will
not crash — the model will silently produce wrong outputs and the resulting
accuracy bug can take days to track down. The cost of asking is minutes; the
cost of a silent accuracy bug is days of debugging.

**Do not invent code to fill gaps.** If the canonical model has a feature
(e.g. attention sinks, sliding window, SwiGLU clamping) and you are not sure
whether the target model needs it, do not insert a no-op version or a
default value. Either confirm from the target model's HuggingFace
config/modeling code that the feature is needed and what the correct
parameters are, or leave it out and flag it for review. Inserting a
"reasonable default" (like `clamp_value=0.0` or `sliding_window=None`) can
mask the fact that you skipped something the model actually needs, or
activate a code path the model should not use.

**Verify every weight mapping against the checkpoint.** Do not assume that
the target model's checkpoint key names follow the same convention as
GPT-OSS. Open the checkpoint index (`model.safetensors.index.json`) or run
`safetensors.safe_open` to inspect actual key names and tensor shapes before
writing `load_weights()`. A mapping that looks right but transposes the wrong
dim produces a model that runs and generates fluent-looking garbage.

---

## Code Structure

The model is organized into 8 sections. Each section is either mostly
parallelism (reusable) or mostly model-specific (change when porting).

```text
Section 1: RMSNorm                      [MODEL-SPECIFIC]
Section 2: Rotary Embedding             [MODEL-SPECIFIC]
Section 3: Attention                    [MIXED — TP/SP infra + model-specific features]
Section 4: MoE Experts                  [MIXED — TP/EP infra + model-specific MoE]
Section 5: MLP Wrapper                  [MODEL-SPECIFIC]
Section 6: Decoder Layer                [MODEL-SPECIFIC layout, PARALLELISM dispatch]
Section 7: Model Backbone               [MIXED — SP embedding + model-specific layers]
Section 8: LM Head + Weight Loading     [MIXED — TP lm_head + model-specific mappings]
```

### What to change per section when porting

| Section | Change | Keep |
|---|---|---|
| 1. Norm | Replace RMSNorm with target norm (LayerNorm, etc). Update unpadded-dim logic if no padding. | — |
| 2. Position Embedding | Replace YaRN RoPE with target (standard RoPE, ALiBi, etc). | — |
| 3. Attention | Change GQA config, remove sinks/sliding window, adjust QKV layout. | TP head sharding, SP all-gather/reduce-scatter, KV cache bind, megakernel call, all-reduce. |
| 4. Experts | Change activation (SwiGLU→GeGLU etc), routing (softmax→sigmoid), clamping, expert counts. For dense models: replace entire section with a simple MLP. | TP intermediate sharding, EP expert partitioning, cross-DP dispatch/combine, all-reduce/reduce-scatter logic. |
| 5. MLP | Adjust wrapper (e.g. add shared expert for DeepSeek-style). For dense: this IS the MLP. | — |
| 6. Decoder Layer | Adjust residual connections, norm placement (pre/post), gating. | `is_decode` dispatch pattern. |
| 7. Backbone | Adjust layer stacking, final norm, any model-specific embeddings. | `VocabDimShardedEmbedding` (SP), TP group setup. |
| 8. LM Head | Change `load_weights()` mappings, adjust tied embeddings if needed. | `ColumnParallelLinear`, `load_sharded_pipelined`, EP weight wrapping. |

---

## Parallelism Reference

**Before porting, read the parallelism design docs in full.** They explain
the theory, collectives, sharding math, and failure modes in detail. Do not
skip them — misunderstanding a collective or shard dimension causes silent
accuracy bugs that are extremely hard to trace.

| Mode | Design doc | Model code impact |
|---|---|---|
| **TP** (Tensor Parallelism) | [Tensor Parallelism](../parallelism/tensor_parallelism.md) | Attention heads divided, MoE intermediate sharded, embedding/lm_head vocab-sharded. SP (Sequence Parallelism) is covered here — it adds all-gather before compute and reduce-scatter after during prefill. |
| **DP** (Data Parallelism) | [Data Parallelism](../parallelism/data_parallelism.md) | **No model code changes.** Framework handles routing. Only requirement: use TP-local rank, not global rank. |
| **EP** (Expert Parallelism) | [Expert Parallelism](../parallelism/expert_parallelism.md) | Each rank holds `total_experts / ep_degree` experts with full intermediate. Includes cross-DP EP (token exchange across DP groups). |

### Which parallelisms apply to which architectures

| Architecture | TP | SP | DP | EP | Cross-DP EP |
|---|---|---|---|---|---|
| Dense (e.g. Llama) | Yes | Yes | Yes | No | No |
| MoE (e.g. GPT-OSS, Mixtral) | Yes | Yes | Yes | Yes | Yes |
| MoE with shared expert (e.g. DeepSeek) | Yes | Yes | Yes | Yes | Yes |

For **dense models**, the MoE Experts section (Section 4) is replaced with a
standard MLP. All EP/cross-DP code is removed. Sections 1-3 and 6-8 apply
unchanged.

---

## Weight Loading

Weights are loaded via `SafetensorsCheckpoint.load_sharded_pipelined`. Each
parameter has a `SafetensorsWeightLoader` that transforms checkpoint tensors.

### Weight loader implementations

Generic weight loaders live in `vllm_neuron/utils/weight_loader.py`.
Read the source to understand the full set and their signatures. These handle
standard operations (QKV fusion, sharding, EP filtering) and work for most
models.

GPT-OSS has its own weight loaders in
`vllm_neuron/model/gpt_oss/weight_loaders_bf16.py` — these are
examples of model-specific weight loaders that handle things like MXFP4
dequantization and hidden dim padding. When porting a new model, you may need
to write your own model-specific loaders following this pattern.

The table below shows common examples, but is not exhaustive — check the
generic loader source for the full list:

| Example | Used for | What it does |
|---|---|---|
| `fused_qkv_weight_loader` | Attention QKV | Fuses Q,K,V from separate checkpoints, TP head sharding |
| `sharding_weight_loader` | O-proj, MoE gate_up/down | Shards on a specified dim |
| `expert_parallel_weight_loader` | EP expert weights | Wraps any loader, filters to local expert indices |
| `VocabDimShardedEmbedding` | Embedding | Vocab-sharded with built-in reduce-scatter |
| `ColumnParallelLinear` | LM head | Vocab-sharded column parallel |

### Model-specific weight mappings

The `load_weights()` method maps HF checkpoint keys to model parameter names.
This is the primary thing to change when porting:

```python
# Example mapping structure (MODEL-SPECIFIC):
{
    "qkv_proj_weight": ["hf.q_proj.weight", "hf.k_proj.weight", "hf.v_proj.weight"],
    "o_proj_weight":   "hf.o_proj.weight",
    "gate_up_proj":    ["hf.experts.gate_up_blocks", "hf.experts.gate_up_scales"],
    "down_proj":       ["hf.experts.down_blocks", "hf.experts.down_scales"],
}
```

For models with different checkpoint formats (e.g. no MXFP4), the weight loaders
simplify — no dequantization step needed.

### Lite weight loading for CPU Compilation (`load_weights_lite`)

During CPU Compilation (`VLLM_NEURON_CPU_COMPILE=1`), the model is instantiated
on the `meta` device — no memory is allocated for parameters. Full weight
loading is unnecessary because compilation does not execute the model.

However, some models require compile-time primitive constants that are baked
into the graph from checkpoint tensors (e.g. scaling factors,
or quantization parameters). Without these values, the compiled graph would
use incorrect defaults and produce invalid NEFFs.

To handle this, CPU Compilation calls `load_weights_lite()` instead of
`load_weights()`. This method:

1. Uses **CPU** as the device (not meta) to read only the required tensors
   from the checkpoint.
2. Converts those tensors into primitive constants that get folded into the
   FX graph during tracing.
3. Leaves all other parameters on the meta device — no full checkpoint load
   occurs.

When porting a new model, implement `load_weights_lite()` if the model checkpoint has
any tensors that must be available as compile-time constants baked into the graph. If the model
has no such requirements, `load_weights_lite()` can be a no-op (the base
class default). Common cases that require `load_weights_lite()`:

- FP8 scale tensors used in dequantization ops

---

## Porting Checklist

### Step 1: Config

Create a new `config.py` by reading the target model's HuggingFace
`config.json` (or `configuration_*.py`) and mapping every relevant field
into the vLLM Neuron config class. Do not just copy GPT-OSS's `config.py` and
swap values — the target model may have fields GPT-OSS doesn't have, or
lack fields GPT-OSS does. Use GPT-OSS's config as a structural reference
for how `from_configs()` works, but derive the field list from the target
model's config. Common fields include:

- `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`
- `num_hidden_layers`, `vocab_size`, `max_position_embeddings`
- `intermediate_size` (for MoE: per-expert; for dense: MLP intermediate)
- `num_local_experts`, `num_experts_per_tok` (MoE only; omit for dense)
- Model-specific fields (e.g. `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`)
- Padding logic in `from_configs()` (if hidden_size needs alignment)

### Step 2: Normalization (Section 1)

Check the target model's HF `modeling_*.py` to see which normalization it
uses, then implement that. Do not assume it uses RMSNorm because GPT-OSS
does. Also check for model-specific details like epsilon values, whether
variance is computed on an unpadded portion, or pre-norm vs post-norm
placement.

### Step 3: Position Embedding (Section 2)

Check the target model's HF `modeling_*.py` to see which position embedding
it uses (RoPE, ALiBi, learned, etc.), then implement that. Pay attention to
the specific variant — e.g. standard RoPE vs YaRN RoPE vs NTK-aware RoPE
are all different. The rotation style (interleaved vs non-interleaved) must
match the checkpoint (see Pitfalls section).

### Step 4: Attention (Section 3)

Change model-specific features, keep parallelism infrastructure:

- **Keep**: TP head sharding math, SP all-gather/reduce-scatter, KV cache
  binding, megakernel integration, weight loader setup pattern
- **Change**: GQA config (Q/KV head counts), sliding window, attention sinks,
  RoPE application, QKV layout (fused vs separate), attention biases

### Step 5: MoE / MLP (Sections 4-5)

**For MoE models**: Change activation function, routing strategy, expert counts,
clamping. Keep TP/EP/cross-DP infrastructure.

**For dense models**: Replace `GptOssExperts` with a standard MLP:

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.tp_group = get_tp_group()
        # gate_proj: [hidden, intermediate/TP]  — TP sharded
        # up_proj:   [hidden, intermediate/TP]  — TP sharded
        # down_proj: [intermediate/TP, hidden]  — TP sharded

    def forward_prefill(self, x):
        x = self.tp_group.all_gather(x, dim=0)    # SP → full
        out = self.down_proj(act(self.gate_proj(x)) * self.up_proj(x))
        return self.tp_group.reduce_scatter(out, dim=0)  # full → SP

    def forward_decode(self, x):
        out = self.down_proj(act(self.gate_proj(x)) * self.up_proj(x))
        return self.tp_group.all_reduce(out)
```

Remove all EP code (`_ep_dispatch`, `_ep_combine`, `_cross_dp_ep`, EP init).

### Step 6: Decoder Layer (Section 6)

Adjust residual connections and norm placement. The `is_decode` dispatch pattern
(calling `forward_prefill` vs `forward_decode`) stays the same.

### Step 7: Backbone (Section 7)

Keep `VocabDimShardedEmbedding` and SP logic. Change layer stacking and final
norm to match target model.

### Step 8: LM Head + Weight Loading (Section 8)

- Keep `ColumnParallelLinear` for LM head
- Keep `load_sharded_pipelined` flow
- **Change `load_weights()`**: Update HF checkpoint key → model parameter mappings
- **Change weight loaders**: Adjust for target checkpoint format (e.g. no MXFP4
  dequant for BF16 checkpoints, different QKV fusion layout)

### Step 9: Registry

Add the new model to `model/registry.py` to route HF architecture name to
the new class.

### Step 10: Model README

Add a `README.md` in `vllm_neuron/model/<model_name>/` that documents:

- **Architecture table**: key parameters (hidden_size, head counts, head_dim,
  layers, vocab, RoPE variant, activation, normalization, etc.)
- **Key differences from reference**: what changed vs the canonical model
  you ported from, and why
- **Feature status table**: every parallelism mode and optional feature with
  a ✅ / ❌ / N/A status and a descriptive note explaining the state

The feature status table makes gaps explicit and reviewable. Every
`PARALLELISM` block that was kept, removed, or deferred must be accounted
for.

Template:

```markdown
# <Model Name>

<One-line description.>

## Architecture

| Parameter              | Value              |
|------------------------|--------------------|
| hidden_size            |                    |
| num_attention_heads    |                    |
| num_key_value_heads    |                    |
| head_dim               |                    |
| num_hidden_layers      |                    |
| intermediate_size      |                    |
| vocab_size             |                    |
| RoPE                   |                    |
| Activation             |                    |
| Normalization          |                    |
| tie_word_embeddings    |                    |
| ...                    |                    |
| *(other parameters)*   |                    |

## Key Differences from Reference

- ...

## Feature Status

Reference model: [<reference>](../path/to/model.py)

| Feature               | Status | Notes                        |
|-----------------------|--------|------------------------------|
| TP (head sharding)    |        |                              |
| SP (seq parallel)     |        |                              |
| DP (data parallel)    |        |                              |
| Dependent DP          |        |                              |
| EP                    |        |                              |
| Cross-DP EP           |        |                              |
| Eagle3 spec decode    |        |                              |
| FP8 KV cache          |        |                              |
| Segmented prefill     |        |                              |
| On-device sampling    |        |                              |
| Prompt embeds         |        |                              |
| ...                   |        |                              |
| *(other features)*    |        |                              |

```

### Step 11: Example Run Script and Sanity Check

Add a run script under `examples/vllm_neuron/models/<model_name>/run.py`.
This serves as a quick smoke test and user-facing documentation for how to
launch the model. Follow the pattern in existing scripts (e.g.
`examples/vllm_neuron/models/llama3/run.py`):

- Default `--model-checkpoint` to the HF model ID
- Set `max_model_len`, `tensor_parallel_size`, and bucket configs appropriate
  for the model size
- Include a few diverse prompts (counting, factual, creative, code)
- Use greedy sampling (`temperature=0.0`) for reproducibility

**Sanity check (required before proceeding to tests):** Run the script on a
Trainium instance and inspect the generated output. Verify that:

1. **Prefill is correct** — the first generated token for each prompt makes
   sense (e.g. "6" after "1 2 3 4 5", "Paris" after "capital of France").
2. **Decode is correct** — subsequent tokens are coherent and don't degrade
   into repetition, garbage, or numerically degenerate patterns (e.g.
   "000000...", looping fragments, or random tokens).

If prefill looks right but decode degrades, this typically indicates a
mismatch between the prefill and decode code paths (e.g. a fused kernel
flag enabled without its associated parameters). Share the full output with
the AI assistant — the pattern of degradation reveals the root cause.

### Step 12: Eagle3 Speculative Decoding (conditional)

Check whether a public Eagle3 draft model checkpoint exists for the target
model (e.g. on HuggingFace). The Eagle3 drafter is Llama-architecture-based
regardless of the target model, so no new draft model code is needed — the
existing `Eagle3LlamaForCausalLM` is reused.

**If a draft checkpoint exists:** confirm with the user that the checkpoint
is a good fit (correct base model, compatible tokenizer, expected layer
count) before proceeding. Once confirmed, add target-side Eagle3 support:

1. Inherit `SupportsEagle3` on the `ForCausalLM` class
2. Add `aux_hidden_state_layers` list to the backbone model
3. Collect hidden states at specified layer indices during forward
4. Thread `aux_hidden_states` through all return paths in `ForCausalLM.forward()`
5. Implement `set_aux_hidden_state_layers()` and
   `get_eagle3_aux_hidden_state_layers()` (defaults: layers 2, mid, near-end)
6. Add an `examples/.../run_eagle3.py` script pointing to the draft checkpoint

See GPT-OSS `model_bf16.py` for the reference pattern — it implements
target-side Eagle3 without a model-specific drafter.

**If no draft checkpoint exists:** skip this step and mark Eagle3 as ❌ in
the model README with a note explaining why.

---

## Testing Structure

Tests are organized as a pyramid: unit tests at the base, module-level accuracy
tests in the middle, E2E logit validation above, and accuracy benchmarks at the
top. Each layer catches a different class of bugs.

### Directory layout

Mirror the model directory structure under `test/vllm_neuron/model/`. Using GPT-OSS
BF16 as the reference:

```text
test/vllm_neuron/model/<model_name>/
├── test_factory.py                  # Model registry/factory selection
├── bf16/                            # One directory per quantization variant
│   ├── modules/                     # Module-level accuracy tests
│   │   ├── test_attention.py        # Attention prefill + decode
│   │   ├── test_experts.py          # MoE experts (or test_mlp.py for dense)
│   │   ├── test_rope.py             # Position embeddings
│   │   └── test_weight_loaders.py   # Weight loader unit tests
│   └── e2e/                         # End-to-end full-model tests
│       ├── test_logits.py           # Logit validation (primary correctness gate)
│       ├── test_gsm8k.py            # Accuracy benchmark
│       ├── test_eagle3_target_logits.py  # Speculative decoding logits
│       └── test_gsm8k_eagle3.py     # Accuracy with speculation
```

For dense models (e.g. Llama), replace `test_experts.py` with `test_mlp.py`.
The rest of the structure is identical.

### Test pyramid

```text
                    ┌─────────────┐
                    │  Accuracy   │  GSM8K, benchmarks
                    │  Benchmarks │  (catches quality regressions)
                    ├─────────────┤
                    │  E2E Logit  │  test_logits.py
                    │  Validation │  (catches integration issues)
                    ├─────────────┤
                    │   Module    │  test_attention.py, test_experts.py, ...
                    │   Tests     │  (catches per-component accuracy bugs)
                    ├─────────────┤
                    │    Unit     │  test_weight_loaders.py, test_rope.py
                    │    Tests    │  (catches data transformation bugs)
                    └─────────────┘
```

**During bringup, work bottom-up**: validate each component in isolation
before integrating. Module tests catch accuracy issues that are much harder
to diagnose in a full-model run. However, add a quick **smoke test early** —
confirm that the model loads, compiles, and generates output (even if the
output is garbage) before investing in module-level accuracy. This catches
registry, weight-loading, and dtype plumbing issues up front.

### Level 1: Unit tests — weight loaders and RoPE

These test data transformations in isolation without Neuron hardware or
distributed execution. They run instantly on CPU.

**Weight loader tests** (`test_weight_loaders.py`):

- Use mock checkpoint slices (no real model weights needed)
- Validate sharding, padding, transposition, and dequantization
- Example: verify `fused_qkv_weight_loader` correctly fuses Q/K/V, applies
  TP sharding, and handles `is_storage_transposed`

**RoPE tests** (`test_rope.py`):

- Two-way comparison: vLLM Neuron cos/sin vs HF reference
- Parametrized across sequence lengths (1 to 131072)
- Catches frequency computation or scaling mismatches

```python
# Pattern: two-way comparison against HF
from vllm_neuron.accuracy.testing import assert_close

neuron_cos, neuron_sin = neuron_rope(positions)
hf_cos, hf_sin = hf_rope(dummy_x, position_ids)
assert_close(neuron_cos, hf_cos, rtol=1e-5, name="rope_cos")
```

### Level 2: Module tests — attention and MoE/MLP

Module tests validate individual model components against HuggingFace using
real checkpoint weights. They use `MPExecutor` to simulate distributed
execution across multiple TP/EP ranks on CPU.

**Three-way comparison** is the primary accuracy gate:

1. **FP32 HF** — numerical baseline (gold standard)
2. **BF16 HF** — expected precision floor from dtype alone
3. **BF16 vLLM Neuron** — target implementation on Neuron

`assert_close_three_way` checks that the vLLM Neuron-to-FP32 error is statistically
comparable to the BF16-to-FP32 error. This avoids hard-coding thresholds —
the BF16 HF result defines the acceptable error budget.

```python
from vllm_neuron.accuracy.testing import assert_close_three_way

assert_close_three_way(
    target=neuron_output,       # what we're testing
    expected=hf_fp32_output,  # gold standard
    baseline=hf_bf16_output,  # precision floor
    rtol=0.01,
    name="attn_prefill",
)
```

**Attention tests** (`test_attention.py`):

- Prefill and decode paths tested separately
- Parametrized: sequence lengths (128–4096), TP sizes (2, 16, 64),
  layer indices (0 = sliding window, 1 = no sliding window)
- Uses `SafetensorsCheckpoint` to load real weights into sharded attention
- Golden outputs are cached via `get_or_compute_goldens()` to avoid
  recomputing HF reference on every run

**MoE experts tests** (`test_experts.py`) — or **MLP tests** for dense:

- Same three-way pattern as attention
- Covers TP-only, EP-only, mixed TP+EP, and cross-DP EP configurations
- Uses `_set_mock_vllm_config()` to control EP/DP settings
- Key configs tested: TP8 EP1, TP1 EP16, TP4 EP4, TP2 DP2 EP8

**How `MPExecutor` works**: it spawns separate processes simulating each
distributed rank, with proper `torch.distributed` initialization. You
dispatch inputs, run the module's forward pass on each rank, and collect
outputs:

```python
executor = MPExecutor(world_size=8)
executor.dispatch(hidden_states=hidden_states, position_ids=position_ids)
outputs = executor.collect()  # one output per rank
```

### Level 3: E2E logit validation

The primary correctness gate for model bringup. Tests the full vLLM pipeline:
weight loading, model construction, compilation, and inference.

**Logit tests** (`test_logits.py`):

- Uses `run_logit_test_flow()` from `test/vllm_neuron/utils/logit_test_flow.py`
- Computes HF golden reference, then compares vLLM Neuron logits
- Supports offline (vLLM `LLM` API) and online (vLLM server) modes
- Parametrized across: TP size, batch size, sequence length, on-device
  sampling (ODS), and EP degree

```python
from test.vllm_neuron.utils.logit_test_flow import run_logit_test_flow
from test.vllm_neuron.utils.test_prompts import PROMPTS_2

run_logit_test_flow(
    model_id="openai/gpt-oss-20b",
    prompts=PROMPTS_2,
    vllm_args=_make_args(tp_size=8, seq_len=1024, ods=True),
    output_path=OUTPUT_PATH,
)
```

**Tolerance maps**: logit tests use per-metric tolerance maps rather than
a single threshold, since different top-K slices have different acceptable
error ranges. For example:

```python
tol_map = {
    "k5":   {"max_observed": 0.0253},
    "k50":  {"max_observed": 0.0479},
    "k1000": {"max_observed": 0.0771},
    "all":  {"max_observed": 0.0857},
}
```

**Pytest markers** make it easy to run subsets of the test grid:

```bash
# Run TP=8, seq_len=256 only
pytest test/vllm_neuron/model/gpt_oss/bf16/e2e/test_logits.py -m "tp8 and seq256"

# Run all batch-size-1 online serving tests
pytest test/vllm_neuron/model/gpt_oss/bf16/e2e/test_logits.py -m "bs1 and online_serving"
```

### Level 4: Accuracy benchmarks

End-to-end evaluation on standard benchmarks to validate model quality.
Uses the shared `eval_runners` API to avoid duplicating server lifecycle,
lm_eval invocation, and result parsing logic across tests.

**Building blocks:**

- **`test.evaluation.eval_runners`** — Per-dataset runner functions. Each
  runs lm_eval against a vLLM server and returns `(metrics_dict, results_path)`.
- **`test.utils.simple_server.start_server`** — Starts a vLLM server from a
  shell command string. Auto-assigns a free port, waits for health, returns
  a `ServerHandle` with `base_url`, `model`, and `stop()`.
- **`test.utils.metric_checks.MetricCheck`** / **`evaluate_all`** — Threshold
  checking with tolerance support. Reports all failures, not just the first.

**Available runners** (see `test/evaluation/eval_runners.py`):

| Runner | lm_eval task | Key metrics |
|---|---|---|
| `run_accuracy_gsm8k` | `gsm8k` | `exact_match,flexible-extract`, `exact_match,strict-match` |
| `run_accuracy_gsm8k_cot` | `gsm8k_cot` | `exact_match,flexible-extract`, `exact_match,strict-match` |
| `run_accuracy_gsm8k_cot_llama` | `gsm8k_cot_llama` | `exact_match,flexible-extract`, `exact_match,strict-match` |
| `run_accuracy_bbh` | `bbh_cot_fewshot` | `exact_match,get-answer` |
| `run_accuracy_gpqa` | `gpqa_main_cot_n_shot` | `exact_match,flexible-extract`, `exact_match,strict-match` |
| `run_accuracy_ifeval` | `leaderboard_ifeval` | `prompt_level_strict_acc,none`, `inst_level_strict_acc,none`, ... |
| `run_accuracy_mbpp` | `mbpp` | `pass_at_1,none` |
| `run_accuracy_mmlu_pro` | `mmlu_pro` | `exact_match,custom-extract` |

**Example — GSM8K accuracy test with eval_runners:**

```python
import json
import pytest
from test.evaluation.eval_runners import run_accuracy_gsm8k
from test.utils.fsx_utils.model_path import resolve_model_dir
from test.utils.metric_checks import MetricCheck, evaluate_all
from test.utils.simple_server import start_server

MODEL, _ = resolve_model_dir("meta-llama/Llama-3.2-1B-Instruct")

@pytest.mark.parametrize("tp_size,threshold", [
    pytest.param(8, 0.33, id="tp8"),
])
def test_gsm8k(tp_size, threshold, tmp_path):
    additional_config = json.dumps({
        "neuron_config": {
            "on_device_sampling_config": {"all_greedy": True},
            "num_batched_tokens_buckets": [4096],
            "num_seqs_buckets": [1],
        }
    })
    handle = start_server(f"""
        vllm serve {MODEL}
            --tensor-parallel-size {tp_size}
            --max-model-len 4096
            --max-num-seqs 1
            --no-enable-log-requests
            --additional-config '{additional_config}'
    """)
    try:
        results, _ = run_accuracy_gsm8k(
            base_url=handle.base_url,
            model=MODEL,
            results_dir=_results_dir(tmp_path, "gsm8k"),
            limit=100,
            max_length=4096,
            gen_kwargs=json.dumps({"max_tokens": 2048}),
        )
        evaluate_all(
            [MetricCheck("exact_match,flexible-extract", value=threshold, op=">=")],
            results,
        )
    finally:
        handle.stop()
```

**Example — multi-dataset accuracy suite (see `test_accuracy_eval.py`):**

For models that need multiple benchmarks, define a dataset registry and
parametrize over it. This runs each dataset as a separate sub-test:

```python
from test.evaluation.eval_runners import run_accuracy_gsm8k_cot, run_accuracy_ifeval

DATASETS = {
    "gsm8k": (run_accuracy_gsm8k_cot, {"exact_match,flexible-extract": 0.435}),
    "ifeval": (run_accuracy_ifeval, {"prompt_level_strict_acc,none": 0.400}),
}

@pytest.fixture()
def server():
    handle = start_server(f"vllm serve {MODEL} --tensor-parallel-size 8")
    yield handle
    handle.stop()

@pytest.mark.parametrize("dataset", DATASETS)
def test_accuracy(server, dataset, tmp_path):
    runner_fn, thresholds = DATASETS[dataset]
    results, _ = runner_fn(
        base_url=server.base_url, model=MODEL,
        results_dir=_results_dir(tmp_path, dataset), limit=200,
    )
    evaluate_all(
        [MetricCheck(k, value=v, op=">=") for k, v in thresholds.items()],
        results,
    )
```

#### Setting accuracy thresholds

Thresholds serve as regression gates. They must be derived from a GPU
baseline, not from a single Neuron run.

**Step 1 — Establish a GPU baseline.** Run the benchmark on a GPU instance
(e.g. p4d.24xlarge with vLLM) at the same `limit` you plan to use. Record
the mean score and standard error across multiple runs.

**Step 2 — Compute the Neuron threshold.** Apply a margin below the GPU
mean to absorb run-to-run variance:

```text
threshold = gpu_mean - k * stderr
```

Where:

- `gpu_mean` — mean score from GPU runs
- `k` — number of standard errors (typically 1)
- `stderr` — standard error from GPU runs

**Example** (Llama-3.2-1B, GSM8K, limit=200):

```text
gpu_mean   = 0.470   (L40S golden, n=200)
stderr     = 0.035
threshold  = 0.470 - 1 * 0.035 = 0.435
```

We typically use a sampled set (e.g. n=200) for both GPU and Neuron runs
to keep evaluation fast while providing enough statistical power for the
threshold formula.

**FP8 configs** need wider margins (increase `k` or add ~1-2% extra) due
to additional quantization noise.

**Document your thresholds.** Add a comment block at the top of the test
file showing the GPU reference values, the formula, and the resulting
thresholds. This makes it possible to recalibrate when the baseline changes.

```python
# GPU Reference (p4d.24xlarge, vLLM 0.19.0, lm_eval 0.4.11, limit=200):
#   Dataset     | Metric                       | Mean  | Stderr | Threshold
#   gsm8k_cot   | exact_match,flexible-extract | 0.470 | 0.035  | 0.435
#   gsm8k_cot   | exact_match,strict-match     | 0.430 | 0.035  | 0.395
```

### What to write when porting a new model

The minimum set of tests for a new model bringup, in suggested order:

| Priority | Test | Why |
|---|---|---|
| 0 | Smoke test (generate any output) | Confirm registry, weight loading, and compilation work before investing in accuracy. Can be as simple as `vllm.LLM(model).generate("Hello")`. |
| 1 | `modules/test_weight_loaders.py` | Validates weight transformations with mock data. Fast to write and run. |
| 2 | `modules/test_rope.py` | Two-way comparison, no distributed setup needed. Catches RoPE variant bugs early. |
| 3 | `modules/test_attention.py` | Three-way comparison with real weights. Isolates attention accuracy issues. |
| 4 | `modules/test_experts.py` or `test_mlp.py` | Three-way comparison for MoE/MLP. Covers TP/EP sharding correctness. |
| 5 | `e2e/test_logits.py` | Full-pipeline logit validation. Run once modules pass. |
| 6 | `e2e/test_gsm8k.py` | Quality gate. Add once logit tests pass. |

### Always validate on CPU first, then move to device

**CPU mode is the primary development loop.** All module tests and E2E logit
tests should pass on CPU before you attempt an on-device run. CPU mode is
fast (seconds vs minutes), does not require Neuron hardware, and catches the
majority of bugs: wrong weight mappings, shape mismatches, incorrect
collectives, bad RoPE variants, and transposition errors. If a test fails on
CPU, it will fail on device — but debugging on CPU is dramatically faster.

CPU mode has two sub-modes:

1. **CPU mode (default, simulator off)**: Fast. NKI kernels use PyTorch
   fallback paths. Catches bugs in weight mappings, shapes, collectives, etc.
   Use this for general development iteration.

2. **CPU mode + NKI simulator (`NKI_SIMULATOR=1`)**: Slower. Runs NKI
   kernels through the CPU simulator for numerical accuracy validation.
   Best for single functions, modules, or layers with small shapes or tiny
   model configs (<10M params). Use this for initial kernel integration and
   accuracy debugging for minimal reproducing examples. Use timeouts
   (e.g. `--timeout 60`) to avoid long-running processes.

**Only move to on-device testing once CPU tests pass.** On-device runs add
compilation time and hardware-specific kernel behavior. You want to be
confident the model logic is correct before introducing those variables.
On-device testing catches a narrower class of issues: kernel precision
differences, NKI-specific numerics, and hardware memory constraints.

To run in CPU mode, set `VLLM_NEURON_CPU_MODE=1`. Pass
`enforce_eager=True` in the vLLM config to skip `torch.compile`. NKI kernels
use CPU fallbacks automatically.

```bash
# CPU mode (fast, no simulator — default for development)
VLLM_NEURON_CPU_MODE=1 pytest test/vllm_neuron/model/gpt_oss/bf16/modules/ -v --timeout=60

# CPU mode + NKI simulator (for kernel accuracy validation, small shapes only)
VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 pytest test/vllm_neuron/model/gpt_oss/bf16/modules/ -v --timeout=60

# Once CPU passes, run on device
pytest test/vllm_neuron/model/gpt_oss/bf16/modules/ -v
```

### Other tips

**Use a small model for bringup.** Llama-3.2-1B (1.2GB) loads in seconds.
GPT-OSS-20B needs a 2-layer checkpoint extract. Smaller models make the
debug cycle much faster.

### Credential-free `_fast` tests for `make test`

Module tests require checkpoints (HF token / S3). To get model coverage in
`make test` without credentials, add `_fast` variants marked with
`@pytest.mark.fast` that use synthetic weights instead.

Pattern: each fast test is self-contained in the model's existing test file.
Build the HF module with seeded fan-in-scaled weights, compute fp32 and bf16
HF goldens in-process, load weights into the vLLM module via `FakeSafeSlice`
and weight loaders, run via `MPExecutor`, then `assert_close_three_way`.

See these files for complete examples:

- Attention: `test/vllm_neuron/model/gpt_oss/bf16/modules/test_attention.py`
  (search for `test_attention_bf16_bs1_tp_prefill_fast`)
- MoE experts: `test/vllm_neuron/model/gpt_oss/bf16/modules/test_experts.py`

Key utilities in `test/vllm_neuron/model/utils.py`:

- `FakeSafeSlice(tensor)` -- stand-in for safetensors slice
- `hf_state_to_fake_slices(state_dict, layer_idx)` -- wraps HF state dict
- `load_weights_from_slices(module, slice_map, mappings, rank, device)` --
  drives weight loaders on synthetic data

---

## Pitfalls & Lessons Learned

These are common mistakes encountered when porting models. Read these before
starting — they will save significant debugging time.

### Weight Loaders

**There are two families of weight loaders — use the right one.**

- **Generic loaders** (`vllm_neuron/utils/weight_loader.py`):
  `fused_qkv_weight_loader`, `sharding_weight_loader`, `sharding_weight_loader_with_padding`.
  These work for standard BF16 checkpoints. Use for most models.

- **GPT-OSS loaders** (`vllm_neuron/model/gpt_oss/weight_loaders_bf16.py`):
  Extended versions with MXFP4 dequantization, hidden dim padding, and different
  parameter signatures (e.g. `num_kv_heads`, `head_dim`, `hidden_size`). Only
  use for models that need these features.

Mixing them up causes `TypeError: got an unexpected keyword argument`. Check the
function signature before calling.

**HuggingFace stores weights transposed.** For linear layers loaded from HF
checkpoints, always pass `is_storage_transposed=True` to the weight loader.
Without this, the sharding dimension is wrong and you get silently incorrect
weights.

**`ColumnParallelLinear` sets its own weight loader.** Don't set a separate
weight loader on `lm_head.weight` unless you need to override the default
(e.g. for hidden dim padding). The default shards on dim 0 which is correct
for LM heads.

### Tied Embeddings (`tie_word_embeddings=True`)

Models like Llama-3.2-1B share the embedding weight with the LM head. This
requires careful handling:

1. **The checkpoint has no `lm_head.weight` key.** You must add an explicit
   mapping: `mappings["lm_head.weight"] = "model.embed_tokens.weight"`.
   Otherwise `load_sharded_pipelined` fails with `KeyError`.

2. **Don't tie before loading.** vLLM creates models on `meta` device
   (`with torch.device("meta")`). If you do `self.lm_head.weight = self.model.embed_tokens.weight`
   before `load_state_dict(assign=True)`, `assign=True` replaces the parameter
   in one module's `_parameters` dict but not the other's. The lm_head ends up
   pointing at a stale meta tensor. Result: `Cannot copy out of meta tensor`
   when the model is moved to device.

3. **Solution**: Map both to the same checkpoint key. `load_sharded_pipelined`
   loads `model.embed_tokens.weight` and `lm_head.weight` independently from
   the same checkpoint tensor, each through their own weight loader. Both end
   up with real (non-meta) data. Optionally tie after loading for memory savings.

### Model Creation on Meta Device

vLLM creates models with `with torch.device("meta")` in `neuron_model_runner.py`.
All `nn.Parameter` tensors are meta tensors (no data). `load_weights` is called
to populate them, then `.to(device)` moves to Neuron.

**Implications:**

- Don't access parameter data in `__init__` (it's meta).
- `load_state_dict(assign=True)` replaces parameter tensors in `_parameters`
  dict. This breaks cross-module references (like weight tying).
- Any parameter not loaded remains meta and causes errors on `.to(device)`.

### RoPE Variants

**The RoPE rotation style must match the checkpoint.** Two common styles:

**Interleaved (rotate_half)**: Used by Llama, Mistral. Splits into first/second
half, rotates as `(-x2, x1)`.

```python
x1, x2 = x[..., :half], x[..., half:]
return torch.cat((-x2, x1), dim=-1) * cos + x * sin  # wrong, simplified
```

**Non-interleaved (split in half)**: Used by GPT-OSS. Applies cos/sin to each
half independently.

```python
first = first_half * cos - second_half * sin
second = second_half * cos + first_half * sin
```

Using the wrong style produces garbage attention outputs that are hard to debug
because the model still runs without errors — just with wrong results.

### Bias Handling

**Pass `None` for bias when the model has no bias.** GPT-OSS has attention bias;
Llama does not. The NF kernel functions (`NF.qkv_proj`, `NF.o_proj`) accept
`None` for bias. Passing a zero tensor is NOT the same — it may trigger
different kernel paths.

### Dense MLP vs MoE — SP collectives are in different places

Both MoE and dense MLP need SP collectives (all-gather before compute,
reduce-scatter after during prefill). The difference is where they live:

- In **GPT-OSS (MoE)**, the experts module (`GptOssExperts.forward_prefill`)
  does the SP all-gather/reduce-scatter internally.
- In **Llama (dense MLP)**, the MLP module itself must do them explicitly.

See `vllm_neuron/model/llama3/model.py` `LlamaMLP.forward()`
for the reference pattern. The MLP must know whether it's prefill (SP active)
or decode (no SP, use all-reduce instead) — pass `is_prefill` from the
decoder layer.

### EP Detection

**Use `vllm_config.parallel_config.enable_expert_parallel`, NOT
`ep_group.world_size`.** The EP group is always initialized by vLLM even when
EP is not enabled. Checking `ep_group.world_size > 1` gives false positives
when DP is enabled (DP creates groups that look like EP groups).

---

## Debugging Accuracy Issues

When a three-way or logit comparison test fails, use these techniques to
isolate the root cause. Work from coarse to fine.

### Step 1: Confirm E2E correctness first

Run the E2E logit test (`test_logits.py`) before module tests. If E2E passes
(cos > 0.999) but module tests fail, the issue is likely in the test harness,
not the model. The E2E test uses the full vLLM pipeline with real weight
loading and is the ground truth.

### Step 2: Isolate TP vs kernel precision

Run the same computation with and without TP to separate parallelism errors
from kernel errors:

```python
# No TP: run NF kernels in a single process (no all-gather/reduce-scatter)
nf_output_no_tp = full_nf_pipeline(hidden_states, weights)

# TP=2: simulate two ranks with split heads and reduce-scatter
nf_output_tp2 = simulate_tp2(hidden_states, weights)

# Compare
print(f"cos(no_tp, tp2) = {cosine_similarity(nf_output_no_tp, nf_output_tp2)}")
```

If no-TP and TP outputs are nearly identical (cos > 0.9999), the issue is NOT
from TP. If they differ significantly, the TP collective logic has a bug.

### Step 3: Per-stage decomposition

Compare each stage independently against HF:

```text
QKV proj  → compare NF.qkv_proj output vs HF q/k/v_proj output
RoPE      → compare canonical apply_rotary_pos_emb vs HF apply_rotary_pos_emb
Attention → compare NF.flash_attention vs HF eager attention
O proj    → compare NF.o_proj vs HF o_proj
```

Use σ-ratio at each stage. If stage N has σ=1.0 but stage N+1 jumps, that
stage introduces the error. Be careful to feed the **same inputs** to both
paths at each stage — don't let errors cascade across stages.

### Step 4: Check weight loading

If outputs are completely wrong (cos < 0.5 or all zeros), the issue is
usually in weight loading:

```python
# After loading, compare a weight tensor against HF
from safetensors import safe_open
with safe_open(checkpoint_path, framework="pt") as f:
    hf_weight = f.get_tensor("model.layers.0.self_attn.q_proj.weight")

# Compare with loaded canonical weight (accounting for TP sharding)
canonical_weight = model.model.layers[0].self_attn.qkv_proj_weight
# The canonical weight is fused [H, q+k+v per rank] and transposed
# Check that the Q portion matches the expected shard of hf_weight
```

Common weight loading bugs:

- **Wrong shard dimension**: `shard_dim=0` vs `shard_dim=1`
- **Missing `is_storage_transposed=True`**: HF stores `[out, in]`, NF expects `[in, out]`
- **Wrong shard size**: Forgetting to divide by TP world_size
- **Tied weights not mapped**: `lm_head.weight` missing from mappings when `tie_word_embeddings=True`

### Step 5: Check shapes at each stage

Many silent accuracy issues come from shape mismatches that broadcast
incorrectly instead of erroring:

```python
# Add shape assertions at key points
assert q.shape == (num_heads_per_rank, seq_len, head_dim), f"Bad q shape: {q.shape}"
assert cos.shape == (seq_len, head_dim), f"Bad cos shape: {cos.shape}"
```

Common shape bugs:

- RoPE cos/sin with wrong dimensions (missing/extra batch dim)
- GQA repeat on wrong dimension
- Flash attention input in wrong layout (tp_q/tp_k/tp_out flags)

---

## GPT-OSS Model Architecture (for reference)

GPT-OSS is a Transformer decoder with Mixture-of-Experts (MoE):

| Parameter | GPT-OSS-20B |
|---|---|
| hidden_size | 2880 (padded to 3072) |
| num_attention_heads (Q) | 64 |
| num_key_value_heads (KV) | 8 |
| head_dim | 64 |
| num_hidden_layers | 24 |
| num_local_experts | 32 |
| experts_per_token (top-k) | 4 |
| intermediate_size | 2880 (padded to 3072) |
| vocab_size | 201088 |
| sliding_window | 128 (even layers only) |
| RoPE | YaRN with theta=150000 |
| Activation | SwiGLU with clamping |

Model-specific features:

- **GQA**: Separate Q (64) and KV (8) head counts
- **Learnable attention sinks**: Per-head logit bias added to attention scores
- **Sliding window**: Applied to even-indexed layers via `layer_types` config
- **SwiGLU with clamping**: Gate clamped to [-inf, 7.0], up clamped to [-6.0, 8.0]
- **Pre-attention and pre-MLP RMSNorm**: Variance computed on unpadded portion
- **Hidden dimension padding**: 2880 → 3072 for hardware alignment
- **MXFP4 checkpoint**: Expert weights stored as packed blocks + scales

---

## Adding Tiny E2E Build-Time Tests

Every new model should include a tiny E2E test that validates the full vLLM pipeline
(model load → weight loading → forward pass → NKI kernel dispatch → sampling → output)
runs correctly on CPU without requiring Neuron hardware or real model weights.

### How it works

1. Create a minimal HuggingFace config with random weights via `save_pretrained()`
2. Load through `vllm.LLM()` — exercises the real weight loading, model construction, and inference path
3. Run in CPU mode with `VLLM_NEURON_CPU_MODE=1`

### Template

Create `test/vllm_neuron/model/<model_name>/tiny/test_tiny_<model>_e2e.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import tempfile

import pytest
import torch
from transformers import <ModelConfig>, <ModelForCausalLM>
from vllm import LLM, SamplingParams

pytestmark = [pytest.mark.fast, pytest.mark.forked]

TINY_CONFIG = <ModelConfig>(
    vocab_size=256,
    hidden_size=<min_valid>,       # Must satisfy H % 256 == 0 for NKI kernels
    intermediate_size=<min_valid>,  # Must be >= 512 for MoE tiling
    num_hidden_layers=1,
    num_attention_heads=<N>,       # head_dim = hidden_size / num_attention_heads <= 128
    num_key_value_heads=2,         # 2 so TP=2 gives 1/rank (decode megakernel requirement)
    max_position_embeddings=128,
    tie_word_embeddings=False,
    # ... model-specific params (num_experts, etc.)
)


def _run_inference(tp_size):
    model_dir = tempfile.mkdtemp()
    torch.manual_seed(42)
    <ModelForCausalLM>(TINY_CONFIG).to(torch.bfloat16).save_pretrained(model_dir)
    llm = LLM(
        model=model_dir, max_num_seqs=1, max_model_len=128, block_size=128,
        tensor_parallel_size=tp_size, enforce_eager=True, enable_prefix_caching=False,
        skip_tokenizer_init=True, num_gpu_blocks_override=4,
        additional_config={"neuron_config": {"num_batched_tokens_buckets": [16, 128]}},
    )
    outputs = llm.generate(
        [{"prompt_token_ids": list(range(1, 11))}],
        SamplingParams(temperature=0.0, max_tokens=3),
    )
    assert len(outputs) == 1 and len(outputs[0].outputs[0].token_ids) > 0


def test_tp2():
    """TP=2: distributed weight loading, multi-worker NKI kernel dispatch."""
    _run_inference(tp_size=2)


def test_tp1():
    """TP=1: single-worker full pipeline."""
    _run_inference(tp_size=1)
```

### Key constraints for the config

Choose config dimensions large enough to trigger NKI kernels in the CPU
simulator, but as small as possible for fast execution. Use the existing tests
(llama3/tiny, gpt_oss/tiny) as reference for working configs.

### Important notes

- **`pytest.mark.forked`** is required — vLLM's `LLM()` can only be created once per process
- **MXFP checkpoints** — for models whose real checkpoint uses MXFP4 (e.g., GPT-OSS), the weight loader auto-detects dense bf16 from `save_pretrained()` vs MXFP4 from real checkpoints. Add `"quantization": "bf16"` to neuron_config.

### Checklist for new models

- [ ] Create `test/vllm_neuron/model/<name>/tiny/__init__.py`
- [ ] Create `test/vllm_neuron/model/<name>/tiny/test_tiny_<name>_e2e.py`
- [ ] Find minimum config that satisfies NKI kernel constraints
- [ ] Verify `make test` passes (both sim-off and sim-on model passes)
- [ ] If weight loader expects non-standard format, add dense bf16 support
