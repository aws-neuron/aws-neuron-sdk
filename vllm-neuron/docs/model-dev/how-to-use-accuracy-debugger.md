# Accuracy debugger tools

<!-- meta: description: Step-by-step guide for running the accuracy
debugger tool including task analysis, prompt analysis, tensor capture,
tensor comparison, and interpreting the HTML report. -->
<!-- meta: date_updated: 2026-06-09 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: NDOC-194 -->

## Task overview

This topic discusses how to run the accuracy debugger pipeline using
vLLM Neuron. The accuracy debugger automates the process of
identifying where model outputs diverge from a HuggingFace reference
by running task evaluation, prompt-level logit validation, KV cache
comparison, and tensor comparison.

## Prerequisites

- **Working vLLM Neuron environment:** See the
  [setup guide](../getting-started/setup-guide.md).
- **Model checkpoint:** Access to the model weights on local disk or
  HuggingFace Hub.
- **Sufficient memory:** The debugger loads HF models (FP32 + BF16)
  on CPU alongside the Neuron model. Ensure enough system RAM for
  your model size.
- **Python dependencies:** `torch`, `transformers`, `vllm`,
  `plotly` (for visualization).

## Instructions

### 1. Pipeline overview

The debugger runs in stages, each usable on its own:

1. **Task analysis** — run a dataset eval against a running vLLM server, then
   score it against thresholds and extract the prompts the model got wrong
   (Section 2).
2. **Prompt analysis** — for those deviated prompts, run logit validation, KV
   cache comparison, and tensor compare (Section 3).
3. **Report** — combine the artifacts into an HTML report (Section 4).

The debugger only *analyzes* results — you run the eval and manage the vLLM
server, then pass the results in. This keeps it decoupled from any particular
eval runner or server harness.

> **Runnable examples.** End-to-end scripts that wire the whole pipeline
> together (launch a server, run each stage, generate the report) live under
> [`examples/vllm_neuron/accuracy/`](https://github.com/vllm-project/vllm-neuron/blob/HEAD/examples/vllm_neuron/accuracy)
> — see `run_accuracy_debugger_llama.py` and `run_accuracy_debugger_gpt_oss.py`.
> The sections below show how to drive the same stages directly.

### 2. Run task analysis programmatically

`run_task_analysis()` does not run the eval itself — it analyzes
pre-computed eval results. Run your eval harness (for example, the
`lm_eval` runners shipped in `vllm_neuron.accuracy.lm_eval`) against a
running server first, then pass the results in via `input_task_results`:

```python
from vllm_neuron.accuracy.lm_eval import run_accuracy_gsm8k_cot
from vllm_neuron.accuracy.accuracy_debugger import run_task_analysis
from vllm_neuron.accuracy.accuracy_debugger.task_plugins.lm_eval_analyzer import LmEvalAnalyzer

# Run the eval yourself against a server you started
# (for example, `vllm serve /path/to/model --tensor-parallel-size 8`):
_scores, results_dir = run_accuracy_gsm8k_cot(
    base_url="http://localhost:8000",
    model="/path/to/model",
    results_dir="./eval_out",
    limit=200,  # cap for a quick run; omit for the full dataset
)

result = run_task_analysis(
    LmEvalAnalyzer(),
    input_task_results=results_dir,
    thresholds={"exact_match,flexible-extract": 0.435},
    output_dir="./accuracy_report",
)

print(f"Passed: {result.passed}")
print(f"Scores: {result.scores}")
print(f"Deviated prompts: {len(result.deviated_prompts)}")
```

**Parameters:**

- `analyzer` -- An analyzer instance with an `analyze_all_results`
  method (for example, `LmEvalAnalyzer()`)
- `input_task_results` -- Pre-computed eval results: either a path to
  an existing results directory (for example, an lm_eval
  `--output_path`) or a results dict. The analyzer interprets it (the
  API stays eval-runner agnostic)
- `thresholds` -- Dict mapping metric names to minimum acceptable
  scores (all lower-bound `>=` checks)
- `output_dir` -- Directory where reports and artifacts are saved

Keeping the eval outside the debugger means it stays decoupled from any
particular eval runner — it only analyzes the results and judges
pass/fail against `thresholds`.

### 3. Run prompt analysis with logit validation and KV cache

Use `run_prompt_analysis()` to perform per-prompt logit validation
and KV cache comparison:

```python
from vllm_neuron.accuracy.accuracy_debugger import run_prompt_analysis
from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins import (
    LogitValPlugin, KvCachePlugin,
)

cfg = {
    "server": {
        "model": "/path/to/model/checkpoint",
        "tp_degree": 8,
        "max_model_len": 256,
    }
}

result = run_prompt_analysis(
    server_config=cfg,
    prompts=task_result.deviated_prompts[:3],
    plugin_steps=[LogitValPlugin(), KvCachePlugin()],
    output_dir="./accuracy_report",
)

for plugin_name, plugin_result in result.plugin_results.items():
    passed = all(v.get("passed") for v in plugin_result.values() if isinstance(v, dict))
    print(f"{plugin_name}: {'PASS' if passed else 'FAIL'}")
```

:::{note}
Set `max_model_len` in `server_config` to control compiled model
length. Smaller values result in faster compilation but limit the
maximum sequence length that can be analyzed.
:::

### 4. Configure tensor capture

To capture intermediate tensors from the Neuron model, configure
`tensor_capture` in `additional_config`:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    additional_config={
        "neuron_config": {
            "tensor_capture": {
                "modules": [
                    "model.layers.0-31",
                    "lm_head",
                ],
                "capture_dir": "/tmp/captures",
            }
        }
    },
)
```

**Module pattern syntax:**

- **Range patterns**: `model.layers.0-31` expands to match layers 0
  through 31
- **Regex patterns**: `model\.layers\.\d+\.self_attn` matches all
  self-attention modules
- **Exact names**: `lm_head` matches the language model head

**Output location:** Captures are saved to the specified
`capture_dir` organized by data-parallel rank, phase (prefill or
decode), and module name:

```text
/tmp/captures/
+-- dp0/
    +-- prefill_s128_0/
    |   +-- model.layers.0/
    |   |   +-- rank0.pt
    |   +-- prefill_s128_0_meta.json
    +-- decode_b1_0/
        +-- model.layers.0/
        |   +-- rank0.pt
        +-- decode_b1_0_meta.json
```

### 5. Run tensor comparison

Use the `TensorComparePlugin` for three-way intermediate tensor
comparison. This must run in a **separate** `run_prompt_analysis()`
call from LogitVal/KvCache plugins because it manages its own Neuron
LLM:

```python
from vllm_neuron.accuracy.accuracy_debugger import run_prompt_analysis
from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins import (
    TensorComparePlugin,
)

cfg = {
    "server": {
        "model": "/path/to/model/checkpoint",
        "tp_degree": 8,
        "max_model_len": 256,
    }
}

tc_result = run_prompt_analysis(
    server_config=cfg,
    prompts=task_result.deviated_prompts[:3],
    plugin_steps=[TensorComparePlugin(
        modules=["model.embed_tokens", "model.layers.0-31", "lm_head"],
        tp_size=8,
    )],
    output_dir="./accuracy_report",
    output_length=3,
)

tc_plugin_res = tc_result.plugin_results["tensor_compare"]
passed = all(v.get("passed") for v in tc_plugin_res.values() if isinstance(v, dict))
print(f"Tensor compare: {'PASS' if passed else 'FAIL'}")
```

**Configurable threshold:** By default, the plugin fails if any
module has L2 ratio >= 3.0. Adjust with:

```python
TensorComparePlugin(
    modules=["model.embed_tokens", "model.layers.0-31", "lm_head"],
    tp_size=8,
    max_l2_ratio=5.0,
)
```

**Standalone tensor comparison** (without the debugger pipeline):

```python
from vllm_neuron.accuracy.tensor_io import read as tensor_io_read
from vllm_neuron.accuracy import (
    align_decode_captures,
    compare_captures_three_way,
    print_three_way_report,
)
from vllm_neuron.accuracy.tensor_alignment_utils import (
    align_and_truncate_hidden,
    hf_reference_reconstruction,
)

# Read captures from disk
fp32 = tensor_io_read("/tmp/captures/hf_fp32")
bf16 = tensor_io_read("/tmp/captures/hf_bf16")
neuron = tensor_io_read("/tmp/captures/neuron")

# Align decode steps by position
fp32 = align_decode_captures(fp32, neuron)
bf16 = align_decode_captures(bf16, neuron)

# Three-way comparison
results = compare_captures_three_way(
    fp32, bf16, neuron,
    reference_reconstruction_fn=hf_reference_reconstruction,
    target_reconstruction_fn=my_reconstruct,
    alignment_fn=align_and_truncate_hidden,
)
print_three_way_report(results)
```

### 6. Generate the HTML report

After running task and/or prompt analysis, generate the combined
interactive report:

```python
from vllm_neuron.accuracy.accuracy_debugger import generate_report

report_path = generate_report("./accuracy_report")
print(f"Report: {report_path}")
# Output: ./accuracy_report/combined_report.html
```

Open the report in your browser:

```bash
open ./accuracy_report/combined_report.html
```

### 7. Interpret the HTML report

The report contains three tabs:

**Overview Tab:**

- Status banner: green (all passed), yellow (minor deviations),
  red (failures)
- Summary cards for each analyzed prompt with pass/fail status
- Triage flowchart for navigating failures

**Task Analysis Tab:**

- Metrics Summary table: score vs threshold with pass/fail indicators
- Deviating samples (expanded): question, expected answer, actual
  response
- Matching samples (collapsed): correctly answered samples
- Reproduce section: copy-paste commands to re-run the analysis

**Prompt Analysis Tab:**

- Sidebar listing analyzed prompts with pass/fail dots
- Logit Validation: per-token L-inf/L2 error, BC, divergent token
  markers
- KV Cache: per-layer heatmaps with L-inf error and BC per attention
  head

**How to read logit validation results:**

| Indicator                          | Meaning                              | Action                                          |
| ---------------------------------- | ------------------------------------ | ----------------------------------------------- |
| Three-way ratio approximately 1.0x | Dtype-inherent BF16 noise, not a bug | No action needed                                |
| Three-way ratio >> 1.0x            | vllm-neuron-specific excess error    | Investigate with tensor compare                 |
| BC >= 0.99                         | Error distributions nearly identical | Model is accurate                               |
| Token 0 fails                      | Prefill bug                          | Check tensor compare for first divergent module |
| Tokens 1+ fail, token 0 passes     | Decode or KV cache bug               | Check KV cache heatmaps                         |

**Pass/fail thresholds (three-way):**

| Check                           | Threshold |
| ------------------------------- | --------- |
| sigma-ratio                     | <= 1.0    |
| Aggregate BC                    | >= 0.99   |
| L-inf ratio (max across tokens) | < 1.5x    |
| L2 ratio (max across tokens)    | < 1.5x    |

**Pass/fail thresholds (two-way static fallback):**

| Top-k | Threshold |
| ----- | --------- |
| K5    | < 0.011   |
| K50   | < 0.02    |
| K1000 | < 0.03    |
| All   | < 0.05    |

The overall verdict passes if any of: sigma-ratio <= 1.0, aggregate
BC >= 0.99, or all two-way thresholds are met.

### 8. Use golden caching to speed up repeated runs

The golden caching system avoids recomputing expensive HF reference
logits across test runs:

```python
from vllm_neuron.accuracy.goldens import get_cached_reference_goldens

goldens = get_cached_reference_goldens(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    model_checkpoint=checkpoint_path,
    model_config=config,
    tokenizer=tokenizer,
    prompts=["The capital of France is", ...],
    output_length=16,
)

# goldens["fp32_logits"]  - FP32 baseline
# goldens["dtype_logits"] - BF16 expected (teacher-forced)
# goldens["input_ids"]    - Tokenized prompts
```

Cache location defaults to `~/.cache/vllm/neuron/goldens`. Override
with the `$VLLM_NEURON_GOLDEN_CACHE_DIR` environment variable.

## Confirm your work

The accuracy debugger confirms your model is accurate when:

- The HTML report shows green status banner (all checks passed)
- Task analysis scores meet or exceed defined thresholds
- Logit validation shows three-way ratios approximately 1.0x and
  BC >= 0.99 for all prompts
- Tensor compare shows L2 ratios < 3.0 for all modules

To verify programmatically:

```python
# Task analysis
assert result.passed, f"Task failed: {result.scores}"

# Prompt analysis
for name, prompt_results in result.plugin_results.items():
    assert all(v.get("passed") for v in prompt_results.values() if isinstance(v, dict)), \
        f"Plugin {name} failed"
```

## Common issues

### HF reference model runs out of memory

- **Possible solution**: The HuggingFace reference runs on CPU. For
  large models (70B+), ensure sufficient system RAM (at least 3x
  model size for FP32 + BF16 + overhead). Alternatively, reduce
  `max_model_len` to limit sequence length or analyze fewer prompts.

### Tensor compare plugin fails with "cannot occupy Neuron cores"

- **Possible solution**: `TensorComparePlugin` must run in a separate
  `run_prompt_analysis()` call from `LogitValPlugin` and
  `KvCachePlugin`. The shared LLM and the tensor compare LLM cannot
  coexist on Neuron cores simultaneously.

### Report shows failures but three-way ratio is close to 1.0x

- **Possible solution**: Ratios between 1.0x and 2.0x are borderline.
  Check if `max_model_len` matches the calibration target used by
  other logit tests. Different compilation targets can cause minor
  numerical differences that are not bugs.

### Logit validation fails at token 0 only

- **Possible solution**: This indicates a prefill bug. Run tensor
  compare to identify which module first introduces excess error
  during prefill. Check embedding weights, input processing, and the
  first attention layer.

### Tensor capture causes torch.compile recompile

- **Possible solution**: Avoid wildcard module patterns like `[".*"]`.
  Use explicit module patterns instead:

```python
# Works correctly:
modules = [
    "model.layers.0-15.input_layernorm",
    "model.layers.0-15.self_attn",
    "lm_head",
]

# Causes recompile guard (avoid):
modules = [".*"]
```

### Golden cache returns stale results after model update

- **Possible solution**: The cache key includes `model_id` and
  `prompts` but not the checkpoint hash. Delete the cache directory
  to force recomputation:

```bash
rm -rf ~/.cache/vllm/neuron/goldens
```

Or set a different cache directory:

```bash
export VLLM_NEURON_GOLDEN_CACHE_DIR=/tmp/fresh_goldens
```

## Related information

- [Accuracy debugging guide](accuracy-debugging-guide.md) -- Deep
  dive into the framework architecture, three-way comparison
  methodology, and threshold tables
- For supported models and features, see the [README](https://github.com/vllm-project/vllm-neuron#supported-models)
  and [model cards](../model-recipes/index.md).
