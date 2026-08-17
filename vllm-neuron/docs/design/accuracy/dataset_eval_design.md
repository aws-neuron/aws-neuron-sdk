# Dataset Evaluation Design

<!-- meta: description: Per-dataset lm_eval accuracy runner design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-07-24 -->

## Overview

Dataset evaluation is **task-level (Level 1)** accuracy validation: run a model
against a standard benchmark (GSM8K, MMLU-Pro, IFEval, …) and compare its score
against a threshold. It is the coarsest, cheapest signal — "is the model
answering questions correctly?" — and the entry point of the accuracy debugger's
pipeline (a task-level failure is what triggers the finer prompt- and
module-level analysis; see [Accuracy debugger design](accuracy_debugging_design.md)).

The runners live in the shipped wheel at `vllm_neuron/accuracy/lm_eval.py`. Each
is a thin, dependency-light wrapper around the [`lm-eval`](https://github.com/EleutherAI/lm-evaluation-harness)
CLI: it invokes `lm_eval` as a subprocess against a running vLLM
(OpenAI-compatible) server, parses the results JSON, and returns a flat metrics
dict the caller can assert on.

```python
from vllm_neuron.accuracy.lm_eval import run_accuracy_gsm8k_cot

results, path = run_accuracy_gsm8k_cot(base_url, model, results_dir, limit=200)
assert results["exact_match,flexible-extract"] >= 0.435
```

## Public API

The public surface is the per-dataset runners `run_accuracy_gsm8k`,
`run_accuracy_gsm8k_cot`, `run_accuracy_gsm8k_cot_llama`, `run_accuracy_bbh`,
`run_accuracy_ifeval`, and `run_accuracy_mmlu_pro`, plus the lower-level
`run_lm_eval` they build on (for running an arbitrary lm_eval task directly).
Each runner takes `(base_url, model, results_dir, ...)` and returns
`(flat_metrics_dict, results_file_path)`.

## Architecture

```text
caller
  │  base_url, model, results_dir, limit, gen_kwargs
  ▼
run_accuracy_<dataset>()          # public per-dataset entrypoint
  │  task name + metric-key list
  ▼
run_lm_eval()                     # public lower-level lm_eval invocation
  │  builds argv, runs `python -m lm_eval`, tees output to a log file
  ▼
lm_eval CLI  ──HTTP──▶  vLLM OpenAI-compatible server
  │  writes results_<timestamp>.json + per-sample logs
  ▼
locate newest results + resolve_metric()
  │  pick newest results file, extract requested metric keys
  ▼
({metric_key: value, ...}, results_file_path)
```

### Core invocation — `run_lm_eval`

Every runner funnels through `run_lm_eval`, which runs the `lm_eval`
subprocess. Responsibilities:

- **Server wiring.** Sets `OPENAI_API_KEY=EMPTY` and `OPENAI_API_BASE`, then
  builds `--model_args` pointing lm_eval at the vLLM server's
  `/v1/chat/completions` (or `/v1/completions` when `use_chat=False`).
  `tokenized_requests=False` / `tokenizer_backend=None` keep tokenization
  server-side.
- **Command assembly.** Toggles `--apply_chat_template`, `--fewshot_as_multiturn`,
  `--num_fewshot`, `--gen_kwargs`, `--limit`, and per-dataset `extra_args`.
- **Streaming + logging.** Runs the subprocess with `Popen`, tees stdout to both
  the parent process and a `<task>_lm_eval.log` file under `results_dir`. A
  non-zero return code raises `CalledProcessError`.
- **Result location.** Globs `results_*.json` and picks the newest by parsing
  the ISO timestamp in the filename (lm_eval writes a fresh timestamped file per
  run).

### Metric extraction — `resolve_metric`

lm_eval nests metrics under the task name, but group/aggregate tasks (e.g.
`bbh_cot_fewshot`, `mmlu_pro`) place the aggregate at `results[task]` alongside
per-subtask entries. `resolve_metric` reads `results[task]` and pulls the
requested `metric_keys` into a flat `{key: value}` dict. A missing key logs an
error and yields `-1` rather than raising — so a partial/renamed metric surfaces
as an obviously-bad value instead of crashing the run.

The metric keys are lm_eval's `"<metric>,<filter>"` convention, e.g.
`exact_match,flexible-extract` (regex-extracted answer) vs
`exact_match,strict-match` (exact string). Each runner hard-codes the keys that
matter for its dataset so callers get a stable, documented surface.

## Available runners

| Runner | lm_eval task | Key metrics |
|---|---|---|
| `run_accuracy_gsm8k` | `gsm8k` | `exact_match,flexible-extract`, `exact_match,strict-match` |
| `run_accuracy_gsm8k_cot` | `gsm8k_cot` | `exact_match,flexible-extract`, `exact_match,strict-match` |
| `run_accuracy_gsm8k_cot_llama` | `gsm8k_cot_llama` | `exact_match,flexible-extract`, `exact_match,strict-match` |
| `run_accuracy_bbh` | `bbh_cot_fewshot` | `exact_match,get-answer` |
| `run_accuracy_ifeval` | `leaderboard_ifeval` | `prompt_level_strict_acc,none`, `inst_level_strict_acc,none`, … |
| `run_accuracy_mmlu_pro` | `mmlu_pro` | `exact_match,custom-extract` |

Every runner takes `(base_url, model, results_dir, limit=None, max_length=16384,
gen_kwargs="", **kwargs)` and returns `(flat_metrics_dict, results_file_path)`.
`limit` defaults to `None` (evaluate the full dataset); pass an integer to cap
the number of samples for a quick run.
`**kwargs` forwards to `run_lm_eval` (e.g. `max_concurrent`,
`num_fewshot`, `use_chat`, `data_dir`).

Per-dataset specializations worth noting:

- **IFEval** returns both strict and loose, prompt- and instruction-level
  accuracies.

## Usage

Point a runner at a running vLLM (OpenAI-compatible) server and assert on the
returned metrics:

```python
from vllm_neuron.accuracy.lm_eval import run_accuracy_gsm8k_cot

scores, results_file = run_accuracy_gsm8k_cot(
    base_url="http://localhost:8000",
    model="/path/to/model",
    results_dir="./eval_out",
    limit=200,  # cap for a quick run; omit for the full dataset
)
assert scores["exact_match,flexible-extract"] >= 0.435
```

## References

- `vllm_neuron/accuracy/lm_eval.py` — the shipped runners.
