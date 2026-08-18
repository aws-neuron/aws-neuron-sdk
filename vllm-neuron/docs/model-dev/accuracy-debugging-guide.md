# Debugging accuracy issues

<!-- meta: description: Diagnose and resolve accuracy issues in vLLM Neuron
deployments, including a workflow for isolating where accuracy drift is
introduced. -->
<!-- meta: keywords: vLLM, Neuron, accuracy debugging, accuracy drift, numerical
mismatch, quantization, precision, LLM -->
<!-- meta: date_updated: 2026-06-11 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: NDOC-191 -->

## Task overview

This topic discusses how to systematically diagnose and resolve accuracy issues
when running models with vLLM Neuron. It provides a workflow for isolating
where accuracy drift is introduced — from checkpoint loading through generation —
so you can fix the underlying cause rather than chasing symptoms.

## Prerequisites

- **vLLM Neuron deployment:** A running or reproducible deployment where you
  observe the accuracy issue. See [setup guide](../getting-started/setup-guide.md).
- **Reference output:** Known-good outputs from a reference implementation (CPU,
  GPU, or a prior Neuron release) that you can compare against.
- **Familiarity with the model:** Understanding of the model's architecture,
  attention variant, and tokenizer.
- **Python environment:** Access to HuggingFace Transformers and the model
  checkpoint for generating reference outputs.

## Validation levels

The accuracy debugging framework operates at three levels. Each level runs
independently. When task evaluation identifies deviated prompts, they can be fed
into prompt-level analysis for deeper investigation.

### Level 1: Task-level validation

Run dataset evaluations (lm_eval, longbench) against a vLLM server and compare
aggregate scores against user-defined thresholds.

- Run accuracy benchmarks (e.g., GSM8K, MMLU Pro, BBH).
- Compare scores against thresholds (simple minimum or mean/std tolerance).
- Report pass/fail with per-dataset scores.

Pass/fail criteria: Accuracy scores must meet thresholds defined by the user.

### Level 2: Prompt-level validation

Validate model outputs at the token level using pre-defined or deviated prompts.
For each prompt, compute HF reference logits (FP32 + BF16) and run token-by-token
comparison with teacher forcing.

- **Logit validation** — Three-way comparison (HF FP32 vs HF BF16 vs Neuron)
  using top-k error maps at k={5, 50, 1000, all} and divergence detection.
- **KV cache analysis** — Three-way comparison of KV caches with per-layer,
  per-head error metrics and Bhattacharyya Coefficient (BC) to classify errors as
  BF16-inherent vs anomalous.

Pass/fail criteria: Logit divergence within tolerance maps; KV cache BC ≥ 0.99.

### Level 3: Module-level validation

Per-module correctness tests validate individual model components (attention, MLP,
RMSNorm, embedding, RoPE, decoder layer) against HuggingFace reference
implementations. These run in CPU mode and on hardware.

Pass/fail criteria: Output tensors match HF reference within tolerance.

:::{note}
Start at Level 1 to confirm there is a real accuracy issue, then use Level 2 to
isolate which tokens and layers diverge, and Level 3 to pinpoint the exact module.
For the internal architecture of this framework (debugger pipeline, plugin system,
KV-cache reconstruction, tensor capture/compare, tensor replacement), see the
[deep dive: accuracy debugging framework](../design/accuracy/accuracy_debugging_design.md).
:::

## Debugging instructions

The following steps walk through a systematic workflow for isolating accuracy
issues, from ruling out false alarms to pinpointing the divergent layer:

1. Rule out sampling variance
2. Reproduce with a minimal, deterministic prompt
3. Compare outputs against a reference
4. Isolate the introduction point with a layered check
5. Interpret the divergence pattern
6. Check for common causes
7. Apply a fix and validate

### Step 1: Rule out sampling variance

Before investigating hardware or framework issues, confirm that the difference you
observe is not normal sampling variance. Many false alarms come from
non-deterministic generation settings.

Set your generation to be fully deterministic:

```python
# Use greedy decoding to eliminate sampling randomness
sampling_params = SamplingParams(
    temperature=0.0,  # greedy
    max_tokens=64,    # short output for fast iteration
)
```

If your issue disappears under greedy decoding, the problem is likely in your
sampling configuration (temperature, top-p, top-k) rather than in the model
computation. Compare the sampling parameters between your Neuron deployment and
your reference environment before proceeding.

:::{note}
Even with greedy decoding, BF16 arithmetic can produce different top-1 tokens than
FP32 when the top two logits are very close. This is expected behavior, not a bug.
If only a small fraction of tokens differ and the overall quality is acceptable,
you may not have an accuracy issue.
:::

### Step 2: Reproduce the issue with a minimal, deterministic prompt

Reduce your reproduction to the smallest possible case:

1. Select a single prompt that demonstrates the issue.
2. Set a fixed random seed if your configuration uses one.
3. Use greedy decoding (`temperature=0.0`).
4. Limit `max_tokens` to the shortest length that still shows the divergence.

```python
# Minimal reproduction setup
prompt = "The capital of France is"  # Use your failing prompt
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

# Generate on Neuron
neuron_output = llm.generate([prompt], sampling_params)

# Compare against reference (e.g., HuggingFace on CPU)
# hf_output = run_hf_reference(prompt, max_tokens=32)
```

A minimal reproduction makes the remaining steps faster and eliminates confounding
variables like batch interactions or long-context effects.

### Step 3: Compare outputs against a reference

Start at the coarsest level and narrow down:

1. **Full text comparison** — Do the generated strings differ? At which token
   position does the first divergence occur?
2. **Top-1 token comparison** — At the first divergent position, what are the
   top-1 tokens from each environment?
3. **Logit distribution comparison** — Compare the full logit vectors at the
   divergent position. Are the top-k rankings similar with only small magnitude
   differences, or is the distribution fundamentally different?

The position of the first divergent token tells you where to focus:

- **Token 0 diverges** — The issue is in prefill (the first forward pass). Check
  weight loading, embedding, and the prefill attention computation.
- **Token 1+ diverges, token 0 matches** — The issue is in decode. Check the KV
  cache, decode attention, or position encoding for generated positions.

:::{note}
The logit validation tool automates this comparison using teacher forcing (forcing
ground-truth token when it diverges to isolate per-position errors from
autoregressive drift).
:::

### Step 4: Isolate the introduction point with a layered check

Work through the model pipeline in order, checking each stage against your
reference. Stop at the first stage that shows unexpected divergence:

| # | Layer | What to check |
| --- | --- | --- |
| 1 | Tokenizer | Confirm both environments produce identical token IDs for your prompt. Tokenizer version mismatches are a common source of apparent "accuracy" issues. |
| 2 | Weight loading | Verify the checkpoint loaded correctly. Compare a sample of weight tensors between Neuron and your reference. Check for dtype mismatches (FP32 vs BF16) or truncated/corrupted weights. |
| 3 | Embedding output | Compare the embedding layer output for your prompt tokens. This isolates weight-loading issues from computation issues. |
| 4 | Attention output | Compare attention outputs at one or more decoder layers. Differences here point to attention implementation issues (RoPE encoding, attention mask, head layout). |
| 5 | KV cache | Compare cached key/value tensors against reference. Issues here cause decode-phase divergence even when prefill is correct. |
| 6 | Post-quantization output | If using quantization, compare the quantized model output against the BF16 baseline. Quantization drift is expected but should be bounded. |
| 7 | Sampling / logit processing | Compare raw logits before and after any logit processors (repetition penalty, temperature scaling). Confirm the sampling implementation matches your reference. |

:::{note}
The logit validation tool performs a three-way comparison (HF FP32 vs HF BF16 vs
Neuron) at the logit and KV cache levels. This distinguishes BF16-inherent
numerical noise from vLLM-on-Neuron-specific bugs.
:::

### Step 5: Interpret the divergence pattern

Use the pattern of divergence to classify the root cause:

| Pattern | Interpretation |
| --- | --- |
| Neuron-vs-FP32 error is similar in magnitude to BF16-vs-FP32 error (ratio ≈ 1.0) | The divergence is BF16 numerical noise, not a vLLM-on-Neuron bug. This is expected behavior. |
| Neuron-vs-FP32 error is much larger than BF16-vs-FP32 error (ratio >> 1.0) | A vLLM-on-Neuron-specific issue exists. Proceed to isolate the specific layer or component. |
| Token 0 fails, subsequent tokens also fail | Prefill bug. The error propagates from the first forward pass. Check weight loading and prefill attention. |
| Token 0 passes, tokens 1+ fail | Decode-phase bug. Check KV cache write/read, decode attention, or position encoding at generated positions. |
| Errors appear only at specific sequence positions | Likely a position encoding or attention mask issue. Check RoPE implementation and context length handling. |

### Step 6: Check for common causes

Review the following common causes of accuracy drift. For each, the symptom
pattern and where to look:

#### Wrong checkpoint format or incomplete weight loading

**Symptom:** Large divergence starting at token 0 across all layers. Output may be
nonsensical.

**Where to check:** Compare weight tensor shapes and values between your loaded
model and the original checkpoint. Look for warnings during model loading about
missing or unexpected keys. Verify the checkpoint format (safetensors vs bin)
matches what the loader expects.

#### Dtype mismatch between reference and deployment

**Symptom:** Small but consistent divergence across all tokens. Three-way
comparison shows Neuron error is proportional to BF16 error.

**Where to check:** Confirm both environments use the same dtype. A common mistake
is comparing FP32 reference outputs against BF16 Neuron outputs and interpreting
the BF16 rounding as a bug. Always compare BF16-to-BF16 when assessing
Neuron-specific accuracy.

#### Quantization-induced drift exceeding tolerance

**Symptom:** Moderate divergence that scales with quantization aggressiveness.
Output is coherent but subtly different from the BF16 baseline.

**Where to check:** Compare BF16 Neuron output against quantized Neuron output to
isolate the quantization contribution. If the drift is within your quality
tolerance, it is working as designed. If not, consider a less aggressive
quantization scheme or a different granularity (per-channel vs per-tensor).

#### KV cache inconsistency under prefix caching or paged attention

**Symptom:** Outputs differ depending on whether a prefix cache hit occurs. The
same prompt produces different results on first vs subsequent runs, or results
change when `block_size` changes.

**Where to check:** Disable prefix caching and compare. If the issue disappears,
check that `block_size` and `max_model_len` are compatible with your prompt
lengths. Verify that KV cache reconstruction from paged blocks produces the same
values as contiguous computation.

#### Speculative decoding verification failure

**Symptom:** Occasional incorrect tokens that do not appear under standard
(non-speculative) decoding.

**Where to check:** Disable speculative decoding and compare. If the issue
resolves, the draft model may be proposing tokens that the verification step
incorrectly accepts. Check that the draft and target models use compatible
tokenizers and that the verification threshold is correctly configured.

#### Unsupported feature combination

**Symptom:** Incorrect output that correlates with enabling a specific feature
(e.g., a particular attention variant, context length extension, or parallelism
configuration).

**Where to check:** Consult the
[model cards](../model-recipes/index.md)
to confirm your feature combination is supported. Some combinations are not yet
validated and may produce incorrect results silently.

### Step 7: Apply a fix and validate

After identifying the root cause, apply your fix and validate systematically:

1. **Rerun the minimal prompt** — Confirm the fix resolves the original failing
   case under greedy decoding.
2. **Test additional prompts** — Run 5–10 diverse prompts that previously showed
   divergence.
3. **Run a broader evaluation** — Use a standard benchmark (e.g., GSM8K, MMLU) to
   confirm the fix does not introduce regressions elsewhere.
4. **Compare against your quality threshold** — Define an acceptable accuracy
   tolerance and verify the deployment meets it.

```bash
# Example: run lm_eval to validate accuracy after a fix
lm_eval --model vllm \
    --model_args pretrained=/path/to/model,tensor_parallel_size=8 \
    --tasks gsm8k \
    --batch_size auto
```

:::{important}
A fix that resolves one prompt but degrades others is not a fix. Always validate
broadly before declaring the issue resolved.
:::

## Confirm your work

To confirm the accuracy issue is resolved:

1. The minimal reproduction prompt now produces output matching your reference
   under greedy decoding.
2. All inspected layers report divergence within the expected BF16 noise range
   (three-way ratio ≈ 1.0).
3. A broader evaluation (multiple prompts or a benchmark) shows scores within your
   defined tolerance of the reference.

## Common issues

### Tokenizer mismatch between reference and deployment

- **Possible solution:** Confirm both environments load the same tokenizer files
  and version. Small tokenizer updates can produce different token IDs for the
  same text, causing apparent model divergence. Always load the tokenizer from the
  same checkpoint path in both environments.

### Quantization-induced drift exceeds quality threshold

- **Possible solution:** Compare BF16 versus quantized outputs to quantify the
  quantization contribution separately from other issues. If drift exceeds your
  tolerance, try a different quantization scheme (e.g., switch from W8A8 to W8A16)
  or increase quantization granularity (per-channel instead of per-tensor).

### Prefix caching produces different output than non-cached runs

- **Possible solution:** This indicates a cache consistency issue. Verify that
  `block_size`, `max_model_len`, and `num_gpu_blocks_override` are compatible with
  your prompt length distribution. As a diagnostic step, disable prefix caching to
  confirm it is the source.

### Output looks correct for short prompts but degrades at longer context

- **Possible solution:** Check that the model's maximum context length
  configuration matches the checkpoint's trained context length. Verify that RoPE
  scaling parameters (if used) are correctly configured. Compare attention outputs
  at positions near the context boundary.

### Accuracy differs between tensor parallelism degrees

- **Possible solution:** Some numerical differences across TP degrees are expected
  due to reduction order. If the differences are large, check that the weight
  sharding is correct for your TP configuration and that the model architecture
  supports the TP degree you selected.

## Accuracy tools reference

For API details on logit validation, KV cache analysis, tensor capture,
tensor comparison, and module-level testing, see [Accuracy examples](https://github.com/vllm-project/vllm-neuron/blob/HEAD/examples/vllm_neuron/accuracy/) and [deep dive: accuracy debugging framework](../design/accuracy/accuracy_debugging_design.md).

For accuracy debugger tools, see [Accuracy debugger tools](how-to-use-accuracy-debugger.md) and [deep dive: accuracy debugging framework](../design/accuracy/accuracy_debugging_design.md).

## Related information

- [Deep dive: accuracy debugging framework](../design/accuracy/accuracy_debugging_design.md)
  — Internal architecture of the debugger pipeline, plugin system, KV-cache
  reconstruction, tensor capture/compare, and tensor replacement.
- [Accuracy debugger tools](how-to-use-accuracy-debugger.md) —
  Step-by-step procedure for running the accuracy debugger tools.
- For supported models and features, see the [README](https://github.com/vllm-project/vllm-neuron#supported-models)
  and [model cards](../model-recipes/index.md).
- [Features guide](../guides/features-guide.md) — Feature configuration options that can
  affect accuracy.
- [Model onboarding](onboarding-models.md) — Model onboarding flow, which includes
  an accuracy validation step.
