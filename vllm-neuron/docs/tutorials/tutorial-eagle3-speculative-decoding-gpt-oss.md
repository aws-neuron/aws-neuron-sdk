# Tutorial: Deploy EAGLE3 speculative decoding with vLLM Neuron using GPT-OSS-120B

<!-- meta: description: Deploy EAGLE3 speculative decoding with vLLM Neuron
using GPT-OSS-120B and a public EAGLE3 draft checkpoint from Hugging
Face. -->
<!-- meta: keywords: vLLM, Neuron, EAGLE3, speculative decoding, GPT-OSS,
Trainium, tutorial, throughput, acceptance rate -->
<!-- meta: date_updated: 2026-07-27 -->
<!-- Content type: procedural-tutorial -->
<!-- Jira: CHRS-1086 -->

This tutorial guides you through deploying EAGLE3 speculative decoding with vLLM
on Neuron using a public EAGLE3 draft checkpoint from Hugging Face. When you have
completed it, you will have a vLLM Neuron server running GPT-OSS-120B
accelerated with an EAGLE3 draft model, observed the draft acceptance rate, and
measured the throughput improvement over the non-speculative baseline on the
public `sonnet` dataset.

## Overview

Speculative decoding uses a small *draft* model to propose several candidate
tokens per step. The larger *target* model then verifies all proposals in a
single forward pass. Each accepted proposal yields an extra token from one target
step, lifting throughput without changing the target model's output.

EAGLE3 ([arxiv.org/abs/2503.01840](https://arxiv.org/abs/2503.01840)) is a draft
architecture that conditions on hidden states from the target model. Because it
can reuse the target's intermediate state, EAGLE3 typically achieves higher
acceptance rates than a vanilla draft-model approach in which an independently
trained smaller model (for example, Llama 3.2 1B as a draft for Llama 3.1 8B)
proposes tokens. vLLM Neuron supports EAGLE3 natively via the
`--speculative-config` flag.

This tutorial uses the following target-draft model pair:

- **Target:** [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
- **Draft:** [RedHatAI/gpt-oss-120b-speculator.eagle3](https://huggingface.co/RedHatAI/gpt-oss-120b-speculator.eagle3)

## Before you start

This tutorial assumes that you have experience in the following areas:

- Running a vLLM Neuron server. See
  [online serving quickstart](../getting-started/quickstart-online-serving.md).
- Working with Hugging Face models.
- Basic familiarity with speculative decoding concepts.

## Model details

This tutorial uses GPT-OSS-120B with the following settings:

- `tensor-parallel-size 16`
- `max-num-seqs 1`
- `max-model-len 768`
- `max-num-batched-tokens 512`
- `num_speculative_tokens 1`

The public GPT-OSS-120B checkpoint stores its MoE expert weights in MXFP4.
Trn2 does not support the MXFP4 runtime path, so vLLM Neuron runs the model in
BF16: the weight loader detects the MXFP4 expert blocks and scales and
dequantizes them to BF16 at load time. You do not need to prepare or convert a
separate BF16 checkpoint — the stock `openai/gpt-oss-120b` checkpoint is used
directly.

**Limitations:**

- The server is configured for a decode concurrency of 1. Higher values require
  additional decode batch-bucket coverage and are out of scope for this tutorial.
- GPT-OSS-120B in BF16 requires `tensor-parallel-size 16` or higher on
  `trn2.48xlarge`; at TP8 the target weights do not fit in per-core HBM.
- If the implementation of either the target model or the draft model is not
  accurate, the acceptance rate will be low and speculative decoding will add
  compute overhead instead of boosting decode throughput.

## Prerequisites

- `trn2.48xlarge` instance with Neuron SDK 2.31 or later. See
  [setup guide](../getting-started/setup-guide.md).
- vLLM Neuron plugin installed.
- Both the target and draft checkpoints are public, ungated repositories — no
  Hugging Face license acceptance or access token is required to download them.

---

## Prepare your environment

You do not need to pre-download the checkpoints — vLLM resolves both model IDs
and downloads them on first launch. Verify you can reach the target repository.
The script below pulls only the small `config.json`; the full model and draft
checkpoints are downloaded by `vllm serve` on first launch.

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="openai/gpt-oss-120b",
    filename="config.json",
)
print("OK")
PY
```

If the script prints `OK`, you are ready.

Download the public `sonnet` dataset, used for the throughput benchmarks in Step 1
and Step 2:

```bash
wget -O sonnet.txt https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt
```

## Step 1: Run the non-speculative baseline

In this step, you will launch GPT-OSS-120B without speculative decoding,
smoke-test it, and benchmark it. You will use this baseline to measure the
throughput improvement EAGLE3 delivers in Step 2.

Launch the baseline server:

```bash
vllm serve openai/gpt-oss-120b \
    --dtype bfloat16 \
    --tensor-parallel-size 16 \
    --max-num-seqs 1 \
    --max-model-len 768 \
    --max-num-batched-tokens 512 \
    --no-enable-prefix-caching \
    --hf-overrides '{"quantization_config": {}}' \
    --additional-config '{
        "neuron_config": {
          "quantization": "bf16",
          "num_batched_tokens_buckets": [512],
          "num_seqs_buckets": [1],
          "kv_segment_size_buckets": [512],
          "on_device_sampling_config": {"all_greedy": true}
        }
      }' \
    --port 8000
```

The `--hf-overrides '{"quantization_config": {}}'` flag clears the checkpoint's
MXFP4 quantization config so the BF16 load path is selected; the weight loader
then dequantizes the MXFP4 expert weights to BF16 automatically. The
`--additional-config` block pins the compile buckets to the benchmark shape
(512-token prefill, single sequence), which keeps first-run compilation time
down.

First-run compilation and warmup take several minutes. Wait until the server log
prints `INFO: Application startup complete.` before continuing.

Confirm the server responds:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "openai/gpt-oss-120b",
      "prompt": "I am gonna keep counting forever, 1 2 3 4 5 ",
      "max_tokens": 10,
      "temperature": 0
    }'
```

The `text` field should continue the counting sequence (for example,
`"6 7 8 9 10 "`).

Run the baseline benchmark:

```bash
vllm bench serve \
    --base-url http://localhost:8000 \
    --model openai/gpt-oss-120b \
    --dataset-name sonnet \
    --dataset-path ./sonnet.txt \
    --sonnet-input-len 512 \
    --sonnet-output-len 128 \
    --num-prompts 30 \
    --max-concurrency 1 \
    --save-result \
    --result-filename baseline.json
```

When the benchmark finishes, stop the baseline server with `Ctrl+C`. Both servers
use the same port and Neuron cores, so the EAGLE3 server in Step 2 cannot run
alongside it.

## Step 2: Run EAGLE3 and compare

In this step, you will launch the EAGLE3 server, smoke-test it, benchmark it on
the same dataset, and compare the results against the baseline from Step 1.

Launch the EAGLE3 server:

```bash
vllm serve openai/gpt-oss-120b \
    --dtype bfloat16 \
    --tensor-parallel-size 16 \
    --max-num-seqs 1 \
    --max-model-len 768 \
    --max-num-batched-tokens 512 \
    --no-enable-prefix-caching \
    --hf-overrides '{"quantization_config": {}}' \
    --additional-config '{
        "neuron_config": {
          "quantization": "bf16",
          "num_batched_tokens_buckets": [512],
          "num_seqs_buckets": [1],
          "kv_segment_size_buckets": [512],
          "on_device_sampling_config": {"all_greedy": true}
        }
      }' \
    --speculative-config '{
        "method": "eagle3",
        "model": "RedHatAI/gpt-oss-120b-speculator.eagle3",
        "num_speculative_tokens": 1
      }' \
    --port 8000
```

The new flag is `--speculative-config`:

- `method: eagle3` — selects the EAGLE3 draft architecture.
- `model` — draft checkpoint, which must be trained against this specific target.
- `num_speculative_tokens: 1` — number of tokens the draft proposes per target
  step. For this MoE pairing, `1` measured fastest; see Step 3.

First-run compilation takes longer than the baseline because both the target and
the draft model compile. Wait again for `INFO: Application startup complete.`.

Confirm the server responds. Under greedy sampling, EAGLE3 is a lossless
acceleration: the target model's output is unchanged regardless of what the draft
proposes, so the response should match the baseline output from Step 1.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "openai/gpt-oss-120b",
      "prompt": "I am gonna keep counting forever, 1 2 3 4 5 ",
      "max_tokens": 10,
      "temperature": 0
    }'
```

Run the EAGLE3 benchmark:

```bash
vllm bench serve \
    --base-url http://localhost:8000 \
    --model openai/gpt-oss-120b \
    --dataset-name sonnet \
    --dataset-path ./sonnet.txt \
    --sonnet-input-len 512 \
    --sonnet-output-len 128 \
    --num-prompts 30 \
    --max-concurrency 1 \
    --save-result \
    --result-filename eagle3.json
```

When the benchmark runs against an EAGLE3 server, `vllm bench serve` prints a
Speculative Decoding block with the draft acceptance metrics. This is the headline
signal for whether EAGLE3 is helping on your workload — the higher the acceptance
rate, the more tokens each target step produces.

```text
---------------Speculative Decoding---------------
Acceptance rate (%):                     69.00
Acceptance length:                       1.68
Drafts:                                  2270
Draft tokens:                            2270
Accepted tokens:                         1566
Per-position acceptance (%):
  Position 0:                            69.00
```

Key fields:

- `Acceptance rate` — overall fraction of drafted tokens that were accepted.
- `Acceptance length` — average tokens produced per target step. `1.0` means no
  draft tokens were accepted (equivalent to no speculation). `num_speculative_tokens + 1`
  is the theoretical maximum.
- `Per-position acceptance` — acceptance per draft position. With
  `num_speculative_tokens: 1` there is a single position, so it equals the
  overall acceptance rate.

Compare the two benchmark runs:

```bash
python - <<'PY'
import json
b = json.load(open("baseline.json"))
e = json.load(open("eagle3.json"))
for k in ("output_throughput", "median_tpot_ms", "median_ttft_ms"):
    print(f"{k:20s}  baseline={b[k]:.2f}  eagle3={e[k]:.2f}")
print(f"\nEAGLE3 output throughput speedup: {e['output_throughput'] / b['output_throughput']:.2f}x")
PY
```

EAGLE3 should report higher `output_throughput` and lower `median_tpot_ms` than
the baseline. Representative numbers for this pairing on `trn2.48xlarge` at
`--max-concurrency 1`:

```text
output_throughput     baseline=112.13  eagle3=134.53
median_tpot_ms        baseline=7.00    eagle3=5.82
median_ttft_ms        baseline=252.33  eagle3=218.21

EAGLE3 output throughput speedup: 1.20x
```

Time-to-first-token can be slightly higher or lower under EAGLE3 depending on
the pairing; for a 120B target the draft's added prefill compute is negligible,
and the headline effect is the per-token speedup.

## Step 3: Tune for your workload

The two settings that have the biggest impact are the draft/target pair and
`num_speculative_tokens`.

- **Confirm that both draft and target models are accurate.** No tuning
  compensates for an inaccurate model on either side. If Step 2 shows an
  acceptance rate below roughly 30%, stop tuning and validate the draft and target
  implementations against a known reference before continuing.
- **`num_speculative_tokens`.** Higher values propose more tokens per step. They
  yield more when acceptance is high and lose more when acceptance is low, because
  rejected drafts still consume compute. Sweep `1`, `2`, and `3` on representative
  traffic and pick the value with the highest `output_throughput` — not the
  highest acceptance rate. For this MoE pairing, `1` was the best value: at
  higher values the per-step verification cost of the 120B MoE target grows
  faster than the extra accepted tokens recover.
- **Workload alignment.** EAGLE3 acceptance depends on the datasets the draft head
  was trained on. Acceptance rates degrade when the request distribution at serving
  time diverges from the draft's training distribution. If your traffic is far from
  the draft's training mix, expect lower speedups and consider a draft trained on
  data closer to your workload.

## Confirmation

You have launched a baseline server and an EAGLE3 server, verified they produce
equivalent greedy output, observed the draft acceptance rate, and measured the
throughput improvement on the sonnet dataset. If you encountered any issues, see
the **Common issues** section below.

---

## Benchmarks

:::{note}
Numbers are illustrative. Throughput depends on prompt mix, input and output
lengths, and draft/target acceptance behavior. Re-measure on traffic that
represents your workload before committing to an EAGLE3 configuration.
:::

| Platform | Metric | Baseline | EAGLE3 (num_speculative_tokens=1) |
| --- | --- | --- | --- |
| trn2.48xlarge, TP16 | output_throughput (tok/s) | 112.1 | 134.5 |
| trn2.48xlarge, TP16 | median_tpot_ms | 7.00 | 5.82 |
| trn2.48xlarge, TP16 | Avg draft acceptance rate | n/a | 69.0% |
| trn2.48xlarge, TP16 | Mean acceptance length | n/a | 1.68 |
| trn2.48xlarge, TP16 | Per-position acceptance (pos 0) | n/a | 69.0% |

Measured with the EAGLE3 `vllm bench serve` command in Step 2
(`--sonnet-input-len 512 --sonnet-output-len 128 --num-prompts 30 --max-concurrency 1`).
This represents a 1.20× output-throughput speedup over the non-speculative
baseline.

## Common issues

- **Acceptance rate is 0% or near 0%.** If the implementation of either the target
  or the draft is not accurate, acceptance collapses. Confirm both models are
  accurate against a known reference and that the draft checkpoint was trained
  against this exact target.
- **Server fails during weight loading with an allocation failure.** GPT-OSS-120B
  in BF16 does not fit at `tensor-parallel-size 8` on `trn2.48xlarge`. Use
  `tensor-parallel-size 16` or higher.
- **Looking for a BF16 GPT-OSS-120B checkpoint.** You do not need one. The stock
  `openai/gpt-oss-120b` checkpoint (MXFP4 expert weights) is used directly; the
  BF16 weight loader dequantizes the expert weights at load time.
- **Throughput regressed with EAGLE3 enabled.** The EAGLE3 acceptance rate is not
  high enough to compensate for the speculative decoding compute overhead.
  Speculative decoding pays compute for every draft token and recovers it only on
  accepted tokens. Check Step 2's Speculative Decoding block; if acceptance is low,
  use a draft trained on data closer to your serving distribution.

## Clean up

Stop the vLLM servers with `Ctrl+C`. If you launched an EC2 instance for this
tutorial, terminate it to avoid ongoing charges.

## Next steps

- [Features guide](../guides/features-guide.md) — Other features you can stack
  with speculative decoding.
- For supported models and features, see the [README](https://github.com/vllm-project/vllm-neuron#supported-models)
  and [model cards](../model-recipes/index.md).
- [Quickstart: Offline serving](../getting-started/quickstart-offline-serving.md)
  — Use EAGLE3 from the offline `vllm.LLM` Python API instead of the server.
- [Prefix caching benchmark tutorial](tutorial-prefix-caching-gpt-oss-benchmarking.md)
  — Combine speculative decoding with prefix caching.
