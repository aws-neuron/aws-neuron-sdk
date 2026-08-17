# Tutorial: Deploy EAGLE3 speculative decoding with vLLM Neuron using Llama 3.1 8B Instruct

<!-- meta: description: Deploy EAGLE3 speculative decoding with vLLM Neuron
using Llama 3.1 8B Instruct and a public EAGLE3 draft checkpoint from Hugging
Face. -->
<!-- meta: keywords: vLLM, Neuron, EAGLE3, speculative decoding, Llama 3.1,
Trainium, tutorial, throughput, acceptance rate -->
<!-- meta: date_updated: 2026-06-11 -->
<!-- Content type: procedural-tutorial -->
<!-- Jira: NDOC-187 -->

This tutorial guides you through deploying EAGLE3 speculative decoding with vLLM
on Neuron using a public EAGLE3 draft checkpoint from Hugging Face. When you have
completed it, you will have a vLLM Neuron server running Llama 3.1 8B Instruct
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

- **Target:** [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **Draft:** [RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3](https://huggingface.co/RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3)

## Before you start

This tutorial assumes that you have experience in the following areas:

- Running a vLLM Neuron server. See
  [online serving quickstart](../getting-started/quickstart-online-serving.md).
- Working with Hugging Face models and access tokens.
- Basic familiarity with speculative decoding concepts.

## Model details

This tutorial uses Llama 3.1 8B Instruct with the following settings:

- `tensor-parallel-size 8`
- `max-num-seqs 2`
- `max-model-len 2048`
- `num_speculative_tokens 3`

**Limitations:**

- The server is configured for a decode concurrency of 2. Higher values require
  additional decode batch-bucket coverage and are out of scope for this tutorial.
- If the implementation of either the target model or the draft model is not
  accurate, the acceptance rate will be low and speculative decoding will add
  compute overhead instead of boosting decode throughput.

## Prerequisites

- `trn2.48xlarge` instance with Neuron SDK 2.32 or later. See
  [setup guide](../getting-started/setup-guide.md).
- vLLM Neuron plugin installed.
- Hugging Face account that has accepted the Llama 3.1 license at
  <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>.
- Hugging Face access token with read access to public gated repositories.
  Fine-grained tokens that lack this permission return HTTP 403 on gated repos
  even when the license is accepted.

---

## Prepare your environment

Set your Hugging Face access token in the shell you will use to launch the
server. You do not need to pre-download the checkpoints — vLLM resolves both model
IDs and downloads them on first launch.

```bash
export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Verify the token has access to the gated target repository. The script below
pulls only the small `config.json` to confirm authentication; the full model and
draft checkpoints are downloaded by `vllm serve` on first launch.

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    filename="config.json",
)
print("OK")
PY
```

If the script prints `OK`, you are ready. If it returns `403 Forbidden`, update
your token permissions at <https://huggingface.co/settings/tokens> and retry.

Download the public `sonnet` dataset, used for the throughput benchmarks in Step 1
and Step 2:

```bash
wget -O sonnet.txt https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt
```

## Step 1: Run the non-speculative baseline

In this step, you will launch Llama 3.1 8B Instruct without speculative decoding,
smoke-test it, and benchmark it. You will use this baseline to measure the
throughput improvement EAGLE3 delivers in Step 2.

Launch the baseline server:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 8 \
    --max-num-seqs 2 \
    --max-model-len 2048 \
    --no-enable-prefix-caching \
    --port 8000
```

First-run compilation and warmup take several minutes. Wait until the server log
prints `INFO: Application startup complete.` before continuing.

Confirm the server responds:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-3.1-8B-Instruct",
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
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sonnet \
    --dataset-path ./sonnet.txt \
    --sonnet-input-len 512 \
    --sonnet-output-len 128 \
    --num-prompts 50 \
    --max-concurrency 2 \
    --save-result \
    --result-filename baseline.json
```

When the benchmark finishes, stop the baseline server with `Ctrl+C`. Both servers
require all 8 Neuron devices for `--tensor-parallel-size 8`, so the EAGLE3 server
in Step 2 cannot run alongside it.

## Step 2: Run EAGLE3 and compare

In this step, you will launch the EAGLE3 server, smoke-test it, benchmark it on
the same dataset, and compare the results against the baseline from Step 1.

Launch the EAGLE3 server:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 8 \
    --max-num-seqs 2 \
    --max-model-len 2048 \
    --no-enable-prefix-caching \
    --speculative-config '{
        "method": "eagle3",
        "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        "num_speculative_tokens": 3
      }' \
    --port 8000
```

The new flag is `--speculative-config`:

- `method: eagle3` — selects the EAGLE3 draft architecture.
- `model` — draft checkpoint, which must be trained against this specific target.
- `num_speculative_tokens: 3` — number of tokens the draft proposes per target
  step.

First-run compilation takes longer than the baseline because both the target and
the draft model compile. Wait again for `INFO: Application startup complete.`.

Confirm the server responds. Under greedy sampling, EAGLE3 is a lossless
acceleration: the target model's output is unchanged regardless of what the draft
proposes, so the response should match the baseline output from Step 1.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-3.1-8B-Instruct",
      "prompt": "I am gonna keep counting forever, 1 2 3 4 5 ",
      "max_tokens": 10,
      "temperature": 0
    }'
```

Run the EAGLE3 benchmark:

```bash
vllm bench serve \
    --base-url http://localhost:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sonnet \
    --dataset-path ./sonnet.txt \
    --sonnet-input-len 512 \
    --sonnet-output-len 128 \
    --num-prompts 50 \
    --max-concurrency 2 \
    --save-result \
    --result-filename eagle3.json
```

When the benchmark runs against an EAGLE3 server, `vllm bench serve` prints a
Speculative Decoding block with the draft acceptance metrics. This is the headline
signal for whether EAGLE3 is helping on your workload — the higher the acceptance
rate, the more tokens each target step produces.

```text
---------------Speculative Decoding---------------
Acceptance rate (%):                     74.14
Acceptance length:                       3.22
Drafts:                                  1976
Draft tokens:                            5928
Accepted tokens:                         4395
Per-position acceptance (%):
  Position 0:                            86.49
  Position 1:                            72.47
  Position 2:                            63.46
```

Key fields:

- `Acceptance rate` — overall fraction of drafted tokens that were accepted.
- `Acceptance length` — average tokens produced per target step. `1.0` means no
  draft tokens were accepted (equivalent to no speculation). `num_speculative_tokens + 1`
  is the theoretical maximum.
- `Per-position acceptance` — acceptance per draft position. The first position is
  usually the highest; a steep drop-off suggests the draft is only useful for the
  immediate next token.

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
`--max-concurrency 2`:

```text
output_throughput     baseline=184.44  eagle3=288.63
median_tpot_ms        baseline=10.52   eagle3=6.41
median_ttft_ms        baseline=51.07   eagle3=63.65

EAGLE3 output throughput speedup: 1.57x
```

Time-to-first-token is slightly higher under EAGLE3 because the draft model adds a
small amount of compute before the first token; this is expected and is more than
offset by the per-token speedup.

## Step 3: Tune for your workload

The two settings that have the biggest impact are the draft/target pair and
`num_speculative_tokens`.

- **Confirm that both draft and target models are accurate.** No tuning
  compensates for an inaccurate model on either side. If Step 2 shows an
  acceptance rate below roughly 30%, stop tuning and validate the draft and target
  implementations against a known reference before continuing.
- **`num_speculative_tokens`.** Higher values propose more tokens per step. They
  yield more when acceptance is high and lose more when acceptance is low, because
  rejected drafts still consume compute. Sweep `2`, `3`, and `4` on representative
  traffic and pick the value with the highest `output_throughput` — not the
  highest acceptance rate.
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

| Platform | Metric | Baseline | EAGLE3 (num_speculative_tokens=3) |
| --- | --- | --- | --- |
| trn2.48xlarge, TP8 | output_throughput (tok/s) | 184.4 | 288.6 |
| trn2.48xlarge, TP8 | median_tpot_ms | 10.52 | 6.41 |
| trn2.48xlarge, TP8 | Avg draft acceptance rate | n/a | 74.1% |
| trn2.48xlarge, TP8 | Mean acceptance length | n/a | 3.22 |
| trn2.48xlarge, TP8 | Per-position acceptance (pos 0 / 1 / 2) | n/a | 86.5% / 72.5% / 63.5% |

Measured with the EAGLE3 `vllm bench serve` command in Step 2
(`--sonnet-input-len 512 --sonnet-output-len 128 --num-prompts 50 --max-concurrency 2`).
This represents a 1.57× output-throughput speedup over the non-speculative
baseline.

## Common issues

- **Acceptance rate is 0% or near 0%.** If the implementation of either the target
  or the draft is not accurate, acceptance collapses. Confirm both models are
  accurate against a known reference and that the draft checkpoint was trained
  against this exact target.
- **HTTP 403 when the server pulls the model.** Your Hugging Face token lacks the
  "read access to public gated repositories" permission, or the account has not
  accepted the Llama 3.1 license. Update the token at
  <https://huggingface.co/settings/tokens>, accept the license at
  <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>, and restart the
  server.
- **Throughput regressed with EAGLE3 enabled.** The EAGLE3 acceptance rate is not
  high enough to compensate for the speculative decoding compute overhead.
  Speculative decoding pays compute for every draft token and recovers it only on
  accepted tokens. Check Step 2's Speculative Decoding block; if acceptance is low,
  reduce `num_speculative_tokens` or use a draft trained on data closer to your
  serving distribution.

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
