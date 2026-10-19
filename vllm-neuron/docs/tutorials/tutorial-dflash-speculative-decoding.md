# Tutorial: Deploy DFlash speculative decoding with vLLM Neuron

<!-- meta: description: Deploy DFlash speculative decoding with vLLM Neuron
using Llama 3.1 8B Instruct and the public DFlash UltraChat draft checkpoint. -->
<!-- meta: keywords: vLLM, Neuron, DFlash, speculative decoding, Llama 3.1,
Trainium, block parallel, throughput, acceptance rate -->
<!-- meta: date_updated: 2026-07-23 -->
<!-- Content type: procedural-tutorial -->

This tutorial guides you through deploying DFlash speculative decoding with
vLLM on Neuron. You will launch Llama 3.1 8B Instruct with a compatible DFlash
draft model, compare its greedy output with a non-speculative baseline, inspect
draft acceptance metrics, and measure performance on your workload.

## Overview

Speculative decoding uses a lightweight *draft* model to propose several
candidate tokens. The larger *target* model verifies those candidates in one
forward pass. Accepted candidates allow one target step to emit multiple tokens
without changing the target model's output distribution.

DFlash proposes a block of tokens in parallel instead of generating draft
tokens autoregressively. It projects auxiliary hidden states from selected
target layers into the draft model's KV cache (KV Injection), then applies bidirectional attention
to a block containing the latest accepted token followed by mask tokens, producing multiple
draft tokens at once. The target model verifies the resulting draft tokens with the standard speculative
decoding rejection sampler.

This tutorial uses the following validated target and draft pair:

- **Target:** [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **Draft:** [z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat)

The draft checkpoint defines a block size of 10: one accepted target token and
nine draft positions. Therefore, this tutorial sets
`num_speculative_tokens` to `9`.

## Before you start

This tutorial assumes that you are familiar with:

- Running a vLLM Neuron server. See the
  [online serving quickstart](../getting-started/quickstart-online-serving.md).
- Working with Hugging Face models and access tokens.
- Basic speculative decoding concepts.

## Model details

The commands use the following settings:

- `tensor-parallel-size 8`
- `max-num-seqs 1`
- `max-model-len 2048`
- `max-num-batched-tokens 2048`
- `num_speculative_tokens 9`

## Prerequisites

- A `trn2.48xlarge` instance with Neuron SDK 2.31 or later. See the
  [setup guide](../getting-started/setup-guide.md).
- The vLLM Neuron plugin installed from a version that includes DFlash support (>= 0.24.0).
- A Hugging Face account that has accepted the Llama 3.1 license at <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>.
- A Hugging Face access token with read access to public gated repositories.

## Prepare your environment

Set your Hugging Face access token in the shell you will use to launch the
server. You do not need to pre-download the checkpoints — vLLM resolves both model
IDs and downloads them on first launch.

```bash
export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Verify that it can access both checkpoint configurations:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

for repo_id in (
    "meta-llama/Llama-3.1-8B-Instruct",
    "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
):
    hf_hub_download(repo_id=repo_id, filename="config.json")
    print(f"Access confirmed: {repo_id}")
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

Launch the target model without speculative decoding:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 8 \
    --max-num-seqs 1 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --no-enable-prefix-caching \
    --additional-config '{
        "neuron_config": {
          "on_device_sampling_config": {"all_greedy": "true"}
        }
      }' \
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
      "max_tokens": 32,
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
    --max-concurrency 1 \
    --save-result \
    --result-filename baseline.json
```

When the benchmark finishes, stop the baseline server with `Ctrl+C`. Both servers
require all 8 Neuron devices for `--tensor-parallel-size 8`, so the DFlash server
in Step 2 cannot run alongside it.

## Step 2: Run DFlash

In this step, you will launch the DFlash server, smoke-test it, benchmark it on
the same dataset, and compare the results against the baseline from Step 1.

Launch the same target with DFlash enabled:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 8 \
    --max-num-seqs 1 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --no-enable-prefix-caching \
    --speculative-config '{
        "method": "dflash",
        "model": "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
        "num_speculative_tokens": 9
      }' \
    --additional-config '{
        "neuron_config": {
          "on_device_sampling_config": {"all_greedy": "true"}
        }
      }' \
    --port 8000
```

The relevant configuration fields are:

- `method: dflash`: Selects the DFlash proposer.
- `model`: Identifies the DFlash checkpoint trained for the target model.
- `num_speculative_tokens: 9`: Uses all nine draft positions in the
  checkpoint's ten-position block.
- `on_device_sampling_config.all_greedy: true`: Declares that every request uses
  greedy sampling. This selects the deterministic argmax path, making baseline
  and DFlash outputs directly comparable.

The first launch takes longer than the baseline because it compiles both target
verification graphs and the DFlash draft graph. Wait again for `INFO: Application startup complete.`.

Confirm the server responds. Under greedy sampling, DFlash is a lossless
acceleration: the target model's output is unchanged regardless of what the draft
proposes, so the response should match the baseline output from Step 1.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-3.1-8B-Instruct",
      "prompt": "I am gonna keep counting forever, 1 2 3 4 5 ",
      "max_tokens": 32,
      "temperature": 0
    }'
```

The generated text should match the baseline. DFlash changes how many tokens the
target verifies per step, not which greedy tokens the target emits.

## Step 3: Inspect acceptance and performance

Run the same benchmark against the DFlash server:

```bash
vllm bench serve \
    --base-url http://localhost:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sonnet \
    --dataset-path ./sonnet.txt \
    --sonnet-input-len 512 \
    --sonnet-output-len 128 \
    --num-prompts 50 \
    --max-concurrency 1 \
    --save-result \
    --result-filename dflash.json
```

The DFlash benchmark reports the following speculative decoding metrics:

```text
---------------Speculative Decoding---------------
Acceptance rate (%):                     71.38
Acceptance length:                       7.42
Drafts:                                  903
Draft tokens:                            8127
Accepted tokens:                         5801
Per-position acceptance (%):
  Position 0:                            97.23
  Position 1:                            96.57
  Position 2:                            90.37
  Position 3:                            83.39
  Position 4:                            81.40
  Position 5:                            63.68
  Position 6:                            55.81
  Position 7:                            45.85
  Position 8:                            28.13
```

Key fields:

- **Acceptance rate:** Fraction of draft tokens accepted by the target.
- **Acceptance length:** Average number of tokens emitted per target step.
  `1.0` means that no draft tokens were accepted. With nine speculative tokens,
  the theoretical maximum is `10.0`.
- **Per-position acceptance:** Acceptance rate for each of the nine positions in
  the DFlash draft block.

Compare the saved benchmark results:

```bash
python - <<'PY'
import json

with open("baseline.json") as f:
    baseline = json.load(f)
with open("dflash.json") as f:
    dflash = json.load(f)

for key in ("output_throughput", "median_tpot_ms", "median_ttft_ms"):
    print(
        f"{key:20s} baseline={baseline[key]:.2f} "
        f"dflash={dflash[key]:.2f}"
    )

speedup = dflash["output_throughput"] / baseline["output_throughput"]
print(f"\nDFlash output throughput speedup: {speedup:.2f}x")
PY
```

Representative output from these benchmark runs:

```text
output_throughput    baseline=158.32 dflash=408.26
median_tpot_ms       baseline=6.15 dflash=2.17
median_ttft_ms       baseline=27.05 dflash=31.75

DFlash output throughput speedup: 2.58x
```

DFlash achieved 2.58x the baseline output throughput and reduced median time per
output token by 64.7%. Median time to first token increased by 4.70 ms because
DFlash adds draft-model work before emitting the first token.

Do not evaluate DFlash from acceptance rate alone. The useful configuration is
the one that improves end-to-end throughput or time per output token for your
request distribution.

## Tune for your workload

- **Keep the target and draft paired.** The checkpoint's target-layer mapping,
  vocabulary, token embedding, and LM head must match the target model. Do not
  substitute another Llama checkpoint based only on architecture similarity.
- **Benchmark representative prompts.** The UltraChat-trained draft can have
  different acceptance behavior on code, mathematics, or domain-specific text.
- **Match `num_speculative_tokens` to the checkpoint block.** This checkpoint
  provides nine draft positions, so `num_speculative_tokens` must be `9`.
- **Size sequence buckets for production concurrency.** This tutorial uses
  `max_num_seqs=1`. For larger batches, include the intended values in
  `num_seqs_buckets` so they compile during startup.

## Confirmation

You have launched a baseline server and a DFlash server, confirmed equivalent
greedy output, inspected speculative acceptance, and compared performance.

## Benchmarks

:::{note}
These numbers are illustrative. Throughput depends on prompt distribution,
input and output lengths, and draft acceptance behavior. Benchmark traffic that
represents your workload before choosing a DFlash configuration.
:::

| Platform | Metric | Baseline | DFlash (`num_speculative_tokens=9`) |
| --- | --- | ---: | ---: |
| trn2.48xlarge, TP8 | Output throughput (tok/s) | 158.32 | 408.26 |
| trn2.48xlarge, TP8 | Median TPOT (ms) | 6.15 | 2.17 |
| trn2.48xlarge, TP8 | Median TTFT (ms) | 27.05 | 31.75 |
| trn2.48xlarge, TP8 | Average draft acceptance rate | n/a | 71.38% |
| trn2.48xlarge, TP8 | Mean acceptance length | n/a | 7.42 |
| trn2.48xlarge, TP8 | Per-position acceptance (positions 0-8) | n/a | 97.23% / 96.57% / 90.37% / 83.39% / 81.40% / 63.68% / 55.81% / 45.85% / 28.13% |

Measured with the Step 1 and Step 3 `vllm bench serve` commands
(`--sonnet-input-len 512 --sonnet-output-len 128 --num-prompts 50
--max-concurrency 1`). Prefix caching was disabled in both runs, and DFlash used
greedy on-device sampling. This configuration produced a 2.58x output
throughput speedup and 64.7% lower median TPOT.

## Common issues

- **Acceptance rate is 0% or near 0%.** Confirm that the target and draft
  implementations are accurate and that the draft checkpoint was trained
  against this exact target.
- **HTTP 403 when the server pulls the model.** Your Hugging Face token lacks
  read access to public gated repositories, or the account has not accepted the
  Llama 3.1 license. Update the token permissions, accept the license, and
  restart the server.
- **Throughput regressed with DFlash enabled.** The acceptance rate is not high
  enough to compensate for speculative decoding overhead. Check the
  speculative decoding metrics in Step 3 and benchmark a lower
  `num_speculative_tokens`.

## Clean up

Stop the vLLM server with `Ctrl+C`. If you launched an EC2 instance for this
tutorial, terminate it to avoid ongoing charges.

## Next steps

- [Features guide](../guides/features-guide.md) - Configure related serving
  features.
- [EAGLE3 speculative decoding tutorial](tutorial-eagle3-speculative-decoding-llama-3-1.md)
  - Compare DFlash with an autoregressive speculative drafter.
- [Quickstart: Offline serving](../getting-started/quickstart-offline-serving.md)
  - Learn more about the `vllm.LLM` Python API.
