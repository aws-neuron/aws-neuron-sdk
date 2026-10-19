# Tutorial: Deploy dense Qwen3-32B with vLLM Neuron

<!-- meta: description: Prepare BF16 and MXFP8 checkpoints and validate dense Qwen3-32B serving on Trainium with vLLM Neuron. -->
<!-- meta: keywords: Qwen3-32B, dense, BF16, MXFP8, Trainium, Trn2, Trn3, Neuron 2.33 -->
<!-- meta: date_updated: 2026-10-08 -->
<!-- Content type: procedural-tutorial -->

This tutorial prepares a dense Qwen3-32B checkpoint, launches an
OpenAI-compatible server, and checks its responses. BF16 targets Trn2 and Trn3;
native MXFP8 targets Trn3.

## Overview

Start with a TP8 single server using BF16 weights and BF16 KV. On Trn3, convert the
same pinned BF16 source to MXFP8 and repeat the smoke procedure. MXFP8 is validated
with both BF16 KV and packed FP8 KV. The DI deployment uses two
Trn3 nodes, packed FP8 KV, TP4/DP16/LNC2 prefill, and TP8/DP16/LNC1 decode.
Check its responses through the proxy.

For hardware, checkpoint formats, and KV-cache restrictions, see the
[model recipe](../model-recipes/qwen3-32b.md). This tutorial uses dense Qwen3-32B.

## Prerequisites

- A Trn2 instance for BF16, or Trn3 for BF16 or MXFP8, with at least eight
  available logical NeuronCores at LNC2.
- For DI, two Trn3 nodes with NIXL/LIBFABRIC connectivity between them.
- A working environment from the [setup guide](../getting-started/setup-guide.md)
  and familiarity with the [online serving quickstart](../getting-started/quickstart-online-serving.md).
- A checkout matching the plugin in the chosen image, including
  `examples/vllm_neuron/models/qwen3/`.
- Access to `Qwen/Qwen3-32B` and local storage for the source and converted
  checkpoints. Conversion also needs host memory for the source model.
- For conversion, `llmcompressor`, `transformers`, and `safetensors` in a separate
  conversion environment. Freeze their versions with the converter commit.

## Step 1: Record the environment and prepare checkpoints

Use a Neuron 2.33 serving environment from the setup guide.
Record its image digest and the installed versions before deployment:

```bash
git rev-parse HEAD
python -m pip show vllm vllm-neuron torch neuronx-cc
neuron-ls
```

Download an immutable source revision. Replace `BF16_REVISION` with the agreed
Hugging Face commit SHA before running:

```bash
export BF16_REVISION=REPLACE_WITH_CHECKPOINT_COMMIT
export BF16_MODEL=/models/Qwen3-32B-bf16
export MXFP8_MODEL=/models/Qwen3-32B-mxfp8
hf download Qwen/Qwen3-32B --revision "$BF16_REVISION" --local-dir "$BF16_MODEL"
```

For BF16 serving, proceed to Step 2. For MXFP8, run the converter in the
conversion environment, using the downloaded directory so the source revision
cannot change during conversion:

```bash
python examples/vllm_neuron/models/qwen3/build_mxfp8_checkpoint.py \
    --model-checkpoint "$BF16_MODEL" \
    --output-dir "$MXFP8_MODEL" \
    --torch-dtype bfloat16 \
    --ignore-modules lm_head
python -m pip freeze > /models/qwen3-conversion-requirements.txt
```

Record the converter's Git commit and complete command. Hash every checkpoint
file, including configuration and tokenizer files, after conversion:

```bash
python - <<'PY_HASH'
import hashlib
from pathlib import Path
import os
for name in ("BF16_MODEL", "MXFP8_MODEL"):
    root = Path(os.environ[name])
    if not root.is_dir():
        continue
    output = root.parent / (root.name + ".sha256")
    with output.open("w") as manifest:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or ".cache" in path.relative_to(root).parts:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            manifest.write(f"{digest.hexdigest()}  {path.relative_to(root)}\n")
PY_HASH
```

Copy the complete checkpoint directory to the serving environment and verify
its checksums there. MXFP8 uses the compressed-tensors checkpoint produced by
this builder; a similarly named repository is insufficient provenance.

## Step 2: Launch a single server

Run one weight format at a time. Select the backend being validated explicitly:

```bash
export VLLM_NEURON_BACKEND=neuron_native
export NEURON_LOGICAL_NC_CONFIG=2
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export NEURON_LIBTORCH_COMPILATION_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export MODEL_PATH="$BF16_MODEL"
```

Use `neuron_native` for the baseline. Set `VLLM_NEURON_BACKEND=vllm_neuron` to
select the XLA backend. Repeat the response checks after a backend change.
The BF16 baseline uses BF16 KV. Prefix caching is on by default and needs
segmented prefill, so the maximum model length (10,240) must exceed the
8192-token batch, and `kv_segment_size_buckets` sets the segment size. The
repository examples use the same initial smoke approach.

```bash
vllm serve "$MODEL_PATH" \
    --served-model-name qwen3-32b \
    --dtype bfloat16 \
    --optimization-level 2 \
    --tensor-parallel-size 8 \
    --max-model-len 10240 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --additional-config '{
        "neuron_config": {
            "on_device_sampling_config": {},
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
```

For MXFP8 on Trn3, set `MODEL_PATH="$MXFP8_MODEL"`. MXFP8 is validated with both
BF16 KV and packed FP8 KV. Set `"quantization": "mxfp8"` in both cases; platform
validation rejects the compressed-tensors checkpoint without it.

For BF16 KV, use the Step 2 command with this additional configuration:

```json
{
    "neuron_config": {
        "quantization": "mxfp8",
        "on_device_sampling_config": {},
        "kv_segment_size_buckets": [8192],
        "num_batched_tokens_buckets": [8192],
        "num_seqs_buckets": [4]
    }
}
```

For packed FP8 KV, also add `--kv-cache-dtype fp8 --block-size 128` and
`"fp8_packed_kv": true`:

```json
{
    "neuron_config": {
        "quantization": "mxfp8",
        "fp8_packed_kv": true,
        "on_device_sampling_config": {},
        "kv_segment_size_buckets": [8192],
        "num_batched_tokens_buckets": [8192],
        "num_seqs_buckets": [4]
    }
}
```

Keep the server's stdout and stderr in the validation record. Cold launch
compiles model graphs; wait for startup completion before sending traffic.
The batch limit and compiled sequence bucket must agree. The maximum batched
tokens must match the largest compiled token bucket.

## Step 3: Validate responses and recovery

In another shell, check health and a deterministic completion:

```bash
curl --fail-with-body http://localhost:8000/health
curl --fail-with-body http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-32b","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

Require HTTP 200 and a coherent `choices[0].text` that gives Paris as the answer.
Write the prompt without a trailing space. A trailing space makes the model
continue with an unrelated sentence. Preserve the request and
response. A qualitative smoke response does not replace logit or task accuracy.

Send concurrent requests, checking every command's exit status and response:

```bash
python - <<'PY_TRAFFIC'
from concurrent.futures import ThreadPoolExecutor
import json
from urllib.request import Request, urlopen

def complete(index):
    body = json.dumps({"model": "qwen3-32b", "prompt": "Count: 1, 2, 3,",
                       "max_tokens": 32, "temperature": 0}).encode()
    request = Request("http://localhost:8000/v1/completions", data=body,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=1200) as response:
        result = json.load(response)
    assert result["choices"][0]["text"], result
    return index, result

with ThreadPoolExecutor(max_workers=4) as pool:
    for index, result in pool.map(complete, range(16)):
        print(json.dumps({"request": index, "response": result}))
PY_TRAFFIC
```

Stop the server normally, restart with the same image, checkpoint, and command,
and repeat health and completion checks. Record cold and warm startup time and
any unexpected OOM, compile timeout, or runtime error.

### Prefix caching and long context

Validate prefix caching and long context as a separate configuration. Use the
single-server command and additional configuration from Step 2 for your weight
format and KV format, and change only `--max-model-len` to `32768`. Keep prefix
caching on: do not add `--no-enable-prefix-caching`, which only the DI command in
Step 4 uses.

Send the same long prefix twice, verify output consistency, and inspect
prefix-cache metrics. Use tokenizer-measured
prompts beyond 8192 tokens to exercise segmented prefill and near the configured
context limit while leaving room for generation. Record token counts, TTFT,
and errors. Do not infer 32k validation from the 10,240-token baseline smoke.

LongBench-v2 at 32k is validated on a single TP8 server at batch 8 with BF16 KV,
for both BF16 and MXFP8 weights. That configuration uses `--max-model-len 32768`,
`--max-num-batched-tokens 8192`, `--max-num-seqs 8`, and `"num_seqs_buckets": [8]`,
with prefix caching on by default. See the
[model recipe](../model-recipes/qwen3-32b.md#accuracy-evaluation) for its acceptance floor.

## Step 4: Deploy disaggregated inference

The MXFP8 DI configuration runs 1P1D on two Trn3 nodes, and each role uses a whole node:

| Setting | Prefill node | Decode node |
|---|---|---|
| Parallelism | TP4 / DP16 / LNC2 | TP8 / DP16 / LNC1 |
| Optimization level | 2 | 3 |
| Maximum batch per replica | 1 | 64 |
| API servers | 16 | 16 |
| KV cache | Packed FP8, block size 128 | Packed FP8, block size 128, 5232 blocks |

Both roles use a 10,240-token context, an 8192-token batched-token bucket, async scheduling,
and a 300 s KV lease. The DI configuration is validated with prefix caching off.
Follow the [DI tutorial](tutorial-di-1p1d-xpyd.md) for NIXL, LIBFABRIC, network
setup, and proxy operation. Use the same image and checkpoint
on both nodes.

> **Note:** This DI configuration supports a maximum model length of up to 10,240 tokens.
> For 32k context, including LongBench evaluation, use the validated single-server
> configuration in [Prefix caching and long context](#prefix-caching-and-long-context).

### Launch the target roles and proxy

Open a shell in the same serving image on each node. In both shells, set `MXFP8_MODEL`,
and set `PREFILL_IP` and `DECODE_IP` to the node addresses reachable by NIXL/LIBFABRIC.
Follow the network checks in the DI tutorial.

On the prefill node:

```bash
export PREFILL_IP=REPLACE_WITH_PREFILL_NODE_IP
export DECODE_IP=REPLACE_WITH_DECODE_NODE_IP
export NEURON_LOGICAL_NC_CONFIG=2
export DI_TP=4
export DI_OPT_LEVEL=2
export DI_MAX_SEQS=1
DI_KV_ARGS=()
export DI_ROLE_CONFIG='"kv_segment_size_buckets": [8192],'
export VLLM_NIXL_SIDE_CHANNEL_PORT=5559
export DI_ROLE=kv_producer
export DI_KV_RANK=0
export DI_ENGINE_ID=qwen3-prefill
export DI_PORT=8100
```

On the decode node:

```bash
export PREFILL_IP=REPLACE_WITH_PREFILL_NODE_IP
export DECODE_IP=REPLACE_WITH_DECODE_NODE_IP
export NEURON_LOGICAL_NC_CONFIG=1
export DI_TP=8
export DI_OPT_LEVEL=3
export DI_MAX_SEQS=64
DI_KV_ARGS=(--num-gpu-blocks-override 5232)
export NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP=0
export VLLM_NEURON_ATTN_MASK_TORCH=1
export VLLM_NEURON_ATTN_PV_SWAP=1
export VLLM_NIXL_SIDE_CHANNEL_PORT=5659
export DI_ROLE=kv_consumer
export DI_KV_RANK=1
export DI_ENGINE_ID=qwen3-decode
export DI_PORT=8200
```

Then run this command on **each** node. Each role sets its own LNC.
Sixteen data-parallel replicas fill the node: 64 logical cores at LNC2 for prefill
and 128 at LNC1 for decode. Each replica of the decode node admits up to 64 sequences,
so the node holds 1,024 concurrent requests.

```bash
export VLLM_NEURON_BACKEND=neuron_native
export VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0
export VLLM_NEURON_NIXL_PROGRESS_THREAD_DELAY_US=200
export NEURON_RT_EXEC_TIMEOUT=600
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_NEURON_COMPILATION_TIMEOUT=1800
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
vllm serve "$MXFP8_MODEL" \
    --served-model-name qwen3-32b \
    --dtype bfloat16 \
    --optimization-level "$DI_OPT_LEVEL" \
    --tensor-parallel-size "$DI_TP" \
    --data-parallel-size 16 \
    --api-server-count 16 \
    --kv-cache-dtype fp8 \
    --block-size 128 \
    "${DI_KV_ARGS[@]}" \
    --max-model-len 10240 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs "$DI_MAX_SEQS" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --port "$DI_PORT" \
    --kv-transfer-config "{
        \"kv_connector\": \"NixlConnector\",
        \"kv_role\": \"$DI_ROLE\",
        \"kv_buffer_device\": \"cuda\",
        \"engine_id\": \"$DI_ENGINE_ID\",
        \"kv_ip\": \"$PREFILL_IP\",
        \"kv_rank\": $DI_KV_RANK,
        \"kv_parallel_size\": 2,
        \"kv_connector_extra_config\": {
            \"backends\": [\"LIBFABRIC\"],
            \"kv_lease_duration\": 300,
            \"prefill_ips\": [\"$PREFILL_IP\"],
            \"decode_ips\": [\"$DECODE_IP\"]
        }
    }" \
    --additional-config "{
        \"neuron_config\": {
            $DI_ROLE_CONFIG
            \"quantization\": \"mxfp8\",
            \"fp8_packed_kv\": true,
            \"on_device_sampling_config\": {},
            \"num_batched_tokens_buckets\": [8192],
            \"num_seqs_buckets\": [$DI_MAX_SEQS]
        }
    }"
```

Each server log must report `Started 16 API server processes`. During traffic,
the log must not contain `Releasing expired KV blocks`.

Stop the single-server process on port 8000 first. Once both roles report startup
complete, launch the proxy in a second shell on the prefill node:

```bash
curl --fail-with-body "http://$PREFILL_IP:8100/health"
curl --fail-with-body "http://$DECODE_IP:8200/health"
python examples/vllm_neuron/vllm/disaggregated_inference/toy_proxy_server.py \
    --port 8000 \
    --prefiller-hosts "$PREFILL_IP" --prefiller-ports 8100 \
    --decoder-hosts "$DECODE_IP" --decoder-ports 8200
```

Repeat Step 3's completion request through port 8000 and retain transfer logs
from both roles. The toy proxy is a validation example; use the DI tutorial to configure a serving deployment.

### KV cache size

The decode engine pins 5232 KV blocks to avoid automatic cache sizing on shared LNC1 HBM.
The allocation is `(10240 / 128) × (64 + 1) + 32` blocks.
It provides a 10,240-token context for the 64 sequences of each replica, plus headroom.
Keep the QK-swap environment settings on the decode role.

The [model recipe](../model-recipes/qwen3-32b.md#accuracy-evaluation) includes
logit and GSM8K results for this mixed-LNC configuration.
Use the same role settings when comparing output or performance.

## Add EAGLE3 speculative decoding

For single-server BF16 or MXFP8 serving, use the matching
`RedHatAI/Qwen3-32B-speculator.eagle3` draft. Pin its revision with the target checkpoint.
Add this option to the Step 2 command for the chosen weight format:

```bash
--speculative-config '{
    "method": "eagle3",
    "model": "RedHatAI/Qwen3-32B-speculator.eagle3",
    "num_speculative_tokens": 5
}'
```

> **Note:** EAGLE3 supports a batch size of up to 8 for an MXFP8 target on Trn3 and
> up to 4 for a BF16 target on Trn2, with TP8 decode at LNC2. Keep `--max-num-seqs`
> and `num_seqs_buckets` within these limits.

EAGLE3 needs more device memory outside the KV cache than the default headroom
gives. Set the headroom before you launch the server:

```bash
# MXFP8 target on Trn3
export VLLM_NEURON_RESERVED_HEADROOM_GIB=17.2
# BF16 target on Trn2
export VLLM_NEURON_RESERVED_HEADROOM_GIB=9.4
```

With the default headroom of 4 GiB, an MXFP8 target on Trn3 starts and serves
one request. The first concurrent batch then fails with a device memory
allocation error, and the engine stops. Each additional GiB of headroom
removes 1 GiB from the KV cache budget.

| Target | Hardware | Validated configuration |
|---|---|---|
| MXFP8, packed FP8 KV | Trn3 | TP8. Batch 8, 32k context, 8192-token bucket, `num_speculative_tokens` 5: headroom 17.2 GiB. Batch 1, 10,240-token context, `num_speculative_tokens` 3, 5, and 7: headroom 18.1 GiB |
| BF16, BF16 KV | Trn2 | TP8, 10,240-token context, headroom 9.4 GiB. Batch 4, `num_speculative_tokens` 3 and 5. Batch 1, `num_speculative_tokens` 3, 5, and 7 |

Each headroom value is validated only for its configuration.

For an MXFP8 target with packed FP8 KV, keep `--kv-cache-dtype fp8 --block-size 128`
from Step 2. Use `num_speculative_tokens` 5. Do not use 3 at batch 4: it gives
incorrect target logits on this path.

This example uses TP8. A BF16 target uses BF16 KV. With EAGLE3, the MXFP8 target
is validated with packed FP8 KV only. The target and draft compile during startup.
Repeat the completion and concurrency checks. Read the proposed and accepted
draft-token counters from `/metrics`:

```bash
curl --fail-with-body http://localhost:8000/metrics | \
    grep -E '^vllm:spec_decode_num_(draft|accepted)_tokens_total'
```

Use counter deltas for the workload. Compute acceptance rate as accepted draft
tokens divided by proposed draft tokens. Accepted draft tokens show that the
speculative path is active. The mean accepted length is
`1 + accepted / (proposed / num_speculative_tokens)`. Over a long generation
(for example GSM8K), a value below 1.5 shows that speculation does not work.
Short outputs give lower values. Compare throughput and latency with the same prompts,
concurrency, and non-speculative command. Keep the target's accuracy checks in the comparison.
The mixed-LNC DI procedure above uses standard decoding.

## Step 5: Benchmark the validated workload

Benchmark the same image, checkpoint, flags, and endpoint used for accuracy and
smoke validation. Start with the [GPT-OSS benchmarking procedure](tutorial-gpt-oss.md)
and the repository's Qwen3 performance targets. Record input/output token counts,
request distribution, concurrency, per-replica batch, warmup, measurement duration,
TP/DP, LNC, KV dtype, optimization level, and context buckets.

Report throughput, TTFT, inter-token latency, cold/warm startup, and memory use.
For DI, benchmark through the proxy. For example, a workload that fills the 16 decode
replicas at batch 64:

```bash
vllm bench serve \
    --base-url http://localhost:8000 \
    --model qwen3-32b \
    --tokenizer "$MXFP8_MODEL" \
    --dataset-name random \
    --seed 42 \
    --random-input-len 8192 \
    --random-output-len 1024 \
    --random-range-ratio 0 \
    --num-prompts 51200 \
    --max-concurrency 1024 \
    --num-warmups 1024 \
    --temperature 0 \
    --percentile-metrics ttft,tpot,itl \
    --ignore-eos
```

The concurrency of 1,024 fills the 16 decode replicas at batch 64. The time to fill and
drain the replicas lowers the reported throughput of short runs, so send many prompts
per unit of concurrency. Compare runs only at the same prompt count.

Compare identical workloads against the
baseline and verify batch admission before reporting a high-batch result.
Measured performance alone does not establish model accuracy.

## Confirm your work

You have completed this tutorial. For each weight format you deployed, you see the following:

- `GET /health` returns HTTP 200, and the completion request returns Paris as the
  answer, both on a cold start and after a restart with the same image, checkpoint,
  and command.
- The 16 concurrent requests all return a non-empty `choices[0].text`, with no OOM,
  compile timeout, or runtime error in the server log.
- For DI, both role servers report `Started 16 API server processes`, the completion
  request succeeds through the proxy on port 8000, and the logs contain no
  `Releasing expired KV blocks`.
- For EAGLE3, `vllm:spec_decode_num_accepted_tokens_total` increases under traffic,
  and the server stays up through the concurrent batch.

Keep the image digest, checkpoint record, and exact commands with these results.
For the logit and task-score accuracy methods, see the
[model recipe](../model-recipes/qwen3-32b.md#accuracy-evaluation).

## Common issues

| Symptom | Action |
|---|---|
| Compile timeout | Check the compile logs, timeout environment variables, and cache setup in the setup guide. Record cold and warm startup separately. |
| OOM or incomplete batch admission | Reduce context or concurrency for diagnosis, keep buckets aligned, and report the configuration that actually ran. Revalidate the intended release point before claiming it. |
| MXFP8 fails to load | Verify compressed-tensors metadata, builder commit, conversion options, checksums, and Trn3 hardware. Keep `lm_head` unquantized. |
| DI health succeeds but proxy requests fail | Check side-channel addresses, NIXL/LIBFABRIC connectivity, core allocation, matching checkpoints, and producer/consumer transfer logs. |
| EAGLE3 compile or numerical failure | Use the baseline non-speculative command while investigating; repeat compile and accuracy checks for the EAGLE3 configuration. |
| EAGLE3 server returns HTTP 500 after the first concurrent batch, and the log shows `Failed to allocate` | Set `VLLM_NEURON_RESERVED_HEADROOM_GIB` to the value for your target. See the EAGLE3 section. |

## Related information

- [Qwen3-32B model recipe](../model-recipes/qwen3-32b.md)
- [Accuracy debugging guide](../model-dev/accuracy-debugging-guide.md)
