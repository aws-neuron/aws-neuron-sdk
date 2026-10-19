# High-performance GPT-OSS 120B implementation for 8K/1K serving on Trn3

<!-- meta: description: The high-performance GPT-OSS 120B implementation for
8K/1K serving on Trn3, and how to run its disaggregated-inference
configuration: six prefill replicas, one TP8 DP16 EP128 decode server with the
decode megakernel, a proxy, and the benchmark client. -->
<!-- meta: keywords: vLLM, Neuron, gpt-oss-120b, megakernel, LNC1, disaggregated
inference, 6P1D, 8K/1K, benchmark, MXFP4, Trn3, Trainium -->
<!-- meta: date_updated: 2026-10-09 -->
<!-- Content type: procedural-tutorial -->

Starting with Neuron SDK 2.33, vLLM Neuron has a high-performance
implementation of GPT-OSS 120B for 8K-input, 1K-output serving on Trn3. The
implementation includes features such as LNC=1 decode and a decode megakernel,
among other optimizations. The megakernel fuses all decoder layers of a decode
step into one NKI kernel. The implementation runs as a disaggregated
deployment, with six prefill replicas and one decode server on two Trn3
instances. The [Overview](#overview) explains each part, and the steps
run the configuration with self-contained scripts.

**Known limitations:**

- The implementation supports Trn3 with MXFP4 weights only.
- The megakernel runs only on the decode server of a disaggregated deployment.
  A single-instance server and the prefill servers use the per-layer path at
  LNC=2.
- The tested shape is 8,192 input tokens and 1,024 output tokens. The servers
  use `--max-model-len 10240`, a block size of 128, and an FP8 KV cache.
- The decode server uses `--max-num-seqs 8` on each of its 16 data-parallel
  ranks. This gives 128 concurrent requests for each decode instance.

Use this tutorial for GPT-OSS 120B with MXFP4 weights on Trn3 at 8K/1K. For a
different case, use the [gpt-oss tutorial](tutorial-gpt-oss.md). It covers
gpt-oss 20B, Trn2 with BF16 weights, a single-instance deployment, and a
context of up to 16,384 tokens.

## Overview

The configuration is a disaggregated-inference (DI) deployment with six prefill
replicas and one decode server (6P1D). It runs on two Trn3 instances.

| Node | Role | Servers | Parallelism | Logical NeuronCores |
| --- | --- | --- | --- | --- |
| A | Prefill | 6 replicas on ports 8100 to 8105 | TP4 DP1 EP2 for each replica, at `NEURON_LOGICAL_NC_CONFIG=2` | 24 of 64 |
| A | Proxy and benchmark client | Proxy on port 8000 | | |
| B | Decode | 1 server on port 8200 | TP8 DP16 EP128, at `NEURON_LOGICAL_NC_CONFIG=1` | 128 of 128 |

:::{note}
The prefill instance of this deployment has idle NeuronCores. The six replicas
use 24 of its 64 logical NeuronCores. For production, keep six prefill replicas
for each decode instance, and fill each prefill instance with 16 replicas.
Three prefill instances and eight decode instances use every NeuronCore. That
deployment has 48 prefill replicas and 8 decode servers on 11 instances.
:::

The decode server differs from the GPT-OSS DI deployment in the
[gpt-oss tutorial](tutorial-gpt-oss.md) in these points:

| Part | What it does |
| --- | --- |
| LNC=1 decode | `NEURON_LOGICAL_NC_CONFIG=1` gives one logical NeuronCore for each physical core. The decode server runs 128 ranks, with one expert on each rank (EP128). |
| Decode megakernel | One NKI kernel runs all decoder layers of a decode step. vLLM Neuron selects it automatically for an MXFP4 decode server at LNC=1 on Trn3. No flag turns it on. |
| Packed FP8 KV cache | `--kv-cache-dtype fp8` with `fp8_packed_kv` on both roles, and a block size of 128. |
| Fused token embedding | The megakernel builds the token embedding (`embedding_dp_size: 1` and `VLLM_NEURON_MEGA_FUSE_EMBED=1`). |
| Fused sampling tail | The final RMSNorm, the lm_head, the top-k and the sampler run in one chunked kernel, with one 128-rank all-to-all (`gpt_oss_tail_hops: [128]`) and one row gather (`VLLM_NEURON_TAIL_FLAT_ROW_GATHER=1`). |
| DMA-order schedule | A schedule file sets the order of the DMA transfers inside the megakernel (`VLLM_NEURON_MEGA_DMA_ORDER_CONSTRAINTS_FILE`). |
| Host-side settings | Four batches in flight (`VLLM_NEURON_BATCH_QUEUE_DEPTH=4`), a backoff on the completion poll, and a delay on the NIXL progress thread. |

The scripts of this tutorial are in the vLLM Neuron repository, in
[`examples/vllm_neuron/models/gpt_oss/120b/mxfp4/multinode_DI_8k1k/`](https://github.com/vllm-project/vllm-neuron/tree/HEAD/examples/vllm_neuron/models/gpt_oss/120b/mxfp4/multinode_DI_8k1k).
They launch each part with the tested configuration. The directory is
self-contained: it carries the DMA-order schedule and the helper that prepares
it.

| File | Where to run | Purpose |
| --- | --- | --- |
| `env_common.sh` | Sourced by every script | The addresses, the shape, the ports, and the environment that both roles need |
| `run_decode.sh` | Node B | The megakernel decode server (`kv_consumer`) with its tuning |
| `run_prefill.sh <index>` | Node A | One prefill replica (`kv_producer`), index 0 to 5 |
| `run_prefill_fleet.sh` | Node A | All six replicas, one after the other |
| `run_proxy.sh` | Node A | The example proxy |
| `run_client.sh` | Node A | A health check and one completion through the proxy |
| `run_bench.sh` | Node A | The two benchmark legs: a warmup leg and the measured leg |
| `prepare_megakernel_schedule.py` | Called by `run_decode.sh` | Writes the schedule for the kernel-call scope of the active compile route |
| `gpt_oss_120b_megakernel_dma_order.json` | Read by the helper | The DMA-order schedule of the decode megakernel for Neuron SDK 2.33 |

## Before you start

This tutorial assumes that you have worked through:

- The [gpt-oss deployment tutorial](tutorial-gpt-oss.md). It explains the
  GPT-OSS DI deployment and its flags.
- The [disaggregated inference tutorial](tutorial-di-1p1d-xpyd.md). It
  explains the NIXL KV connector, the proxy, and the EFA setup.

## Tested versions

| Component | Version |
| --- | --- |
| Neuron SDK | 2.33 |
| vLLM Neuron plugin | Shipped with the Neuron SDK 2.33 release |
| vLLM (upstream) | 0.29.0 |
| gpt-oss 120B checkpoint | `openai/gpt-oss-120b` (Hugging Face) |

## Prerequisites

- **Two Trn3 instances with EFA connectivity.** Put them in the same subnet
  and placement group. The security group must allow this traffic between the
  two instances: port 8000 (proxy), ports 8100 to 8105 (prefill replicas), port
  8200 (decode), the NIXL side-channel ports 5559, 5659, 5759, 5859, 5959 and
  6059 (prefill) and 8759 (decode), and all EFA traffic (a self-referencing
  rule).
- **Neuron SDK 2.33 with vLLM Neuron on both instances.** See the
  [setup guide](../getting-started/setup-guide.md). Install `nixl` with its
  LIBFABRIC runtime libraries, as the DI tutorial describes.
- **The vLLM Neuron examples on both instances.** Clone the
  [vLLM Neuron repository](https://github.com/vllm-project/vllm-neuron), or
  copy its `examples/vllm_neuron/` directory. The scripts are in
  `examples/vllm_neuron/models/gpt_oss/120b/mxfp4/multinode_DI_8k1k/`, and the
  proxy is in `examples/vllm_neuron/vllm/disaggregated_inference/`.
- **The checkpoint on a local NVMe volume of both instances.** Each server
  loads the checkpoint at startup. A local NVMe volume gives a faster startup
  than a network volume or a download at startup. Download
  `openai/gpt-oss-120b` from Hugging Face to the same path on both instances,
  and set `MODEL_ID` to that path in step 1. Put the compile cache on local
  NVMe too, as the [setup guide](../getting-started/setup-guide.md) describes.

  ```bash
  hf download openai/gpt-oss-120b --local-dir /path/to/nvme/gpt-oss-120b
  ```

## Step 1: Set the addresses

In this step, you tell the scripts where the two nodes are.

On **both** nodes, export the private IP of each node:

```bash
export PREFILL_HOST=10.0.1.10                # node A
export DECODE_HOST=10.0.1.20                 # node B
export MODEL_ID=/path/to/nvme/gpt-oss-120b   # or openai/gpt-oss-120b
```

Replace the two addresses with the private IPv4 addresses of your instances.
Every script sources `env_common.sh`. That file reads these variables and
exports the environment that both roles need. Every other value has a default.
Open the file to see each value and the reason for it.

The logs of the servers go to `~/gpt-oss-120b-8k1k-results/logs/`, and the
benchmark results go to `~/gpt-oss-120b-8k1k-results/`. Set `BENCH_RESULT_DIR`
or `SERVER_LOG_DIR` to change the location.

## Step 2: Launch the decode server on node B

In this step, you start the one decode server. It takes every NeuronCore of the
node.

```bash
cd examples/vllm_neuron/models/gpt_oss/120b/mxfp4/multinode_DI_8k1k
./run_decode.sh
```

The script does three things before it calls `vllm serve`:

1. It sets `NEURON_LOGICAL_NC_CONFIG=1` and the decode-only environment.
2. It writes the DMA-order schedule for the decode graph into
   `MEGAKERNEL_SCHEDULE_DIR` (default
   `~/.cache/vllm-neuron/megakernel_schedule`), and it exports the path of the
   file. The path is part of the compile-cache key, so keep one directory.
3. It exports `VLLM_NEURON_MEGA_FUSE_EMBED=1` and
   `VLLM_NEURON_TAIL_FLAT_ROW_GATHER=1`.

The first launch compiles the decode graph. Later launches use the compile
cache. The server reports `Application startup complete` when it is ready. You
can start the prefill replicas (step 3) at the same time.

## Step 3: Launch the six prefill replicas on node A

In this step, you start six TP4 prefill servers. Replica `i` binds NeuronCores
`4i` to `4i+3`, serves HTTP on port `8100 + i`, and uses the NIXL side-channel
port `5559 + 100 * i`.

```bash
cd examples/vllm_neuron/models/gpt_oss/120b/mxfp4/multinode_DI_8k1k
./run_prefill_fleet.sh
```

The fleet script starts replica 0 and waits until it answers `/health`, so the
compile cache holds the prefill graph. Then it starts replicas 1 to 5 the same
way. Replica 0 compiles the prefill graph, so it takes the longest to start.
The later replicas use the compile cache. The replicas keep running after the
script exits. Their logs are in
`~/gpt-oss-120b-8k1k-results/logs/prefill_<i>.log`. To start one replica by
hand, run `./run_prefill.sh <index>`.

The six replicas are six separate TP4 DP1 EP2 servers, and not one DP6 server.
Expert parallelism cannot share data-parallel ranks on the prefill graph.

## Step 4: Launch the proxy on node A

In this step, you start the proxy and send one request through it. Start the
proxy after the decode server and all six replicas report
`Application startup complete`:

```bash
./run_proxy.sh &
./run_client.sh
```

The health check of the proxy (`/healthcheck`) returns `HTTP/1.1 200 OK`, and
the completion returns an OpenAI-compatible JSON payload with a coherent
`choices[0].text`.

:::{note}
The proxy is `toy_proxy_server.py` from the vLLM Neuron examples. It sends the
prompts to the prefill replicas in turn (round robin). It is a minimal example
router, and not a production router. For production, use a production router.
:::

## Step 5: Run the benchmark

In this step, you run two benchmark legs against the proxy. Both legs send
random prompts of 8,192 tokens, ask for 1,024 output tokens with
`--ignore-eos`, and keep 128 requests in flight. 128 is the decode capacity: 8
sequences on each of the 16 data-parallel ranks.

```bash
./run_bench.sh
```

| Leg | Prompts | Purpose |
| --- | --- | --- |
| Warmup | 1,280 | The first leg after a server launch is not a valid measurement. Discard it. |
| Measured | 12,800 | The result. The script prints the key metrics and keeps `bench_clean.json` in `BENCH_RESULT_DIR`. |

Before the measured leg sends its first request, the client generates the
12,800 random prompts of 8,192 tokens. That step takes several minutes, and the
servers are idle while it runs. Read these keys of `bench_clean.json`:

- `median_tpot_ms` and `p99_tpot_ms`: the time for each output token, as the
  client sees it. `1000 / median_tpot_ms` is the output tokens per second for
  each user.
- `output_throughput`: the output tokens per second over all 128 users.
- `total_token_throughput`: the input plus output tokens per second.
- `mean_ttft_ms` and `p99_ttft_ms`: the time to the first token. The prefill
  replicas and the KV transfer set it.

## Configuration reference

### Shape

These values are the tested configuration of the megakernel decode.

| Setting | Value | Reason |
| --- | --- | --- |
| `--max-model-len` | 10240 | Holds the 8,192 input tokens and the 1,024 output tokens. |
| `--max-num-batched-tokens` | 8192 | One prefill segment holds the whole prompt. Both roles use one bucket, `[8192]`, for the KV segment and for the batched tokens. |
| `--block-size` | 128 | The KV block size of the packed FP8 cache. |
| `--kv-cache-dtype fp8` and `fp8_packed_kv: true` | Both roles | The packed FP8 KV layout that the prefill writes and that the megakernel reads. |
| `--max-num-seqs` (decode) | 8 | The partition axis caps the per-rank decode batch at `128 * TP / EP = 8`. A larger value stops the trace with `Partition dim size must be preserved`. |
| `--max-num-seqs` (prefill) | 1 | One prompt for each prefill step. |
| `--data-parallel-size 16`, `--enable-expert-parallel`, `ep_degree: 128` (decode) | | 16 TP8 ranks give 128 ranks, with one expert on each rank. |
| `embedding_dp_size: 1` (decode) | | The megakernel builds the token embedding, so the embedding is not sharded over the DP group. |
| `lm_head_dp_size: 16` (decode) | | The lm_head is replicated over the 16 data-parallel ranks. |
| `gpt_oss_tail_hops: [128]` (decode) | | One 128-rank all-to-all in the fused tail. |
| `--optimization-level 2` (decode) | | The compiler optimization level of the decode graph. |
| `--max-logprobs 0` | Both roles | No logprob gather. |
| `--no-disable-hybrid-kv-cache-manager` | Both roles | Keeps the hybrid KV cache manager that the sliding-window layers need. |
| `--hf-overrides '{"quantization_config": {}}'` | Both roles | Lets vLLM load the published MXFP4 checkpoint. `quantization: "mxfp4"` in `neuron_config` selects the Neuron MXFP4 path. |
| `--distributed-executor-backend mp` | Both roles | The multiprocess executor. |

### Environment

Both roles, from `env_common.sh`:

| Variable | Value | Reason |
| --- | --- | --- |
| `VLLM_NEURON_BACKEND` | `neuron_native` | The compile route. libtorch-neuronx-lite reads this variable itself, and it selects its native-route setup only for an explicit value. |
| `NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND` | `1` | The native route needs the graph-capture backend off. |
| `TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE` | `1` | Turns off the vocab-sharding FX pass, so the per-rank vocab offset stays out of the graph. |
| `NEURON_RT_DBG_VECTOR_DGE_SKIP_NOTIF` | `1` | Packed-FP8 KV writes benign out-of-bound skip notifications. Without this setting, the notifications stop the decode server on a long run. |
| `VLLM_NEURON_NIXL_PROGRESS_THREAD_DELAY_US` | `250` | The NIXL progress thread polls with no delay by default and takes a core from the worker. |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | `0.0.0.0` | The side channel must accept the KV handshake from the other node. |
| `NEURON_LIBTORCH_COMPILATION_TIMEOUT` | `2400` | The limit in seconds for the compile of one graph. The default is 600, and the compile of one prefill graph takes several minutes. |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `14400` | The limit in seconds for the startup of one server. The default is 600, and the first prefill replica compiles its graph during the startup. |

Decode only, from `run_decode.sh`:

| Variable | Value | Reason |
| --- | --- | --- |
| `NEURON_LOGICAL_NC_CONFIG` | `1` | The megakernel needs one logical NeuronCore for each physical core. |
| `VLLM_NEURON_MEGA_COLLECTIVE_PRIVATE_SRC` | `1` | Keeps the source tensors of a collective off shared HBM. A toolchain that requires this stops the trace without it. |
| `VLLM_NEURON_BATCH_QUEUE_DEPTH` | `4` | Four batches in flight hide the host time of a step. |
| `NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION` | `0` | The completion poll backs off and gives host cores back to the workers. |
| `VLLM_NEURON_MEGA_DMA_ORDER_CONSTRAINTS_FILE` | The schedule path | The DMA-order schedule. When this variable is set, vLLM Neuron also sets the kernel options that the schedule was made with. |
| `VLLM_NEURON_MEGA_FUSE_EMBED` | `1` | The token embedding inside the megakernel. |
| `VLLM_NEURON_TAIL_FLAT_ROW_GATHER` | `1` | One row gather in the fused tail. |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `8759` | Past the last prefill side-channel port. |

Prefill only, from `run_prefill.sh`:

| Variable | Value | Reason |
| --- | --- | --- |
| `NEURON_LOGICAL_NC_CONFIG` | `2` | The prefill graph runs the per-layer path at LNC=2. |
| `NEURON_VISIBLE_DEVICES` | `4i-(4i+3)` | The four cores of replica `i`. |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `5559 + 100 * i` | One side-channel port for each replica. The same value goes into `nixl_side_channel_port` of the additional config. |

Script options:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ID` | `openai/gpt-oss-120b` | A Hugging Face model ID, or the local path of the checkpoint. |
| `BENCH_RESULT_DIR` | `~/gpt-oss-120b-8k1k-results` | The benchmark results. |
| `SERVER_LOG_DIR` | `$BENCH_RESULT_DIR/logs` | The logs of the servers and of the proxy. |
| `MEGAKERNEL_SCHEDULE` | The JSON file of this directory | The shipped DMA-order schedule. |
| `MEGAKERNEL_SCHEDULE_DIR` | `~/.cache/vllm-neuron/megakernel_schedule` | Where `run_decode.sh` writes the schedule copy. Keep one directory: its path is in the compile-cache key. |
| `MEGAKERNEL_SCOPE` | `auto` | The kernel-call scope of the schedule. `auto` gives 250 on the native route and 261 on the XLA route, the values of the Neuron SDK 2.33 release. |
| `PROXY_SCRIPT` | The proxy of the examples tree | The path of `toy_proxy_server.py`, if you copied the directory somewhere else. |
| `PROXY_BIND_HOST` | `0.0.0.0` | The address that the proxy listens on. |
| `PROXY_URL` | `http://localhost:8000` | The proxy address for `run_client.sh` and `run_bench.sh`. |
| `BENCH_NETRC` | `/dev/null` | The netrc file that the benchmark client reads before each request. Keep a local path. See [Common issues](#common-issues). |

## Common issues

### The benchmark reports a low throughput and a high TTFT, but the servers are healthy

The benchmark client can be the bottleneck. `vllm bench serve` opens its HTTP
session with `trust_env`, so the client looks up `~/.netrc` before it sends each
request. On a network file system, each lookup can take hundreds of
milliseconds. The client then sends requests late. The request throughput
decreases and the TTFT increases, but the TPOT and the server metrics stay
normal.

`run_bench.sh` points the lookup at `/dev/null` (`BENCH_NETRC`). If you run
`vllm bench serve` yourself, set `NETRC=/dev/null` on the command. You can also
run the client as a user whose `$HOME` is on a local disk.

### The decode TPOT is much larger than expected

Start the decode server with `run_decode.sh`. The script sets the megakernel
tuning. Each setting is off by default, and without them the server gives
correct output at a larger step time.

After a change to a kernel option or to the schedule file, clear the compile
caches on both nodes. The cache key does not include the content of the
schedule file or the kernel options.

```bash
rm -rf ~/.cache/neuron_libtorch ~/.cache/vllm/neuron /var/tmp/neuron-compile-cache /var/tmp/nki-*cache*
```

### The decode trace stops with `Partition dim size must be preserved`

`--max-num-seqs` on the decode server is above 8. Keep 8 for TP8 and EP128.

### The decode server stops with `At LNC=1 a collective's src tensors must not be on shared_hbm`

`VLLM_NEURON_MEGA_COLLECTIVE_PRIVATE_SRC=1` did not reach the server. Run the
server through `run_decode.sh`.

### A server stops during compilation

A compile or a startup took longer than its limit. Raise
`NEURON_LIBTORCH_COMPILATION_TIMEOUT`, the limit for one graph, and
`VLLM_ENGINE_READY_TIMEOUT_S`, the limit for the startup of one server, in
`env_common.sh`.

### Requests stall after the prefill, or the NIXL handshake fails

Check `VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0` on both roles. Check the
side-channel ports in the security group: 5559 to 6059 in steps of 100 for the
replicas, and 8759 for the decode server. Check the EFA rule. Enable the NIXL
logs as the [disaggregated inference tutorial](tutorial-di-1p1d-xpyd.md)
describes.

### The decode server stops on a long run with many out-of-bound skip notifications

`NEURON_RT_DBG_VECTOR_DGE_SKIP_NOTIF=1` is absent on one role. `env_common.sh`
sets it for both roles.

## Clean up

Stop each server with SIGINT, so it shuts down in order:

```bash
pkill -INT -f "vllm serve"       # on both nodes
pkill -f toy_proxy_server        # on node A
```

## Related information

- [gpt-oss model recipe](../model-recipes/gpt-oss.md): the supported
  checkpoints, the features, and the accuracy results.
- [gpt-oss deployment tutorial](tutorial-gpt-oss.md): the single-instance and
  the 1P1D deployments, and the GPT-OSS flags.
- [Disaggregated inference tutorial](tutorial-di-1p1d-xpyd.md): the NIXL KV
  connector, the proxy, and the xPyD scaling.
- [Disaggregated inference design](../design/vllm/disaggregated-inference.md):
  how the DI control and data flow are implemented on Neuron.
- [Features guide](../guides/features-guide.md): the feature configuration for
  GPT-OSS and other models.
- [Profiling workloads](../guides/how-to-profile-workloads.md): how to profile
  a vLLM Neuron server.
