# Tutorial: Configure disaggregated inference with 1P1D and xPyD

<!-- meta: description: Configure disaggregated inference with vLLM Neuron
using simple 1P1D and xPyD examples on AWS Trainium and Inferentia. -->
<!-- meta: keywords: disaggregated inference, 1P1D, xPyD, prefill, decode,
KV cache, NIXL, vLLM, Neuron, Trainium, Inferentia -->
<!-- meta: date_updated: 2026-08-12 -->
<!-- Content type: procedural-tutorial -->
<!-- Jira: NDOC-188 -->

This topic guides you through configuring disaggregated inference (DI) with vLLM
on Neuron using a simple 1P1D (one prefill, one decode) example and scaling up to
a general xPyD topology. When you have completed it, you will have a working DI
deployment and will understand the control and data flow between prefill and
decode servers.

## Overview

Disaggregated inference separates the prefill phase (prompt processing) from the
decode phase (token generation) across different servers. This separation lets
you:

- Scale prefill and decode independently based on your traffic mix.
- Use different parallelism configurations for each phase (for example, TP=4 for
  prefill, DP4×TP8 with expert parallelism for decode).
- Reduce head-of-line blocking where long prompts delay short decode requests.

The architecture has three components:

1. **Prefill server** — runs the model forward pass over the prompt and produces
   the KV cache.
2. **Decode server** — pulls the KV cache from the prefill server and generates
   tokens iteratively.
3. **Proxy server** — routes client requests to prefill, then hands off to decode
   for token generation.

KV cache transfer between prefill and decode uses
[NIXL](https://github.com/ai-dynamo/nixl) over LIBFABRIC (EFA on AWS). This
tutorial uses the default **read mode**, in which the decode server pulls KV
blocks directly from the prefill server's device memory via a NIXL RDMA READ.

**Request flow (read mode):**

1. Client sends request to the proxy server.
2. Proxy dispatches the request to a prefill server with `max_tokens=1` and
   blocks on the response.
3. Prefill server processes the prompt, produces KV cache in device memory, and
   returns the KV transfer parameters. The single token it samples is discarded.
4. Proxy dispatches the request to a decode server, attaching those KV transfer
   parameters.
5. The decode server pulls the KV cache from prefill via a NIXL RDMA READ over
   LIBFABRIC, waiting until the transfer completes.
6. Decode server generates all output tokens (including the first) and streams
   them back through the proxy to the client.

For more detail on the architecture, transfer modes (read vs. write), and the
vLLM/Neuron integration points, see the
[Disaggregated inference design document](../design/vllm/disaggregated-inference.md).

## Before you start

This tutorial assumes that you have experience in the following areas:

- Running vLLM Neuron on a single instance. See
  [online serving quickstart](../getting-started/quickstart-online-serving.md).
- Familiarity with KV cache and its role in LLM inference.
- Understanding of tensor parallelism and data parallelism concepts.

## Prerequisites

- **Neuron instance**: DI is supported on Trn2 and Trn3. For single-node DI, use
  an instance with enough free NeuronCores for both engines. For multi-node DI,
  use two or more instances with EFA connectivity.
- **vLLM Neuron environment**: Installed and verified. See
  [setup guide](../getting-started/setup-guide.md).
- **NIXL and its runtime libraries**: The NIXL KV transfer library. Install with:

  ```bash
  pip install nixl
  ```

  NIXL's LIBFABRIC backend also loads the `libcuda.so.1` and `libfabric.so.1`
  shared libraries at runtime. If your environment does not already ship them
  (some container images do not), follow
  [Install dependencies for disaggregated inference](../getting-started/setup-guide.md#install-dependencies-for-disaggregated-inference)
  before you start. Otherwise the servers fail at startup with
  `unsupported backend 'LIBFABRIC'`.

- **Model access**: Ability to pull the model on each server (for example, via
  Hugging Face Hub or a shared filesystem).

:::{note}
The GPT-OSS commands in this tutorial run on Trn3 with MXFP4 weights.
GPT-OSS DI also runs on Trn2 with BF16 weights; see the
[GPT-OSS model recipe](../model-recipes/gpt-oss.md) for platform-specific model
support.
:::

## Prepare your environment

Run these commands in every terminal that will launch a prefill or decode
server. Each server runs in its own foreground process, and exports do not carry
between shells.

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_*/bin/activate
export VLLM_NIXL_SIDE_CHANNEL_HOST="0.0.0.0"
export FI_EFA_ENABLE_SHM_TRANSFER=0

# GPT-OSS compilation and startup can exceed the vLLM defaults.
export VLLM_NEURON_COMPILATION_TIMEOUT=2400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_READY_TIMEOUT_S=1800
```

The activation command assumes that only one matching vLLM environment is
installed. If multiple versions are present, replace the wildcard with the exact
environment path for your installed release.

`FI_EFA_ENABLE_SHM_TRANSFER=0` is required for single-node DI. It prevents
Libfabric from selecting its shared-memory transport for the on-device KV cache
transfer between the colocated engines. Without it, both servers can start
successfully but requests can produce inaccurate output.

Set the model for the single-node example:

```bash
MODEL="openai/gpt-oss-20b"
```

:::{note}
The `NEURON_VISIBLE_DEVICES` ranges in this tutorial assume direct host
execution and host-level NeuronCore numbering. With EKS and Kubernetes Dynamic
Resource Allocation (DRA), a pod can expose its allocation under different,
pod-local NeuronCore IDs. Run `neuron-ls` inside each prefill and decode pod,
then adjust `NEURON_VISIBLE_DEVICES` to the IDs visible in that pod instead of
copying the host ranges verbatim. Ensure the DRA allocations give each engine
enough NeuronCores and keep the prefill and decode allocations disjoint.
:::

## Step 1: Launch the prefill server

In this step, you will start the vLLM server in the `kv_producer` role. This
server handles prompt processing and produces the KV cache that the decode server
will consume.

```bash
NEURON_VISIBLE_DEVICES=0-3 VLLM_NIXL_SIDE_CHANNEL_PORT=5559 vllm serve "$MODEL" \
    --port 8100 \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --max-num-seqs 1 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --no-disable-hybrid-kv-cache-manager \
    --hf-overrides '{"quantization_config": {}}' \
    --kv-transfer-config '{"kv_connector": "NeuronNixlConnector", "kv_role": "kv_producer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "mxfp4",
            "ep_degree": 2,
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [1]
        }
    }'
```

Key parameters:

- `NEURON_VISIBLE_DEVICES=0-3` — exposes exactly four host-level NeuronCores to
  the TP4 prefill process. The collective world size is determined by TP, so a
  wider visible range is not required.
- `VLLM_NIXL_SIDE_CHANNEL_PORT=5559` — the port NIXL uses for metadata exchange
  between servers.
- `--kv-transfer-config` — configures the NIXL connector with
  `kv_role: "kv_producer"` (this server produces KV cache).
- `NeuronNixlConnector` — required for GPT-OSS because the generic
  `NixlConnector` does not include Neuron's sliding-window transfer alignment.
- `--tensor-parallel-size 4` and `ep_degree: 2` — use the tested GPT-OSS
  prefill sharding.
- `--port 8100` — the prefill server's API port.

Wait for the server to print `Application startup complete`.

## Step 2: Launch the decode server

In this step, you will start the decode server in the `kv_consumer` role. This
server pulls KV cache from the prefill server and generates tokens.

The decode server can use a different parallelism configuration. This example
uses DP=4 with TP=8 and expert parallelism on NeuronCores 32–63:

```bash
NEURON_VISIBLE_DEVICES=32-63 VLLM_NIXL_SIDE_CHANNEL_PORT=5659 vllm serve "$MODEL" \
    --port 8200 \
    --tensor-parallel-size 8 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --optimization-level 2 \
    --dtype bfloat16 \
    --max-num-seqs 4 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-logprobs 0 \
    --no-disable-hybrid-kv-cache-manager \
    --hf-overrides '{"quantization_config": {}}' \
    --kv-transfer-config '{"kv_connector": "NeuronNixlConnector", "kv_role": "kv_consumer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "mxfp4",
            "embedding_dp_size": 4,
            "lm_head_dp_size": 4,
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
```

Key parameters:

- `NEURON_VISIBLE_DEVICES=32-63` — uses the second half of the instance (32
  NeuronCores for DP4×TP8).
- `VLLM_NIXL_SIDE_CHANNEL_PORT=5659` — a different port than the prefill server to
  avoid conflicts on the same host.
- `kv_role: "kv_consumer"` — this server consumes KV cache from the producer.
- `--data-parallel-size 4 --enable-expert-parallel` — with expert parallelism
  enabled, the decode EP degree is derived from TP x DP. This configuration uses
  TP8 x DP4 = EP32, so `ep_degree` does not need to be set separately.
- `--optimization-level 2` — uses the compiler optimization level validated by
  the GPT-OSS MXFP4 decode workload.
- `embedding_dp_size: 4` and `lm_head_dp_size: 4` — match the embedding and
  LM-head replication to the decode data-parallel degree.

Wait for the server to print `Application startup complete`.

:::{note}
The prefill and decode servers do not need identical parallelism configurations.
This example uses attention TP4 for prefill and DP4×TP8 for decode. This is
called **hybrid TP** and is a key advantage of disaggregated inference. These
settings match the GPT-OSS 20B configuration in the
[GPT-OSS tutorial](tutorial-gpt-oss.md).
:::

## Step 3: Launch the proxy server

In this step, you will start the proxy server that routes client requests between
prefill and decode.

```bash
python3 examples/vllm_neuron/vllm/disaggregated_inference/toy_proxy_server.py \
    --port 8000 \
    --prefiller-ports 8100 \
    --decoder-ports 8200
```

The proxy listens on port 8000 and coordinates the request lifecycle: it sends
prompts to the prefill server, then hands off to the decode server for token
generation.

:::{note}
The `toy_proxy_server.py` is included in the vLLM Neuron repository under
`examples/vllm_neuron/vllm/disaggregated_inference/`. For production deployments,
consider using an orchestrator for routing and autoscaling.
:::

## Step 4: Validate the 1P1D deployment

In this step, you will send a request through the proxy and confirm it completes
end-to-end.

```bash
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"prompt\": \"Count the number 1, 2, 3\",
      \"max_tokens\": 200
    }"
```

A successful JSON response with generated text confirms:

1. The proxy routed the request to the prefill server (port 8100).
2. The prefill server processed the prompt and produced KV cache.
3. The decode server (port 8200) pulled the KV cache via NIXL over LIBFABRIC.
4. The decode server generated tokens and returned them through the proxy.

## Step 5: Scale to xPyD (multi-node)

In this step, you will generalize the topology to multiple prefill and decode
nodes across separate instances.

For a multi-node deployment, the configuration changes are:

1. Each server runs on its own instance, with its own
   `NEURON_VISIBLE_DEVICES` assignment.
2. The proxy server specifies remote hosts.
3. Security groups must allow traffic between instances on the server ports and
   NIXL side channel ports.

:::{note}
The addresses `10.0.1.10`, `10.0.1.20`, and so on used throughout this step are
**example VPC private IPs**. Replace each one with the actual private IPv4 address
of the corresponding instance in your VPC (find it with `hostname -I` on the
instance, or in the EC2 console under **Private IPv4 addresses**). The specific
values do not matter as long as the instances can reach each other on the ports
listed below.
:::

**EFA / LIBFABRIC prerequisites (multi-node).** Read-mode KV transfer between
instances rides NIXL over LIBFABRIC, which requires EFA. Before you start:

- Launch instances that have EFA enabled and place them in the **same subnet and
  placement group** so RDMA traffic stays on the EFA fabric.
- Confirm the EFA driver and Libfabric are present on each instance:

  ```bash
  fi_info -p efa   # should list at least one EFA provider
  ```

- The Neuron DLAMI ships EFA and Libfabric. If you built a custom AMI, install the
  [AWS EFA installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html)
  before running the servers.

**Security group ports.** For each pair of communicating instances, the security
group must allow inbound traffic on:

- The vLLM API ports (`8100` for prefill, `8200` for decode in this example) so the
  proxy can reach each server.
- The NIXL side channel ports (`5559` for prefill, `5659` for decode) so the
  servers can exchange KV transfer metadata.
- EFA traffic — allow **all traffic between members of the same security group**
  (a self-referencing rule), which EFA/Libfabric requires for the RDMA path.

The 120B servers can take longer to compile and start than the single-node 20B
servers. Override the earlier engine-ready timeout in both server terminals:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=5400
```

**Prefill server** (on instance at `10.0.1.10`):

```bash
MODEL="openai/gpt-oss-120b"
NEURON_VISIBLE_DEVICES=0-3 VLLM_NIXL_SIDE_CHANNEL_PORT=5559 vllm serve "$MODEL" \
    --port 8100 \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --max-num-seqs 1 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --no-disable-hybrid-kv-cache-manager \
    --hf-overrides '{"quantization_config": {}}' \
    --kv-transfer-config '{"kv_connector": "NeuronNixlConnector", "kv_role": "kv_producer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "mxfp4",
            "ep_degree": 2,
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [1]
        }
    }'
```

**Decode server** (on instance at `10.0.1.20`):

```bash
MODEL="openai/gpt-oss-120b"
NEURON_VISIBLE_DEVICES=0-63 VLLM_NIXL_SIDE_CHANNEL_PORT=5659 vllm serve "$MODEL" \
    --port 8200 \
    --tensor-parallel-size 8 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --optimization-level 2 \
    --dtype bfloat16 \
    --max-num-seqs 4 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-logprobs 0 \
    --no-disable-hybrid-kv-cache-manager \
    --hf-overrides '{"quantization_config": {}}' \
    --kv-transfer-config '{"kv_connector": "NeuronNixlConnector", "kv_role": "kv_consumer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "mxfp4",
            "embedding_dp_size": 8,
            "lm_head_dp_size": 8,
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
```

These commands use TP4 DP1 EP2 on prefill and TP8 DP8 EP64 on decode. As in the
single-node example, the decode EP degree is derived from TP x DP. Run the
environment preparation commands, including `FI_EFA_ENABLE_SHM_TRANSFER=0`, in
both server terminals before launching them.

**Proxy server** with remote hosts:

```bash
python3 examples/vllm_neuron/vllm/disaggregated_inference/toy_proxy_server.py \
    --port 8000 \
    --prefiller-host 10.0.1.10 --prefiller-port 8100 \
    --decoder-host 10.0.1.20 --decoder-port 8200
```

To scale to **2P3D** (2 prefill, 3 decode), add more hosts:

```bash
python3 examples/vllm_neuron/vllm/disaggregated_inference/toy_proxy_server.py \
    --port 8000 \
    --prefiller-host 10.0.1.10 10.0.1.11 \
    --prefiller-port 8100 8100 \
    --decoder-host 10.0.1.20 10.0.1.21 10.0.1.22 \
    --decoder-port 8200 8200 8200
```

The proxy round-robins independently across the prefill host list and the decode
host list — prefill and decode selection are decoupled, so a request routed to
prefill instance *i* is not tied to decode instance *i*. This lets you scale the
two pools asymmetrically (for example, 2P3D above), and any prefill server can hand
off to any decode server because the decode worker pulls KV directly from whichever
prefill server produced it (read mode). The `--prefiller-port` / `--decoder-port`
lists must line up positionally with their host lists; repeat the port when
multiple instances share the same port on different hosts.

:::{note}
The `toy_proxy_server.py` round-robin is intentionally simple and stateless — it
does no load-aware routing or health checking. For production xPyD deployments,
front the pools with an orchestrator that handles routing, health checks, and
autoscaling.
:::

## Confirmation

You have a working DI deployment that:

- Separates prefill and decode across different NeuronCore partitions or
  instances.
- Transfers KV cache via NIXL over LIBFABRIC.
- Routes requests through a proxy server.
- Supports asymmetric parallelism (different TP/DP/EP on prefill vs decode).
- Scales to arbitrary xPyD topologies by adding more servers and updating the
  proxy.

## Common issues

### Enable NIXL and EFA debug logs

If the servers start but the NIXL handshake or KV transfer fails, enable
transport logging in both the prefill and decode terminals before starting the
servers:

```bash
export NIXL_LOG_LEVEL=trace
export NIXL_DEBUG_LOGGING=yes
export FI_LOG_LEVEL=info
```

`NIXL_LOG_LEVEL` and `NIXL_DEBUG_LOGGING` expose NIXL agent setup, memory
registration, and transfer activity. `FI_LOG_LEVEL` enables Libfabric provider
logs, including EFA endpoint selection and connection errors. These settings
produce verbose output; remove them after debugging:

```bash
unset NIXL_LOG_LEVEL NIXL_DEBUG_LOGGING FI_LOG_LEVEL
```

### Troubleshoot the deployment

- **Requests stall after prefill completes**: Verify that
  `VLLM_NIXL_SIDE_CHANNEL_HOST="0.0.0.0"` is set on both servers. Check that the
  NIXL side channel ports (5559, 5659) are reachable between the servers.

- **Inaccurate output from single-node DI**: Verify that
  `FI_EFA_ENABLE_SHM_TRANSFER=0` is set in both server processes.

- **`Connection refused` from proxy to servers**: Both vLLM servers must be fully
  started before the proxy can route requests. First-time Neuron compilation can
  take several minutes — wait for `Application startup complete` on each server.

- **KV transfer timeout or errors**: Ensure LIBFABRIC/EFA is available. On
  multi-node, confirm instances are in the same placement group and subnet.
  Check that `nixl` is installed (`python -c "import nixl"`), then enable the
  NIXL and EFA debug logs above in both server terminals.

- **Decode server OOM**: The decode server must hold KV cache for all in-flight
  requests. Reduce `--max-num-seqs` or `--max-model-len` if memory is tight.

- **Port conflicts on single-node**: When running both servers on one instance,
  use different `VLLM_NIXL_SIDE_CHANNEL_PORT` values (for example, 5559 for
  prefill, 5659 for decode) and different `--port` values.

## Clean up

Stop all processes:

```bash
pkill -f "vllm serve"
pkill -f "toy_proxy_server"
```

If you launched EC2 instances specifically for this tutorial, terminate them to
avoid ongoing charges.

## Next steps

- [Features guide](../guides/features-guide.md) — configure prefix caching,
  speculative decoding, and other features alongside DI.
- [Disaggregated inference design document](../design/vllm/disaggregated-inference.md)
  — architecture, read vs. write transfer modes, and vLLM/Neuron integration
  internals.
