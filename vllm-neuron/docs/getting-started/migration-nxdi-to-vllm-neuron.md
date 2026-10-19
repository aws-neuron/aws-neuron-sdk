# How to migrate from NxD Inference to vLLM Neuron

<!-- meta: description: Migration guide covering the practitioner flow and
performance engineer flow for moving from NxDI to vLLM Neuron. -->
<!-- meta: date_updated: 2026-07-15 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: NDOC-190 -->

## Task overview

Follow this guide when moving from the legacy NxD Inference integration to the
vLLM Neuron plugin. Migration covers two distinct paths, each documented as its
own sub-guide below:

- **[Migrate a deployment](#guide-1-migrate-a-deployment)** — for ML
  practitioners and performance engineers moving an existing NxDI deployment of
  a supported model to the plugin: mapping server flags and `neuron_config`,
  validating accuracy, and benchmarking.
- **[Migrate a custom model](#guide-2-migrate-a-custom-model)** — for teams who
  authored a custom model architecture against NxDI and need to re-implement it
  on the vLLM Neuron plugin's building blocks.

Start with the deployment guide if you serve a model that vLLM Neuron already
supports. Use the custom-model guide if you maintain your own modeling code.

## Prerequisites

- Existing NxD Inference deployment (using `neuronx-distributed-inference`
  with the legacy vLLM Neuron component)
- A new Trn2 or Trn3 instance with the vLLM Neuron plugin installed (see
  [setup guide](setup-guide.md))
- Access to the same model weights for comparison testing

## Guide 1: Migrate a deployment

Use this path when vLLM Neuron already supports your model and you are moving an
existing NxDI serving deployment to the plugin.

### 1. Assess migration scope

Determine which features of NxDI your deployment uses:

| NxDI Feature                    | vLLM Neuron                              | Notes                                   |
| ---------------------------------- | --------------------------------------------------- | --------------------------------------- |
| Continuous batching                | Enabled by default                                  | No configuration needed                 |
| Prefix caching                     | Enabled by default                              | No configuration needed     |
| Speculative decoding | EAGLE3 only, via `--speculative-config`               | Vanilla draft-model speculative decoding is not supported; migrate to an EAGLE3 draft            |
| Disaggregated inference            | Supported (1P1D, xPyD)                              | Topology configuration changed          |
| Quantization (INT8/FP8)            | `quantized` + `quantization_dtype` in neuron_config | Same quantization approach              |
| On-device sampling                 | Enabled by default                                  | Supports temperature, top_k, top_p      |
| Sequence parallelism                 | Enabled by default                                  | No configuration needed      |
| Data parallelism                 | Native vLLM data parallel support                                  | Server configuration changed     |
| Expert parallelism                 | Native vLLM expert parallel support                                 | Server configuration changed      |
| Decode bucketing                 | vLLM Neuron buckets decode by num_seqs by default. Decode context length bucketing is supported via `decode_context_length_buckets`      | Server configuration changed      |

### 2. Map API and parameters

**Server startup:**

```bash
# NxDI (legacy vLLM Neuron integration)
VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference' \
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --device neuron \
    --tensor-parallel-size 32 \
    --max-num-seqs 8 \
    --max-model-len 4096 \
    --override-neuron-config '{"on_device_sampling_config": {"dynamic": true}}'

# vLLM Neuron plugin (new)
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --hf-overrides '{"quantization_config": {}}' \
    --additional-config '{
      "neuron_config": {
        "on_device_sampling_config": {}
      }
    }'
```

Key differences:

- No `--device neuron` flag needed — the plugin auto-detects Neuron hardware
- No `VLLM_NEURON_FRAMEWORK` environment variable needed — the plugin registers
  itself with vLLM via its entry point
- Neuron-specific settings move from `--override-neuron-config` to
  `--additional-config`, nested under `neuron_config` (see below)

**Configuration override:**

```bash
# NxDI: flat neuron config via --override-neuron-config
--override-neuron-config '{"context_encoding_buckets": [128, 256, 512, 1024]}'

# vLLM Neuron: nested under additional_config, with renamed fields
--additional-config '{"neuron_config": {"num_batched_tokens_buckets": [128, 256, 512, 1024]}}'
```

Field names also changed — for example, NxDI's `context_encoding_buckets`
maps to `num_batched_tokens_buckets`. See the
[configuration reference](../guides/reference-configuration.md) for the full
list of `neuron_config` options.

### 3. Migrate deployment configuration

1. Update your server launch command using the parameter mapping above
2. Remove `--device neuron`, the `VLLM_NEURON_FRAMEWORK` environment variable,
   and any other NxDI-specific flags
3. Move any `--override-neuron-config` settings under `--additional-config`,
   nested inside `neuron_config`, renaming fields as needed (for example,
   `context_encoding_buckets` → `num_batched_tokens_buckets`)
4. Drop `--num-gpu-blocks-override` and `--block-size` unless you were using
   them for advanced tuning — the plugin sizes the KV cache automatically

To learn how to enable and tune vLLM Neuron features (bucketing, prefix
caching, quantization, disaggregated inference, and more) in your new
deployment, see the [features guide](../guides/features-guide.md).

### 4. Validate output accuracy

Run the same prompts through both deployments and validate that results are comparable.
Use datasets or logit validation to evaluate accuracy.

If the model is inaccurate, see the
[accuracy debugging guide](../model-dev/accuracy-debugging-guide.md).

### 5. Benchmark performance

Compare latency and throughput between deployments:

- **TTFT** (time to first token): Should be comparable or improved with prefix
  caching
- **ITL** (inter-token latency): Should be comparable or improved
- **Throughput** (tokens/sec): Should be comparable at the same batch size

### 6. Staged rollout

1. Deploy vLLM Neuron alongside the existing NxDI deployment
2. Route a small percentage of traffic to the new deployment
3. Monitor accuracy metrics and latency SLAs
4. Gradually increase traffic to the new deployment
5. Decommission the NxDI deployment once validated

### Confirm your work

Deployment migration is successful when:

- Model output accuracy is similar between NxDI and vLLM Neuron for your test
  prompts
- Latency (TTFT, ITL) meets your SLA requirements
- Throughput matches or exceeds the NxDI deployment
- All required features (quantization, prefix caching, etc.) are functioning

### Common issues

#### Out-of-memory at startup

- **Possible solution**: Default KV sizing respects memory utilization and a
  fractional cap. Reduce `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` after an HBM
  allocation failure. If you explicitly enable headroom sizing, increase
  the active reserve instead. You can also reduce
  `--max-num-seqs` or `--max-model-len`.

#### Performance regression after migration

- **Possible solution**: Verify bucket configurations match your traffic
  pattern. Reduce `num_batched_tokens_buckets` to match your actual prompt length
  distribution. Ensure compilation artifacts are cached.

## Guide 2: Migrate a custom model

Use this path when you authored a custom model architecture against NxDI (for
example, subclassing NxDI modeling code or `NeuronConfig`) and vLLM Neuron does
not yet support it out of the box. On the plugin, you re-implement the model
using vLLM Neuron's building blocks and register it with vLLM's model registry —
there is no direct port of NxDI modeling classes.

At a high level:

1. **Implement the model** against the plugin's building blocks (attention, MLP,
   and parallel layer primitives), applying the plugin's tensor-parallel
   sharding conventions rather than NxDI's.
2. **Register the model** with vLLM's model registry so `vllm serve` can resolve
   your architecture.
3. **Compile and test** the model on Neuron to confirm it compiles and runs.
4. **Validate accuracy** against a reference (HuggingFace) implementation.
5. **Benchmark and tune** performance (bucketing, parallelism, quantization).

This is a code migration, not a configuration change, so it is documented in
full separately. For the complete step-by-step instructions, code patterns, and
per-component details, follow the
[model onboarding guide](../model-dev/onboarding-models.md). Once your model is
onboarded, return to [Guide 1](#guide-1-migrate-a-deployment) to migrate its
serving deployment.

## Related information

- For supported models and features, see the [README](https://github.com/vllm-project/vllm-neuron#supported-models)
  and [model cards](../model-recipes/index.md).
- [Model onboarding guide](../model-dev/onboarding-models.md)
- [Accuracy debugging guide](../model-dev/accuracy-debugging-guide.md)
- [Features guide](../guides/features-guide.md)
