# vLLM Metrics Integration

<!-- meta: description: Production metrics design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

This document describes how vLLM Neuron publishes custom Prometheus metrics alongside vLLM's built-in metrics at the `/metrics` endpoint.

## Overview

vLLM exposes a `/metrics` HTTP endpoint that returns all registered metrics in the [Prometheus text exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/). vLLM Neuron extends this endpoint with Neuron-specific metrics.

## Architecture

vLLM uses `prometheus_client` with a global default registry. In the default single-server configuration, vLLM runs metrics collection in the API server process and uses IPC (`SchedulerStats` via `EngineCoreOutputs`) to transport stats from the EngineCore process.

vLLM Neuron cannot use this IPC path because `SchedulerStats` is not extensible by plugins. Instead, vLLM Neuron enables `prometheus_client` multiprocess mode so that metrics observed in the EngineCore and Worker processes are written to shared mmap files and aggregated by the API server's `/metrics` endpoint.

### Multiprocess Prometheus Setup

`PROMETHEUS_MULTIPROC_DIR` is set at module level in `vllm_neuron/__init__.py` before `prometheus_client` is imported anywhere. This ensures all processes (API server, EngineCore, workers) use multiprocess mode for metric storage.

``` python
# vllm_neuron/__init__.py (top of file)
if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(
        prefix="vllm_neuron_prometheus_"
    )
```

Users can override this by setting `PROMETHEUS_MULTIPROC_DIR` before starting the server. The directory must exist and should be wiped between server restarts to avoid stale metrics.

## vLLM Neuron Metrics

vLLM Neuron reports Neuron-specific metrics under the `vllm_neuron:` prefix. Metrics are defined at module scope in `vllm_neuron/metrics.py` so they auto-register with the `prometheus_client` global default registry.

vLLM Neuron uses labels to characterize metrics according to common dimensions like model. Common labels include:

- `model_name` -- The name of the model for that server process. Present on all metrics.
- `bucket_name` -- The name of the relevant bucket, such as for compilation or execution metrics. The bucket name follows a format similar to `prefill_s1024`.
- `rank_id` -- The process local rank ID. This rank ID is the local rank (i.e. relative to start rank) rather than the core ID.

### General Metrics

| Metric Name | Type | Description |
|----|----|----|
| `vllm_neuron:num_seqs_padding` | Histogram | Number of padded batch lines processed by the model, labeled by `model_name` and `bucket_name`. |
| `vllm_neuron:num_batched_tokens_padding` | Histogram | Number of padded sequence lengths processed by the model, labeled by `model_name` and `bucket_name`. |
| `vllm_neuron:neff_execution_count` | Counter | Number of NEFF executions, labeled by `model_name` and `bucket_name`. |

### Server Startup Metrics

| Metric Name | Type | Description |
|----|----|----|
| `vllm_neuron:startup_time_seconds` | Gauge | Total server startup time from worker spawn to ready, labeled by `model_name`. |
| `vllm_neuron:compilation_time_seconds` | Gauge | Time spent compiling Neuron graphs (FX trace + neuronxcc compile). Includes compile cache hits. Labeled by `model_name` and `bucket_name`. |
| `vllm_neuron:model_load_time_seconds` | Gauge | Time spent loading model weights to device (host to HBM transfer), labeled by `model_name`. |
| `vllm_neuron:model_load_size_bytes` | Gauge | Size of model weights transferred to device (host to HBM transfer), labeled by `model_name`. |

### Adding New Metrics

1. Add the metric definition to `vllm_neuron/metrics.py`.
2. Use the `vllm_neuron:` prefix.
3. Include a `model_name` label for consistency with vLLM.
4. Update the metric from the relevant code path.
5. Add unit tests and integration tests.
