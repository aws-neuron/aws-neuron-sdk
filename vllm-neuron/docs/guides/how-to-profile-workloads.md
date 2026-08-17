# How to profile vLLM Neuron workloads

<!-- meta: description: Capture NRT device and system profiles on a live
vLLM Neuron serving instance using the built-in profiler interface. -->
<!-- meta: date_updated: 2026-06-09 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: No Jira ticket found -->

## Task overview

This guide explains how to capture Neuron Runtime (NRT) profiles on a
live vLLM serving instance using the `/start_profile` and
`/stop_profile` HTTP endpoints. Profiles can be viewed in Neuron
Explorer for device-level performance analysis.

## Prerequisites

- A running vLLM server on a Neuron instance (Trn2 or Inf2)
- The `neuron-explorer` tool for viewing `.ntff` files
- `curl` or equivalent HTTP client

## Architecture

For the full profiler architecture and NRT API design, see
[Neuron profiling design](../design/vllm/neuron-profiling.md).

## Instructions

### 1. Start the server with profiling enabled

Enable the profiler at server startup. We reuse `"cuda"` as the
profiler kind to mount the HTTP endpoints:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 32 \
    --profiler-config '{"profiler": "cuda"}' \
    --additional-config '{
      "neuron_profiler": {
        "activities": ["device_profile", "system_profile"],
        "neuron_cores": [0, 1, 2, 3],
        "output_dir": "/tmp/nrt_profile"
      }
    }'
```

### 2. Start profiling

Once the server is ready and handling requests:

```bash
curl -X POST http://localhost:8000/start_profile
```

### 3. Send representative workload

Send requests that represent your production traffic pattern:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-3.3-70B-Instruct",
      "prompt": "Explain quantum computing in simple terms.",
      "max_tokens": 200
    }'
```

### 4. Stop profiling

```bash
curl -X POST http://localhost:8000/stop_profile
```

### 5. View profiles in Neuron Explorer

Open the `.ntff` files in Neuron Explorer:

```bash
neuron-explorer view -d /tmp/nrt_profile/
```

## Configuration reference

### Profiler config (via `--profiler-config`)

| Field              | Type | Default | Description                          |
| ------------------ | ---- | ------- | ------------------------------------ |
| `profiler`         | str  | —       | Must be `"cuda"` to enable endpoints |
| `delay_iterations` | int  | 0       | Skip N engine steps before starting  |
| `max_iterations`   | int  | —       | Auto-stop after N engine steps       |

### Neuron profiler config (via `--additional-config`)

Nested under `"neuron_profiler"`:

| Field                         | Type              | Default                                | Description                       |
| ----------------------------- | ----------------- | -------------------------------------- | --------------------------------- |
| `activities`                  | list[str]         | `["device_profile", "system_profile"]` | Activity types to capture         |
| `neuron_cores`                | list[int] or null | null (rank 0)                          | Which worker ranks to profile     |
| `output_dir`                  | str               | `./neuron_profiles`                    | Output directory for profile data |
| `sys_trace_max_events_per_nc` | int or null       | null (NRT default)                     | Max system trace events per NC    |

Valid `activities` values: `"system_profile"`, `"device_profile"`,
`"host_memory"`, `"cpu_util"`, `"all"`.

### Iteration control

One iteration equals one `execute_model` call (one batched forward
pass). The state machine in `WorkerProfiler.step()` handles
delay/max iteration logic:

| Config field       | Behavior                              |
| ------------------ | ------------------------------------- |
| `delay_iterations` | Skip N steps before calling NRT begin |
| `max_iterations`   | Auto-call NRT stop after N steps      |

Example — profile steady-state only (skip warmup):

```bash
vllm serve <model> \
    --profiler-config \
      '{"profiler": "cuda", "delay_iterations": 50, "max_iterations": 20}' \
    --additional-config \
      '{"neuron_profiler": {"output_dir": "/tmp/nrt_profile"}}'

curl -X POST http://localhost:8000/start_profile
# Auto-starts after 50 iterations, auto-stops after 20 more
```

## Profiling with vllm bench

For quick benchmarking with profiling:

```bash
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 5
```

## Disaggregated inference profiling

In a disaggregated inference setup, prefill and decode run as separate
vLLM server instances. Each instance has its own `/start_profile` and
`/stop_profile` endpoints.

### Proxy server pass-through

The DI proxy server forwards profile requests to all backend servers:

```text
Client                    Proxy                  Prefill Server    Decode Server
  │                         │                         │                 │
  ├─ POST /start_profile ──→│                         │                 │
  │                         ├─ POST /start_profile ──→│                 │
  │                         ├─ POST /start_profile ──────────────────→│
  │                         │                         │                 │
  │              (200 OK after all backends respond)  │                 │
  │←────────────────────────┤                         │                 │
  │                         │                         │                 │
  │         ... requests profiled on both servers ... │                 │
  │                         │                         │                 │
  ├─ POST /stop_profile ───→│                         │                 │
  │                         ├─ POST /stop_profile ───→│                 │
  │                         ├─ POST /stop_profile ────────────────────→│
  │                         │                         │                 │
  │←────────────────────────┤                         │                 │
```

For deployments without a proxy, manually hit the endpoints on each
server you want to profile.

## Profile output structure

After `/stop_profile`, the output directory contains:

```text
output_dir/
  i-<instance_id>_pid_<pid>/<timestamp>/
    profile_nc_0_session_0.ntff    # Device profile (NTFF)
    ntrace.pb                      # System trace
    trace_info.pb                  # Trace metadata
    cpu_util.pb                    # CPU utilization (if enabled)
    host_mem.pb                    # Host memory (if enabled)
  neffs/
    graph_<hash1>.neff             # Compiled NEFFs
    graph_<hash2>.neff
```

NTFF files are viewed with Neuron Explorer. NEFFs are automatically
copied from the compile cache — they are required for Neuron Explorer
to render device profiles and correlate with instruction data.

## Confirm your work

After stopping a profile, verify output was generated:

```bash
ls /tmp/nrt_profile/
# Should show i-<instance>_pid_<pid>/ directory with .ntff files
```

Open in Neuron Explorer to verify traces were captured correctly.

## Common issues

### No profile output generated

- **Possible solution**: Ensure `--profiler-config '{"profiler":
  "cuda"}'` was passed at server startup. The endpoints are only
  mounted when this flag is set.

### Empty or truncated system traces

- **Possible solution**: Increase `sys_trace_max_events_per_nc` in
  the neuron profiler config. The default ring buffer may overflow
  during long profiling sessions.

### Neuron Explorer cannot render device profiles

- **Possible solution**: Ensure NEFFs are present in the `neffs/`
  subdirectory of the output. They should be auto-copied from the
  compile cache. If missing, set the compile cache path via
  `NEURON_COMPILED_ARTIFACTS`.

### Profile shows no activity

- **Possible solution**: Ensure requests were being processed between
  `/start_profile` and `/stop_profile`. If using `delay_iterations`,
  verify enough iterations elapsed before profiling began.

## Related information

- [CPU development workflow](../model-dev/cpu-development.md) — developing without
  hardware
- [Configuration reference](reference-configuration.md) — all Neuron
  config options
- [Neuron Explorer documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html)
