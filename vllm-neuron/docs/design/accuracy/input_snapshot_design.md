# Input Snapshot Design

<!-- meta: description: Capturing NRT-boundary input tensors for off-chip replay -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-07-20 -->

## Overview

Input snapshot capture writes the flattened NRT-boundary input tensor vector of a
selected forward to disk, so that exact forward can be replayed off-chip for
accuracy debugging. Where [Tensor capture](tensor_capture_design.md) extracts
*intermediate* values inside the graph, input snapshots capture the *inputs*
handed to the compiled NEFF — the starting point a replay needs to reproduce a
divergence.

Capture is opt-in and defaults off; when off it adds nothing to the forward
path. Selection is proactive: rather than dumping broadly and searching after,
the targeted forward is identified live and only that one is written.

## Architecture

Capture policy lives in the Python plugin (`vllm_neuron/snapshot/`): it decides
*whether*, *what*, and *where* to capture. The actual device-to-host tensor
write is delegated to the Neuron runtime via a standalone serialize op the
plugin calls immediately before a plain `execute` on selected forwards — the
runtime is an opaque writer here, and capture is a no-op on a runtime backend
that does not expose the op.

:::{note}
The compiled NEFF passes tensors across the boundary but not Python
objects, so the request/token identity that makes a snapshot answerable for
a regression exists only in the model runner. The runner and the compiled
executable run on the same thread within one synchronous forward, so the
runner resolves the capture verdict and publishes it into a process-global
holder *before* calling the model; the executable reads it while deciding
whether to dump.
:::

``` text
┌───────────────────────────────────────────────────────────────────┐
│ NeuronModelRunner.execute_model (before the compiled forward)       │
│  - CaptureSelector.evaluate_forward(req_ids, positions, is_decode)  │
│  - set_current_forward(SnapshotForwardContext(step, capture, ...))  │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│ Executable._execute_with_snapshot (compile/backend.py)              │
│  - get_current_forward(); skip if absent (warmup)                   │
│  - OR per-forward token/request verdict with this NEFF's call-index │
│  - consume process-global capture budget                           │
│  - selected: serialize op (inputs, call_dir, format), then a        │
│              plain execute(); + write_call_meta(call_dir, ...)      │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│ Neuron runtime serialize op (opaque writer)                         │
│  - copies each tensor device->host and writes it under call_dir     │
│    (called on synchronized inputs, before the plain execute)        │
└───────────────────────────────────────────────────────────────────┘
```

## Selection

Three rules, OR'd together. A forward is captured if any rule fires, subject to
the capture budget.

- **Call index** (`VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL`) — the Nth
  (0-based) post-warmup call of a NEFF; `-1` selects every call, and with no
  rule configured this defaults to call `0`. It counts arrival order, so it
  reproduces the same forward only under a deterministic schedule (batch size 1,
  single sequence, sync). Under batching / chunked prefill / speculative decode,
  forwards interleave across NEFFs and call `N` may land on a different request
  run to run — use the token/request rules below for reproducible targeting.
- **Token** (`VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN`) — a decode step that
  generates a target token position. Fires on decode when a row at position
  `p` will generate a targeted token `p + offset` (`offset` in
  `1..num_speculative_tokens+1`); skipped on prefill, where that relationship
  does not hold.
- **Request** (`VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST`) — a request id. vLLM
  appends a unique `-<suffix>` to the caller's id (`abc` becomes
  `abc-9a8546d5`), so the caller-supplied base id is matched as a prefix; an
  exact full id also matches.

Selection is bounded by a process-global **capture budget**
(`VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES`, default 4) so a broad rule cannot dump
without end. Budget is consumed only after a forward is selected.

Malformed call/token selections raise at startup (a typo should not silently
disable a debugging run); a malformed rank list degrades to the default instead
of aborting.

## Configuration

Set via environment variables. Defaults preserve current behavior (capture off).

| Variable | Meaning |
| --- | --- |
| `VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE` | Master switch. Capture is off unless set. |
| `VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL` | Call-index rule (comma-separated 0-based indices, or `-1` for all calls — not "all tokens"). Unset defaults to the first call. |
| `VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN` | Token rule (comma-separated target token positions). |
| `VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST` | Request rule (comma-separated request ids, prefix-matched). |
| `VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES` | Process-global capture budget (default 4). |
| `VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS` | tp-ranks to capture (comma-separated). Unset captures all ranks — each worker writes its own `rank<global_rank>/` bundle, which is what a tensor-parallel replay needs since each rank holds only its shard. Set it to narrow to specific tp-ranks. |
| `VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT` | Artifact format: `pt` (pickled, default) or `npy` (raw value-preserving bytes). Validated at startup. |

The bundle root is derived from `VLLM_CACHE_ROOT` (`.../neuron/snapshots`),
mirroring the compile cache dir.

## Gating

`NeuronPlatform.check_and_update_config` enforces the preconditions once at
startup:

- `VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE` must be set.
- The runtime backend must implement the snapshot write; the default backend
  does, so no action is needed. On a backend that does not, capture is a no-op.
- **Sync scheduling** is required. Capture reads inputs inline before
  scheduling; under async scheduling the buffers may still belong to an
  in-flight prior forward, so a config with `async_scheduling` raises rather
  than capture the wrong tensors.

The snapshot config (selector, format, ranks, budget) is resolved once here and
cached for the process, so a malformed selection or format fails at startup
rather than mid-request, and the model runner and every executable reuse that
single resolution instead of re-parsing the environment.

## Output Structure

Each selected forward produces a `call` directory holding the positional
input tensors plus a `meta.json` identity tag. The directory is keyed on the
worker's **global rank**, which is unique per process across any parallelism
combination (tp/dp/pp/…), so workers sharing a compilation hash never write to
the same directory. The tp/dp/pp breakdown is recorded in `meta.json` rather
than encoded in the path.

``` text
$VLLM_CACHE_ROOT/neuron/snapshots/
└── <compilation_hash>/
    └── rank<global_rank>/               # unique per worker process
        └── call<N>/                     # one per selected forward
            ├── tensor0.npy              # or tensor0.pt
            ├── tensor1.npy
            └── meta.json                # identity + input dtype/shape
```

`meta.json` records the compilation hash, the artifact `format`
(`npy`/`pt`, so a reader can locate `tensor{i}.{format}`), the rank
breakdown (`global_rank`, `tp_rank`, `dp_rank`), call index, the rules
that selected this call (`selected_by`), the per-input `dtype`/`shape`
(needed to reinterpret raw `.npy` bytes on replay), and the forward identity
(`global_step`, `is_prompt`, `req_ids`, `positions`, and the matched
rows).

The `call<N>` directory is named by the NEFF's call index, regardless of which
rule fired the capture. So when a token/request rule selects a forward, the
directory is still `call<N>` for whichever call it landed on, not named after
the token or request — you identify the one you targeted via `meta.json`
(`selected_by` and `matches`). For example, to capture the forward that
generates decode step `k` of a request whose prompt length is `L`, set
`VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN=<L+k>`; the resulting
`call<N>` bundle carries `selected_by=["token"]` and that position in
`matches`.

Capture fails loudly: an unwritable directory, a failed tensor write, or a
failed `meta.json` write aborts the forward rather than leaving a partial or
missing bundle that looks captured but is not. Capture is opt-in debug, so a
hard stop is preferable to a silently incomplete artifact.

## Encoding

The runtime writes each input either as `.npy` (raw element bytes,
value-preserving) or as a pickled `.pt` (any dtype). `bf16` and `fp8`
carry no numpy scalar type, so they are stored as fixed-width raw bytes
(`|V2` / `|V1`) that reload unchanged. A dtype with no numpy representation
raises under `.npy` (asking the caller to re-run with `.pt`) rather than
silently mixing formats. The encoding is a property of the on-disk artifact,
independent of which runtime backend produced it.

## Components

Policy layer (`vllm_neuron/snapshot/`):

- `config.py` — `SnapshotConfig` (resolved settings) and `CaptureSelector`
  (the three selection rules, parsed from the environment).
- `context.py` — `SnapshotForwardContext` and the process-global holder the
  runner publishes to; also the process-global capture budget.
- `capture.py` — `resolve_capture_spec` (builds the global-rank-scoped
  output directory and bundles the selector/budget for one executable) and
  `SnapshotCapturer`, which owns the per-forward capture decision — selection,
  budget, the serialize-op call, and `meta.json` — so the concern lives here
  rather than in the executable.
- `meta.py` — `write_call_meta`, the `meta.json` writer (raises on
  failure; see the fail-loud note above).

Integration points:

- `compile/backend.py` — constructs a `SnapshotCapturer` per executable
  (via `build_capturer`) and, before each plain `execute`, calls
  `capturer.capture(inputs)`; capture is a no-op unless the forward is
  selected, keeping the execute path free of debug logic.
- `vllm/worker/neuron_model_runner.py` — publishes the per-forward context
  before the compiled forward and clears it after.
- `vllm/platform.py` — startup gating (see [Gating](#gating)).
- Neuron runtime serialize op — called with `(inputs, call_dir, format)`
  before a plain `execute`; performs the host-side tensor write (opaque to
  the plugin).

## Limitations

- **Sync scheduling only**: async scheduling is rejected at startup (see
  [Gating](#gating)).
- **Rank-local writes**: each worker writes its own bundle; multi-host captures
  must be aggregated across hosts for comparison.
- **Anonymous positional inputs**: dumps are a positional vector; request/token
  identity lives in `meta.json`, not in the tensor files.
