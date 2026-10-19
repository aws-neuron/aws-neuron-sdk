# Compile-Artifact Manifest

<!-- meta: description: Per-engine compile-artifact manifest for vLLM Neuron -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-08-07 -->

## Overview

Each engine emits one `manifest.json` that lists every graph it compiled
(HLO / NEFF) alongside the config that produced them, so a reader consumes one
file instead of crawling the raw compile-cache directories.

- **Per engine** — one manifest per engine, named by the engine's core window,
  so a data-parallel or co-located deployment produces one file each.
- **Self-describing** — each graph entry carries its cache paths, shapes,
  compiler flags, and the ranks that traced it; `config` records the model,
  parallelism, and resolved Neuron / vLLM settings.
- **Best-effort and off by default** — gated by `VLLM_NEURON_MANIFEST_DIR`; it
  never affects compilation, warmup, or serving.

The feature is enabled by setting **`VLLM_NEURON_MANIFEST_DIR`** to a directory.
Unset or empty, nothing is recorded and no manifest is written — there is no
default location.

### Why a Manifest?

A deployment's compiled artifacts are otherwise discoverable only by walking the
compile cache: a flat tree of `<hash>/` directories with no record of which
engine produced them, which config they came from, or — under data and expert
parallelism — which shard each graph belongs to. The manifest turns that into a
single per-engine file that:

- names the graphs an engine actually compiled this run, with their config;
- attributes each graph to the ranks that traced it, so a consumer can tell a
  complete deployment from a partial one;
- is cheap to collect and diff across runs.

## Architecture

The manifest is assembled in two phases from a compile-time **ledger**: every
worker records the graphs it traces during compilation, and one process per
engine reads that ledger after warmup and writes the manifest.

### Directory Layout

```text
$VLLM_NEURON_MANIFEST_DIR/
├── ledger/
│   ├── dev0_7/                  # engine 0 (cores 0-7): compile-time ledger
│   │   ├── 41234.jsonl          #   one JSONL file per writer process
│   │   └── 41240.jsonl
│   └── dev8_15/                 # engine 1 (cores 8-15)
│       └── …
├── dev0_7/
│   └── manifest.json            # emitted by local_rank 0 of engine 0
└── dev8_15/
    └── manifest.json
```

`<engine-id>` is `get_server_prefix()` — `dev{first}_{last}` over
`NEURON_VISIBLE_DEVICES`, or `nodev` when no device is visible (CPU-compile).
The manifest filename is fixed and written atomically (temp file + `os.replace`),
so relaunching an engine overwrites its manifest and nothing accumulates.

### Record — Compile Time, Every Rank

While a graph is captured (Lite's `capture()` backend), the process appends one
line to its own ledger file `ledger/<engine-id>/<pid>.jsonl`:

```json
{"key": "<graph_hash>", "rank": 3, "input_shapes": ["4,", "..."], "compiler_flags": "--framework XLA --target trn2 ..."}
```

- `key` — the graph's content hash (its compile-cache directory name).
- `rank` — the writer's TP-group-relative rank (`get_tp_group().rank_in_group`);
  the emitter maps this to the writer's global rank for the manifest's `ranks`.
- `input_shapes` / `compiler_flags` — captured inline from the traced inputs and
  the resolved compile options; no artifact files are read.

Writes are raw `os.write` so a line survives the trace fork children's
`os._exit()` (which skips buffered flushes), and each process writes its own file
so concurrent lanes never contend. The first write of a run truncates the file —
discarding any stale same-pid file left by a prior run — and later writes append.
When the feature is off, nothing is computed or written.

### Emit — After Warmup, `local_rank == 0`

One process per engine reads its own engine's ledger and writes
`<engine-id>/manifest.json`:

1. **Group** the ledger lines by `key`, accumulating the ranks that traced each
   graph and mapping each writer's TP-local rank to its global rank (via the
   engine's TP group).
2. **Enrich** each graph from its `<hash>/` folder in the compile cache —
   `hlo_path`, `neff_path`, `neff_size`, and `created_timestamp` come from one
   `stat` of the NEFF; `output_shapes` is read from the HLO module's entry result
   shape.
3. **Join** the live `VllmConfig` (`config`) and a package-version snapshot
   (`environment`).
4. **Write** the file atomically.

The manifest is written **whenever the feature is on**, even if the ledger is
empty (→ `"graphs": []`). Because the filename is fixed.

### Cleanup

After emitting, `local_rank == 0` removes its engine's `ledger/<engine-id>`
subdirectory. Other ranks remove nothing, and the shared `ledger/` root is left
in place — engines clean up independently.

Every step above is **best-effort**: a ledger-write, enrichment, or emit failure
is logged and swallowed, so the manifest can never break compilation, warmup, or
a request.

## Manifest Schema

The schema is versioned (`manifest_version`, currently `"1"`) and defined in
`vllm_neuron/manifest/schema.py`.

Top level:

| Field | Contents |
| ---- | ---- |
| `manifest_version` | Schema version (`"1"`) |
| `config` | How the run was compiled/served (below) |
| `graphs` | One entry per compiled graph (below) |
| `environment` | Installed package versions + Python (a pip-freeze snapshot) |

`config`:

| Field | Contents |
| ---- | ---- |
| `model`, `target`, `dtype` | Model path, compiler target (e.g. `trn2`), dtype |
| `max_num_seqs`, `max_model_len` | Max concurrent sequences, max sequence length |
| `topology` | `world_size`, `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`, `attention_dp_size`, `ep_degree` (resolved), `decode_context_parallel_size`, `has_collectives` |
| `neuron_config` | Full resolved Neuron config |
| `vllm_config` | Full resolved vLLM `VllmConfig` — every sub-config (model / cache / parallel / scheduler / compilation / …), JSON-coerced |

`graphs[]`:

| Field | Contents |
| ---- | ---- |
| `graph_hash` | Compile-cache hash (its directory name) |
| `hlo_path`, `neff_path` | Paths relative to the compile-cache root |
| `neff_size`, `created_timestamp` | NEFF size in bytes and compile time (ISO 8601 — the NEFF mtime) |
| `input_shapes`, `output_shapes` | Ordered shape strings (`""` for a scalar) |
| `compiler_flags` | `neuronx-cc` flags, machine-specific paths removed |
| `ranks` | Global ranks (`0..world_size-1`) that traced this graph |
| `input_snapshot_paths`, `output_snapshot_paths` | Reserved; a graph maps to 0..N snapshots, so each is a list (currently empty) |

Example (TP=2 dense model, trimmed):

```json
{
  "manifest_version": "1",
  "config": {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "target": "trn2",
    "dtype": "torch.bfloat16",
    "max_num_seqs": 4,
    "max_model_len": 256,
    "topology": {"world_size": 2, "tensor_parallel_size": 2, "data_parallel_size": 1,
                 "ep_degree": 1, "has_collectives": true},
    "neuron_config": {"...": "..."},
    "vllm_config": {"...": "..."}
  },
  "graphs": [
    {
      "graph_hash": "fc5c06b7...",
      "hlo_path": "fc5c06b7.../graph.hlo",
      "neff_path": "fc5c06b7.../graph_fc5c06b7....neff",
      "neff_size": 949498,
      "created_timestamp": "2026-08-06T20:37:11.818072+00:00",
      "input_shapes": ["4,", "4,", "..."],
      "output_shapes": ["4", "4, 49152", "..."],
      "compiler_flags": "--framework XLA --target trn2 --auto-cast=none ...",
      "ranks": [0, 1],
      "input_snapshot_paths": [],
      "output_snapshot_paths": []
    }
  ],
  "environment": {"python": "3.13.7", "torch": "2.11.0", "vllm": "0.24.0"}
}
```

## Engine Identity

`<engine-id>` is the engine's physical core window, from `get_server_prefix()`.
On Neuron, whenever `data_parallel_size > 1` each engine is assigned a disjoint
physical device slice and narrows `NEURON_VISIBLE_DEVICES` to it, so two engines
on one host never resolve to the same window. It is the same value the compile
cache uses to scope its per-rank subdirectories, so a manifest correlates
directly with the cache it describes.

The prefix is per-engine only *after* that narrowing, which happens in
`init_device`. Every manifest and ledger path is resolved later, during warmup,
so the value is always per-engine by the time it is used.

## Environment Variables

| Variable | Default | Description |
| ---- | ---- | ---- |
| `VLLM_NEURON_MANIFEST_DIR` | (unset) | Directory that ledgers and manifests are written under. A path enables the feature; unset or empty disables it. There is no default location. |
