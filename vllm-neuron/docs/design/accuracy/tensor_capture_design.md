# Tensor Capture Design

<!-- meta: description: Tensor capture design for accuracy debugging -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-07-20 -->

## Overview

Tensor capture extracts intermediate tensor values from compiled models for accuracy debugging. It enables comparison to identify which module/op first diverges.

## Architecture

Hook-based capture using `ModelCapture` that works with `torch.compile(fullgraph=True)`.

:::{note}
`torch.compile()` is a JIT compiler — it returns a wrapper immediately
and defers tracing/compilation to the first forward call (warmup). Hooks
registered before warmup are traced into the compiled NEFF.
:::

``` text
┌─────────────────────────────────────────────────────────────────┐
│ 1. torch.compile(model, backend="vllm_neuron", fullgraph=True)  │
│    - Returns OptimizedModule wrapper (NO tracing/compilation)   │
│    - Original model stored as wrapper._orig_mod                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. ModelCapture (after torch.compile(), before warmup)          │
│    - Registers forward hooks on _orig_mod submodules            │
│    - No compilation has happened yet — hooks are pre-trace      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Warmup (first forward call triggers actual compilation)      │
│    - Dynamo traces _orig_mod forward, sees hooks                │
│    - Hook clone()/detach() ops baked into FX graph              │
│    - neuronx_cc compiles FX graph → single NEFF per bucket      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Runtime: setup_tensor_capture() / model(**kwargs) / save_tensor_captures()│
│    - setup_tensor_capture: activates this model's TensorRegistry as   │
│      the global singleton so inline capture_tensor() calls      │
│      route to the correct registry (needed for multi-model)     │
│    - model(**kwargs): compiled NEFF executes, hooks fire,       │
│      tensors are cloned into registry (inside the graph)        │
│    - save_tensor_captures: reads tensors from registry and writes      │
│      them to disk with metadata (outside the graph); restores   │
│      previous global registry for multi-model isolation         │
└─────────────────────────────────────────────────────────────────┘
```

`ModelCapture` is used uniformly for target, draft, and vision encoder models.
Each model gets its own instance with an isolated `TensorRegistry`.

## API

### vLLM Configuration

Configure tensor capture via `neuron_config` in vLLM's `additional_config`:

``` python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    additional_config={
        "neuron_config": {
            "tensor_capture": {
                "modules": ["model.layers.0-31", "lm_head"],
                "capture_dir": "/tmp/captures"
            }
        }
    }
)
```

### Pattern Syntax

- **Range**: `model.layers.0-31` expands to individual patterns before matching
- **Regex**: All patterns use Python regex (e.g., `model\.layers\.\d+`, `.*self_attn$`)
- **Vision prefix**: `visual.blocks.0` routes to vision encoder capture

### Programmatic Usage

``` python
from vllm_neuron.accuracy import ModelCapture

# After compile — no wrapping needed
compiled = torch.compile(model, backend="vllm_neuron", fullgraph=True)
raw_model = compiled._orig_mod

capture = ModelCapture(
    model=raw_model,
    modules=["model.layers.0-31"],
    capture_dir="/tmp/captures",
)
capture.register_clear_hook(compiled)
capture.enable()

# Each forward pass:
capture.setup_tensor_capture()
output = compiled(**kwargs)
capture.save_tensor_captures(positions=positions, is_prefill=True, req_ids=[...])
```

### Manual Capture

For capturing tensors inside model code:

``` python
from vllm_neuron.accuracy import capture_tensor

class LlamaAttention(nn.Module):
    def forward(self, hidden_states, ...):
        attn_output = self._compute_attention(...)
        capture_tensor("attn_output", attn_output)  # Captured
        return attn_output
```

Inline `capture_tensor()` calls require `TensorRegistry._instance` to be set
before warmup so dynamo traces them as active code.

### Multi-Prompt Capture

When running multiple prompts, the capture system automatically organizes captures by request ID and phase (prefill/decode).

### Output Structure

``` text
/tmp/captures/
├── dp0/                                # Target model captures
│   ├── prefill_s128_0/                 # First prefill, bucket size 128
│   │   ├── model.layers.0/
│   │   │   └── rank0.pt                # TP-local rank
│   │   └── prefill_s128_0_meta.json    # Request IDs, positions
│   └── decode_b1_0/                    # First decode, batch size 1
│       ├── model.layers.0/
│       │   └── rank0.pt
│       └── decode_b1_0_meta.json
├── draft/
│   └── dp0/                            # Draft model captures
│       └── decode_b4_0/
└── vision/
    └── dp0/                            # Vision encoder captures
        └── prefill_s4096_0/
```

## Core Classes

### ModelCapture

Unified tensor capture for any compiled model (target, draft, or vision):

- Registers forward hooks on `_orig_mod` submodules after `torch.compile()`
  but before warmup (i.e., before Dynamo traces the graph)
- Hooks are traced by Dynamo during the first forward (warmup) and captured
  tensors become part of the compiled NEFF — single compilation, no recompile
- `register_clear_hook(compiled_model)` registers a pre-forward hook on the
  compiled wrapper to clear the registry before each forward pass
- `setup_tensor_capture()` activates this instance's registry as the global singleton
- `save_tensor_captures()` writes captures to disk and restores the previous registry
- `enable()` / `disable()` control whether disk writes actually happen
  (disabled during warmup to avoid capturing synthetic inputs)
- Each instance has its own `TensorRegistry` for multi-model isolation

### TensorRegistry

Stores captured tensors during forward pass. Two capture modes:

1. **Hook-based**: Hooks call `register_module_tensor()` during forward
2. **Inline**: `capture_tensor()` calls `register_manual_tensor()` via global singleton

Methods:

- `register_module_tensor(name, tensor)` — Called by hooks
- `register_manual_tensor(name, tensor)` — Called by `capture_tensor()`
- `get_all_tensors()` — Returns tensors in registration order
- `clear()` — Called at start of each forward pass
- `configure(enabled=True)` — Enable/disable tensor registration

### CaptureWriter

Disk I/O component (internal to `ModelCapture`):

- Saves tensors to disk when enabled via `write()`
- Organizes captures by phase (prefill/decode) and bucket size
- Supports per-DP-rank and per-TP-rank directories
- Optional `capture_filter` for write-time filtering

## Integration Points

### Model Runner Integration

In `neuron_model_runner.py`:

``` python
# In init_tensor_capture() — called after load_model(), before warmup
def init_tensor_capture(self):
    capture_config = self.neuron_config.tensor_capture
    if not capture_config:
        return

    # Target model
    self._target_tensor_capture = self._setup_capture(
        self.model, text_modules,
        capture_filter=...,  # None means write all; set filters disk output
    )
    TensorRegistry._instance = self._target_tensor_capture._registry

    # Vision encoder
    if vision_modules:
        self._vision_tensor_capture = self._setup_capture(
            inner_model.visual, vision_modules, subdirectory="vision"
        )

    # Draft model
    if self.drafter:
        self._draft_tensor_capture = self._setup_capture(
            self.drafter.model, draft_modules, subdirectory="draft"
        )
```

In `neuron_worker.py` (after warmup):

``` python
self.model_runner.enable_capture()
self.model_runner._draft_tensor_capture.enable()
```

### Forward Pass Integration

Each model's forward is bracketed with `setup_tensor_capture()` / `save_tensor_captures()`:

- `setup_tensor_capture()` — saves the current global `TensorRegistry._instance`
  and replaces it with this model's registry. This ensures inline
  `capture_tensor()` calls route to the correct registry when multiple
  models (target, draft, vision) each have their own capture instance.

- `save_tensor_captures()` — writes captured tensors to disk with scheduler metadata
  (request IDs, positions, phase) and restores the previous global registry.

Example with speculative decoding (target + draft):

``` python
# Target forward — registry_target is active
self._target_tensor_capture.setup_tensor_capture()    # _instance = registry_target
model_output = self.model(**kwargs)             # hooks + inline captures → registry_target
self._target_tensor_capture.save_tensor_captures(...)  # write to disk, _instance = previous

# Draft forward — registry_draft is active
self._draft_tensor_capture.setup_tensor_capture()     # _instance = registry_draft
draft_output = self.drafter.propose(...)        # hooks + inline captures → registry_draft
self._draft_tensor_capture.save_tensor_captures(...)   # write to disk, _instance = previous
```

``` python
# Target model (in execute_model)
self._target_tensor_capture.setup_tensor_capture()
model_output = self.model(**model_kwargs)
self._target_tensor_capture.save_tensor_captures(positions, is_prefill, req_ids)

# Draft model (in propose)
self._draft_tensor_capture.setup_tensor_capture()
draft_output = self.drafter.propose(...)
self._draft_tensor_capture.save_tensor_captures(positions, is_prefill, req_ids)

# Vision encoder (in _execute_mm_encoder)
self._vision_tensor_capture.setup_tensor_capture()
encoder_outputs = self.model.embed_multimodal(...)
self._vision_tensor_capture.save_tensor_captures(positions, is_prefill, req_ids)
```

### Multi-Model Support

For speculative decoding and multimodal models, each sub-model has its own
`ModelCapture` instance with isolated registry and output directory.

Configuration:

``` python
"tensor_capture": {
    "modules": ["model.layers.0", "lm_head"],       # target model
    "draft_modules": ["model.layers.0"],             # draft model
    "capture_dir": "/tmp/captures",
    "capture_filter": ["model.layers.0"],            # write-time filter
}
```

For vision models, use the `visual.` prefix:

``` python
"modules": [
    "visual.blocks.0",              # vision encoder
    "language_model.layers.0",      # text decoder
]
```

## Limitations

- **Sharding must match**: CPU and Neuron runs must use identical sharding. If sharding differs, users must reconcile tensor shapes manually.
- **Multi-host captures**: Each rank saves locally. For multi-host setups, users must aggregate files from each host for comparison.
- **Single tensor per module**: Currently captures only the first tensor from module output. Complex output structures (nested tuples, dicts) are not fully supported.
