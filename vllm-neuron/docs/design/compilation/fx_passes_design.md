# FX Passes Architecture Design

<!-- meta: description: FX passes architecture and design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-22 -->

> This document describes the FX passes stage of the [compilation pipeline](index.md) — the graph transformations that run between FX tracing and HLO lowering.

## Overview

Scalable architecture for managing FX graph transformations in the vLLM Neuron compilation pipeline.

## Architecture

### Directory Structure

``` text
vllm_neuron/fx_passes/
├── __init__.py              # get_default_pass_manager()
├── base.py                  # FXPass interface
├── aliasing_pass.py         # AliasingOutputRewritePass
├── backend_config_pass.py   # NkiKernelWriteBackendConfigPass
├── collective_replica_groups_pass.py  # CollectiveReplicaGroupsPass
├── device_rewriter.py       # DeviceRewriterPass
├── inplace_rewrite_pass.py  # InPlaceToOutOfPlacePass
└── pass_manager.py          # FXPassManager with timing logs
```

### Base Pass Interface

``` python
class FXPass(ABC):
    @abstractmethod
    def run(self, gm: torch.fx.GraphModule, **kwargs) -> torch.fx.GraphModule:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
```

### Pass Manager with Debug Logging

``` python
def run_passes(self, gm: torch.fx.GraphModule, **kwargs) -> torch.fx.GraphModule:
    for pass_obj in self.passes:
        start_time = time.perf_counter()
        gm = pass_obj.run(gm, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        self.logger.debug(f"FX Pass '{pass_obj.name}' completed in {elapsed_time:.4f}s")
    return gm
```

### Default Pass Manager

``` python
def get_default_pass_manager() -> FXPassManager:
    manager = FXPassManager()
    manager.add_pass(DeviceRewriterPass())
    manager.add_pass(AliasingOutputRewritePass())
    manager.add_pass(InPlaceToOutOfPlacePass())
    manager.add_pass(NkiKernelWriteBackendConfigPass())
    manager.add_pass(CollectiveReplicaGroupsPass())
    return manager
```

Pass ordering matters. The aliasing pass must run before the in-place rewrite pass because it analyzes mutations while in-place operations are still present in the graph.

### Individual Pass Documentation

<div class="toctree" maxdepth="1">

device_rewriting_fx_pass aliasing_output_rewrite_pass inplace_to_outofplace_pass

</div>

## Integration

### Pipeline Integration

``` python
def compile(gm: torch.fx.GraphModule, example_inputs, options: dict = {}):
    # Apply FX passes before XLA tracing
    if options.get('enable_fx_passes', True):
        pass_manager = get_default_pass_manager()
        gm = pass_manager.run_passes(gm, target_device='xla')

    # Existing XLA tracing
    model = torch_neuronx.trace(lambda *args: gm(*args), ...)
```

### Configuration Options

- `enable_fx_passes` (bool): Enable/disable FX pass pipeline (default: True)
- `target_device` (str): Target device for passes (default: 'xla')

## Benefits

- **Modularity**: Self-contained passes with clear interface
- **Extensibility**: New passes added by implementing `FXPass`
- **Observability**: Debug logs track individual pass performance
- **Testability**: Individual passes can be unit tested independently

## Adding New Passes

1. **Implement FXPass interface**:

``` python
class MyNewPass(FXPass):
    @property
    def name(self) -> str:
        return "my_new_pass"

    def run(self, gm: torch.fx.GraphModule, **kwargs) -> torch.fx.GraphModule:
        # Transform logic here
        return gm
```

1. **Add to default pass manager**:

``` python
def get_default_pass_manager() -> FXPassManager:
    manager = FXPassManager()
    manager.add_pass(DeviceRewriterPass())
    manager.add_pass(MyNewPass())  # Add new pass
    return manager
```

1. **Unit test the pass independently**
