# FX Device Rewriting for XLA Compilation

<!-- meta: description: FX device rewriting pass for XLA compilation -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-22 -->

> This pass runs as part of the [FX passes stage](index.md) of the compilation pipeline. It rewrites hardcoded CPU device references so the graph compiles correctly on XLA/Neuron devices.

## Problem Statement

During torch.compile tracing, tensor devices get hardcoded based on the tracing environment (typically CPU). This causes graph breaks during XLA compilation when vLLM Neuron backend moves tensors to XLA devices.

### Root Cause

When users write code that creates new tensors by copying device from existing tensors:

``` python
def user_function(input_tensor):
    # input_tensor is on CPU during tracing
    # User creates new tensor by copying device from input
    output = torch.empty(64, dtype=torch.float32, device=input_tensor.device)

    # Some collective operation that returns XLA tensor
    reduced = torch.ops._c10d_functional.reduce_scatter_tensor(input_tensor, 'sum', 4, '1')
    result = torch.ops._c10d_functional.wait_tensor(reduced)

    # Copy result into output buffer
    output.copy_(result)
    return output
```

**Traced FX Graph (during CPU tracing):**

``` text
placeholder  l_input_tensor_ L_input_tensor_                      ()                {}
call_function output      <built-in method empty of type object at 0x7f273d6ca460> (64,)               {'dtype': torch.float32, 'device': device(type='cpu')}
call_function tensor      _c10d_functional.reduce_scatter_tensor          (l_input_tensor_, 'sum', 4, '1') {}
call_function res       _c10d_functional.wait_tensor               (tensor,)             {}
call_method  copy_      copy_                           (output, res)           {}
output     output_1     output                          ((output,),)           {}
```

### The Issue

**During XLA compilation:**

- The trace operation moves `l_input_tensor_` to XLA device
- `res` from collective ops is on XLA device
- But `output` is still hardcoded as `device(type='cpu')` in the graph
- `copy_(output, res)` tries to copy between XLA→CPU, causing graph break

The pattern `device=input_tensor.device` gets "frozen" to `device(type='cpu')` during tracing rather than adapting to the runtime XLA device.

## Implemented Solution

### Overview

The DeviceRewriterPass implements a simple, reliable single-pass algorithm that rewrites all non-XLA device parameters to XLA. This universal approach ensures compatibility with XLA compilation while maintaining simplicity and robustness.

### Algorithm Design

**Core Strategy**: Universal device rewriting - for any node that has a device parameter in kwargs, if the device is not XLA, replace it with XLA.

**Implementation Logic**:

1. Single pass through all nodes in the FX graph
2. For each node with a 'device' parameter in kwargs:
    - Extract the current device type (handling various device formats)
    - If current device is not the target device (default: XLA), rewrite it
    - Preserve the original device format (string vs torch.device object)
    - Replace the node in the graph with updated device metadata

### Algorithm Flow

``` text
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   FX Graph      │    │   Single Pass:       │    │   Updated Graph     │
│   Input         │───▶│  Universal Rewrite   │───▶│   with XLA devices  │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
                                │
                                ▼
                       ┌─────────────────────┐
                       │ For each node:      │
                       │ • Has device param? │
                       │ • Not target device?│
                       │ • Rewrite & replace │
                       └─────────────────────┘
```

### Device Format Preservation

The implementation preserves the original device format to maintain graph compatibility:

``` python
# Original was a string, replace with string
if isinstance(current_device, str):
    new_kwargs['device'] = target_device
# Original was a torch.device object, replace with torch.device object  
elif hasattr(current_device, 'type'):
    new_kwargs['device'] = torch.device(target_device)
# Fallback: use string format
else:
    new_kwargs['device'] = target_device
```

### Why Universal Rewriting?

**Simplicity**: The universal approach eliminates complex device propagation analysis, making the implementation straightforward and maintainable.

**Reliability**: By rewriting all device parameters, we ensure no edge cases are missed where device mismatches could cause graph breaks.

**Performance**: Single-pass algorithm with O(n) complexity where n is the number of nodes in the graph.

**Correctness**: Since all tensors will eventually be moved to XLA device during compilation, preemptively setting device metadata to XLA is semantically correct.

### Trade-offs

**Pros**:

- Simple implementation and maintenance
- Guaranteed to catch all device-related issues
- No complex dependency analysis required
- Robust against future PyTorch changes

**Cons**:

- May rewrite devices for tensors that don't strictly need it

## Compatibility

The current implementation is designed to be forward-compatible with PyTorch changes:

- Uses standard FX graph manipulation APIs
- Handles various device format representations
- Includes fallback logic for unknown device formats
