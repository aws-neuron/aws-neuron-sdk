# Asynchronous Execution Framework Capability

<!-- meta: description: Asynchronous execution double buffering framework -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-22 -->

## Overview

The vLLM Neuron framework provides asynchronous execution capabilities that enable overlapped model inference on AWS Neuron hardware. This document demonstrates the framework's async execution features through two showcase examples that highlight double buffering patterns for improved throughput.

## Problem Statement

### Sequential Execution Bottleneck

In sequential mode, model execution and result processing are serialized:

``` text
Sequential Mode:
Request 1: [Model Exec] → [Wait & Process Result] → [Submit Next]
Request 2:                                           [Model Exec] → [Wait & Process Result]
Request 3:                                                                                   [Model Exec]

Problem: Neuron hardware sits idle during Python-level result processing
```

This serialization means the next model execution must wait for the previous request's result processing to complete, underutilizing Neuron hardware during Python overhead periods.

## Framework Solution: Async Double Buffering

The vLLM Neuron framework enables async execution where Python-level overheads are overlapped with Neuron hardware execution.

### Architecture

Double buffering allows submitting the next model execution before waiting on result processing:

``` text
Async Double Buffering:
Request 1: [Model Exec A] ────────────────→ [Result A Ready]
Request 2:      [Model Exec B] ────────────────→ [Result B Ready]
Request 3:           [Model Exec C] ────────────────→ [Result C Ready]
Python:    [Submit A] [Submit B] [Process A] [Submit C] [Process B]

Benefit: Python processing overhead overlapped with Neuron execution
```

### Key Limitation: Data Dependencies

**Critical Constraint**: Async execution requires that the next model invocation cannot depend on CPU-processed results from the previous invocation.

**LLM Example - CPU Sampling (Cannot Use Async)**:

``` python
# This pattern CANNOT use async execution
for step in range(sequence_length):
    logits = model(input_ids)                    # Neuron execution
    next_token = sample_on_cpu(logits)           # CPU processing
    input_ids = torch.cat([input_ids, next_token])  # Next input depends on CPU result
```

**LLM Example - On-Device Sampling (Can Use Async)**:

``` python
# This pattern CAN use async execution
for step in range(sequence_length):
    input_ids = model_with_sampling(input_ids)   # Neuron execution + on-device sampling
    # No CPU dependency - next input ready immediately
```

**Requirement**: To leverage async execution patterns, sampling and next-token generation must occur on-device rather than on CPU.

## Framework Showcase Examples

The framework's async capabilities are demonstrated through two comprehensive examples that showcase different execution patterns and use cases.

### Showcase 1: Single Process Comparison

**File**: `examples/vllm_neuron/basics/async/async_vs_sequential.py`

**Purpose**: Demonstrates the framework's ability to switch between async and sequential execution modes within a single process, showcasing the performance benefits of double buffering.

**Sample Usage**:

``` python
class Inference:
    def _run_double_buffered(self, inputs):
        # Framework handles async execution automatically
        input1 = torch.tensor([inputs[0]]).to("neuron:0")
        input2 = torch.tensor([inputs[1]]).to("neuron:0")

        current_output = self.model(input1)
        next_output = self.model(input2)

        # Framework manages buffer swapping
        for i in range(2, len(inputs)):
            results.append(current_output.to("cpu").item())
            next_input = torch.tensor([inputs[i]]).to("neuron:0")

            # Swap buffers and start next request
            current_output = next_output
            next_output = self.model(next_input)
```

### Showcase 2: Multiprocessing Distributed Execution

**File**: `examples/vllm_neuron/basics/async/async_llm.py`

**Purpose**: Demonstrates the framework's distributed execution capabilities with async patterns across multiple worker processes, showcasing scalability and coordination.

**Framework Integration**:

``` python
def worker_process(rank, world_size, input_queue, output_queue, load, master_port, use_double_buffer):
    # Framework configures each worker for async execution
    if use_double_buffer:
        # Framework manages double buffering per worker
        input1 = torch.tensor([input_queue.get()]).to("neuron:0")
        input2 = torch.tensor([input_queue.get()]).to("neuron:0")
        current_output = model(input1)
        next_output = model(input2)

        while True:
            new_input = input_queue.get()
            if new_input == "STOP":
                break

            # Framework handles buffer coordination
            output_queue.put(current_output.to("cpu").item())
            current_output = next_output
            next_output = model(torch.tensor([new_input]).to("neuron:0"))
```

**Process Coordination**:

``` python
class AsyncExecutor:
    def __init__(self, world_size, model_load, use_double_buffer=True):
        # Framework manages distributed worker processes
        for rank in range(world_size):
            p = mp.Process(target=worker_process, args=(..., use_double_buffer))
            p.start()
```

### Showcase 3: Data dependency

You can find the extended examples that handle data dependencies in auto-regressive workflow in the files linked below. Please note that the model code has been updated to include on-device sampling, which avoids moving data to CPU for processing between invocations.

**File**: `examples/vllm_neuron/basics/async/async_vs_sequential_autoregressive.py`

**File**: `examples/vllm_neuron/basics/async/async_llm_autoregressive.py`

## Performance Characteristics

**Performance Comparison Screenshots**:

Here is a comparison of the system profile for `examples/vllm_neuron/basics/async/async_llm.py`.

This shows that in synchronous execution mode, we get an overhead of 437μs. This includes dispatch overheads, tensor allocations, and Python overheads. For 437μs between each invocation, the hardware remains idle.

The same workload when run in asynchronous mode shows no idle time. This is because hardware executions are queued up and there is no dependency on CPU processing.

The profiled NEFF execution time is 5.18 ms. Overall execution time ranges from approximately 5.20 ms to 5.70 ms.

We also explore jitter across each core. In a trn2 instance at maximum world size, we can run 64 processes, each executing on one Neuron core. For the `async_llm.py` example, we see jitter on the order of ~80μs.

### How can I reproduce these profiles?

### Showcase 1: Single Process

``` bash
# Demonstrate framework async execution
python async_vs_sequential.py --double-buffer

# Compare with framework sequential mode
python async_vs_sequential.py --sequential
```

### Showcase 2: Multiprocessing

``` bash
# Framework distributed async execution
python async_llm.py --double-buffer

# Framework scalability demonstration (64 workers)
python async_llm.py --double-buffer --world-size 64

# Framework distributed sequential mode
python async_llm.py --sequential --world-size 64
```

### Showcase 3: Data dependency (continued)

``` bash
# Demonstrate framework async execution
python async_vs_sequential_autoregressive.py --double-buffer

# Compare with framework sequential mode
python async_vs_sequential_autoregressive.py --sequential

# Framework distributed async execution
python async_llm_autoregressive.py --double-buffer

# Framework scalability demonstration (64 workers)
python async_llm_autoregressive.py --double-buffer --world-size 64

# Framework distributed sequential mode
python async_llm_autoregressive.py --sequential --world-size 64
```

### Profiling and Analysis

``` bash
# Profile framework async capabilities
NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_OUTPUT_DIR=./output_double \
    python async_vs_sequential.py --double-buffer

# Profile framework sequential mode
NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_OUTPUT_DIR=./output_sequential \
    python async_vs_sequential.py --sequential

# Analyze framework execution patterns
neuron-explorer view -d ./output_double --output-format perfetto
# View system_profile.pftrace on https://ui.perfetto.dev/
```
