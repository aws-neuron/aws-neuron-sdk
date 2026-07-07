.. meta::
    :description: Glossary of terms used in Neuron Explorer profiling, hardware, metrics, and UI.
    :date-modified: 06/18/2026

.. _neuron-explorer-glossary:

Neuron Explorer Glossary
========================

Definitions of key terms used throughout Neuron Explorer documentation,
the profiling UI, and profile data tables.

.. contents:: On this page
   :local:
   :depth: 2


File types and artifacts
------------------------

NEFF
   Neuron Executable File Format. The compiled binary that runs on NeuronCores.
   Contains the computation graph, instructions, and metadata. Produced by the
   Neuron compiler (``neuronx-cc``).

NTFF
   Neuron Trace File Format. Device-level profiling output captured during
   execution on NeuronCores. Each NTFF is paired with a NEFF for analysis.
   Contains raw hardware event data (instructions, DMA packets, semaphores).

.pb files
   Protocol Buffer files containing system-level trace data:

   - ``trace_info.pb`` — Runtime event trace metadata
   - ``ntrace.pb`` — Neuron runtime trace
   - ``cpu_util.pb`` — CPU utilization data
   - ``host_mem.pb`` — Host memory usage data

kernel_debug_info.json
   Debug metadata inside a NEFF enabling source code linking for NKI kernels.
   Maps hardware instructions to NKI source file and line number. Generated
   when ``NKI_DEBUG_INFO=True`` is set before compilation.

Parquet files
   Columnar data files containing processed profile data. After ingestion,
   Explorer converts raw device profiles (NEFF/NTFF) and system profiles (.pb
   files) into queryable tables (Instruction, DmaPacket, DmaPacketAggregated,
   Summary, etc.).

trace.json
   Framework-level trace (PyTorch/JAX profiler output) showing CPU operations
   and call stacks. Must be in the same directory as system trace files for
   Explorer to display framework events alongside runtime events.


Hardware
--------

NeuronCore (NC)
   A single compute core on a Trainium or Inferentia chip. Contains multiple
   execution engines (DMA, Tensor, Vector, Scalar, Sync, GPSIMD) operating
   in parallel.

VNC (Virtual NeuronCore)
   A logical grouping of physical NeuronCores. Configured via
   ``NEURON_RT_VIRTUAL_CORE_SIZE``. A VNC of size 2 means two physical cores
   act as one logical core.

DMA Engine
   Direct Memory Access engine that moves data between HBM and SBUF.
   Operates independently of compute, enabling data movement to overlap
   with computation (pipelining).

Tensor Engine (TE)
   Systolic array for matrix multiplication. Executes ``MATMUL`` and
   ``LDWEIGHTS`` instructions. Performance depends on tile size alignment
   with hardware dimensions (free dim, partition dim).

Vector Engine (VE)
   Engine for elementwise ops, activations, and reductions. Executes
   ``TENSOR_TENSOR``, ``SCALAR``, and ``REDUCE`` instructions.

Scalar Engine
   Engine for activation functions and DMA coordination.

GPSIMD Engine
   General-Purpose SIMD engine for custom C code and operations not
   supported by other engines.

Sync Engine
   Handles synchronization between engines using semaphores.

SBUF (State Buffer)
   On-chip SRAM. 256 KiB per partition. Data is staged here from HBM
   before compute engines process it. Much faster than HBM but limited
   in capacity.

PSUM (Partial Sum Buffer)
   Accumulator memory for matrix multiplication results. 8 banks.
   Tensor Engine writes partial sums here.

HBM (High Bandwidth Memory)
   Off-chip DRAM with large capacity (e.g., 32 GB per NeuronCore on Trn2).
   Higher latency than SBUF.

NeuronLink
   High-speed interconnect between NeuronCores on the same chip or across
   chips within a node. Used for collective communication.

LNC (Logical NeuronCore)
   A logical NeuronCore identifier used in system profile summaries and
   device profile tables. Represents the logical index of a NeuronCore
   as seen by the runtime. Not to be confused with NeuronLink (which
   is sometimes abbreviated LNC in hardware documentation).

Partition (P)
   A parallel processing lane in SBUF and compute engines. Hardware supports
   up to 128 partitions. Full utilization (P=128) is critical for peak
   performance.

PCIe
   Host-to-device interconnect. Used for Host-to-Device (H2D) and
   Device-to-Host (D2H) data transfers. Appears as an engine track in
   system profiles.

EFA (Elastic Fabric Adapter)
   High-performance network interface for inter-node communication in
   distributed workloads. Relevant for multi-node profiling.


Metrics
-------

MFU (Model FLOPs Utilization)
   Ratio of *useful model FLOPs* to hardware's theoretical peak. "Useful"
   means only FLOPs contributing to the model's mathematical result (excludes
   padding, transposes, overhead). Key measure of hardware effectiveness.

HFU (Hardware FLOPs Utilization)
   Ratio of *total hardware FLOPs* (including overhead) to peak. Always ≥ MFU.
   A gap between HFU and MFU indicates wasted compute (padding, transposes).

MBU (Memory Bandwidth Utilization)
   Ratio of actual HBM bandwidth used to theoretical peak. High MBU + low MFU
   = memory-bound. Low MBU + high MFU = compute-bound.

mm_arithmetic_intensity
   Ratio of matrix multiplication FLOPs to total HBM transfer bytes.
   Compare against ``peak_flops_bandwidth_ratio`` to classify whether a
   workload is compute-bound or memory-bound.

peak_flops_bandwidth_ratio
   Hardware's theoretical peak FLOPs / peak memory bandwidth. The crossover
   point for classifying workloads as compute-bound (above) or memory-bound
   (below).

adjusted_flops
   FLOPs count adjusted for data type throughput. FP8×FP8 achieves 2x
   throughput of BF16, so a 0.5x multiplier normalizes to peak. For
   mixed-precision (FP8×BF16), execution runs at BF16 speed — multiplier
   should be 1.0x.

raw_flops
   Unadjusted FLOPs count. Literal number of floating-point operations
   executed by hardware.

total_time
   Total on-device execution time in seconds. Excludes host-device data
   movement and framework overhead.

useful_read_percent
   Fraction of HBM reads that are "useful":
   ``(hbm_read_bytes - hbm_reload_bytes) / hbm_read_bytes``.
   Low values = data being reloaded redundantly from HBM.

average_dma_size
   Average DMA transfer size. Larger = better efficiency per byte.
   Target ≥ 32768 bytes.


FLOPs breakdown
^^^^^^^^^^^^^^^

model_flops
   Percentage of Tensor Engine FLOPs performing useful matrix operations.

transpose_flops
   Percentage of Tensor Engine FLOPs performing data movement (transposes).
   Ideally minimal.

active_flops
   Percentage of FLOPs during active periods where the engine was not
   fully utilized (undersized tiles, poor pipelining).

throttled_flops
   FLOPs wasted due to hardware throttling. Worth investigating if
   significant.


Memory metrics
^^^^^^^^^^^^^^

sbuf_reload_ratio
   Fraction of SBUF access that is reloading previously-evicted data from
   HBM. High values (>60%) = spilling.

sbuf_save_ratio
   Fraction of SBUF access that is saving (evicting) data to HBM to make
   room.

weight_queue_bytes
   Total bytes of weight data queued for loading. If much larger than
   ``weight_size_bytes``, weights are being redundantly reloaded.

hbm_reload_bytes
   Bytes reloaded from HBM that were previously evicted from SBUF.


Profiling concepts
------------------

Device profile
   Profile of hardware execution on NeuronCores. Per-instruction timing,
   DMA activity, semaphores, engine utilization. Uses device clock
   (cycle-accurate).

System profile
   Profile of runtime/framework activity on host CPU. Framework ops, runtime
   events, CPU overhead. Uses host CPU clock.

Eager mode
   PyTorch execution mode with dynamic graphs. Compiles many small NEFFs
   (potentially hundreds), one per operator. Contrast with compiled/graph
   mode (fewer, larger NEFFs).

Session-based profile
   Captures device activity across an entire runtime session (not just one
   NEFF execution). Required for inference workloads and vLLM.

Multi-rank profile
   Profile from a distributed job with one NEFF/NTFF pair per rank. Enables
   straggler detection and communication bottleneck analysis.

MPMD (Multiple Program Multiple Data)
   Distributed execution where different ranks run different NEFFs. Contrast
   with SPMD where all ranks run the same NEFF.

DGE notifications
   Data Gather Engine notifications that label DMA transfers with tensor
   variable names. Without them, DMAs show ``variable: unknown``.


Timeline events
---------------

nc_exec_running
   Runtime event marking when a NEFF is executing on a NeuronCore. The
   bridge between system and device views — clicking it reveals device
   execution.

kbl_exec_wait
   Runtime event indicating the host is waiting for NeuronCore execution
   to complete. Long durations = host blocked on device.

kmgr_exec_core
   Kernel manager event showing dispatch to a specific NeuronCore.

Framework events (aten ops)
   CPU-side operations from PyTorch (``aten::matmul``, ``aten::linear``,
   etc.) visible in the system timeline.

HLO (High-Level Operations)
   XLA compiler intermediate representation. The Hierarchy Viewer organizes
   execution at this level, between framework ops and hardware instructions.

record_function regions
   Named annotations added via ``torch.profiler.record_function`` (PyTorch)
   or ``jax.profiler.TraceAnnotation`` (JAX). Group related operations in
   the timeline.


UI and viewer terms
-------------------

Device Trace Viewer
   Zoomable timeline showing per-engine hardware instruction execution.
   The primary view for device-level analysis. Instructions are color-coded
   by their associated PyTorch operator.

System Trace Viewer
   Timeline showing runtime API events, framework operations, CPU
   activity, and host-device transfers. Uses host clock. Supports multiple
   grouping modes (CPU vs Device, NeuronCore, Thread, Process, Instance).

Hierarchy Viewer
   Tree-structured view organized by framework layers → HLO operations →
   hardware instructions. Right-clicking an operator highlights all
   corresponding instructions in the Device Trace Viewer.

Source Code Viewer
   Bidirectional linking between hardware instructions and source code.
   Click an instruction → see the source line. Click a source line →
   highlight all associated instructions. Requires ``NEURON_FRAMEWORK_DEBUG=1``
   (for model code) or ``NKI_DEBUG_INFO=True`` (for NKI kernels).

Summary Viewer
   Single-page performance overview showing key metrics, FLOPs breakdown,
   memory bandwidth, collective operations, and AI-generated recommendations.
   Supports region selection for per-layer analysis.

Database Viewer
   Widget for running custom SQL queries or natural-language queries against
   processed profile data (Parquet tables). Useful for ad-hoc analysis not
   covered by other viewers.

Tensor Viewer
   Widget displaying tensor information including names, sizes, shapes, and
   memory usage details (SBUF allocation per tensor).

Memory Viewer
   Widget visualizing SBUF memory allocation and usage patterns across
   partitions over time. Shows spill/reload activity and helps identify
   when the working set exceeds on-chip memory.

Region Highlighter
   Widget that automatically identifies optimization regions in profiles
   based on collective operations, hierarchy data, and kernel stack frames.
   Highlights regions of interest without manual annotation.

AI Recommendation Viewer
   Widget providing AI-powered bottleneck analysis and optimization
   recommendations for NKI profiles. Uses Amazon Bedrock (Claude) to analyze
   profile data and source code. Returns top 2-3 ranked recommendations with
   symptoms, suggested optimizations, and expected speedup.

Dependency Chain Viewer
   Widget showing dependency relationships between system profile events.
   Click a system event with a flow-id to see upstream (predecessors) and
   downstream (successors) events with connecting arrows.

Dependency highlighting
   Click an instruction to see predecessors (what it waited for) and
   successors (what waited for it). Shows data flow across engines.

Semaphore
   Hardware synchronization primitive. One engine increments; another waits
   before proceeding.

Annotations
   Named regions on the timeline. User-defined (via record_function) or
   created manually by right-clicking in the Device Trace Viewer. Support
   saving, loading, renaming, and time-difference calculation between markers.

CC box (Collective Communication box)
   Visual marker showing boundaries of a collective operation (AllReduce,
   AllGather, etc.).

Operator Table
   Widget aggregating hardware metrics into framework layers/operations.
   Shows MFU, data moved, time per operator. Rows expand progressively.

Overall Summary
   Metrics across the entire profile.

Current Selection Summary
   Metrics for the currently-visible time window. Updates on zoom.

Box Selection Summary
   Metrics for a rectangular region you select in the Device Trace Viewer.
   Activated via the box selection button or keyboard.

Event Details
   Panel showing full metadata for a selected instruction (opcode,
   operands, timing, source location, dependencies).

Profile Manager
   Landing page for uploading, browsing, and managing profiles.

Deep link
   URL preserving the exact view state (time range, zoom, selections,
   color scheme) for sharing. 

Layout Manager
   Save and load custom widget arrangements. Layouts persist across profiles.

flow-id
   A property on system profile events that indicates a dependency
   relationship. Events with matching flow-ids are connected in the
   Dependency Chain Viewer.

System Timeline Grouping Modes
   The System Trace Viewer supports multiple ways to organize events:

   - **CPU vs Device** (default) — groups by event source
   - **NeuronCore** — groups by individual NeuronCore
   - **Thread** — groups by thread identifier
   - **Process** — groups by process identifier
   - **Instance** — groups by EC2 instance

Host-to-Device / Device-to-Host transfers
   PCIe data transfer events visible in the System Trace Viewer. Each
   direction shows a transfer events track (individual transfers) and a
   transfer bandwidth track (throughput over time). Useful for identifying
   PCIe bottlenecks.

NKI instruction coverage
   Percentage of instructions on each engine (tensor, vector, scalar)
   that are NKI-generated vs compiler-generated. Shown in the NKI Engine
   Statistics chart. When below 50%, the Summary Viewer recommends writing
   NKI kernel code for those operations.


Collective operations
---------------------

AllReduce
   Reduce a tensor across all ranks and distribute the result back.

AllGather
   Gather tensors from all ranks and concatenate.

ReduceScatter
   Reduce across ranks, scatter different portions to different ranks.

AllToAll / AllToAllV
   Redistribute data between all ranks. "V" = variable-sized messages.

SB2SB (SBUF-to-SBUF)
   Collective mode transferring data directly between SBUF memories
   without HBM staging. Lower latency.

HBM Collectives
   Collectives staged through HBM. Higher capacity, higher latency.

CC-core
   Dedicated NeuronCore or stream for collective communication. Appears
   as separate tracks in the device timeline.

CC Active
   Percentage of time spent in collective communication operations.
   Shown as a column in the System Profile Summary device profiles table.

Collective operation outliers
   Operations whose duration significantly exceeds the group median for
   that operation type and size. The Summary Viewer's collective operations
   chart highlights these as scatter points far from the cluster, indicating
   potential straggler behavior or interference.


DMA categories
--------------

DMA Data Types
   How DMA transfers are categorized by what they carry:

   - **Instruction** — DMA triggered by compute instructions (loads/stores for compute)
   - **IO** — Input/output tensor transfers (model inputs and outputs)
   - **Weights** — Model weight loading from HBM to SBUF
   - **Dynamic** — Runtime-generated transfers not known at compile time

DMA Source Types
   How DMA transfers are categorized by what generated them:

   - **Static** — Compiler-generated transfers (determined at compile time)
   - **Software Dynamic** — GPSIMD-generated transfers (runtime decisions in custom code)
   - **Hardware Dynamic** — DGE (Data Gather Engine) hardware-generated transfers

Spill
   When the working set exceeds SBUF capacity, data is evicted ("spilled")
   to HBM to make room. The evicted data must be reloaded later when needed
   again, consuming HBM bandwidth for non-useful work.

Reload
   Loading previously-spilled data back from HBM into SBUF. Appears as
   ``hbm_reload_bytes`` in metrics. High reload ratios indicate the working
   set doesn't fit in on-chip memory.


System profile summary
----------------------

System Overview Card
   Aggregate information about the profiling session: number of instances,
   processes, total system profile time, cumulative device runtime, and
   total device profiles captured.

HBM Memory Usage Chart
   Line chart showing HBM memory usage over time, with a separate line per
   Logical NeuronCore when per-core data is available.

Device Profiles Table
   Table listing all device profiles in a system profile. Columns include
   Profile Name, LNC, Neuron Cores, Total Duration, Calls, MFU, HFU, MBU,
   and CC Active. Rows are expandable for per-engine details.


Additional hardware terms
-------------------------

psum_zero
   A flag in Tensor Engine instruction operands indicating whether the
   PSUM accumulator is zeroed before the operation begins. When
   ``psum_zero=1``, the accumulator starts fresh. When ``psum_zero=0``,
   the instruction accumulates onto existing PSUM contents.

fast weight load
   A compiler optimization that preloads weights into SBUF before they
   are needed, hiding load latency behind computation. Not shown as a
   distinct indicator in profiles — visible as overlapping DMA and
   Tensor Engine activity in the timeline.

Profiler overhead
   Additional resource consumption from profiling itself.
   ``ProfileMode.DEVICE`` reserves ~5 GB HBM on Trn2 for hardware
   notifications. DGE notifications add DMA traffic. System profiling
   adds CPU overhead for event recording. Designed to be <2% for
   system-only profiles.

NKI source location depth
   The Source Code Viewer currently maps instructions to a single NKI
   source file and line — not a full call stack within the kernel. If
   the same helper function is called from multiple sites, the profiler
   shows the helper's line but not which call site invoked it. Full
   intra-kernel stack trace is a planned feature.


Common instruction opcodes
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Opcode
     - Engine
     - Description
   * - MATMUL
     - Tensor
     - Matrix multiplication
   * - LDWEIGHTS
     - Tensor
     - Load weight data for matmul
   * - TENSOR_TENSOR
     - Vector
     - Elementwise operation between two tensors
   * - SCALAR
     - Vector
     - Scalar operation on tensor elements
   * - REDUCE
     - Vector
     - Reduction (sum, max, etc.)
   * - ACTIVATE
     - Scalar
     - Activation function
   * - DMA_DIRECT2D
     - DMA
     - Direct 2D data transfer
   * - EVENT_SEMAPHORE
     - Sync
     - Semaphore increment or wait
   * - NONZERO_WITH_COUNT
     - GPSIMD
     - Count non-zero elements


DMA packet fields
^^^^^^^^^^^^^^^^^

variable
   Tensor name associated with a DMA transfer. Shows ``unknown`` when DGE
   notifications are not enabled or for collective DMAs.

queue_type
   DMA queue classification. ``instruction`` often indicates a collective DMA.

queue_idx
   DMA queue index. ``11`` is commonly associated with collective operations
   (not guaranteed).

DMA direction (src/dst)
   DMA transfers move data between memory regions. The profiler shows
   source and destination addresses but does not currently label them as
   "read" or "write" explicitly. Determine direction from context:
   HBM address → SBUF address = read (load); SBUF → HBM = write (store).

Skipped DMA
   A DMA transfer that was scheduled but not executed (e.g., due to
   conditional logic or runtime-skipped regions). Skipped DMAs do not
   contribute to ``DmaTransferTotalBytes`` in the Summary but their static
   allocation may still appear in ``StaticDmaSize``. This can cause
   ``StaticDmaSize > DmaTransferTotalBytes``.

Unknown opcode
   Instructions showing as "unknown" or "Operation(N)" in the timeline
   indicate the profiler's instruction decoder doesn't recognize the
   opcode. Common examples: ``activate2``, ``select_reduce``,
   ``range_select``, ``NONZERO_WITH_COUNT``. These are typically newer
   instructions added after the profiler version was built. Updating
   ``neuron-explorer`` to the latest SDK version usually resolves this.


Processing status
-----------------

PROCESSING
   Profile uploaded, being converted to viewable Parquet tables.

ERROR_PROCESS_INCOMPLETE
   Processing failed. Common causes: profile too large, malformed artifacts,
   disk space, or NEFF/NTFF UUID mismatch.

No metadata found for system profile
   Upload directory structure doesn't match expected format. Missing
   ``trace_info.pb`` or layout incorrect.
