.. meta::
   :description: Localize performance bottlenecks in a vLLM Neuron device and system profile on Trainium using the Neuron Framework Profile Analysis skill, with session workload timing, roofline bounds, collective partitioning, and per-component attribution
   :keywords: profile analysis, vllm, vllm-neuron, roofline, performance bounds, collectives, attribution, system trace, session workload, MFU, MBU, neuron-explorer, parquet, Trainium, trn2, Neuron Agentic Development
   :date-modified: 2026-10-08

.. _neuron-framework-profile-analysis-vllm-neuron:

Deep dive: Localize performance bottlenecks with the Profile Analysis skill
===========================================================================

**Why read this guide?** You have a capture of a vLLM Neuron model and you need to know
where the time goes: which compiled graphs ran and how often, which engine binds a step, how
much of it is collective traffic, how far the measured step sits above its hardware floor,
and which part of the model is responsible. This skill turns a capture into those answers.

It reads **both halves of a capture**. The system trace gives the session-level picture, and
is where the step time that every device-level ratio divides by comes from. The device trace
gives the per-instruction, per-engine detail inside one step. Neither answers the question
alone: the device trace cannot tell you which step is representative, and the system trace
cannot tell you what happened inside it.

**How to use this guide:** Read :ref:`analysis-vllm-maturity` first — it tells you which
outputs to trust, which matters here more than in most tooling. Jump to
:ref:`analysis-vllm-workflow` for the stages, or :ref:`analysis-vllm-concepts` for what the
numbers mean.

**Use this skill interactively.** Unlike a capture, an analysis has no single correct
output: the useful answer depends on the question. Tell the agent what you are trying to
decide — whether a kernel change paid off, why decode is slower than the roofline predicts,
whether a tensor-parallel degree is collective-bound — and it will drive the stages that
bear on it. Given no direction it will produce a general survey, which is a reasonable
starting point and rarely the thing you needed.

**You need experience with** reading a roofline argument, pandas or SQL over parquet, and
the vLLM Neuron serving configuration whose profile you are analyzing.

.. _analysis-vllm-maturity:

Maturity: which outputs to trust
--------------------------------

.. important::
   **Ingestion is stable. The derived analysis is approximate.** Treat a floor, a ratio, or
   a gap attribution as an indication of where to look rather than a number to quote in a
   commitment. What is read directly from the profile is reliable: row counts, table
   inventories, engine active-wall times, collective counts, and the session timing of
   :ref:`Stage 3 <analysis-vllm-workflow>`.

Known limits, stated plainly, so a reader can calibrate:

* **Latency floor (roofline) is self-derived by default.** The skill computes its floors from
  the device constants recorded in the capture, so a floor is internally consistent rather
  than independently confirmed. Cross-validating against an analytical model is strongly
  recommended and is what turns a floor into evidence — see
  :ref:`analysis-vllm-entitlement` and the
  :ref:`Hardware Entitlement deep dive <neuron-hardware-entitlement>`.
* **There is no report template.** The skill ships a worked example report, and agents
  reproduce its shape with varying fidelity. Two analyses of the same capture may lay out
  the same findings differently.

Prerequisites
-------------

* **A completed capture**, with a ``manifest.json``. See
  :ref:`neuron-framework-profiling-vllm-neuron`.
* **A capture taken with at least two recorded iterations.** At one iteration
  ``CcOp.parquet`` is empty and the collective partitioning stage cannot run at all.
* **Roofline reasoning.** What a memory floor and a compute floor are, and why the larger
  of the two bounds the step.
* **Parquet querying.** pandas, pyarrow, or DuckDB over the ingested tables.
* **Logical NeuronCore semantics.** At the default LNC=2 a profile covers both physical
  cores of one Logical NeuronCore, which changes how times must be aggregated. See
  :ref:`active-wall time <analysis-vllm-activewall>`.

Hardware and software requirements
----------------------------------

Analysis is a host-side activity. No Trainium device is required once the capture exists,
so this can run on any machine the artifacts can reach.

* **neuron-explorer**, from ``aws-neuronx-tools``, to ingest the capture.
* **pandas and pyarrow**, or DuckDB, to query the result.
* **Host memory and disk.** The binding constraint. Ingesting a multi-gigabyte device
  trace needs tens of GB of RAM, and the parquet output is roughly an order of magnitude
  larger than the trace. A 4.9 GB trace takes about 40 minutes to ingest and yields a
  ``DmaPacket`` table of about 2.2 GB on its own.

.. _analysis-vllm-workflow:

Workflow
--------

Stage 1. Ingest to parquet
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Read the manifest first.** It resolves the two nested glob paths — the
``i-<instance>_pid_<pid>/<timestamp>/`` session directory and the NEFF hashes from
``trace_info.pb`` — so nothing has to be searched for. Every path in it is absolute. If a
capture has no manifest, because it predates the contract or died before teardown, fall
back to locating the session directory by hand; the manifest is a shortcut, not a
requirement.

**Stage a minimal capture directory**, containing the system traces, the NEFFs, and one
rank's device trace per NEFF. Ingesting more ranks multiplies cost without adding
information, because the ranks run the same graph.

**Ingest with** ``neuron-explorer view --ingest-only``, pointing ``--data-path`` under the
project root so the parquet joins this run's evidence trail rather than landing in the
tool's default location.

.. note::
   ``neuron-explorer view`` binds a fixed port even in ingest-only mode, so two phases
   cannot be ingested concurrently on one host. Ingest them in sequence.

**Verify quality before analyzing.** Check that the expected tables exist with non-zero
rows. Expect roughly 47 tables per phase. Two absences are diagnostic rather than
mysterious:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Observation
     - Meaning
   * - ``CcOp.parquet`` is empty
     - The capture recorded only one iteration. Collective aggregation needs begin/end
       pairing across at least two. The collective stage cannot run; recapture.
   * - ``DmaPacket`` tables are empty
     - DMA packet metadata was not enabled at server start. DMA-efficiency bounds are
       unavailable; everything else works.

Stage 2. Identify the hardware constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read them from the capture's metadata rather than from a datasheet, so the floors derive
from the machine that produced the trace.

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - Constant
     - Example
     - Use
   * - ``hbm_ddr_bandwidth``
     - 716 GB/s
     - Memory floor at LNC=2. Peak for one HBM stack, shared across both cores.
   * - ``dma_ddr_bandwidth``
     - 435 GB/s
     - Memory floor at LNC=1 only.
   * - Tensor Engine peak
     - 157.3 TFLOPS
     - Compute floor. Derived from the engine's row and column counts and clock, times
       ``num_physical_cores``.
   * - ``num_physical_cores``
     - 2
     - 2 at LNC=2. The profile already covers both, so the peaks above are combined
       values.

.. important::
   At LNC=2, use ``hbm_ddr_bandwidth`` and the combined Tensor Engine peak. Using the
   single-core constants on an LNC=2 profile halves both floors and makes a healthy step
   look pathological.

Stage 3. Profile the session from the system trace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the system-level half of the analysis, and it comes first because it decides which
step the device-level stages look at. The system trace records every NEFF dispatch during
the run. From it, establish:

* **which compiled graphs executed**, and how many times each ran;
* **the execution pattern** — the order of dispatches, which distinguishes a prefill from
  a steady-state decode without trusting the run's label;
* **the mean duration** of each graph;
* **the inter-dispatch gaps** in steady state, which are host-side time the device trace
  cannot see;
* **a throughput estimate** derived from the trace itself, independent of the benchmark's
  own reported number.

A prefill capture normally shows two NEFFs — the prefill graph and the batch-size-1 decode
graph that the request warmed — so filter by the prefill NEFF hash.

This stage also establishes the **step time** that every later ratio divides by, taken from
the live dispatch records. Keep it as a cross-check on the device-side execution window: the
two are measured differently, so agreement between them is evidence that the right step was
analyzed. A capture in this family reported 208,137.8 µs from the dispatch records against a
208,296.2 µs device window — a 0.08% spread, which is what a healthy pairing looks like.

Stage 4. Compute per-NEFF performance bounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each NEFF, compute the memory floor and the compute floor, then report the measured
step against the larger of the two, the per-engine active time with the binding engine
identified, and the gap families — memory-side (DMA idle, DMA inefficiency) and
compute-side (engine idle, underutilization, transposes).

Where an analytical floor is available, compare against it here rather than trusting the
trace-derived floor alone — see :ref:`analysis-vllm-entitlement`.

Three adaptations apply to framework-compiled NEFFs and are easy to get wrong:

* Use ``hbm_ddr_bandwidth``, not ``dma_ddr_bandwidth``, for the memory floor.
* Use the combined Tensor Engine peak for the compute floor.
* Skip the no-reloads variant of the memory floor. It is not well defined here, because
  KV cache tensors are allocated at runtime and the recorded tensor sizes do not reflect
  an algorithmic minimum.

.. _analysis-vllm-activewall:

.. important::
   **Engine active time means active-wall, not summed instruction time.** Report the
   interval-merged wall-clock time each engine was busy. A raw sum double-counts both the
   two physical cores of an LNC=2 logical core and overlapping issue slots on one engine,
   so it can exceed the step — a 44 ms "busy" figure on a 21 ms step is the sum, not the
   wall. Merge intervals, and merge across both physical cores into one timeline.

Stage 5. Partition each NEFF by collectives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Collectives are the only synchronization points in the compiled graph, which makes them
natural segment boundaries. Partition the step at them, then bound each segment
independently.

The collective mix also identifies the tensor-parallel strategy: an all-reduce pattern and
a reduce-scatter plus all-gather pattern are different shardings, and reading which one
executed is faster than inferring it from the configuration.

Summed collective wall time must not exceed the step. If it does, the aggregation is
wrong — most often summed rather than interval-merged.

Stage 6. Roll up by component and investigate the hotspot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Group the instruction table by the framework module scope to get time and HBM traffic per
model component, then investigate whichever component binds. See
:ref:`analysis-vllm-attribution` for how the grouping works and what it can and cannot
tell you.

.. _analysis-vllm-concepts:

Key concepts
------------

The ingested table set
^^^^^^^^^^^^^^^^^^^^^^

Roughly 47 tables per phase. The authoritative field-by-field reference is
:ref:`neuron-explorer-profile-schema-reference`; the ones that carry most analyses:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Table
     - What it carries
   * - ``Instruction``
     - Per-instruction timing with engine, plus the attribution columns. The widest and
       most useful table.
   * - ``ActiveTime``
     - Per-engine active intervals, per physical core. Not pre-merged across cores.
   * - ``CcOp``
     - Aggregated collective operations: operation, count, wall time, bytes.
   * - ``FrameworkNode``
     - The framework-level node tree with parent and child links and per-node engine
       times.
   * - ``SystemProfileEvents``
     - **System trace.** NEFF dispatch records: the source of step time, the execution
       pattern, and the inter-dispatch gaps.
   * - ``Metadata``
     - The device constants of Stage 2.
   * - ``DmaPacket``, ``DmaPacketAggregated``
     - Per-DMA-packet detail. Populated only when DMA metadata was enabled at capture.

Floors, and the measured-to-floor ratio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The memory floor is the time the step would take if it moved its bytes at peak bandwidth
and computed instantly; the compute floor is the inverse. The larger bounds the step. The
ratio of measured to floor is the headroom: near 1 means the step is at the hardware limit
and only an algorithmic change helps, while a large ratio means time is going somewhere
neither floor explains — idle engines, serialization, or collective waits.

A floor below the measured step is expected. A floor *above* it means the floor is wrong,
usually from single-core constants on an LNC=2 profile.

.. _analysis-vllm-entitlement:

Cross-validating with an analytical roofline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Recommended: pair this skill with the** :ref:`Hardware Entitlement skill
<neuron-hardware-entitlement>`. This skill's floors are derived from the trace — the bytes
and FLOPs the compiled graph actually issued. The Hardware Entitlement skill derives them
independently, from the model configuration and the hardware constants: for every matmul-class operator it computes the
theoretical minimum per-device time the operator *must* pay, split into a compute floor
(operator FLOPs over tensor-engine peak) and an HBM floor (weight plus KV bytes over HBM
bandwidth), and reports which of the two binds.

The two are answering the same question from opposite directions, which is exactly what
makes comparing them informative:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - This skill
     - Hardware entitlement
   * - Floor derived from
     - the captured trace
     - the model configuration
   * - Resolution
     - per NEFF, per segment, per component
     - per operator
   * - Answers
     - where the time went
     - where it was obliged to go

Comparing the profiled step against the analytical floor is what lets you separate the two
things a large measured-to-floor ratio can mean:

* **The profiled latency is near the analytical floor.** The work is mandatory. No kernel
  change will help; only a different parallelism configuration, a different sequence shape,
  or a different model will.
* **The profiled latency is well above the analytical floor, and the trace-derived floor
  agrees.** The gap is implementation — idle engines, serialization, collective waits,
  unfused interfaces — and the component roll-up tells you where to look.
* **The two floors disagree with each other.** Something is mis-modeled rather than slow.
  Resolve that before drawing any conclusion about the model's performance.

Both skills accept the same project root, so pointing them at one directory puts the
capture, the analytical floor it is compared against, and the analysis in a single evidence
trail. The analysis block records which floor was used in its ``entitlement`` field: a
``source`` of ``self`` means trace-derived only, and naming the entitlement skill means the
two were reconciled.

.. important::
   **The analytical floor is scoped to the tensor engine and HBM.** Vector and scalar engine
   work is not modeled and is excluded from its binding maximum, so it is a valid lower
   bound that may be loose. This matters in practice: a decode step whose busiest engine is
   neither the tensor engine nor a DMA path can sit well above the analytical floor without
   that gap being waste. Check the binding engine from
   :ref:`Stage 4 <analysis-vllm-workflow>` before concluding a gap is recoverable.

.. _analysis-vllm-attribution:

Component attribution
^^^^^^^^^^^^^^^^^^^^^

The ``Instruction`` table carries attribution at two resolutions, and both are usable when
the NEFF was compiled with the debug variables set.

**Module scope — the ``layer`` column.** The framework path an instruction came from, for
example a decoder layer's attention or MLP submodule. Grouping by it, with the layer index
collapsed, gives time and HBM traffic per component. This is the fastest route to the
attention-versus-MLP question and needs no joins.

**Source file and line — the ``nki_source_location`` and
``bir_debug_info_source_location`` columns.** Both carry a real ``path:line``. Measured on
one decode capture of 179,747 instruction rows, with the three debug variables set:

.. list-table::
   :header-rows: 1
   :widths: 44 18 38

   * - Column
     - Populated
     - What it resolves to
   * - ``layer``
     - 88%
     - Module scope, for example
       ``.../Gemma4DecoderLayer/Gemma4Attention[self_attn]``.
   * - ``nki_source_location``
     - 86%
     - A file and line. On that capture, 20 distinct files.
   * - ``bir_debug_info_source_location``
     - 86%
     - The same, from the compiler's debug records.

Where those lines land is worth knowing before you follow one:

.. list-table::
   :header-rows: 1
   :widths: 40 18 42

   * - Location family
     - Share
     - Meaning
   * - The NKI kernel library
     - ~88%
     - The kernel that implements the operation. The right place to look for a kernel-level
       question, and not editable in your model.
   * - ``vllm_neuron/nn``, ``vllm_neuron/functional``
     - ~9%
     - Shared layers and operators, for example an embedding or an argmax
       implementation.
   * - ``vllm_neuron/model/<model>/``
     - ~3%
     - **The model's own source**, for example ``model/granite/model.py:158``. This is the
       line to edit when the answer is in the model definition.
   * - ``torch``
     - <1%
     - Framework glue.

So most instructions attribute to the kernel that implements them rather than to the model
file that called it, which is expected: that is where the work is. The minority that lands
in ``model/`` is often the more actionable set, because it points at code you own.

.. important::
   All of this depends on a decision already made. The debug information is baked into the
   NEFF at compile time, so a capture taken against a cache built without those variables
   has an unpopulated ``layer`` column and empty source locations, and no post-processing
   recovers it. See :ref:`profiling-vllm-config`.

Rolling up to a component means collapsing the layer index. A roll-up that lists one entry
per layer per instruction has not answered the question it was asked: forty attention
entries are one component, not forty.

Known dead ends
---------------

Do not re-test these. Each was investigated and did not pay off for collective-bound
vLLM Neuron models:

* Replicated tensor parallelism, and fusing reduce-scatter with all-gather.
* Windowed einsum and collective-matmul variants.
* Optimization level 3 at tensor-parallel degrees above 1.
* vLLM async scheduling, for the capture path.

.. _analysis-vllm-debugging:

Common issues
-------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Symptom
     - Likely cause and fix
   * - Ingestion runs for hours, or emits zero-byte parquet
     - The device trace is too large. Captures should record two iterations, not fifty.
       Recapture rather than waiting.
   * - ``CcOp.parquet`` is empty
     - The capture recorded one iteration. Recapture with two.
   * - ``DmaPacket`` tables are empty
     - DMA packet metadata was not enabled at capture. Not recoverable
       post-hoc.
   * - A second ingest fails to start
     - ``neuron-explorer view`` binds a fixed port. Ingest phases sequentially.
   * - Engine active time exceeds the step
     - Instruction durations were summed instead of interval-merged, and both physical
       cores were counted. See :ref:`active-wall time <analysis-vllm-activewall>`.
   * - The computed floor is above the measured step
     - Single-core device constants were used on an LNC=2 profile. Use
       ``hbm_ddr_bandwidth`` and the combined engine peak.
   * - Component labels are mostly unknown
     - The NEFF was compiled without framework debug information. It cannot be added
       afterwards; recompile and recapture.
   * - Device time does not match the benchmark latency
     - Expected. Device time excludes queueing, scheduling, and the serving stack. Say
       which quantity you are reporting.
   * - Collective wall time exceeds the step
     - Collective intervals were summed rather than merged.

Limitations
-----------

* **Bounds, partitioning, and investigations are approximate.** See
  :ref:`analysis-vllm-maturity`.
* **Entitlement is self-derived by default**, so a floor is internally consistent rather
  than independently cross-validated. Pair the skill with
  :ref:`Hardware Entitlement <neuron-hardware-entitlement>` to close that gap; see
  :ref:`analysis-vllm-entitlement`.
* **No report template**, so two analyses of one capture may be laid out differently.
* **Attribution depends on a compile-time decision.** Without the debug variables set at
  compile, neither module scope nor source location is available, and it cannot be added
  afterwards.
* **Phases cannot be ingested in parallel** on one host.
* **Host CPU and host memory are not analyzed.** A capture can record them and they ingest,
  but no stage consumes them, so a host-side bottleneck will not surface even when the data
  is present. Session-level host time is visible only indirectly, as the inter-dispatch gaps
  of Stage 3.

Related resources
-----------------

* :ref:`neuron-framework-profiling-vllm-neuron` — produce the capture this skill consumes.
* :ref:`profile-analysis-vllm-neuron-tutorial` — a worked analysis, end to end.
* :ref:`neuron-hardware-entitlement` — the analytical per-operator roofline to
  cross-validate a measured step against, and :ref:`hardware-entitlement-tutorial` for a
  worked example.
* :ref:`neuron-parallelism-planner` — the same bound at model level, for choosing a config
  before you capture.
* :ref:`neuron-explorer-profile-schema-reference` — the authoritative table and field
  reference.
* :ref:`neuron-explorer-troubleshooting` — ingestion and rendering problems.
* :doc:`/vllm-neuron/docs/design/vllm/neuron-profiling` — how the profiler integration
  works.
* `Neuron Agentic Development on GitHub <https://github.com/aws-neuron/neuron-agentic-development>`_
