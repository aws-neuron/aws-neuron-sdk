.. meta::
   :description: Tutorial for analyzing a vLLM Neuron device and system profile with the Neuron Agentic Development Profile Analysis skill, producing session workload timing, roofline bounds, a collective breakdown, and a per-component attribution.
   :keywords: profile analysis, vllm, vllm-neuron, tutorial, roofline, performance bounds, collectives, attribution, MFU, MBU, neuron-explorer, parquet, Trainium, trn2
   :date-modified: 2026-10-08

.. _profile-analysis-vllm-neuron-tutorial:

Tutorial: Analyze a vLLM Neuron profile using the Profile Analysis skill
========================================================================

This tutorial walks you through analyzing a profile you have already captured. You point the
agent at a capture manifest, tell it what you want to learn, and work through the answers
with it: which compiled graphs ran, where a step sits relative to its hardware floor, which
engine binds it, how much of it is collective traffic, and which part of the model is
responsible.

The agent reads both halves of the capture — the system trace for the session-level picture
and the step time, the device trace for the per-instruction detail inside a step.

By the end you will have a per-phase analysis block of machine-readable findings and a
written report, and you will know how to check that both are sound rather than merely
plausible.

.. important::
   **Use this skill interactively, and say what you want to know.** An analysis has no
   single correct output — the useful answer depends on the question. Asking "analyze this
   profile" produces a general survey; asking "why is decode 2.5 times its memory floor"
   drives the stages that bear on it. Plan to converse with the agent rather than to hand
   it a job and come back.

.. note::
   This tutorial starts from a capture. Produce one first with
   :ref:`profiling-vllm-neuron-tutorial`. For what the numbers mean and which to trust, see
   the :ref:`Profile Analysis deep dive <neuron-framework-profile-analysis-vllm-neuron>`.

Prerequisites
-------------

Have a completed capture
^^^^^^^^^^^^^^^^^^^^^^^^

You need a run directory with a ``manifest.json`` whose device trace is real. This tutorial
uses the Gemma 4 31B IT capture from
:ref:`profiling-vllm-neuron-tutorial` — tensor parallel 16, 8192 tokens in and 1024 out.

.. code-block:: bash

   cd ~/profiling-tutorial/agent_artifacts/profiles
   python3 -c "
   import json, os
   m = json.load(open('gemma4_decode/manifest.json'))
   n = m['profile']['device_ntff']
   print('recipe:', m['recipe'])
   print('trace :', n, os.path.getsize(n) if n else 'MISSING')
   print('neffs :', len(m['neff_hashes']))
   "

A ``null`` device trace, or a size of a few hundred kilobytes, means the capture failed and
there is nothing to analyze. Go back to the capture tutorial.

.. important::
   The capture must have recorded **at least two iterations**. At one, ``CcOp.parquet``
   comes out empty and the collective stage cannot run. The capture tutorial's recipes use
   two by default; if you captured by hand, check ``profiler_config.max_iterations`` in
   ``run_spec.json`` before continuing.

No Trainium device needed
^^^^^^^^^^^^^^^^^^^^^^^^^

Analysis is host-side. You can do it on the instance that captured, or copy the capture
elsewhere. What you do need is headroom:

.. code-block:: bash

   df -h .          # disk
   free -g          # host memory

Ingesting a long-sequence prefill trace is the slow, memory-hungry step of this tutorial:
it runs for a while, writes parquet several times larger than the trace, and holds a
substantial amount of host RAM throughout. Decode is far cheaper. If you are short on
either, start with the decode phase. For measured figures, see
:ref:`the deep dive's requirements <neuron-framework-profile-analysis-vllm-neuron>`.

Install Neuron Agentic Development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have not already, follow the
:ref:`Getting Started guide <neuron-agentic-development-getting-started>` and confirm the
deploy step completed.

.. code-block:: bash

   # For Claude Code
   deploy-neuron-agentic-development-to-claude

   # For Kiro
   deploy-neuron-agentic-development-to-kiro

Activate your Python environment and verify the tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   source ~/opt/aws_neuronx_venv_pytorch_2_9/bin/activate
   which neuron-explorer
   python3 -c "import pandas, pyarrow; print(pandas.__version__, pyarrow.__version__)"

``neuron-explorer`` ingests the capture; pandas and pyarrow query the result.

Step 1. Point the agent at the capture
--------------------------------------

The manifest is the entry point. Every path in it is absolute, so the agent needs nothing
else to locate the artifacts.

.. code-block:: bash

   export NEURON_PROJECT_ROOT=~/profiling-tutorial

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Input
     - What it is
     - Example value
   * - manifest path
     - The capture to analyze. One per phase.
     - ``.../profiles/gemma4_decode/manifest.json``
   * - ``NEURON_PROJECT_ROOT``
     - Where the ingested parquet and the report land, beside the capture.
     - ``~/profiling-tutorial``
   * - your question
     - What you want to decide. This shapes the whole analysis.
     - "Is decode memory-bound or collective-bound?"

Step 2. Invoke the analysis agent
---------------------------------

Open your agentic IDE and state the capture **and the question**.

.. code-block:: text

   Analyze the decode profile at
   ~/profiling-tutorial/agent_artifacts/profiles/gemma4_decode/manifest.json

   I want to know whether this step is memory-bound or collective-bound,
   and which model component accounts for most of the device time.

The agent ingests, then works through the stages that bear on your question. Expect it to
come back to you — to confirm which NEFF to treat as the decode graph, or to ask whether a
gap matters for your decision. That is the skill working as intended; see
:ref:`analysis-vllm-workflow` for what each stage does.

Useful follow-ups once the first pass lands:

.. code-block:: text

   Break the step down by collective segment and tell me which segment
   holds the binding engine.

   Compare the MLP and attention roll-up by HBM bytes read, not just by
   engine time.

   The measured step is well above the floor. Where is the unexplained
   time going?

Step 3. What happens during the analysis
----------------------------------------

The agent ingests the capture into parquet, reads the device constants from the capture
itself, establishes the step time from the live dispatch records, computes the floors,
partitions the step at the collectives, and rolls the instructions up by model component.
The deep dive's :ref:`analysis-vllm-workflow` explains each stage and why it is ordered
that way.

Two things you will see that are worth understanding as they happen:

* **The ingest is the slow part**, and its cost tracks trace size, not model size.
* **A prefill capture shows two NEFFs**, because the single request also warms the
  batch-size-1 decode graph. The agent filters to the prefill NEFF; if it asks which to
  use, the manifest's ``neff_hashes`` is the answer.

.. _analysis-tutorial-output:

Step 4. Read the analysis block
-------------------------------

The machine-readable findings land as a per-phase ``analysis`` block. This is the output to
check first, because every claim in the written report should be traceable to it.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``measured_step_us``
     - The step time from the live dispatch records.
   * - ``memory_floor_us``, ``compute_floor_us``
     - The two roofline floors. The larger one bounds the step.
   * - ``mbu_percent``, ``mfu_percent``
     - Achieved fraction of peak bandwidth and peak FLOPS.
   * - ``binding_engine``, ``binding_engine_pct``
     - The engine busiest for the largest share of the step.
   * - ``engine_active_pct``
     - Per-engine active-wall share, interval-merged.
   * - ``collectives``
     - One entry per operation, with count, wall time, and bytes.
   * - ``component_rollup``
     - Device time and HBM bytes per model component.
   * - ``entitlement``
     - The floor the measured step is compared against, its source, and the ratio.

Here is the real decode block from this tutorial's capture — Gemma 4 31B IT at tensor
parallel 16, 8192 in and 1024 out. Trimmed to the largest roll-up entries; the full block
listed fourteen.

.. code-block:: json

   {
     "measured_step_us": 208296.158,
     "memory_floor_us": 82611.251,
     "compute_floor_us": 1406.57,
     "mbu_percent": 39.66,
     "mfu_percent": 0.68,
     "binding_engine": "sync",
     "binding_engine_pct": 67.35,
     "engine_active_pct": {
       "tensor": 3.99, "vector": 4.21, "scalar": 2.77,
       "gpsimd": 6.25, "sync": 67.35
     },
     "collectives": [
       {"operation": "ALLREDUCE", "count": 121, "wall_us": 3481.15, "bytes_out": 10407936},
       {"operation": "BARRIER",   "count": 1,   "wall_us": 1172.08, "bytes_out": 0},
       {"operation": "ALLGATHER", "count": 3,   "wall_us": 74.25,   "bytes_out": 8454144}
     ],
     "component_rollup": [
       {"module": "Gemma4Model[model]/Gemma4DecoderLayer/Gemma4Attention[self_attn]",
        "engine_us": 344971.35, "hbm_read_bytes": 56407115950},
       {"module": "Gemma4Model[model]/Gemma4DecoderLayer/Gemma4MLP[mlp]",
        "engine_us": 21250.15, "hbm_read_bytes": 2622412800},
       {"module": "Gemma4Model[model]/Gemma4DecoderLayer",
        "engine_us": 3510.06, "hbm_read_bytes": 86228},
       {"module": "ColumnParallelLinear[lm_head]",
        "engine_us": 1459.37, "hbm_read_bytes": 0}
     ],
     "entitlement": {"source": "self", "predicted_us": 82611.25, "ratio": 2.5214}
   }

How to read it
^^^^^^^^^^^^^^

Working through it in the order that answers the question asked in Step 2:

* **Memory-bound or compute-bound?** The memory floor is 82,611 µs against a compute floor
  of 1,407 µs — a factor of 59. The utilization figures agree: 39.7% of peak bandwidth
  against 0.68% of peak FLOPS. This step moves bytes; it barely computes. Decode at a long
  context is expected to look like this.
* **At the floor, or above it?** 208,296 µs measured against an 82,611 µs floor is a ratio
  of 2.52, so roughly 126 ms is going somewhere neither floor explains. That gap is the
  thing worth chasing.
* **Collective-bound?** No — and this is where reading only one field would mislead you.
  The collectives total about 4,728 µs, a little over 2% of the step. The 121 all-reduces
  are the expected tensor-parallel pattern at this degree, and a useful check that the
  sharding is what you configured, but they are not where the time is.
* **So what binds?** The ``sync`` engine, at 67.35% of the step, while every compute engine
  sits below 7%. The step is dominated by synchronization that is *not* collective traffic —
  the collectives account for 2% while sync accounts for 67%. That distinction is the
  finding, and it points the investigation at serialization and waits rather than at
  kernels.
* **Which component?** Attention at 344,971 µs against MLP at 21,250 µs, reading 56.4 GB
  against 2.6 GB. For a memory-bound step the bytes column is the one that decides, and
  attention reads 21 times the MLP's. At an 8192-token context that is the KV cache, which
  is exactly where decode attention time goes.

Two checks worth making on any roll-up, including this one:

* **Is there an ``Unknown`` entry, and how large?** This block has none, which means module
  scope was populated for every attributed instruction. A large unknown share means the
  attribution is incomplete — usually a NEFF compiled without framework debug information.
  See :ref:`analysis-vllm-attribution`.
* **Does it name components or instructions?** Here each entry is a module path with the
  layer index collapsed, so attention across all layers is one row. A list of per-layer
  entries has not answered the question.

.. note::
   Engine times in a roll-up are per-engine interval sums and will overshoot the step where
   engines overlap — 344,971 µs of attention inside a 208,296 µs step is not a
   contradiction. Use them to rank components, not to partition the step.

.. important::
   This capture's **prefill** phase is a worked example of a failure, not a success: its
   ingest completed only 30 of the expected tables, dropping the framework and BIR layers
   while leaving the instruction and collective tables intact. Five of the six analysis
   dimensions still scored clean on it. Only the table inventory caught it — which is why
   check 1 in :ref:`analysis-tutorial-confirm` is not a formality.

Step 5. Read the report
-----------------------

The agent also writes a report under ``analysis/`` in the run's output directory,
``<phase>_report.md``. A sound one runs a few thousand bytes and covers: the bottom line,
both floors, per-engine active-wall, the collectives, the component roll-up, the
entitlement comparison, and a verification that the phase is what the label claims.

Read it against the block from Step 4. If a sentence in the report has no corresponding
field in the block, treat it as unsupported.

.. _analysis-tutorial-confirm:

Confirmation
------------

An analysis fails quietly — a plausible but wrong floor reads exactly like a right one. Run
these five checks before acting on any of it.

**1. Ingestion is complete.** Around 47 tables for the phase, and the attribution tables
non-empty.

.. code-block:: bash

   NE=$NEURON_PROJECT_ROOT/agent_artifacts/data/gemma4_decode/profiles
   find "$NE" -name '*.parquet' | wc -l
   find "$NE" -name 'CcOp.parquet' -size -1k   # prints a path only if it is empty

An empty ``CcOp.parquet`` means the capture recorded one iteration and the collective
findings are absent rather than zero.

**2. The measured step is at or above the larger floor.** If a floor exceeds the measured
step, the floor is wrong — almost always single-core device constants applied to an LNC=2
profile.

**3. No engine's active time exceeds the step.** If one does, the times were summed rather
than interval-merged — see :ref:`active-wall time <analysis-vllm-activewall>`.

**4. Summed collective wall time is less than the step.** Collectives happen inside the
step, so exceeding it means the intervals were summed rather than merged.

**5. The roll-up names components, not instructions.** Entries should read like
``.../Gemma4MLP[mlp]``, one per component with the layer index collapsed. A list of
per-layer entries has not answered the attention-versus-MLP question. And check the
``Unknown`` share: if it dominates, the attribution is not usable.

Step 6. Act on the result
-------------------------

Rank the components, then optimize the binding one — and confirm it is binding before you
start, because a correct optimization aimed at a non-binding operation buys nothing.

To verify a change, capture again and compare against this analysis. Compare numbers from
the **same** capture only — a device time from one run against a wall clock from another is
arithmetic, not measurement.

.. important::
   Device time is not end-to-end latency. It excludes queueing, scheduling,
   detokenization, and the serving stack. When you report a result, say which quantity you
   measured.

Troubleshooting
---------------

The deep dive carries the full reference at :ref:`analysis-vllm-debugging`. Three things
bite specifically during the steps above.

**The ingest runs for hours, or writes zero-byte parquet.** The trace is too large. Check
``profiler_config.max_iterations`` in the capture's ``run_spec.json`` — it should be 2. A
capture made with a large iteration cap is faster to redo than to ingest.

**A second ingest will not start.** ``neuron-explorer view`` binds a fixed port even in
ingest-only mode, so prefill and decode cannot be ingested at the same time on one host.
Run them in sequence.

**The component roll-up is mostly ``Unknown``.** The NEFF was compiled without framework
debug information, which cannot be added after the fact. Recompile the capture with the
three debug variables set — see :ref:`profiling-vllm-config` — and recapture.

Everything else is in :ref:`analysis-vllm-debugging`, which covers empty tables, floors
above the measured step, engine times exceeding the step, device time not matching
benchmark latency, and the profiler schema defects that cause a misreading.

Related resources
-----------------

- :ref:`neuron-framework-profile-analysis-vllm-neuron` for the deep dive: the stages, what
  the numbers mean, the verification rules, and which outputs to trust.
- :ref:`profiling-vllm-neuron-tutorial` for producing the capture this tutorial consumes.
- :ref:`neuron-framework-profiling-vllm-neuron` for the capture deep dive.
- :ref:`neuron-explorer-profile-schema-reference` for the authoritative table and field
  reference.
- :doc:`/tools/neuron-agentic-development/getting-started` for installation and setup.
- `Neuron Agentic Development GitHub <https://github.com/aws-neuron/neuron-agentic-development>`_
  for source code and issue tracker.
