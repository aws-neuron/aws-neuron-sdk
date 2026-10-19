.. meta::
   :description: Capture prefill and decode device and system profiles from a running vLLM Neuron server on Trainium using the Neuron Framework Profiling skill
   :keywords: profiling, vllm, vllm-neuron, neuron profiler, NTFF, device profile, system profile, prefill, decode, Trainium, trn2, neuron-explorer, tensor parallelism
   :date-modified: 2026-10-08

.. _neuron-framework-profiling-vllm-neuron:

Deep dive: Capture device and system profiles from a vLLM Neuron server with the Profiling skill
================================================================================================

**Why read this guide?** You are an ML engineer or performance engineer who needs a
device trace of a vLLM Neuron model running on Trainium, and you need the prefill pass
and the steady-state decode pass captured *separately* so that each can be attributed on
its own. Mixing the two in one trace is the single most common reason a capture cannot
answer the question that motivated it.

**How to use this guide:** Read :ref:`profiling-vllm-requirements` and
:ref:`profiling-vllm-config` before your first capture, because two of the settings there
must be in place at *compile* time and cannot be fixed afterwards. Jump to
:ref:`profiling-vllm-workflow` if your environment is already set up, or to
:ref:`profiling-vllm-debugging` if a capture produced no trace.

**This skill is distinct from the manual procedure.** The mechanics of enabling the
Neuron profiler on a vLLM server — the ``neuron_profiler`` configuration block, the
``/start_profile`` and ``/stop_profile`` endpoints, and the output directory layout — are
documented in :doc:`/vllm-neuron/docs/design/vllm/neuron-profiling` and
:doc:`/vllm-neuron/docs/guides/how-to-profile-workloads`, and a single-bracket worked
example is in :doc:`/tools/tutorials/performance-profiling-vllm`. This guide does not
restate them. What this skill adds is **phase isolation** (a trace containing only
prefill, or only decode), a managed server lifecycle that does not leak device-holding
processes, and a recorded input and output contract that makes a capture reproducible.

**You need experience with** running ``vllm serve``, tensor parallelism, and reading a
profile in Neuron Explorer. You do not need to have written a profiler integration.

Prerequisites
-------------

* **vLLM Neuron serving.** You can bring up ``vllm serve`` against a local checkpoint on
  Trainium and send it a request. See
  :doc:`/vllm-neuron/docs/getting-started/quickstart-online-serving`.
* **The Neuron profiler integration.** How the profiler is configured and what it writes.
  See :doc:`/vllm-neuron/docs/design/vllm/neuron-profiling`.
* **Tensor parallelism and Logical NeuronCores.** The capture pins one worker rank, and
  the number of Logical NeuronCores a run occupies equals its ``tensor_parallel_size``.
* **Neuron Explorer ingestion.** A capture is not an answer; it is an input to ingestion
  and analysis. See :ref:`neuron-explorer-profile-schema-reference`.

Overview
--------

A capture produces a device trace (an NTFF file), a system trace, and the compiled NEFFs
needed to render them. The skill drives three things you would otherwise coordinate by
hand: *when* the profiler is armed relative to the workload, *what* workload is driven,
and *how* the server is torn down so the next run can start.

It captures two phases, in separate runs, because prefill and decode are different
compiled graphs with different bottlenecks:

* **Prefill isolation** arms the profiler while the server is idle, fires one request,
  and stops on the request's return. One iteration, no decode.
* **Decode isolation** arms on a scheduler event — the engine reporting that the prefill
  queue has drained — so the recorded iterations are steady-state decode at the target
  batch size.

Both write a ``manifest.json`` naming every artifact by absolute path. That file is the
entry point for the :ref:`neuron-framework-profile-analysis-vllm-neuron` skill.

.. _profiling-vllm-requirements:

Hardware and software requirements
----------------------------------

* **Instance.** A Trainium instance with enough Logical NeuronCores for the model's
  ``tensor_parallel_size``, plus HBM headroom for the profiler's buffers (see
  :ref:`profiling-vllm-artifacts`). The examples in this guide use ``trn2.48xlarge``.
* **Neuron SDK.** Installed on the instance, with ``aws-neuronx-dkms``,
  ``aws-neuronx-runtime-lib``, ``aws-neuronx-collectives``, and ``aws-neuronx-tools``.
  All are pre installed on the Amazon Linux 2023 Neuron DLAMI, which is a good starting
  point. ``aws-neuronx-tools`` provides ``neuron-explorer``, which ingests a capture, and
  ``neuron-ls``, which is how you confirm a teardown released its devices.
* **Python.** 3.10 or later.
* **vLLM Neuron.** The ``vllm_neuron`` package, importable in the environment you profile
  from, and ``libtorch-neuronx-lite``, which exposes the runtime profiling bindings. The
  latter is the hard dependency for serve-based capture: without it the profile routes
  mount but no device trace is written.
* **Model weights.** Downloaded locally, for example with ``huggingface-cli download``.
  The local path is what you pass as ``--model``.
* **Disk and host memory.** A capture costs far less than ingesting it. Budget for the
  device trace (21 MB to about 5 GB, see :ref:`profiling-vllm-artifacts`), roughly an
  order of magnitude more for the parquet tables ingestion produces, and tens of GB of
  host RAM to ingest a multi-gigabyte trace.

.. note::
   This guide describes a single interactive instance, and profiling a
   disaggregated-inference deployment is out of scope. On a managed cluster the launch
   step becomes a job submission, but nothing about how profiling works changes — adapt
   the launch, not the capture.

.. _profiling-vllm-config:

Configuration
-------------

Required environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``VLLM_NEURON_VENV_PATH``
     - The Python environment containing the Neuron SDK packages. On the DLAMI this is
       ``~/opt/aws_neuronx_venv_pytorch_2_9``. The skill prepends its ``bin`` directory
       to ``PATH`` so the server child resolves the right ``vllm``.
   * - ``VLLM_NEURON_DIR``
     - The vLLM Neuron source tree. Set it to the directory containing the installed
       ``vllm_neuron`` package; the skill uses it as the server's working directory.
   * - ``NEURON_PROJECT_ROOT``
     - The project directory this run belongs to. Defaults to the current directory.
       Every artifact lands beneath it.
   * - ``HF_HOME``
     - Where model weights live. Defaults to
       ``$NEURON_PROJECT_ROOT/agent_artifacts/models``.

Where outputs go
^^^^^^^^^^^^^^^^

There is **one knob and no override**. Every artifact is placed relative to
``NEURON_PROJECT_ROOT``, under ``agent_artifacts/`` in ``profiles/<run_name>/``,
``bench/<run_name>/``, and ``models/``.

Separate per-artifact directory variables were deliberately removed. Four names for one
idea meant that a campaign setting some of them scattered its plan, its profiles, and its
reports across unrelated directories, and no reader could tell which run produced what.
If the disk you want to profile onto is not where your project lives, point the project
root at that disk — ``export NEURON_PROJECT_ROOT=/local/scratch/$USER`` — and the paths
still read the same way everywhere. The other skills in this family accept the same value
as ``--project-root``, so one export puts a profile and the analysis derived from it in
one evidence trail.

Compile-time debug variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``XLA_IR_DEBUG=1``, ``XLA_HLO_DEBUG=1``, and ``NEURON_FRAMEWORK_DEBUG=1``. The first
two put debug information in the compiled NEFF; the third adds the framework-level
module-scope path that lets the analysis skill say which part of the model a device
instruction came from.

.. important::
   These must be set **before** ``vllm serve`` compiles the model, not before the
   capture. The information is baked into the NEFF, and a warm compile cache only carries
   it if the original compile had them set. A capture taken against a cache populated
   without them yields a trace whose instructions cannot be attributed, and the only fix
   is a recompile.

Other settings that change what you capture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Setting
     - Why it matters
   * - ``VLLM_NEURON_WORKER_TERMINATION_TIMEOUT=60``
     - The default of 5 seconds kills tensor-parallel workers before the runtime has
       flushed its system-trace buffers, and the ``.pb`` files come out zero bytes or
       missing. At 60 seconds, expect 4 to 5 MB of system trace per worker.
   * - ``VLLM_NEURON_RESERVED_HEADROOM_GIB``
     - The dial that makes room for the profiler. The KV budget is
       ``HBM per logical core - weights - headroom``, so raising the headroom trades KV
       cache the capture does not need for the buffers the profiler does. Changing it
       forces a one-time recompile, because the block count changes.
   * - ``VLLM_NEURON_MIN_KV_BUDGET_GIB``
     - The floor the runtime refuses to start below, 1.0 GiB by default. Raising the
       headroom far enough to free the profiler can push the KV budget under it, so the
       two usually move together.
   * - ``activities`` in the ``neuron_profiler`` block
     - Defaults to ``["device_profile", "system_profile"]``. Host CPU and host memory are
       captured only if you add ``cpu_util`` and ``host_memory``.
   * - ``NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1``
     - Adds per-DMA-packet metadata, which is what populates the ``DmaPacket`` tables
       needed for DMA-efficiency analysis. The runtime reads it **once at
       initialization**, so it must be set before ``vllm serve``; the profiler config
       block cannot toggle it. Off by default — keep it off for mixture-of-experts decode
       and for dense models with tight KV budgets, where it floods the notification
       queue.
   * - Model-specific compile flags
     - The compile cache keys on the full compiler argument list, and any environment
       variable that alters that list alters the key. Export the same set across cold and
       warm reruns or the warm cache silently misses. Declare their names so the skill
       snapshots them into the run specification.

.. important::
   **Sizing the headroom, worked.** A 35B mixture-of-experts model at tensor-parallel 4
   holds 16.25 GiB of weights in a 24 GiB logical core. At a headroom of 6 that leaves
   about 0.24 GiB free — less than the profiler's 128 MB notification buffers — and the
   capture writes no trace at all while the server appears healthy. A headroom of 7.5
   frees roughly 1.7 GiB, which is enough; the KV budget it leaves is then 0.25 GiB, under
   the 1.0 GiB default floor, so ``VLLM_NEURON_MIN_KV_BUDGET_GIB`` has to come down with
   it. Both are environment variables, so the pinned serving configuration is unchanged.

.. important::
   If ``cpu_util`` and ``host_memory`` are absent from ``activities``, the
   ``cpu_util.pb`` and ``host_mem.pb`` files are still created but are **zero bytes**.
   They appear in the manifest and look like evidence. Check their size before drawing
   any host-side conclusion.

.. _profiling-vllm-methods:

Choosing a capture method
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Situation
     - Method
   * - Default. Any model, including mixture-of-experts, expert-parallel, and large
       tensor-parallel degrees. Profiles a running server, can target steady-state
       decode, and can select which ranks profile.
     - **Serve plus HTTP endpoints**
   * - You need one NEFF's device trace in isolation, or a scripted offline capture on a
       dense model where bringing up a server is inconvenient.
     - **Offline capture** (fallback)

Prefer the serve-based method. It is the one that works for the hard cases: standalone
NEFF capture fails on large-tensor-parallel NEFFs with baked-in collectives, and
low-level device profiling crashes on mixture-of-experts kernels. Both methods produce
the same artifacts, so either feeds the same analysis.

The offline method runs a throughput benchmark with profiling enabled to emit NEFFs and
system traces, then captures each NEFF's device trace standalone. It does **not** work
for mixture-of-experts or large-tensor-parallel models.

.. _profiling-vllm-workflow:

Workflow
--------

Capture with the serve endpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Profiling is toggled on a live server, with no restart and no separate capture step.
Setting ``--profiler-config`` at startup is what mounts the ``/start_profile`` and
``/stop_profile`` routes; omit it and the routes do not exist. The canonical command
forms are in :doc:`/vllm-neuron/docs/design/vllm/neuron-profiling`; the values that
matter for phase isolation are:

.. code-block:: text

   --profiler-config  '{"profiler": "cuda", "delay_iterations": 0, "max_iterations": 2}'
   --additional-config '{"neuron_profiler": {
                           "activities": ["device_profile", "system_profile"],
                           "neuron_cores": [0],
                           "output_dir": "<profiles>/<run_name>/output"}}'

* ``"profiler": "cuda"`` is a deliberate misnomer on Neuron. It is the only profiler slot
  a plugin can reinterpret, and on Neuron it dispatches to the Neuron profiler. Any
  CUDA-related warning it produces is harmless.
* ``neuron_cores`` selects **worker ranks**, not physical cores. Leave it at ``[0]``: a
  capture is a per-replica object, so one rank of one replica characterizes the graph at
  any data-parallel degree.
* ``delay_iterations: 0`` arms on the next iteration. One iteration is one
  ``execute_model`` call, which is one batched forward pass.

Poll ``http://127.0.0.1:8041/v1/models`` for readiness rather than guessing. Compilation
dominates the wait — budget 7 to 25 minutes per graph for a dense model, potentially
hours for a large mixture-of-experts model, with a warm cache saving about 4 minutes.
Always redirect the server log to a file: a run that is compiling looks exactly like a
run that is hung until you read it.

Bracket the window with ``POST /start_profile`` and ``POST /stop_profile``. For
instruction-tuned models drive ``/v1/chat/completions``, not ``/v1/completions``, so the
chat template applies — otherwise the model emits an end-of-turn token immediately and
you capture empty generations. Keep ``prompt_tokens + max_tokens`` within
``--max-model-len``.

.. important::
   ``max_iterations: 2`` is the smallest value that preserves collective-operation
   analysis, because the aggregation needs begin/end pairing across at least two recorded
   iterations. At ``max_iterations: 1`` the trace ingests cleanly and the instruction
   tables populate, but ``CcOp.parquet`` is **zero bytes** and the collective
   partitioning stage of analysis cannot run at all. Verified on a 31B model at
   tensor-parallel 16.

Isolate prefill
^^^^^^^^^^^^^^^

Prefill is one iteration long, so ``delay_iterations`` cannot isolate it: by the time the
counter has advanced past the prefill dispatch, the dispatch has already happened. Arm
the profiler while the server is **idle** instead, and make the stop the trigger.

1. Launch with ``delay_iterations: 0`` and ``max_iterations: 2``. The cap here is
   defensive, not the stop trigger. Wait for readiness and send **no** warmup requests.
2. ``POST /start_profile``. The profiler is armed and the iteration counter stays at zero
   because nothing is executing.
3. Fire exactly one synchronous request with an output length of 1. The next
   ``execute_model`` after it lands is the prefill dispatch. For reference this returns in
   roughly 250 ms at an input length of 8192 on a 31B dense model.
4. ``POST /stop_profile`` immediately when the request returns.

**Why an output length of 1 and not 0.** Zero is rejected by vLLM. One is exactly one
iteration, because the prefill pass emits the first output token itself through on-device
sampling; there is no separate decode step for that token.

Afterwards, ``trace_info.pb`` lists **both** the prefill NEFF and the batch-size-1 decode
NEFF, because the request warms the decode graph even though no decode iteration
executed. Filter by the prefill NEFF hash when you ingest.

Isolate decode
^^^^^^^^^^^^^^

An iteration counter is a static heuristic that assumes how many prefills precede steady
state. Arm on a scheduler event instead.

1. Launch and wait for readiness.
2. Start the benchmark workload at the target concurrency.
3. Watch the engine log for a stats line reporting an average prompt throughput of zero
   together with the target number of running requests and nothing waiting. That
   combination means no prefill is in flight.
4. ``POST /start_profile``. The next ``execute_model`` is a decode step at the target
   batch size, and ``max_iterations: 2`` stops the profiler two iterations later.
5. Let the workload finish and tear down. The explicit stop is idempotent.

Because the trigger is an event the scheduler emits rather than a count you guessed, this
is robust to warmup variability and to different prompt distributions.

The decode-only fallback
~~~~~~~~~~~~~~~~~~~~~~~~

The scheduler trigger can fail to fire, and the cause is structural rather than flaky.
The engine emits that stats line on a fixed tick, and an average prompt throughput of
zero is a property of the *window*: it needs a whole tick containing no prefill. At
concurrency 1 the benchmark runs requests back to back, so if the decode leg is shorter
than the tick, no tick is ever prefill-free. The signature is a prompt throughput of
exactly input-length-divided-by-tick on every tick.

The skill then falls back automatically to serving with a KV-transfer connector in
consumer mode, which reports every prompt's KV cache as already present. vLLM skips
prefill entirely, so no prefill graph is compiled and none is ever scheduled — there is
nothing that could leak into the trace, and the strict trigger fires immediately. The
run specification, lifecycle, validation, and manifest are unchanged, and the connector
is recorded alongside the configuration rather than merged into it, so a drift check
still sees the pinned configuration served unmodified.

.. important::
   In this mode the KV cache is never populated, so decode runs against uninitialized KV
   and the generated tokens are meaningless. This does not affect a profile or a
   throughput measurement, because the fixed-shape decode graph performs the same
   computation regardless of KV contents. Never use a trace captured this way for
   anything that reads model outputs.

Server lifecycle
^^^^^^^^^^^^^^^^

A profiling server holds Neuron devices, and if it is not torn down cleanly the next
launch fails with a runtime initialization error. The skill launches the server in its
own isolated process group, records the group id, polls readiness, and kills by recorded
group id after the capture, with a self-guard so teardown cannot reach the launching
session. Teardown is recorded separately from the capture result, because cleanup can
neither rescue a failed capture nor erase a successful one. Afterwards, confirm with
``neuron-ls`` that the cores show no owning process.

.. important::
   Never tear down a profiling server with a pattern-based kill such as ``pkill -f``. A
   bare background launch shares its process group with whatever started it, which is
   what makes a group kill hit the wrong processes.

.. _profiling-vllm-artifacts:

Capture inputs and outputs
--------------------------

The run specification (input)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``run_spec.json`` is written at **launch**, before compilation, so it survives a compile
or launch failure. It carries the structured serving configuration — the source of truth
for the launch, with the exact command line derived from it — the workload knobs, and a
snapshot of the model-specific compile environment:

.. code-block:: text

   schema_version, run_name, phase, recipe, workload, serve_config, model_env

Because the command line is derived from ``serve_config`` rather than built
independently, the recorded configuration cannot drift from what actually ran.

The manifest (output)
^^^^^^^^^^^^^^^^^^^^^

``manifest.json`` is written before cleanup, from a block that runs even on failure, so a
partial run still produces one. It names every artifact by absolute path and
back-references the run specification.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Contents
   * - ``recipe``
     - ``prefill_isolation`` or ``decode_isolation``.
   * - ``profile.device_ntff``
     - The device trace. **A null or zero-byte value here is the failure to look for**:
       the server can serve correctly and still write no device trace.
   * - ``profile.system_trace_pb``
     - The system trace, which is where per-request device time is read from. The sibling
       ``trace_info_pb`` maps the session to the NEFFs it executed.
   * - ``profile.neff_dir``
     - The NEFFs. Required for Neuron Explorer to render the device profile.
   * - ``neff_hashes``
     - The graphs this session executed, which is how you filter a prefill capture.

``profile.cpu_util_pb`` and ``profile.host_mem_pb`` are also listed, and are zero bytes
unless requested in ``activities``.

The session directory name contains the instance id, the process id, and a nanosecond
timestamp that cannot be known in advance. That is precisely why the manifest exists:
never hard-code a session path, read it from the manifest.

Output layout and artifact sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   $NEURON_PROJECT_ROOT/agent_artifacts/profiles/<run_name>/
   ├── run_spec.json      # input contract, written at launch
   ├── manifest.json      # output contract, written at teardown
   ├── logs/serve.log
   └── output/            # the per-session trace tree and neffs/, see the design doc

Measured device-trace sizes, to calibrate what a healthy capture looks like:

.. list-table::
   :header-rows: 1
   :widths: 34 20 10 18 18

   * - Model class
     - Input/output
     - TP
     - Prefill
     - Decode
   * - 1B dense
     - 1024 / 512
     - 8
     - 92 MB
     - 21 MB
   * - 31B dense, hybrid KV
     - 8192 / 1024
     - 16
     - 4.94 GB
     - 483 MB
   * - 35B mixture-of-experts
     - 1024 / 512
     - 4
     - 4.48 GB
     - 4.61 GB

The device profiler itself costs roughly 3.4 GB of HBM for its buffer pool plus about
128 MB of notification buffer per profiled core. If that cannot be allocated the capture
fails *silently*: the server serves, the requests succeed, and the output directory stays
at a few hundred kilobytes. Raise ``VLLM_NEURON_RESERVED_HEADROOM_GIB`` to make room: it
trades KV cache for profiler space, and a capture needs far less KV than a serving
deployment does. See :ref:`profiling-vllm-config`.

.. _profiling-vllm-debugging:

Common issues
-------------

Launch and compile failures
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symptom
     - Likely cause and fix
   * - ``No available shared memory broadcast block found in 60 seconds``
     - Usually compilation progressing, not a hang. Check the log for forward progress
       before intervening.
   * - Maximum peak HBM usage exceeds the HBM limit, at compile time
     - A stale cache entry, oversized input/output shapes, or too high a memory
       utilization target. Purge the cache entry and reduce
       ``--gpu-memory-utilization``.
   * - Byte-identical source re-fails with the same error after a failed compile
     - The compile cache recorded the failure. Purge the entry before retesting.
   * - Next launch fails with a runtime initialization error
     - Device-holding workers from a previous run were orphaned. Tear down by recorded
       process group; confirm with ``neuron-ls``.
   * - A rerun recompiles despite a warm cache
     - A workload-affecting environment variable changed, which changed the compile key.
       See :ref:`profiling-vllm-config`.

Capture failures
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symptom
     - Likely cause and fix
   * - Requests succeed but the output directory is a few hundred kilobytes
     - The profiler buffer pool could not allocate. Raise
       ``VLLM_NEURON_RESERVED_HEADROOM_GIB``. If the server then refuses to start with
       "Computed KV cache budget is below minimum threshold", lower
       ``VLLM_NEURON_MIN_KV_BUDGET_GIB`` below the resulting budget as well.
   * - ``manifest.json`` reports a null or zero-byte device trace
     - The run served without producing a device profile. Confirm
       ``libtorch-neuronx-lite`` is installed and that ``device_profile`` is in
       ``activities``.
   * - ``.pb`` files are zero bytes or missing
     - Workers were killed before the runtime flushed its buffers. Set
       ``VLLM_NEURON_WORKER_TERMINATION_TIMEOUT=60``.
   * - ``cpu_util.pb`` and ``host_mem.pb`` are zero bytes
     - They were not requested. Add ``cpu_util`` and ``host_memory`` to ``activities``.
   * - Decode arming never fires
     - The decode leg is shorter than the engine's stats tick. The skill falls back
       automatically; see the decode-only fallback above.
   * - An instruction-tuned model captures empty generations
     - Drive ``/v1/chat/completions`` so the chat template applies.
   * - Notification queue overflow on a mixture-of-experts session capture
     - Disable DMA packet metadata for this capture.

Ingestion failures
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symptom
     - Likely cause and fix
   * - Ingestion runs for hours or emits zero-byte parquet
     - The session trace is too large. Keep ``max_iterations`` small; 2 is the correct
       value for both isolation recipes.
   * - ``CcOp.parquet`` is zero bytes
     - The capture used ``max_iterations: 1``. Collective aggregation needs at least two
       recorded iterations. Recapture.
   * - ``DmaPacket`` tables are empty
     - DMA packet metadata was not enabled at server start.
   * - Neuron Explorer cannot render the device profile
     - The NEFFs are missing. They must accompany the trace; see the ``neff_dir`` entry
       in the manifest.

Limitations
-----------

* **Host activities are opt-in and fail quietly.** With the default ``activities``, the
  ``cpu_util.pb`` and ``host_mem.pb`` files exist at zero bytes rather than being absent,
  so they can be mistaken for captured data.
* **The system trace window is short.** ``max_iterations: 2`` is required for collective
  analysis, and it caps the system trace to a window of a few seconds. A longer host-side
  session needs a separate system-only capture.
* **Component attribution depends on a decision made at compile time.** If the three
  debug variables were not set when the NEFF was compiled, the capture cannot be
  attributed back to model components and no post-processing recovers it.
* **Version compatibility is recorded, not asserted.** The skill records the package
  versions it observed for reproducibility rather than validating them against a
  supported matrix. A diagnostic that succeeds does not establish that a capture
  succeeded — check that the requested traces exist and are non-empty.

Related resources
-----------------

* :ref:`neuron-framework-profile-analysis-vllm-neuron` — analyze the capture this guide
  produces.
* :doc:`/vllm-neuron/docs/design/vllm/neuron-profiling` — how the profiler integration
  works: configuration fields, the call chain, and iteration control.
* :doc:`/vllm-neuron/docs/guides/how-to-profile-workloads` — the manual operator
  procedure.
* :doc:`/tools/tutorials/performance-profiling-vllm` — a single-bracket manual profiling
  walkthrough, including viewing a profile in Neuron Explorer.
* :doc:`/vllm-neuron/docs/getting-started/setup-guide` — set up the vLLM Neuron
  environment.
* :ref:`neuron-explorer-profile-schema-reference` — the parquet tables ingestion
  produces.
* :ref:`neuron-explorer-troubleshooting` — Neuron Explorer ingestion and rendering
  problems.
* `Neuron Agentic Development on GitHub <https://github.com/aws-neuron/neuron-agentic-development>`_
