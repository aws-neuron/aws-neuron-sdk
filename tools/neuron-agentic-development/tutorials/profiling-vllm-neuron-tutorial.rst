.. meta::
   :description: Tutorial for capturing separate prefill and decode device profiles from a vLLM Neuron server on Trainium using the Neuron Agentic Development Profiling skill.
   :keywords: profiling, vllm, vllm-neuron, tutorial, device profile, system profile, NTFF, prefill, decode, Trainium, trn2, neuron-explorer, Neuron Agentic Development
   :date-modified: 2026-10-08

.. _profiling-vllm-neuron-tutorial:

Tutorial: Capture prefill and decode profiles using the Profiling skill
=======================================================================

This tutorial walks you through capturing two device profiles of a vLLM Neuron model on
Trainium: one containing only the prefill pass, and one containing only steady-state
decode. You provide a model path, a tensor parallel size, and a workload shape; the agent
brings up the server, arms the profiler at the right moment, drives the workload, tears
the server down, and writes a manifest naming every artifact it produced.

By the end of this tutorial you will have, for each phase, a device trace, a system trace,
the compiled NEFFs needed to render them, and a ``manifest.json`` that the
:ref:`profile-analysis-vllm-neuron-tutorial` takes as its input.

.. note::
   This tutorial captures profiles with an agent that isolates each phase. For the manual
   single-bracket procedure, which is a good way to see the underlying endpoints, see
   :doc:`/tools/tutorials/performance-profiling-vllm`. For how the isolation works and
   why, see the
   :ref:`Profiling deep dive <neuron-framework-profiling-vllm-neuron>`.

Prerequisites
-------------

Set up a Trainium instance
^^^^^^^^^^^^^^^^^^^^^^^^^^

You need a Trainium instance with at least as many Logical NeuronCores as the tensor
parallel size you intend to use. This tutorial uses ``trn2.48xlarge`` and a tensor
parallel size of 16, which occupies 16 of its 64 Logical NeuronCores. Launch from the
Neuron Deep Learning AMI (DLAMI) and SSH in.

Verify that Neuron devices are available and free.

.. code-block:: bash

   neuron-ls

If you see 0 devices, your instance does not have Neuron hardware attached or the driver
is not loaded. Stop here and fix that first. If devices are listed but show an owning
process ID, a previous run is still holding them — see :ref:`profiling-tutorial-cleanup`.

Install Neuron Agentic Development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have not already installed the package, follow the
:ref:`Getting Started guide <neuron-agentic-development-getting-started>`, and make sure
the deploy step completed.

.. code-block:: bash

   # For Claude Code
   deploy-neuron-agentic-development-to-claude

   # For Kiro
   deploy-neuron-agentic-development-to-kiro

Activate your Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   source ~/opt/aws_neuronx_venv_pytorch_2_9/bin/activate

Verify the vLLM Neuron package and the profiling bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The skill needs ``vllm_neuron`` to serve, and ``libtorch-neuronx-lite`` to capture. Check
both, because a missing profiling binding is the failure that lets a run look successful
while writing no device trace.

.. code-block:: bash

   python3 -c "import vllm_neuron; print(vllm_neuron.__file__)"
   python3 -c "import libtorch_neuronx_lite; print(libtorch_neuronx_lite.__file__)"

Verify Neuron Explorer
^^^^^^^^^^^^^^^^^^^^^^

``neuron-explorer`` ingests the capture. It ships with ``aws-neuronx-tools`` and is pre
installed on the DLAMI.

.. code-block:: bash

   which neuron-explorer

Download your model weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This tutorial profiles Gemma 4 31B IT, which is also the model used in the
:ref:`profile-analysis-vllm-neuron-tutorial`, so the capture you produce here is the one
you analyze there.

.. code-block:: bash

   huggingface-cli download google/gemma-4-31B-it --local-dir ~/models/gemma-4-31b-it

.. note::
   This is a full serving recipe, not a toy one. Compilation dominates the run, and a
   long prefill bucket on a model this size is at the slow end of it. If you want a
   faster first pass, run the same steps against a small dense model at a shorter
   sequence length — the procedure is identical and the artifacts are far smaller. See
   :ref:`profiling-vllm-artifacts` for measured compile and artifact figures.

Set your environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Value for this tutorial
   * - ``VLLM_NEURON_VENV_PATH``
     - ``~/opt/aws_neuronx_venv_pytorch_2_9``
   * - ``VLLM_NEURON_DIR``
     - The directory containing the installed ``vllm_neuron`` package
   * - ``NEURON_PROJECT_ROOT``
     - ``~/profiling-tutorial``. Every artifact lands under its ``agent_artifacts/``;
       there is no per-artifact override.
   * - ``HF_HOME``
     - ``~/models``
   * - ``XLA_IR_DEBUG``, ``XLA_HLO_DEBUG``, ``NEURON_FRAMEWORK_DEBUG``
     - ``1`` each
   * - ``VLLM_NEURON_WORKER_TERMINATION_TIMEOUT``
     - ``60``

.. code-block:: bash

   export VLLM_NEURON_VENV_PATH=~/opt/aws_neuronx_venv_pytorch_2_9
   export VLLM_NEURON_DIR=$(python3 -c "import vllm_neuron, pathlib; print(pathlib.Path(vllm_neuron.__file__).parent.parent)")
   export NEURON_PROJECT_ROOT=~/profiling-tutorial
   export HF_HOME=~/models
   export XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 NEURON_FRAMEWORK_DEBUG=1
   export VLLM_NEURON_WORKER_TERMINATION_TIMEOUT=60

.. important::
   Export the three debug variables **now**, before the agent compiles anything. They
   cannot be added to a capture afterwards — see :ref:`profiling-vllm-config`.

Step 1. Choose your run name and workload
-----------------------------------------

The agent needs a run name, a model path, a tensor parallel size, and a workload shape.

.. list-table::
   :header-rows: 1
   :widths: 24 41 35

   * - Argument
     - What it is
     - Example value
   * - ``run_name``
     - Names the output directory under
       ``$NEURON_PROJECT_ROOT/agent_artifacts/profiles/``. Use one per phase.
     - ``gemma4_prefill``, ``gemma4_decode``
   * - ``model``
     - Local path to the checkpoint.
     - ``~/models/gemma-4-31b-it``
   * - ``tensor_parallel_size``
     - Must divide the model's key/value head count. Determines how many Logical
       NeuronCores the run occupies.
     - ``16``
   * - ``isl``
     - Input sequence length, in tokens.
     - ``8192``
   * - ``osl``
     - Output sequence length. Pinned to ``1`` for prefill; your target decode length
       otherwise.
     - ``1024`` for decode
   * - ``conc``
     - Concurrency, which becomes the steady-state decode batch size.
     - ``8``

Step 2. Invoke the profiling agent
----------------------------------

Open your agentic IDE on the Trainium instance and ask for both phases.

.. code-block:: text

   Capture prefill and decode profiles of ~/models/gemma-4-31b-it at
   tensor-parallel size 16, with an input length of 8192 and an output
   length of 1024 at concurrency 8. Use run names gemma4_prefill and
   gemma4_decode.

The agent confirms the inputs, writes a run specification for each phase, and starts. You
do not need to do anything else while it runs.

To capture only one phase, ask for that phase alone:

.. code-block:: text

   Capture a decode-only profile of ~/models/gemma-4-31b-it at
   tensor-parallel size 16, input length 8192, output length 1024,
   concurrency 8. Call the run gemma4_decode.

.. _profiling-tutorial-input:

What the agent writes before it launches: ``run_spec.json``
-----------------------------------------------------------

Before any compilation happens, the agent writes the input contract to
``$NEURON_PROJECT_ROOT/agent_artifacts/profiles/<run_name>/run_spec.json``. Read this
file to confirm the agent understood you. It is written *first* precisely so that it
survives a compile or launch failure, and the ``vllm serve`` command line is derived from
its ``serve_config`` rather than built separately, so what is recorded cannot drift from
what ran.

For the decode phase of this tutorial it looks like this. Annotated, so it is shown as
text rather than strict JSON:

.. code-block:: text

   {
     "schema_version": "1.0",
     "run_name": "gemma4_decode",
     "phase": "decode",
     "recipe": "decode_isolation",
     "workload": {
       "isl": 8192,
       "osl": 1024,
       "conc": 8,
       "nprompts": 16,
       "steady_state_bs": 8
     },
     "serve_config": {
       "model": "/home/ubuntu/models/gemma-4-31b-it",
       "tensor_parallel_size": 16,
       "data_parallel_size": 1,
       "dtype": "bfloat16",
       "kv_cache_dtype": "fp8_e4m3",
       "max_model_len": 10240,
       "max_num_batched_tokens": 8192,
       "max_num_seqs": 8,
       "block_size": 16,
       "optimization_level": 2,
       "enable_prefix_caching": false,
       "async_scheduling": false,
       "port": 8041,
       "additional_config": {
         "neuron_config": {
           "kv_segment_size_buckets": [8192],
           "num_batched_tokens_buckets": [8192],
           "num_seqs_buckets": [8],
           "block_table_align_tokens": 2048,
           "on_device_sampling_config": {"max_top_k": 64}
         },
         "neuron_profiler": {
           "activities": ["device_profile", "system_profile"],
           "neuron_cores": [0],
           "output_dir": ".../agent_artifacts/profiles/gemma4_decode/output"
         }
       },
       "profiler_config": {
         "profiler": "cuda",
         "delay_iterations": 0,
         "max_iterations": 2
       }
     },
     "model_env": {
       // Model-specific and compile-affecting environment variables, snapshotted
       // from your shell. These vary by model and by backend, so there is no
       // default set -- inject whatever your workload needs here. Anything that
       // changes the compiler invocation belongs in this field.
       "<NAME>": "<value>"
     }
   }

Check three fields before you walk away. The deep dive explains why each matters:
``max_iterations`` and ``neuron_cores`` in :ref:`profiling-vllm-workflow`, ``model_env``
in :ref:`profiling-vllm-config`.

* ``max_iterations`` is ``2``, not ``1``.
* ``neuron_cores`` is ``[0]``.
* ``model_env`` contains every model-specific flag your workload needs. A capture of the
  wrong compilation is worse than no capture.

Step 3. What happens during the capture
---------------------------------------

Each phase runs as its own server bring-up. How the isolation works, and why each phase
needs a different trigger, is in
:ref:`the deep dive's workflow section <profiling-vllm-workflow>`.

**Phase 1: prefill isolation.** The agent launches the server and polls
``http://127.0.0.1:8041/v1/models`` until it answers — compilation dominates this wait.
It then arms the profiler while the server is idle, fires one request, and stops the
profiler when that request returns.

**Phase 2: decode isolation.** The agent launches a fresh server, starts the benchmark
workload, waits for the engine to report that no prefill is in flight, and arms there.
The profiler stops itself two decode iterations later.

If that trigger cannot fire, the agent automatically re-runs the phase on a decode-only
server and reports that it did so. The resulting trace is a valid decode profile.

.. important::
   In the fallback mode the KV cache is never populated, so the generated tokens are
   meaningless. This does not affect the profile, because the fixed-shape decode graph
   does the same computation regardless of KV contents, but never use such a trace for
   anything that reads model outputs.

**Teardown.** After each phase the agent kills the server by its recorded process group
and records the teardown separately from the capture result.

.. _profiling-tutorial-output:

Step 4. Check the results
-------------------------

Each run directory looks like this.

.. code-block:: text

   $NEURON_PROJECT_ROOT/agent_artifacts/profiles/gemma4_decode/
   ├── run_spec.json        # the input contract, from Step 2
   ├── manifest.json        # the output contract — start here
   ├── capabilities.json    # profiler interface diagnostic
   ├── capture.json         # which trace files appeared, and their sizes
   ├── cleanup.json         # teardown receipt
   ├── bench_c8.json        # benchmark result
   ├── logs/serve.log
   └── output/
       ├── i-<instance>_pid_<pid>/<timestamp>/
       │   ├── profile_nc_0_session_0.ntff
       │   ├── ntrace.pb
       │   ├── trace_info.pb
       │   ├── cpu_util.pb
       │   └── host_mem.pb
       └── neffs/graph_<hash>.neff

``manifest.json`` is the final product. It names every artifact by absolute path, so
nothing downstream has to guess at the session directory — whose name contains a
nanosecond timestamp that cannot be known in advance.

.. code-block:: json

   {
     "schema_version": "1.1",
     "recipe": "decode_isolation",
     "run_name": "gemma4_decode",
     "workload": {"isl": 8192, "osl": 1024, "conc": 8, "nprompts": 16, "steady_state_bs": 8},
     "run_spec": ".../profiles/gemma4_decode/run_spec.json",
     "profile": {
       "session_dir": ".../output/i-0c6ee4eb5d1add807_pid_735169/1788460887886211545",
       "device_ntff": ".../1788460887886211545/profile_nc_0_session_0.ntff",
       "system_trace_pb": ".../1788460887886211545/ntrace.pb",
       "trace_info_pb": ".../1788460887886211545/trace_info.pb",
       "cpu_util_pb": ".../1788460887886211545/cpu_util.pb",
       "host_mem_pb": ".../1788460887886211545/host_mem.pb",
       "neff_dir": ".../output/neffs"
     },
     "neff_hashes": ["94d0d6227a3b1ad1b7cdfbf2c0af3a94"],
     "bench_result_json": ".../profiles/gemma4_decode/bench_c8.json"
   }

.. _profiling-tutorial-confirm:

Confirmation
------------

A server that served correctly can still write no device trace, so confirm the artifacts
rather than the exit status.

**1. The manifest names a device trace, and it is not empty.**

.. code-block:: bash

   cd "$NEURON_PROJECT_ROOT/agent_artifacts/profiles"
   for run in gemma4_prefill gemma4_decode; do
     ntff=$(python3 -c "import json;print(json.load(open('$run/manifest.json'))['profile']['device_ntff'] or '')")
     echo "$run: ${ntff:-NO DEVICE TRACE}"
     [ -n "$ntff" ] && du -h "$ntff"
   done

A ``null`` value, or a size of a few hundred kilobytes for the whole output directory,
means the profiler's buffers could not be allocated. See
:ref:`profiling-tutorial-troubleshooting`.

**2. The sizes are plausible.** A real device trace is megabytes to gigabytes, and prefill
is substantially larger than decode at the same configuration. Compare yours against the
measured figures for this model class in :ref:`profiling-vllm-artifacts`; a trace orders of
magnitude smaller than those means the profiler wrote a stub.

.. note::
   The prefill trace is the one that matters downstream — ingesting it is the slow step of
   the analysis tutorial. Check the cost before you start it.
   See :ref:`neuron-framework-profile-analysis-vllm-neuron`.

**3. The recipe matches the phase you asked for.** ``prefill_isolation`` for one run,
``decode_isolation`` for the other. A prefill run that reports ``decode_isolation`` means
the phases were mixed up, and the trace will not answer a prefill question.

**4. The NEFFs are present.** ``neff_dir`` must contain at least one
``graph_<hash>.neff``. Neuron Explorer cannot render a device profile without them.

**5. The devices are released.**

.. code-block:: bash

   neuron-ls

Every core should show no owning process ID.

Step 5. Hand the capture to the analysis skill
----------------------------------------------

The capture is an input, not an answer. Point the analysis skill at the manifest to get
performance bounds, a collective breakdown, and a per-component attribution:

.. code-block:: text

   Analyze the decode profile at
   ~/profiling-tutorial/agent_artifacts/profiles/gemma4_decode/manifest.json

See :ref:`profile-analysis-vllm-neuron-tutorial` for the full walkthrough.

.. _profiling-tutorial-cleanup:

Clean up
--------

The agent tears down its own server, but if a run was interrupted the workers may still
hold devices, and the next launch will then fail during runtime initialization.

.. code-block:: bash

   neuron-ls                  # any core with a PID is still held
   fuser -v /dev/neuron*      # which processes hold the devices

.. important::
   Do not clear a stuck server with a pattern-based kill such as ``pkill -f``. A server
   launched in the background shares its process group with whatever started it, so a
   group kill can take out your shell or your IDE session. Ask the agent to tear down by
   recorded process group, or kill the specific process IDs that ``fuser`` reports.

The captures themselves are large. Remove the run directories you no longer need, and
terminate the instance when you are finished.

.. _profiling-tutorial-troubleshooting:

Troubleshooting
---------------

The deep dive carries the full failure reference, grouped by phase, at
:ref:`profiling-vllm-debugging`. Three things bite specifically during the steps above.

**The agent stops immediately saying an environment variable is unset.** The skill
hard-fails on ``VLLM_NEURON_VENV_PATH`` and ``VLLM_NEURON_DIR`` rather than guessing.
Re-export them in the shell your IDE inherits, not only in a different terminal.

**Generation succeeds but the output directory is a few hundred kilobytes.** The device
profiler's buffer pool could not be allocated, which fails silently — requests succeed and
the server looks healthy. Ask the agent to raise ``VLLM_NEURON_RESERVED_HEADROOM_GIB``,
which trades unused KV cache for profiler room; it may need to lower
``VLLM_NEURON_MIN_KV_BUDGET_GIB`` too. See :ref:`profiling-vllm-config`.

**Decode arming times out.** Expected on short decode legs. The agent falls back to a
decode-only server automatically; confirm from its output that it did, and re-read the
caveat in Step 3 before using the trace for anything that reads model outputs.

Everything else is in :ref:`profiling-vllm-debugging`, which groups symptoms into launch
and compile failures (peak HBM errors, unexpected recompiles, runtime initialization
errors on the next launch), capture failures (zero-byte ``.pb`` files, a null device
trace, zero-byte host activities, empty generations), and ingestion failures (hours-long
ingests, zero-byte parquet, an empty ``CcOp.parquet``, Neuron Explorer refusing to
render).

Related resources
-----------------

- :ref:`neuron-framework-profiling-vllm-neuron` for the deep dive on how the Profiling
  skill isolates each phase and manages the server lifecycle.
- :ref:`profile-analysis-vllm-neuron-tutorial` for analyzing the capture you just made.
- :doc:`/tools/tutorials/performance-profiling-vllm` for the manual single-bracket
  procedure and for viewing a profile in Neuron Explorer.
- :doc:`/vllm-neuron/docs/design/vllm/neuron-profiling` for the profiler integration
  design.
- :ref:`neuron-explorer-profile-schema-reference` for the parquet tables ingestion
  produces.
- :doc:`/tools/neuron-agentic-development/getting-started` for installation and setup.
- `Neuron Agentic Development GitHub <https://github.com/aws-neuron/neuron-agentic-development>`_
  for source code and issue tracker.
