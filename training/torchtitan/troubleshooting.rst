.. meta::
   :description: Troubleshooting common errors and symptoms for TorchTitan running on AWS Neuron with TorchNeuron, including the TorchTitan profiler interface.
   :keywords: TorchTitan, TorchNeuron, AWS Neuron, Trainium, troubleshooting, profiler, FSDP, OOM
   :date-modified: 2026-10-15

.. _torchtitan-neuron-troubleshooting:

================================================
Troubleshooting: TorchTitan Neuron
================================================

Overview
--------

This topic covers common failure modes for TorchTitan on Neuron runs and the
TorchTitan profiler interface you use to diagnose them. Use it when your
TorchTitan run is failing, hanging, or producing unexpected loss trajectories
on Trainium. Scope is TorchTitan-specific — for lower-level TorchNeuron issues,
see :doc:`/torch/troubleshooting`.

Common symptoms and solutions
-----------------------------

- **Symptom: TorchTitan exits immediately during process-group init.**

  **Possible causes:**

  - torch-neuronx not installed, or ``device='neuron'`` does not resolve.
  - The launcher's world size does not match the instance's NeuronCore count.

  **Solutions:**

  - Verify that TorchNeuron works standalone by following
    :doc:`/torch/get-started/quickstart-torch-neuron` first.
  - Confirm ``NGPU`` matches the number of NeuronCores on the instance (64 on
    a ``trn2.48xlarge``). ``run_train.sh`` passes ``NGPU`` to ``torchrun`` as
    ``--nproc_per_node``.

- **Symptom: Multi-node run hangs or fails during collectives bootstrap.**

  **Possible causes:**

  - EFA is not available or not configured on one or more nodes.
  - ``RDZV_ENDPOINT`` is unreachable from every node, so the rendezvous cannot
    complete.

  **Solutions:**

  - Confirm EFA is present on every node and that each node reports a non-zero
    EFA endpoint count.
  - Verify ``RDZV_ENDPOINT`` points at an address all nodes can reach and that
    the port is open. A bootstrap failure typically surfaces as:

    .. code-block:: text

       ERROR ENC:ncclBootstrapRecv  failed neuronBootstrapRecv request to NCCL

- **Symptom: FSDP shard initialization fails with a shape or dtype error.**

  A mixed-dtype parameter set, for example, fails with:

  .. code-block:: text

     AssertionError: FSDP expects uniform original parameter dtype but got {torch.float32, torch.bfloat16}

  **Possible causes:**

  - The FSDP wrap policy does not align with the model definition.
  - The mixed-precision configuration is inconsistent across modules.

  **Solutions:**

  - Use a shipped per-family config (selected by name via ``CONFIG``) rather
    than assembling parallelism settings by hand.
  - Keep the mixed-precision configuration unified; do not override it
    per-module without a specific reason.

- **Symptom: Out of memory.**

  Device-memory exhaustion surfaces as a ``Failed to allocate`` message naming
  the size the runtime could not reserve:

  .. code-block:: text

     Failed to allocate x.yGB

  **Possible causes:**

  - The config does not fit the device-memory budget for this instance —
    activation checkpointing, batch size, or the parallelism degrees are not
    sized for the available memory per rank.

  **Solutions:**

  - Reduce peak memory by tuning, roughly in this order:

    - **Activation checkpointing** — enable it, or widen from selective (SAC)
      to full AC.
    - **Local batch size** — lower ``training.local_batch_size``.
    - **FSDP degree** — increase the data-parallel shard degree so parameters,
      gradients, and optimizer state are sharded across more ranks.
    - **TP degree** — increase the tensor-parallel degree to split individual
      layers across more NeuronCores.

    Change one lever at a time so you can attribute the effect. See
    :doc:`/training/torchtitan/best-practices`.

Diagnostic steps
----------------

When a run misbehaves, work from the known-good baseline outward:

1. Confirm TorchNeuron works standalone with
   :doc:`/torch/get-started/quickstart-torch-neuron`.
2. Confirm TorchTitan installs cleanly and the verification run in
   :doc:`/training/torchtitan/get-started/setup-torchtitan-neuron` succeeds.
3. Run the basic tutorial
   :doc:`/training/torchtitan/tutorials/tutorial-basic-training` against a
   small, short run.
4. Bisect between the last known-good run and the current failing run by
   reverting one change at a time.

The TorchTitan profiler
-----------------------

When a run is slower than expected, use the TorchTitan profiler to find where
the time goes. TorchTitan has a built-in profiler for both execution traces and
memory snapshots, configured through ``profiler.*`` options (set in a config or
overridden on the command line). This section documents the TorchTitan-facing
interface; for how the underlying PyTorch profiler and memory-snapshot
mechanics work in depth, see
:doc:`/torch/guides/memory-profiling-guide` and :doc:`/torch/troubleshooting`.

Execution tracing
~~~~~~~~~~~~~~~~~~

Enable the profiler to capture Kineto execution traces:

- ``profiler.enable_profiling`` — turn tracing on (default off).
- ``profiler.save_traces_folder`` — output directory (default
  ``profiling/traces``).
- ``profiler.profile_freq`` — capture every N steps (default 10).
- ``profiler.profiler_warmup`` / ``profiler.profiler_active`` — warmup and
  active step counts within a capture window (defaults 3 and 1).

On Neuron the profiler is driven by ``torch_neuronx``: the Neuron runtime is
exposed as the profiler's device activity. Lightweight runtime, CPU, and host
metrics — the **system** profile — are always captured and are usually enough
to locate a bottleneck, so start there. Only enable the heavy **device**
profile (hardware instruction traces) if the system profile does not explain
the problem: the device trace reserves a large amount of HBM, so it is
**opt-in** via environment variables:

- ``NEURON_PROFILE_DEVICE=1`` — add the device (hardware-trace) profile. Leave
  unset for runtime/host-only profiling.
- ``NEURON_PROFILE_CAPTURE_NC`` — which NeuronCores to capture the device trace
  for (default ``0,1``). Each core's trace is large, so capturing every core of
  a 64-core run produces a very large profile; capture a subset. Accepts a
  comma-separated list or a range, for example ``0,1,2,3`` or ``0-63``.
- ``NEURON_PROFILE_RANKS`` — global-rank allowlist for profiling (default
  ``0``). With one process per NeuronCore, this is distinct from
  ``NEURON_PROFILE_CAPTURE_NC``.
- ``NEURON_PROFILE_NEFF_CACHE_DIR`` — directory the runtime reads compiled NEFFs
  from so the device trace can be resolved (default ``/tmp/neff_cache``).
  Override this if the run's NEFF cache lives elsewhere.

Viewing captured traces in Neuron Explorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze a captured profile with **Neuron Explorer**, which ingests native
PyTorch profiles and renders an interactive timeline. The Neuron Explorer docs
cover the end-to-end flow:

- :doc:`/tools/neuron-explorer/how-to-profile-pytorch` — how to capture a
  native PyTorch profile from a Neuron workload.
- :doc:`/tools/neuron-explorer/get-started` — how to launch Neuron Explorer and
  upload the captured profile for interactive analysis.

Memory snapshots
~~~~~~~~~~~~~~~~~

To capture periodic memory snapshots (useful for diagnosing OOM and memory
growth):

- ``profiler.enable_memory_snapshot`` — turn memory snapshots on (default off).
- ``profiler.save_memory_snapshot_folder`` — output directory (default
  ``profiling/memory_snapshot``).

On OOM, TorchTitan dumps a final snapshot into a ``step_<N>_exit`` directory
(under the ``profiling/memory_snapshot`` folder) so you can inspect the
allocation state at the point of failure. For a worked example
of capturing and visualizing a snapshot from a TorchTitan run, see
:ref:`Profile memory in TorchTitan <torchneuron-memory-profiling-titan-example>`.

Troubleshooting tools
---------------------

- **TorchTitan profiler:** execution traces and memory snapshots, configured as
  above. Start here for TorchTitan-specific performance and memory questions.
- **Neuron Explorer:** inspect where memory or compute goes intra-step. See
  :doc:`/tools/neuron-explorer/index`.
- **torch-neuronx memory APIs:** step-level memory snapshots at the TorchNeuron
  level. See :doc:`/torch/guides/memory-profiling-guide`.

Collecting information for support
----------------------------------

Before opening a support case, collect:

- TorchTitan, torch-neuronx, PyTorch, and Neuron SDK versions.
- The exact command line used to launch the run (including ``NGPU``,
  ``MODULE``, ``CONFIG``, and any overrides).
- The model revision and dataset revision in use.
- The full stack trace or error output.
- Output of ``neuron-ls`` on the instance.

.. include:: /includes/torch/trn3-sizing-note.txt

Related topics
--------------

- :doc:`/torch/troubleshooting` — Lower-level TorchNeuron troubleshooting.
- :doc:`/torch/guides/memory-profiling-guide` — TorchNeuron memory profiling in
  depth.
- :doc:`/training/torchtitan/about-torchtitan-neuron`
- :doc:`/training/torchtitan/best-practices`
