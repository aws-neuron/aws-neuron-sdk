.. meta::
    :description: Troubleshooting guide, FAQ, and error code reference for Neuron Explorer.
    :date-modified: 06/18/2026

.. _neuron-explorer-troubleshooting:

Neuron Explorer Troubleshooting & FAQs
=======================================

This page covers common issues, error codes, and frequently asked questions
when using Neuron Explorer for profiling Trainium and Inferentia workloads.

.. contents:: On this page
   :local:
   :depth: 2


Error codes and processing failures
-----------------------------------

Profile processing errors
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Error
     - Cause & Resolution
   * - ``ERROR_PROCESS_INCOMPLETE``
     - Profile processing failed before completion. Common causes:

       - Profile too large (NTFF >5 GB may exceed processing limits)
       - NEFF/NTFF UUID mismatch (files from different compilations)
       - Disk space exhaustion on the Explorer server
       - Malformed artifacts (corrupted during transfer)

       **Fix:** Verify NEFF and NTFF UUIDs match (numeric hash in filenames
       must be identical). Check total size <5 GB. Check the Explorer
       processing logs for more specific error details.

   * - ``No metadata found for system profile: <name>``
     - Explorer cannot parse the uploaded directory as a valid system profile.

       **Common causes:**

       - Uploaded a device-only profile (NEFF+NTFF) via "Directory Upload"
         which requires system trace files
       - Missing required files (``trace_info.pb`` and/or ``ntrace.pb``)
       - Directory structure doesn't match expected layout
       - Multi-rank device profiles uploaded as directory (not yet supported
         as a single system profile view)

       **Fix:** For device-only profiles, use "Individual Files" upload or
       CLI ``neuron-explorer view -n <neff> -s <ntff>``. For system profiles,
       ensure directory contains ``trace_info.pb``.

   * - ``Failed to upload NTFF (HTTP 400)``
     - Upload rejected by the server.

       **Common causes:**

       - NTFF file exceeds upload size limit
       - Network timeout during large file transfer
       - Server-side validation failure

       **Fix:** For files >5 GB, consider
       filtering capture to specific NeuronCores to reduce file size.
       Check the Explorer processing logs for detailed error information.

   * - Profile stuck in ``PROCESSING`` or ``UPLOADED`` state
     - Processing started but never completed, or upload succeeded but
       processing didn't begin.

       **Common causes:**

       - Very large profile (>5 GB NTFF)
       - Server resource exhaustion (OOM)
       - Silent failure in processing pipeline

       **Fix:** Wait 10-15 minutes for large profiles. If still stuck after
       30 minutes, try uploading without source code (if that works, check
       source is ``.tar.gz`` format). For consistently large profiles, filter
       capture to fewer NeuronCores or shorter time windows.


Non-fatal processing messages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These messages appear in logs but don't affect profile usability:

.. code-block:: text

   ERRO[0012] Unable to process node with uid <hash> for exec 6
   ERRO[0110] invalid DMA duration - transfer rate is invalid
   ERRO[0183] Unable to convertToInt64. Cannot convert empty string "" to int64 for field ModelId.

These indicate minor issues with individual events. The overall profile is
still viewable and accurate for the remaining data.


UI and connection errors
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Resolution
   * - ``neuron-explorer`` command not found
     - Tools not installed, or ``/opt/aws/neuron/bin`` is not in PATH.
       Run ``sudo apt install aws-neuronx-tools`` or use the Neuron DLAMI.
       If the tools are installed but the command is still not found, add
       ``/opt/aws/neuron/bin`` to your PATH:
       ``export PATH=/opt/aws/neuron/bin:$PATH``
   * - UI doesn't load (connection refused)
     - Explorer server not running, or SSH tunnel misconfigured. Run
       ``neuron-explorer view`` on instance first, then verify both ports
       are forwarded: ``ssh -L 3001:localhost:3001 -L 3002:localhost:3002 ...``
   * - UI loads but shows no data / blank widgets
     - Only port 3001 forwarded. Must tunnel **both** 3001 and 3002:
       ``ssh -L 3001:localhost:3001 -L 3002:localhost:3002 ...``
   * - 500 error when opening a profile
     - Profile processed on a different Explorer version than the one
       serving it. Re-process: upload again or re-run
       ``neuron-explorer view -d <dir> --ingest-only``.
   * - Browser tab freezes/crashes on profile open
     - Profile has too many instructions for the browser to render.
       Try filtering to specific NeuronCores, or use
       ``--output-format summary-text`` for a text summary instead.
   * - ``The requested file could not be read``
     - File permission issue after reference was acquired. Usually occurs
       when profile files are moved/deleted while Explorer is running.
       Restart Explorer and re-upload.


Profiling and capture issues
----------------------------

No output / empty profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - Resolution
   * - Empty output directory when capturing a system profile
     - Profiling wasn't enabled or workload didn't execute on Neuron.
       Verify ``NEURON_RT_INSPECT_ENABLE=1`` is set. Verify the model
       is running on NeuronCores (not CPU fallback).
   * - "No profiling data" in viewer
     - Viewer pointed at the wrong directory. Use
       ``neuron-explorer view`` without ``--data-path``. Use ``-d <dir>``
       for profile output directory.
   * - ``.ntff`` files appear empty or contain no meaningful data
     - In PyTorch, this is expected when the process initializes more
       NeuronCores than it executes on. For example, if the process
       controls 64 NeuronCores but runs on NeuronCore 0 only, NTFFs
       for cores 1–63 will be empty. Use ``torchrun`` or set
       ``NEURON_RT_NUM_CORES`` to match the number of cores actually used.
       If all NTFFs are empty, verify device profiling is enabled and the
       model is warmed up (3+ iterations) before profiling starts.
   * - No ``.neff`` files in output
     - For system profiles: NEFFs are in a separate compiler cache. Set
       ``TORCH_NEURONX_NEFF_CACHE_DIR=./profile_output`` before running,
       or set ``NeuronConfig(neff_cache_dir=<dir>)`` programmatically,
       or copy from ``/tmp/neff_cache/``.
   * - Only ``.pb`` files, no ``.ntff``
     - System-only capture (``NEURON_RT_INSPECT_DEVICE_PROFILE`` not set
       or set to ``0``). Set to ``1`` or ``session`` for device traces.
   * - Unequal number of NEFF and NTFF files
     - With session-based device profiling (the default in PyTorch), a 1:1
       NEFF-to-NTFF mapping is not required — this is expected behavior.
       For ``model`` mode, some NEFFs may not have executed during the
       profiled window (common with vLLM). Explorer processes available
       matching pairs.
   * - NEFF/NTFF UUID mismatch
     - Files are from different compilations. Recompile and recapture
       in the same session to ensure matching UUIDs.


Missing data in profiles
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - Resolution
   * - DMA variable shows ``unknown``
     - DGE notifications not enabled. Set
       ``NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1`` and recapture.
       Note: collective DMAs may still show ``unknown`` even with DGE.
   * - Source Code Viewer widget is empty
     - Source code was not uploaded alongside the profile. Upload source
       as a ``.tar.gz`` with the profile.
   * - Device Trace events missing source code information
     - Debug info not captured. Set ``NEURON_FRAMEWORK_DEBUG=1`` (for model
       code) or ``NKI_DEBUG_INFO=True`` (for NKI kernels) **before
       recompilation**. Existing cached NEFFs won't have debug info.
   * - NKI source location points to non-existent files
     - Source paths in debug info are absolute from the compilation host.
       If viewing on a different machine, paths won't resolve. Upload
       source as a ``.tar.gz`` alongside the profile.
   * - No framework events in system trace
     - Framework trace (``trace.json``) not in the expected directory.
       Move ``neuron_framework_trace_rank_<N>.json`` into the matching
       ``<instance-id>_pid_<pid>/`` directory.
   * - No arrows from aten ops to device execution
     - Dependency chain requires Native PyTorch profiler integration
       (``ProfilerActivity.CPU`` + ``PrivateUse1``). If using env-var
       capture only, framework-to-device linking isn't available.
   * - Tensor Engine events missing from profile
     - Known issue with certain workloads. Verify the workload actually
       uses matmul operations. If yes, try recapturing with the latest
       ``neuron-explorer`` version.
   * - Instructions show as "unknown" or "Operation(N)"
     - Profiler version doesn't recognize newer opcodes (e.g.,
       ``activate2``, ``select_reduce``, ``range_select``,
       ``AllToAllV``). Update ``aws-neuronx-tools`` to the latest version.
   * - MFU/HFU shows 0 despite continuous PE activity
     - Known issue when the profile contains only non-matmul Tensor Engine
       instructions (e.g., transposes only). MFU counts only matmul FLOPs.
       Check ``hfu_estimated_percent`` which includes all TE instructions.
   * - ``DMA results may not be accurate`` warning
     - DGE notifications are not collected by default. Recapture with
       ``--enable-dge-notifs`` for accurate DMA metrics. Warning: this
       can result in timeout errors for large NEFFs. If an error occurs
       you can run with the flag off.
   * - Dependency highlighting doesn't show for some instructions
     - Known issue with E2E (end-to-end) profiles where dependency
       metadata may be incomplete for certain instruction types.
   * - Events appear/disappear based on zoom level
     - Virtualization optimization: Explorer only renders events visible
       at the current zoom. Zoom in to see smaller events. This is
       expected behavior for performance.
   * - Duplicate DMA packets in timeline
     - Known profiler issue where the same DMA packet appears multiple
       times. Under investigation. Does not affect DMA size calculations
       in the summary.
   * - Dropped events in system profile (``Warning: N trace events were dropped``)
     - Trace buffers filled and oldest events were overwritten.

       1. Increase buffer: set ``NeuronConfig(max_events_per_nc=<N>)``
          in PyTorch (default: 1,000,000). Uses more host memory.
       2. Apply capture-time filters (NeuronCore or event type).
       3. Shorten the profiled code region.
   * - Incomplete JAX profiles (fewer events than expected)
     - Check:

       1. Is ``jax.profiler.stop_trace`` called inside a
          ``with jax.profiler.trace`` block? Use ``stop_trace`` only with
          ``start_trace``.
       2. Is ``NEURON_RT_INSPECT_ENABLE`` set to ``1``? It should NOT be set
          when using ``jax.profiler``.
       3. Is ``NEURON_RT_INSPECT_OUTPUT_DIR`` set to the same directory passed
          to ``jax.profiler.trace``?


Performance issues
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - Resolution
   * - Profile too slow to interact with (laggy pan/zoom)
     - Large profiles with many instructions degrade UI performance.
       Use region selection or annotations to focus on a subset.
       Filter to specific NeuronCores during capture.
   * - Processing takes >30 minutes
     - Expected for very large profiles (>2 GB NTFF). Use
       ``--ignore-system-profile`` or ``--ignore-device-profile`` to
       process only what you need.
   * - Out-of-memory during profiling
     - ``ProfileMode.DEVICE`` reserves ~5 GB HBM on Trn2/Trn3. Remove from
       modes list if device traces aren't needed. Also reduce
       ``max_events_per_nc`` to limit buffer size.
   * - ``neuron-explorer`` assertion failure with multiple process groups
     - Known issue profiling MPMD workloads (e.g., TP2+EP2). Workaround:
       profile with single process group, or use session-based capture.


Timing and measurement issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - Resolution
   * - Model execution shows ~0.2 ms (impossibly fast)
     - Async dispatch: you're measuring queue submission time, not
       execution. Add ``torch.neuron.synchronize()`` before and after
       the timed region.
   * - Profile shows compilation instead of execution
     - Model wasn't warmed up. Run 3+ forward passes before starting the
       profiler to ensure you capture execution, not compilation.
   * - Collective input/output sizes off by 2x
     - Known issue with SB2SB collective reporting. The profiler may
       report 2x more data transfer than actually occurs for SB2SB
       collectives. Under investigation.
   * - ProfilerMFU vs mfu_estimated_percent discrepancy
     - ``ProfilerMFU`` in QoR CSV uses ``adjusted_hardware_flops`` (closer
       to HFU). ``mfu_estimated_percent`` in Explorer uses model flops
       only. They measure different things — see glossary for definitions.


Frequently asked questions
--------------------------

Capture and setup
^^^^^^^^^^^^^^^^^

**How do I determine NEFF execution time without profiling?**
   There is no built-in non-profiling timer for NEFF execution. The
   recommended approach is to use ``torch.neuron.synchronize()`` around
   your workload and measure wall-clock time:

   .. code-block:: python

      torch.neuron.synchronize()
      t0 = time.time()
      for _ in range(50):
          model(x)
      torch.neuron.synchronize()
      avg_ms = (time.time() - t0) / 50 * 1000

**What's the difference between ``model`` and ``session`` device profiling?**
   - ``model`` (or ``1``): Captures the first execution of each NEFF per
     NeuronCore. Good for compiled-graph workloads (``torch.compile``).
   - ``session``: Captures all device activity in one continuous NTFF.
     Required for inference serving (vLLM), eager mode, or when you need
     to see multiple executions of the same NEFF.

**How do I profile vLLM / inference serving workloads?**
   Set ``NEURON_RT_INSPECT_DEVICE_PROFILE=session`` (not ``1``/``model``).
   The standard ``model`` mode only captures the first execution, which
   misses the continuous serving behavior.

**How do I profile eager mode (torch.eager) workloads?**
   Eager mode generates many NEFFs (one per op). Profile from the 1st
   iteration (no compilation step). Use ``session`` mode and set
   ``neff_cache_dir`` in NeuronConfig to ensure all NEFFs are captured.
   Expect potentially hundreds of NEFF/NTFF pairs.

**Can I profile MPMD workloads (different NEFF per rank)?**
   Currently limited. ``neuron-explorer capture`` on a single NEFF captures
   all NeuronCores running that NEFF. For true multi-NEFF-per-rank
   visibility, use environment-variable capture with
   ``NEURON_RT_INSPECT_DEVICE_PROFILE=session`` and collect per-rank output.

**How much profiler overhead is there?**
   - System-only: <2% CPU overhead
   - Device profiling: reserves ~5 GB HBM on Trn2/Trn3 for notification buffers.
     Runtime overhead to NEFF execution is negligible due to hardware support.
     The main overhead comes from transferring profile data from device to
     host memory and saving it to disk.
   - DGE notifications: adds DMA traffic proportional to transfer count

**Why do I need to recompile after setting debug environment variables?**
   ``NEURON_FRAMEWORK_DEBUG``, ``NKI_DEBUG_INFO``, ``XLA_IR_DEBUG``, and
   ``XLA_HLO_DEBUG`` affect what metadata the compiler embeds in NEFFs.
   Previously-cached NEFFs don't have this metadata. Delete the compiler
   cache or set a new ``neff_cache_dir`` to force recompilation.


Viewing and analysis
^^^^^^^^^^^^^^^^^^^^

**What is the difference between MFU and HFU?**
   - **MFU (Model FLOPs Utilization):** Only counts FLOPs from useful
     matrix multiplications (model progress). The metric you optimize toward.
   - **HFU (Hardware FLOPs Utilization):** Counts all Tensor Engine FLOPs
     including transposes, padding, and overhead. Always ≥ MFU.
   - If HFU >> MFU: hardware is busy but doing non-useful work (transposes, padding).

**Why is MFU 0 even though Tensor Engine is active?**
   MFU only counts MATMUL instructions. If the Tensor Engine is active but
   only running transposes or weight loads, MFU will be 0. Check HFU for
   total Tensor Engine utilization.

**How do I export annotations or profile data to CSV/Excel?**
   Use the Database Viewer to run SQL queries and export results. For
   annotations, there is no direct CSV export — use the annotation save/load
   feature to persist them as JSON.

**Can I view multi-rank profiles (one NEFF, many NTFFs per rank)?**
   Multi-rank device-only profiles (without system trace) are not yet
   supported as a single unified view. You can upload each rank's
   NEFF+NTFF pair as a separate profile and compare side-by-side.
   For unified multi-rank viewing, capture a system profile which
   aggregates all ranks.

**How do I isolate metrics for specific model layers (e.g., MoE vs attention)?**
   Use the Hierarchy Viewer to identify layer boundaries, then create
   annotations at those boundaries. The Current Selection Summary and
   Box Selection Summary show metrics for the selected region only.

**Why do system and device timelines use different clocks?**
   System profiles use the host CPU clock (wall time). Device profiles use
   the NeuronCore device clock (cycle-accurate). They are correlated via
   ``nc_exec_running`` events but are currently not exactly synchronized due to
   clock domain differences.

**What does "NKI instruction coverage" mean in the Summary?**
   The percentage of instructions on each engine that were generated from
   NKI kernel code vs the Neuron compiler. Low NKI coverage (<50%) means
   most execution is compiler-generated — writing NKI kernels for those
   operations could improve performance.

**Is ``summary-json`` still supported?**
   Yes. Use ``neuron-explorer view -d <dir> --output-format summary-json``
   to get machine-readable summary metrics. The output schema may change
   between versions — pin to a specific Explorer version for automation.

**How do I see PSUM and SB usage in profiles?**
   SBUF and PSUM buffer usage data is included by default. If it appears
   missing, verify you are using the latest ``neuron-explorer`` version.
   To explicitly disable it (e.g., for faster processing of very large
   profiles), use ``-F ignore-nc-buf-usage=true``.

**How do rows reorder on zoom reset in system profiles?**
   Dragged row positions in the System Trace Viewer reset on zoom changes.
   This is a known UX limitation. Row positions don't persist across
   zoom levels.


Compatibility
^^^^^^^^^^^^^

**Are my existing Neuron Profiler/Profiler 2.0 profiles compatible?**
   Yes. Existing profile files must be reprocessed by Neuron Explorer but
   don't need recapturing. Upload them and Explorer will re-ingest.

**Is Neuron Explorer replacing Neuron Profiler?**
   Yes. Neuron Profiler and Profiler 2.0 entered end-of-support in
   Neuron 2.29. Use Neuron Explorer for all new profiling work.

**Which Python versions are supported?**
   Neuron Explorer supports Python 3.9, 3.10, 3.11, 3.12, and 3.13
   (version support follows the Neuron SDK release).

**Does Neuron Explorer work on Inf1?**
   No. Neuron Explorer is not supported on Inf1. It supports Inf2, Trn1,
   Trn2, and Trn3 instances.
