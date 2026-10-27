.. meta::
    :description: Reference for the files that Neuron Explorer captures and processes, including NEFF, NTFF, and system profile files, what each one contains, which files each profile type needs, and how files are named and paired.
    :date-modified: 10/26/2026

.. _neuron-explorer-profile-artifacts:

Neuron Explorer profile files
=============================

Capturing a profile writes raw profile files to disk. When you add them to the Neuron Explorer
server, it processes them into Parquet tables stored under its data path. This page explains which
files each profile type needs, what each file contains, and how files are named and paired.

.. contents:: On this page
   :local:
   :depth: 1


Files each profile type needs
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Profile type
     - Files you need
   * - **Device**
     - The ``.ntff`` files and every ``.neff`` they executed.
   * - **System**
     - A capture output directory that contains ``trace_info.pb`` and ``ntrace.pb``. Optional:
       ``cpu_util.pb``, ``host_mem.pb``, ``trace.json``, and a memory snapshot.
   * - **System + device**
     - The same capture output directory, which also contains the ``.neff`` and ``.ntff`` files.

Source code is added separately from the profile files. To add files to the server, see
:ref:`Step 5 of Get Started <neuron-explorer-step5-upload-profile>`.


Raw profile files
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 27 30 25

   * - File
     - Produced by
     - What it contains
     - What needs it
   * - ``.neff``
     - The Neuron compiler. For example, ``torch_neuronx.trace(..., compiler_workdir=...)``
       writes ``graph.neff``. PyTorch profiling copies NEFFs into the profile directory when
       ``neff_cache_dir`` or ``TORCH_NEURONX_NEFF_CACHE_DIR`` is set.
     - Neuron Executable File Format: the compiled program, with its instructions and metadata.
       When compiled with ``NKI_DEBUG_INFO=True``, it also embeds ``kernel_debug_info.json``,
       which maps instructions to NKI source lines.
     - Every device profile. An NTFF can't be processed without the NEFF it was captured from.
   * - ``.ntff``
     - Device profiling: ``NEURON_RT_INSPECT_DEVICE_PROFILE`` (environment variable),
       ``ProfileMode.DEVICE`` (PyTorch), ``neuron-explorer capture`` (CLI), or
       ``neuron-explorer inspect`` (CLI).
     - Neuron Trace File Format: the raw device execution trace for a NeuronCore.
     - The Device Trace Viewer.
   * - ``ntrace.pb``
     - System profiling: ``NEURON_RT_INSPECT_ENABLE=1`` (environment variable),
       ``ProfileMode.RUNTIME`` (PyTorch), or ``neuron-explorer inspect`` (CLI).
     - Neuron Runtime events: API calls, model loads, and executions.
     - The System Trace Viewer. Required for **Directory Upload**.
   * - ``trace_info.pb``
     - System profiling, written alongside ``ntrace.pb``.
     - Metadata about the system trace capture session.
     - Every system profile. **Directory Upload** requires it.
   * - ``cpu_util.pb``
     - CPU utilization sampling: ``NEURON_RT_INSPECT_CPU_UTIL=1`` (environment variable),
       ``ProfileMode.CPU_UTIL`` (PyTorch), or ``neuron-explorer inspect`` (CLI).
     - Sampled host CPU utilization per core.
     - The System Trace Viewer. Optional.
   * - ``host_mem.pb``
     - Host memory sampling: ``NEURON_RT_INSPECT_HOST_MEMORY=1`` (environment variable),
       ``ProfileMode.HOST_MEMORY`` (PyTorch), or ``neuron-explorer inspect`` (CLI).
     - Sampled host memory usage over time.
     - The System Trace Viewer. Optional.
   * - ``trace.json``
     - PyTorch profiling, through ``NeuronProfiler.export_trace``.
     - The ``torch.profiler`` framework trace: CPU operations and Python call stacks.
     - The System Trace Viewer. Optional.
   * - ``memory_snapshot_rank<N>.pickle``
     - ``NeuronProfiler.export_snapshot`` (PyTorch).
     - PyTorch caching-allocator history with allocation stack traces.
     - The **PyTorch Allocator** tab of the Memory Viewer. Optional. See
       :ref:`capture-pytorch-memory-snapshot`.
   * - Source code (``.py`` files, a folder, or a ``.tar.gz``)
     - You.
     - Your NKI or framework source.
     - The Source Code Viewer. Uploaded separately from the profile.

.. tip::

   Keep the raw files after you upload them. If you upgrade Neuron Explorer and an older profile
   fails to open, re-process the raw files rather than recapturing. Processed profiles are tied to
   the Explorer version that processed them.


How files are named and paired
------------------------------

If an NTFF and the NEFFs you upload with it come from different compilations, processing fails with
``ERROR_PROCESS_INCOMPLETE`` (see :doc:`troubleshooting`).

**Captured with environment variables or** ``neuron-explorer inspect``
   The NEFF and its NTFFs share a numeric hash. ``neff_395760075800974.neff`` pairs with
   ``395760075800974_instid_0_vnc_0.ntff``, ``395760075800974_instid_0_vnc_1.ntff``, and so on.
   There is one NTFF per NeuronCore. A NeuronCore that the process initialized but didn't run on
   produces an empty NTFF, which you can ignore.

**Captured with** ``neuron-explorer capture``
   The NTFF name comes from ``-s``. Explorer adds suffixes depending on how you capture:

   .. list-table::
      :header-rows: 1
      :widths: 50 50

      * - Command
        - Output
      * - ``-s profile.ntff``
        - ``profile.ntff``
      * - ``-s profile.ntff --profile-nth-exec=2``
        - ``profile_exec_2.ntff`` (the 2nd execution)
      * - ``-s profile.ntff --collectives-workers-per-node 4``
        - ``profile_rank_0.ntff`` through ``profile_rank_3.ntff``
      * - Both of the above
        - ``profile_rank_<rank>_exec_<n>.ntff``

   The NEFF is the file you passed to ``-n``.

**Captured with session-mode device profiling** (``NEURON_RT_INSPECT_DEVICE_PROFILE=session``)
   Each NeuronCore writes a single NTFF for the whole session, such as
   ``profile_nc_0_session_0.ntff``, next to the NEFFs it executed.


Capture output directory layout
-------------------------------

Capture with environment variables, PyTorch, JAX, or ``neuron-explorer inspect`` writes one
subdirectory per process, named ``<instance-id>_pid_<pid>``, so many processes and nodes can share
one output directory. PyTorch profiling also adds a timestamped session directory inside each one.
For an example tree, see :ref:`Expected output <neuron-explorer-profile-expected-output>`.

Pass the top-level output directory to ``--profile-directory`` or ``-d``, or select it in
**Directory Upload**.


Processed profile data
----------------------

The server writes processed profiles as Parquet tables under its data path (see
:ref:`neuron-explorer-step4-launch-server`). Manage them through the
Profile Manager rather than editing the data path directly. To delete a profile, see
:ref:`neuron-explorer-get-started-delete-profile`.

To get the Parquet tables in a folder of your choice for scripts or coding agents, use
``neuron-explorer view --output-format parquet --output-file <dir>``. See
:doc:`how-to-analyze-profile-output` and :doc:`profile-schema-reference`.
