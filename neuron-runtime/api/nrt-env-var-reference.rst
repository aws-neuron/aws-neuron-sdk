.. meta::
   :description: Reference for AWS Neuron Runtime (NRT) environment variables that configure NeuronCore allocation, collectives, memory, logging, profiling, and debugging on Trainium and Inferentia.
   :date-modified: 2026-09-27
   :keywords: Neuron, Neuron Runtime, NRT, libnrt, environment variables, NEURON_RT, NEURON_RT_VISIBLE_CORES, NEURON_RT_LOG_LEVEL, configuration, logging, profiling

.. _nrt-env-var-reference:

================================================
Reference: Neuron Runtime environment variables
================================================

The Neuron Runtime (``libnrt.so``) reads its configuration from process-level environment variables that control NeuronCore allocation, collectives, memory, execution, logging, profiling, and debugging. Neuron framework extensions set most of these on your behalf, so set them explicitly only when you need a specific behavior. Use this page to look up a variable, its type and default, and when to apply it. It lists every runtime variable that is supported for customer use; variables not listed here are internal diagnostics and are unsupported.

Applies to
----------

- The Neuron Runtime (``libnrt.so``) on AWS Trainium and Inferentia instances.

.. _nrt-env-how-to-use:

How to use these environment variables
---------------------------------------

Set a variable inline for a single process, or export it to apply it to every
process in the shell session:

.. code-block:: bash

   # Apply to one invocation only
   NEURON_RT_VISIBLE_CORES=0-3 python myapp.py

   # Apply to every process in this shell
   export NEURON_RT_LOG_LEVEL=INFO
   export NEURON_RT_LOG_LOCATION=console
   python myapp.py

Boolean variables accept ``0``, ``1``, ``false``, or ``true``,
case-insensitively.

To partition one instance across several processes, give each process its own
slice of cores with ``NEURON_RT_VISIBLE_CORES``:

.. code-block:: bash

   # Partition one instance across parallel processes
   NEURON_RT_VISIBLE_CORES=0-1 python worker.py &
   NEURON_RT_VISIBLE_CORES=2-3 python worker.py &

By default the runtime logs at ``WARN`` to both console and syslog; setting only
``NEURON_RT_LOG_LEVEL`` sends logs to the console alone and stops syslog output,
and setting only ``NEURON_RT_LOG_LOCATION`` lowers the level to ``ERROR``, so
warnings are no longer logged.

.. _nrt-env-functional:

Functional
----------

These variables control normal runtime behavior — which cores a process uses,
how models load, distributed bootstrap, and numerics. Set them to make the
runtime do something specific that the framework does not configure for you.

.. list-table::
   :header-rows: 1
   :widths: 30 22 16 62

   * - Variable
     - Type / default
     - Subsystem
     - Description
   * - ``NEURON_RT_VISIBLE_CORES``
     - Range list (``0-3``, ``0,2,4``); default: all
     - Core allocation & topology
     - Restrict the process to an explicit set of logical NeuronCores, so multiple processes can partition one instance. Validated against actually-visible ``/dev/neuron*`` devices, so a bad range fails fast at init.
   * - ``NEURON_LOGICAL_NC_CONFIG``
     - Unsigned integer; default: ``2`` (``1`` on Trn1)
     - Core allocation & topology
     - Set the LNC size: how many physical NeuronCores fuse into one logical core. Must be a power of two dividing the per-device core count; only ``2`` is publicly supported on Trn2+.
   * - ``NEURON_RT_ROOT_COMM_ID``
     - Host-and-port string; default: unset
     - Networking & barriers
     - Socket address (``host:port``) of rank 0 for collectives bootstrap. The variable that makes multi-node distributed work at all.
   * - ``NEURON_RT_EXEC_TIMEOUT``
     - Unsigned integer seconds; default: ``30``
     - DMA & execution
     - Per-execution timeout in seconds. Distributed workloads may legitimately need far more than the default because of network interruptions — the usual cause of spurious timeouts.
   * - ``NEURON_RT_ALLOW_LEGACY_NEFF``
     - Unsigned integer; default: ``0``
     - Model loading & firmware
     - Accept older NEFF format versions that would otherwise be rejected. When unset, NRT uses ``0``.
   * - ``NEURON_RT_VALIDATE_HASH``
     - Boolean; default: ``false``
     - Model loading & firmware
     - Verify the NEFF package hash at load, catching a corrupt or tampered artifact before execution. When unset, NRT uses ``false``.
   * - ``NEURON_RT_ENABLE_RG_VALIDATION``
     - Boolean; default: ``false``
     - Collectives
     - Pre-validate replica groups before ``nrt_execute`` — catches a malformed group with a clear error instead of a hang. When unset, NRT uses ``false``.
   * - ``NEURON_RT_ASYNC_SENDRECV_BOOTSTRAP_PORT``
     - Unsigned integer; default: ``0``
     - Networking & barriers
     - Bootstrap port for the async send/recv feature above. When unset, NRT uses ``0``.
   * - ``NEURON_RT_ASYNC_SENDRECV_EXPERIMENTAL_ENABLED``
     - Boolean; default: ``false``
     - Networking & barriers
     - Enable the experimental asynchronous EFA send/recv path. When unset, NRT uses ``false``.
   * - ``NEURON_RT_ENABLE_P2P``
     - Boolean; default: driver-dependent
     - Networking & barriers
     - Selects the driver P2P interface instead of dma-buf for EFA memory registration. NRT defaults to P2P when dma-buf v2 is unavailable and forces P2P back on if the user disables it with an incompatible driver.
   * - ``NEURON_FAIL_ON_NAN``
     - Boolean; default: ``false``
     - Numerics
     - Stop treating ``NRT_EXEC_COMPLETED_WITH_NUM_ERR`` as success: abort or raise when a NaN is detected. Also forces numerical-error verbosity to at least ``critical``.
   * - ``NEURON_RT_STOCHASTIC_ROUNDING_EN``
     - Boolean; default: ``false``
     - Numerics
     - Enable stochastic rounding across DMA — reduces systematic bias when accumulating in low precision. When unset, NRT uses ``false``.
   * - ``NEURON_RT_STOCHASTIC_ROUNDING_SEED``
     - Unsigned integer; default: ``0``
     - Numerics
     - Seed the stochastic rounding RNG, making an otherwise nondeterministic run reproducible. When unset, NRT uses ``0``.

.. warning::

   ``NEURON_RT_ALLOW_LEGACY_NEFF`` is for testing only and should not be used in
   production. Running a NEFF compiled for a different target architecture can
   cause silent data corruption in model outputs or hangs at runtime, among
   other unexpected failures.

.. _nrt-env-performance:

Performance tuning
------------------

These variables trade off throughput, latency, and memory. The defaults are
tuned for common cases; reach for these when profiling shows a specific
bottleneck in collectives, scratchpad memory, or the networking proxy path.

.. list-table::
   :header-rows: 1
   :widths: 30 22 16 62

   * - Variable
     - Type / default
     - Subsystem
     - Description
   * - ``NEURON_RT_DBG_DMA_PACKETIZATION_SIZE``
     - Byte count; default: ``4096``
     - DMA & execution
     - Global DMA packetisation size. Hard-capped at ``UDMA_M2S_CFG_LEN_MAX_PKT_SIZE_MASK`` (1 MiB − 1 B); exceeding it fails init. Large packets can starve other DMA queues, because a queue isn't rescheduled until its packet ends — which is exactly how they delay collectives.
   * - ``NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS``
     - Unsigned integer; default: ``8``, max ``63``
     - DMA & execution
     - In-flight request depth for the hardware execution queue. Capped at 63 by the 4 MB host DMA allocation limit; higher values warn and clamp rather than fail.
   * - ``NEURON_RT_DISABLE_DEBUG_INFO``
     - Boolean; default: ``false``
     - Model loading & firmware
     - Skips construction of the model debug-information graph during NEFF load. This can reduce load work, but source-level out-of-bounds diagnostics will not be available for that model.
   * - ``NEURON_RT_CC_CHANNEL_BUFFER_N``
     - Unsigned integer; default: ``8``
     - Collectives
     - Number of collectives channel buffers. Does not affect the P2P path (``ENC_P2P_NR_CHANNEL_CHUNKS`` is a compile-time constant).
   * - ``NEURON_RT_CC_CHUNK_SIZE``
     - Byte count; default: ``2 MiB``
     - Collectives
     - Collectives chunk size. Does not affect the P2P path.
   * - ``NEURON_RT_CC_STREAM_SPLIT``
     - ``W0:W1``; default: ``3:1``
     - Collectives
     - Divide TOP_SPs and DMA qbundles between streams as ``W0:W1`` — first stream's weight, then the weight shared among the rest. Every stream keeps at least one of each resource. Malformed input logs an error and keeps the default.
   * - ``NEURON_RT_DBG_CC_DMA_PACKET_SIZE``
     - Unsigned integer list; default: unset
     - Collectives
     - Per-*stream* max DMA packet size for collectives (list starting at stream 0). Kept for backward compatibility only: the runtime's per-priority packet sizing takes precedence, and when that is in effect this variable is ignored with a warning.
   * - ``NEURON_RT_DISABLE_EXECUTION_BARRIER``
     - Boolean; default: ``false``
     - Networking & barriers
     - Drop the collectives barrier between executions. Valid only when the workload is genuinely SPMD *and* the caller runs its own barrier (typical distributed inference) — a real latency win, and a hang if the precondition doesn't hold.
   * - ``NEURON_RT_ENABLE_HW_EXECUTION_BARRIER``
     - Boolean; default: arch-dependent
     - Networking & barriers
     - Use the hardware execution barrier. Defaults on for Trn2-family and Neuron Switch v1; left off on Trn1 due to a suspected deadlock with the barrier's traffic pattern.
   * - ``NEURON_RT_LOW_LATENCY_TASKS_CPU_AFFINITY``
     - Integer range list; default: unset
     - Networking & barriers
     - Confine latency-sensitive runtime threads to a chosen CPU set (e.g. ``40-47,88-95``) — keeps proxy threads off the cores your framework is using. Validated against the real CPU count.
   * - ``NEURON_RT_ONE_THREAD_PER_CORE``
     - Boolean; default: ``false``
     - Networking & barriers
     - Pin each proxy thread to a single CPU core and drop ``sched_yield``. Recommended when the core-to-proxy ratio is 1:1.
   * - ``NEURON_RT_RANKS_PER_NETWORK_PROXY``
     - Unsigned integer; default: ``1``
     - Networking & barriers
     - Consolidate *N* ranks onto one network proxy thread, trading proxy parallelism for CPU cores. Requires the HW barrier: init fails if ``>1`` with ``ENABLE_HW_EXECUTION_BARRIER=0``.
   * - ``NEURON_RT_REGISTER_HBM_MR``
     - Boolean; default: ``true``
     - Networking & barriers
     - Register one EFA memory region covering all HBM instead of one per tensor — far fewer registrations at scale. Requires ``NEURON_RT_MAP_HBM``; forced off without it.
   * - ``NEURON_RT_CONTIGUOUS_SCRATCHPAD``
     - Boolean; default: driver-dependent
     - Memory & scratchpad
     - Allocates scratchpad pages adjacent in HBM and defaults on where the platform and driver support it. Set it to ``0`` as a workaround for rare HBM-fragmentation load failures. On Trn2+ LNC1 the default becomes ``0`` because the feature is unsupported there, but an explicit ``1`` is **not** overridden — it is honoured at config time and then rejected by the tdrv safeguard, failing ``nrt_init()`` with ``NRT_INVALID``.
   * - ``NEURON_RT_MAP_HBM``
     - Boolean; default: ``true`` (Cayman+)
     - Memory & scratchpad
     - mmap HBM into the process address space during device init (Trn2+ only), enabling direct host access. Prerequisite for ``NEURON_RT_REGISTER_HBM_MR``.
   * - ``NEURON_SCRATCHPAD_PAGE_SIZE``
     - Unsigned integer MiB; default: ``512``
     - Memory & scratchpad
     - Scratchpad page size in MiB. Ignored when contiguous scratchpad is in use, because that path forces 64 MiB pages. When a model's largest variable does not fit the current page size, the runtime logs a recommended value for this variable alongside the matching compiler flag.

.. _nrt-env-observability:

Observability
-------------

These variables control what the runtime logs and which metrics and diagnostic
events it emits. Use them to increase visibility during bring-up, performance
analysis, or a support investigation. Profile capture has its own section,
:ref:`Profiling <nrt-env-profiling>`, and core dumps are under
:ref:`Debugging <nrt-env-debugging>`.

.. list-table::
   :header-rows: 1
   :widths: 30 22 16 62

   * - Variable
     - Type / default
     - Subsystem
     - Description
   * - ``NEURON_RT_GPSIMD_STDOUT_QUEUE_SIZE_BYTES``
     - Byte count; default: ``0``
     - Model loading & firmware
     - Size the per-Pool-engine stdout queue, enabling ``printf``-style output from GPSIMD kernels. ``0`` disables stdout entirely, so this must be set to see any kernel output.
   * - ``NEURON_RT_NUMERICAL_ERRORS_VERBOSITY``
     - String enum; default: ``none``
     - Numerics
     - Choose how loudly NaN/Inf and other numerical error notifications are reported: ``none``, ``critical``, ``debug``, ``verbose``. Auto-raised to ``critical`` when ``NEURON_FAIL_ON_NAN=1``.
   * - ``NEURON_RT_DBG_VERBOSE_DEV_MEM``
     - Boolean; default: ``false``
     - Logging & diagnostics
     - Break the device-memory usage log into every allocation category instead of just aggregates — the tool for "who ate my HBM". When unset, NRT uses ``false``.
   * - ``NEURON_RT_DEBUG_MEMLOG_MAX_SIZE``
     - Byte count; default: ``1 MiB``
     - Logging & diagnostics
     - Size the in-memory ring log retained for post-mortem dumps. When unset, NRT uses ``1 MiB``.
   * - ``NEURON_RT_ENABLE_DEVICE_MEMORY_OOM_LOG``
     - Boolean; default: ``true``
     - Logging & diagnostics
     - Emit a verbose device-memory report when an allocation OOMs. When unset, NRT uses ``true``.
   * - ``NEURON_RT_ENABLE_MEMORY_METRICS``
     - Boolean; default: ``true``
     - Logging & diagnostics
     - Emit a metric on every device-memory alloc and free. When unset, NRT uses ``true``.
   * - ``NEURON_RT_LOG_LEVEL``
     - String enum; default: ``WARN``; ``HAL`` is ``ERROR``
     - Logging & diagnostics
     - Set the global log verbosity (``ERROR``…``TRACE``) for every runtime submodule at once. When unset, NRT applies no override and the compiled-in defaults stand — ``WARN`` everywhere except ``HAL``, which is ``ERROR``. Setting this alone disables syslog; see the log level/location pairing note above.
   * - ``NEURON_RT_LOG_LOCATION``
     - String enum; default: console *and* syslog
     - Logging & diagnostics
     - Send logs to ``console`` or ``syslog`` globally. Selecting one **disables the other**; when unset NRT applies no override and both sinks are active. See the log level/location pairing note above.
   * - ``NEURON_RT_LOG_LEVEL_<submodule>``
     - Log-level enum; default: inherits global
     - Logging
     - Set verbosity for one submodule without changing the others. Valid submodules: ``ENC``, ``ENCD``, ``HAL``, ``NDBG``, ``NEFF``, ``NLOG``, ``NMGR``, ``NRT``, ``RUST``, ``TDRV``, ``TEST``, ``UTIL``. ``HAL`` defaults to ``ERROR``, the rest to ``WARN``.
   * - ``NEURON_RT_LOG_LOCATION_<submodule>``
     - Log-destination enum; default: inherits global
     - Logging
     - Send one submodule's logs to ``console`` or ``syslog``. Same submodule list as above; selecting one sink disables the other.
   * - ``NEURON_RT_MIN_MEM_ERROR_INTERVAL``
     - Unsigned integer seconds; default: ``3600``
     - Logging & diagnostics
     - Minimum seconds between repeated NC memory-error reports, to keep a flapping cell from flooding logs. When unset, NRT uses ``3600``.
   * - ``NEURON_RT_OOM_DUMP_DIRECTORY``
     - Filesystem path string; default: ``/tmp``
     - Failure forensics
     - Destination for device-memory OOM dumps. Path capped at 200 chars; over-long paths log an error and fall back to the default.
   * - ``NEURON_RT_DEBUG_STREAM_BUFFER_SIZE``
     - Unsigned integer count; default: ``262144``
     - Profiling & tracing
     - Size the ``ndebug`` debug-event stream ring. **Must be a power of two** — init fails otherwise.

.. note::

   ``NEURON_RT_NUMERICAL_ERRORS_VERBOSITY`` and ``NEURON_FAIL_ON_NAN`` are
   subject to a hardware limitation: once a numerical condition is flagged on an
   engine, the hardware status can stay set, so the same error may be reported
   again for later operations that did not actually produce it. Treat the
   reports at any verbosity level as a signal that a numerical condition
   occurred somewhere in execution, not as exact per-operation attribution, and
   expect possible false positives after the first genuine occurrence.
   Restarting the application clears the sticky state.

.. _nrt-env-profiling:

Profiling
---------

These variables control the runtime's own capture of device profiles and system
traces, and the size of the on-device buffers that hold the captured records.
``NEURON_RT_INSPECT_ENABLE`` is the master switch; the other
``NEURON_RT_INSPECT_*`` variables select what is captured and where it is
written, and the ``NEURON_RT_PROFILE_BUF_*`` variables size each category of
notification buffer. Raise a buffer only when a capture reports truncated
notifications for that category — the memory comes out of the device.

.. list-table::
   :header-rows: 1
   :widths: 30 24 66

   * - Variable
     - Type / default
     - Description
   * - ``NEURON_RT_INSPECT_ENABLE``
     - Boolean; default: ``false``
     - Master switch for profile capture — captures one profile per unique NEFF execution. The single variable that turns profiling on.
   * - ``NEURON_RT_INSPECT_CPU_UTIL``
     - Boolean; default: ``false``
     - Add host CPU-utilisation sampling to the profile. When unset, NRT uses ``false``.
   * - ``NEURON_RT_INSPECT_DEVICE_PROFILE``
     - String enum; default: disabled
     - Enable on-device hardware profile capture: ``model`` (or ``1``) profiles per model execution, ``session`` profiles the whole session, ``0`` disables. When unset, NRT uses disabled.
   * - ``NEURON_RT_INSPECT_DEVICE_PROFILE_NUM_WARMUP``
     - Unsigned integer; default: ``0``
     - Discard the first *N* executions before profiling, so cold-start artifacts stay out of the data. Only valid in ``model`` mode; warns and is ignored otherwise.
   * - ``NEURON_RT_INSPECT_EVENT_FILTER_NC``
     - Integer range list; default: all visible
     - Restrict system-trace capture to specific logical cores, cutting profile size on a 64-core box. Errors if you name a core that is not visible.
   * - ``NEURON_RT_INSPECT_EVENT_FILTER_TYPE``
     - Comma-separated token list; default: all events
     - Filters which event types are traced. It accepts categories (``hardware``, ``software``), explicit event names, and ``^name`` to subtract, such as ``hardware,^cc_running``. Invalid names fail with the full valid list.
   * - ``NEURON_RT_INSPECT_HOST_MEMORY``
     - Boolean; default: ``false``
     - Add host memory-usage sampling to the profile. When unset, NRT uses ``false``.
   * - ``NEURON_RT_INSPECT_ON_FAIL``
     - Boolean; default: ``false``
     - Capture a device profile *only* when an execution fails — zero steady-state overhead, evidence on the failing run. Implies ``INSPECT_ENABLE`` and forces device-profile-only, model mode.
   * - ``NEURON_RT_INSPECT_OUTPUT_DIR``
     - Filesystem path string; default: cwd
     - Destination directory for captured profile artifacts. When unset, NRT uses cwd.
   * - ``NEURON_RT_INSPECT_PRECACHE_ENABLE``
     - Boolean; default: ``false``
     - Pre-cache profile metadata on warmup executions so the profiled iterations carry less overhead — improves profile fidelity. When unset, NRT uses ``false``.
   * - ``NEURON_RT_INSPECT_SYSTEM_PROFILE``
     - Boolean; default: ``true``
     - Enable/disable host-side system profile (API and runtime event timeline). When unset, NRT uses ``true``.
   * - ``NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC``
     - Unsigned integer count; default: ``0`` (unbounded)
     - Bound system-trace events retained per core. When unset, NRT uses ``0`` (unbounded).
   * - ``NEURON_RT_PROFILE_BUF_CC_CORE_EVENT_MB``
     - Unsigned integer MiB; default: ``2``
     - Sets the collectives-core event notification buffer used by device profiling. Increase it when collectives event records are truncated.
   * - ``NEURON_RT_PROFILE_BUF_CC_CORE_INSTRUCTION_MB``
     - Unsigned integer MiB; default: ``64``
     - Sets the collectives-core instruction notification buffer used by device profiling. Increase it when collectives instruction records are truncated.
   * - ``NEURON_RT_PROFILE_BUF_DMA_MB``
     - Unsigned integer MiB per DMA engine; default: ``128``
     - Sets the device-profile DMA notification buffer for each DMA engine, so its total HBM cost scales with the number of engines. Increase it when DMA records are truncated and reduce it when profile memory use is too high.
   * - ``NEURON_RT_PROFILE_BUF_EVENT_MB``
     - Unsigned integer MiB; default: ``64``
     - Sets the device-profile notification buffer for event records. Increase it when event notifications are truncated, while accounting for the additional device-memory use.
   * - ``NEURON_RT_PROFILE_BUF_H2D_DMA_MB``
     - Unsigned integer MiB; default: ``64``
     - Sets the notification buffer for top-level host-to-device and device-to-host DMA profiling; ``0`` disables that capture. NRT also disables this capture with a warning when the installed driver lacks H2D DMA notification support.
   * - ``NEURON_RT_PROFILE_BUF_INSTRUCTION_MB``
     - Unsigned integer MiB; default: ``128``
     - Sets the device-profile notification buffer for instruction trace records. A value of ``0`` creates a zero-sized buffer and therefore prevents useful instruction capture.
   * - ``NEURON_RT_PROFILE_BUF_THROTTLE_MB``
     - Unsigned integer MiB; default: ``64``
     - Sets the device-profile notification buffer for throttle records. Increase it when throttle notifications are truncated, while accounting for the additional device-memory use.
   * - ``NEURON_RT_ENABLE_DGE_NOTIFICATIONS``
     - Boolean; default: ``false``
     - Enable notifications from dynamic DMA (DGE) rings. When unset, NRT uses ``false``.

.. note::

   These are the complete set of runtime profiling controls. For the end-to-end
   profiling workflow that uses them, see the
   :ref:`Neuron Profiler documentation <new-neuron-profiler-setup>`.

.. _nrt-env-debugging:

Debugging
---------

These variables capture extra evidence — input tensors, a device core dump, or a
known fill pattern in device memory — to reproduce and diagnose a failure. They
add overhead or change behavior, so enable them only for a targeted
investigation and remove them afterward.

.. list-table::
   :header-rows: 1
   :widths: 30 22 16 62

   * - Variable
     - Type / default
     - Subsystem
     - Description
   * - ``NEURON_RT_DBG_DUMP_INPUTS_ON_ERR``
     - Signed integer; default: ``0``
     - Failure forensics
     - Persist all input tensors when execution fails, so a failure is reproducible offline. ``0`` off, ``-1`` always, or a specific ``NRT_STATUS`` to dump on only that failure class.
   * - ``NEURON_RT_DBG_INPUT_DUMP_DIRECTORY``
     - Filesystem path string; default: ``/tmp/neuron-input-dump``
     - Failure forensics
     - Destination for the above input dumps. When unset, NRT uses ``/tmp/neuron-input-dump``.
   * - ``NEURON_RT_LOCAL_CORE_DUMP_DIRECTORY``
     - Filesystem path string; default: ``/tmp/neuron-core-dump/dt-%d-cid-%c``
     - Failure forensics
     - Local destination for runtime core dumps. Supports ``%d`` (device) and ``%c`` (collectives cluster id) substitution so concurrent ranks don't collide. Set to ``""`` **together with** ``S3_CORE_DUMP_PREFIX`` to disable dumps entirely.
   * - ``NEURON_RT_S3_CORE_DUMP_PREFIX``
     - S3 URI string; default: unset
     - Failure forensics
     - S3 prefix handed to ``neuron-dump`` for upload (the runtime does not upload itself). Same ``%d``/``%c`` substitution.
   * - ``NEURON_RT_HBM_INIT_VAL``
     - Enum or 32-bit unsigned integer; default: ``0``
     - Memory & scratchpad
     - Scrub HBM at init with a chosen pattern so stale data can't masquerade as real output. Accepts ``off``, ``fp32_nan``, ``bf16_nan``, or a uint32. Note the default ``0`` still scrubs (with zeros); only ``off`` disables.

.. _nrt-env-deprecated:

Deprecated
----------

These names are obsolete. Remove them from your environment, launch scripts, and
configuration files, and use the replacement instead. The **Status** column says
what the runtime does with each one today: a name that is *still applied* keeps
working for now but may stop being honored in a future release, while a name that
is *parsed but ignored* already has no effect on how the runtime runs.

.. list-table::
   :header-rows: 1
   :widths: 34 20 16 60

   * - Variable
     - Status
     - Subsystem
     - Description
   * - ``NEURON_RT_NUM_CORES``
     - Deprecated; still applied
     - Core allocation & topology
     - Requests *N* cores without specifying which, leaving the choice to the runtime. Not topology-aware, so it cannot express a placement that works for collectives. Still parsed and still applied: use ``NEURON_RT_VISIBLE_CORES`` to pin an explicit set of cores, or ``NEURON_LOGICAL_NC_CONFIG`` to choose a logical core configuration.
   * - ``NEURON_RT_VIRTUAL_CORE_SIZE``
     - Deprecated; still applied
     - Core allocation & topology
     - Older spelling of ``NEURON_LOGICAL_NC_CONFIG``, consulted only when the new name is unset. When unset, NRT does not apply this override. Use ``NEURON_LOGICAL_NC_CONFIG`` instead.
   * - ``NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS``
     - Parsed but ignored
     - DMA & execution
     - This legacy variable used to enable implicit asynchronous execution, a mode that has been removed. Positive values produce a migration warning and are ignored; use the explicit NRT async APIs.

Related reference
-----------------

- :ref:`nrt-configuration` — the Neuron Runtime configuration guide, with worked examples for core allocation, logging, and timeouts.
- :ref:`nrt-troubleshooting` — diagnose and resolve runtime failures on Inf1, Inf2, and Trn1.
- :doc:`Neuron Runtime core dumps </neuron-runtime/explore/core-dump-deep-dive>` — how the core-dump directory variables are used.

Learn more
----------

- :ref:`neuron-runtime-guides` — all Neuron Runtime how-to guides.
- :ref:`nrt-migrate-to-explicit-async` — move off the deprecated implicit async variable and onto the ``nrta_*`` APIs.
- :ref:`Neuron Profiler documentation <new-neuron-profiler-setup>` — the full set of profiling controls.
