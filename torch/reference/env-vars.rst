.. meta::
   :description: Reference for environment variables that configure torch-neuronx behavior, including the neuron backend for torch.compile, caching, eager execution, distributed collectives, and the memory allocator.
   :keywords: TorchNeuron, torch-neuronx, environment variables, configuration, NEFF cache, torch.compile, AWS Neuron, Trainium
   :date-modified: 2026-10-14

.. _torchneuron-env-vars:

====================================
TorchNeuron environment variables
====================================

This reference lists the environment variables available to configure
``torch-neuronx`` behavior. Use this page to look up variables that control
the ``neuron`` backend for ``torch.compile``, compilation caching, eager
execution, distributed collectives, metrics, profiling, and the Neuron memory
allocator.

Applies to
----------

- ``torch-neuronx``
- Trn2 and Trn3 Trainium instances

``neuron`` backend for ``torch.compile``
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 8 17 45

   * - Variable
     - Type
     - Default
     - Description
   * - ``TORCH_NEURONX_DEBUG_DIR``
     - str
     - ``/tmp/neuron_backend``
     - Directory where compilation artifacts (FX graphs, StableHLO IR, NEFF files) are saved.
   * - ``TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS``
     - bool
     - ``False``
     - When ``1``, keeps compilation artifacts after execution. Useful for debugging via IR inspection.
   * - ``TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION``
     - bool
     - ``True``
     - When ``1`` (the default), a ``torch.compile(backend="neuron")`` graph that fails to compile raises the error instead of falling back to eager execution of the original graph. Set to ``0`` to fall back to eager execution on a compile failure. Affects the ``torch.compile`` path only; the ``fallback_execution`` compile option overrides it.
   * - ``TORCH_NEURONX_COMPLEX_DECOMPOSE``
     - bool
     - PyTorch >= 2.13: ``True``; otherwise ``False``
     - Enables framework decomposition of complex operations before lowering. When disabled, verified native complex operations use compiler lowering and other operations follow the existing fallback policy.
   * - ``TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY``
     - bool
     - ``False``
     - When ``1``, raises an error if CPU tensors are passed as inputs instead of auto-copying to Neuron.
   * - ``TORCH_NEURONX_COMPILE_ONLY``
     - bool
     - ``False``
     - When ``1``, compile to NEFF without device execution. Ops compile but never run; ``nrt_init`` failure is non-fatal. Also set by ``torch_neuronx.compiler.compile_only()``.
   * - ``NEURON_PLATFORM_TARGET_OVERRIDE``
     - str
     - (query live instance)
     - Target platform (for example ``trn2``, ``trn3``) for cross-compilation on a device-less host. Also set via ``CompileOptions(target=...)``.

Artifact directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both ``torch.compile`` and eager execution use the same unified job-based
artifact hierarchy. The working directory used by all ``neuronx-cc`` calls is
preserved under ``TORCH_NEURONX_DEBUG_DIR``.

.. code-block:: text

   {TORCH_NEURONX_DEBUG_DIR}/torch_neuronx/rank{rank}/{job_id}/
   ├── fx_graphs/{model}_{ts}.fx.txt
   ├── transformed_fx/{model}_{ts}.transformed.fx.txt
   ├── raw/{model}_{ts}.torch_raw.mlir
   ├── stablehlo/{model}_{ts}.stablehlo.mlir
   ├── torch_mlir_error/{model}_{ts}.mlir  # Populated when torch-mlir fails
   ├── inputs/   # Only when TORCH_NEURONX_DYNAMO_TENSOR_DUMPS is set
   └── outputs/  # Only when TORCH_NEURONX_DYNAMO_TENSOR_DUMPS is set

   {TORCH_NEURONX_DEBUG_DIR}/neuronx-cc/{hash}/
   ├── log-neuron-cc.txt
   ├── module.mlir
   ├── module.neff
   ├── metadata.txt
   └── compiler_args.txt

When ``TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1``, all directories and
their contents are preserved for debugging and analysis.

Compiler
--------

See :ref:`Neuron Compiler CLI reference <neuron-compiler-cli-reference-guide>`
for available compiler flags.

Persistent NEFF / HLO cache
---------------------------

Compiled NEFFs and lowered HLO entries are cached on disk to avoid
recompilation across runs. The cache is content-addressed; entries from one
run can be reused by another as long as the inputs (HLO bytes, compiler
config, compiler version) are unchanged.

.. list-table::
   :header-rows: 1
   :widths: 30 8 17 45

   * - Variable
     - Type
     - Default
     - Description
   * - ``TORCH_NEURONX_NEFF_CACHE_DIR``
     - path
     - ``/tmp/neff_cache``
     - Directory for persistent NEFF cache. Point at a shared mount (for example, FSx) to reuse compiled NEFFs across runs and instances.
   * - ``TORCH_NEURONX_HLO_CACHE_DIR``
     - path
     - ``/tmp/hlo_cache``
     - Directory for persistent HLO cache, shared by the eager and ``torch.compile`` entry points.
   * - ``TORCH_NEURONX_HLO_CACHE``
     - bool
     - ``True``
     - When ``1``, eager ops cache their HLO lowering to disk, keyed by the dispatch key.
   * - ``TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR``
     - path
     - ``/tmp/local_cache``
     - Local directory for filelocks and scratch. **Must stay on a local POSIX filesystem** (NFS/Lustre lock semantics are unreliable).
   * - ``TORCH_NEURONX_NEFF_DISABLE_CACHE``
     - bool
     - ``False``
     - When ``1``, disables the persistent NEFF cache entirely. Every compile runs fresh.
   * - ``TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT``
     - int
     - ``1200``
     - Seconds to wait for the per-key compile lock before giving up.
   * - ``TORCH_NEURONX_NEFF_KEY_FROM_HLO``
     - bool
     - ``True``
     - When ``1`` (the default), the persistent NEFF key is derived from the lowered HLO bytes plus compiler config (instead of the in-memory base_key). Lets benign Python edits re-trace and re-lower without busting the on-disk cache. Set to ``0`` to use the in-memory base_key.
   * - ``NKI_ENABLE_TRACE_CACHE``
     - bool
     - ``True``
     - When ``1``, the NKI kernel compile cache is persisted across processes at ``/var/tmp/nki-trace-cache-uid<uid>/`` (or ``NKI_TRACE_CACHE_URL`` if set). Only files under a ``site-packages``/``dist-packages`` tree that map to an installed wheel get captured in the caching process. An editable install, a vendored copy, or any tree with no ``.dist-info`` does not resolve to a version and will not be tracked, potentially resulting in a stale cache hit. Additionally, the kernel is assumed to be a pure function of its declared arguments and its source. Any value that steers control flow should be passed as an argument to the kernel; otherwise a change to that value might not be detected and could also result in a stale cache hit.
   * - ``TORCH_CACHING_PRECOMPILE``
     - bool
     - ``False``
     - When ``1``, enables the PyTorch Dynamo precompile cache, which caches the backend result of the Dynamo trace of any ``torch.compile``-ed modules or functions. Saved at ``TORCHINDUCTOR_CACHE_DIR``, split by subdirectories per rank if applicable. For the ``neuron`` backend the HLO result is cached, separate from the HLO cache. Delete ``TORCHINDUCTOR_CACHE_DIR`` to clear this cache. Off by default.
   * - ``TORCHINDUCTOR_CACHE_DIR``
     - path
     - ``/tmp/torchinductor_<username>/``
     - Directory for the torch inductor cache. This is a PyTorch variable and affects the location of other caches as well. Refer to the PyTorch documentation for details on other caches.

General (eager execution)
-------------------------

These variables affect eager execution (non-compiled operations).

.. list-table::
   :header-rows: 1
   :widths: 30 8 17 45

   * - Variable
     - Type
     - Default
     - Description
   * - ``NEURON_LAUNCH_BLOCKING``
     - bool
     - ``False``
     - When ``1``, forces synchronous execution. Useful for debugging.
   * - ``TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS``
     - bool
     - ``False``
     - When ``1``, only falls back to CPU for unimplemented ops, not execution errors.
   * - ``TORCH_NEURONX_ENABLE_STACK_TRACE``
     - bool
     - ``False``
     - When ``1``, captures Python stack traces per operation. Adds overhead.
   * - ``TORCH_NEURONX_LOG_LEVEL``
     - int
     - ``1``
     - Log verbosity: ``0`` = ERROR, ``1`` = WARNING (the default), ``2`` = INFO, ``3`` = DEBUG.
   * - ``TORCH_NEURONX_USE_PINNED_MALLOC``
     - bool
     - ``False``
     - (Experimental) When ``1``, uses driver-pinned host memory.

Neuron Runtime
--------------

- See :ref:`Neuron Runtime configuration <nrt-configuration>` for all runtime variables.
- See :doc:`Logical NeuronCore Configuration </about-neuron/arch/neuron-features/logical-neuroncore-config>` for LNC details.

Metrics
-------

.. list-table::
   :header-rows: 1
   :widths: 30 8 17 45

   * - Variable
     - Type
     - Default
     - Description
   * - ``TORCH_NEURONX_ENABLED_METRIC_TABLES``
     - str
     - ``""``
     - Comma-separated metric tables to enable for CSV export.
   * - ``TORCH_NEURONX_METRICS_DIR``
     - str
     - ``/tmp``
     - Directory for metric CSV files.

Profiling
---------

These variables set the options of the default Neuron profiling session. A
``NeuronConfig`` overrides them. CPU-only profiling does not start a session,
and a run without the profiler is not affected.

``torch-neuronx`` starts the default session when ``torch.profiler.profile``
gets ``ProfilerActivity.PrivateUse1`` and no ``NeuronConfig`` in
``experimental_config``.

.. list-table::
   :header-rows: 1
   :widths: 30 8 17 45

   * - Variable
     - Type
     - Default
     - Description
   * - ``TORCH_NEURONX_PROFILE_DEFAULT_SESSION``
     - bool
     - ``1``
     - Set to ``0`` to disable the default session. Then ``ProfilerActivity.PrivateUse1`` without a ``NeuronConfig`` records no Neuron data.
   * - ``TORCH_NEURONX_PROFILE_MODES``
     - str
     - ``runtime,cpu_util,host_memory``
     - Comma-separated profile modes. Add ``device`` for the device profile. The device profile reserves device memory for hardware notifications.
   * - ``TORCH_NEURONX_PROFILE_RANKS``
     - str
     - ``0``
     - Comma-separated global ranks that capture a profile. The rank comes from the ``RANK`` environment variable.
   * - ``TORCH_NEURONX_PROFILE_MAX_EVENTS_PER_NC``
     - int
     - ``10000000``
     - Maximum number of system trace events for each NeuronCore.
   * - ``TORCH_NEURONX_PROFILE_CAPTURE_NC``
     - str
     - ``0,1``
     - NeuronCore indices or ranges to capture, for example ``0,1,2-5``.
   * - ``TORCH_NEURONX_PROFILE_OUTPUT_DIR``
     - path
     - ``neuron_profile``
     - Directory for the profile session output.

With the ``device`` mode, the session copies the NEFFs from
``TORCH_NEURONX_NEFF_CACHE_DIR`` (default ``/tmp/neff_cache``) to the ``neffs``
directory in ``TORCH_NEURONX_PROFILE_OUTPUT_DIR``. The device profile needs
these NEFFs.

Distributed
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 8 17 45

   * - Variable
     - Type
     - Default
     - Description
   * - ``TORCH_NEURONX_ENABLE_STREAMS``
     - bool
     - ``True``
     - When ``1`` (the default), Neuron streams are enabled and work can be submitted to the default stream or non-default streams. By default, compiled operations containing collectives are routed to the default stream while preserving execution ordering. When ``0``, all streams alias the default stream. Only ``0`` and ``1`` are accepted.
   * - ``TORCH_NEURONX_ENABLE_HOST_CC``
     - bool
     - ``False``
     - When ``1``, enables collectives to execute on non-default streams, allowing communication to overlap with computation. Requires TORCH_NEURONX_ENABLE_STREAMS=1 (the default).
   * - ``NEURON_USE_SPLIT_ALLGATHER``
     - bool
     - on (default stream)
     - Enables split-based allgather for better memory efficiency. When unset, the default depends on the stream; when set, only ``1`` enables it and any other value disables it.
   * - ``NEURON_USE_SPLIT_REDUCE_SCATTER``
     - bool
     - on (default stream)
     - Enables split-based reduce_scatter. When unset, the default depends on the stream; when set, only ``1`` enables it and any other value disables it.
   * - ``NEURON_USE_NKI_ALLTOALLV``
     - bool
     - ``True``
     - Enables native NKI ``all_to_all_v``. Set to ``0`` to use the compatibility padding fallback. Native mode requires ``aws-neuronx-runtime-lib >= 2.x.62579`` and ``aws-neuronx-collectives >= 2.x.63596``.
   * - ``COLLECTIVE_BUCKETSIZE_IN_MB``
     - int
     - ``512``
     - Maximum bucket size (MB) for collective operations.

Neuron memory allocator configuration
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 8 17 45

   * - Variable
     - Type
     - Default
     - Description
   * - ``TORCH_NEURONX_ENABLE_SCALAR_CACHE``
     - bool
     - ``True``
     - When ``1`` (the default), caches device-side scalar constants to avoid repeated host-to-device copies.

The Neuron caching allocator's segment-pool tuning knobs are exposed through a
single environment variable:

.. code-block:: text

   TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG=key1=value1,key2=value2,...

The value is a comma-separated list of ``key=value`` pairs. The first call to
the allocator parses the variable, validates the result, and snapshots it into
the per-device ``NeuronSegmentPool`` at construction. The snapshot is fixed for
the pool's lifetime. To change tuning, set the variable and restart the
process.

**Format**

- **Sizes**: raw bytes as a non-negative decimal integer. No size suffixes;
  ``1048576`` is valid, ``1MB`` is rejected with a warning.
- **Plain integers** (only ``kRoundupPower2Divisions`` today): same syntax,
  bare digits.
- Unknown keys are ignored with a warning.
- Invalid values are ignored with a warning; the field keeps its default.

Default values are tuned from training-workload experiments. Override them if
you run into out-of-memory (OOM) errors resulting from fragmentation.

.. warning::

   Changing these values can have drastic effects on memory allocation in
   training workloads and can potentially induce OOMs.

.. list-table::
   :header-rows: 1
   :widths: 27 8 20 45

   * - Key
     - Type
     - Default (bytes)
     - Definition
   * - ``kPoolBoundary``
     - bytes
     - ``1048576`` (1 MB)
     - Allocations below this go to the small pool, at or above to the large pool. The two pools have separate free sets so small allocations cannot fragment large-pool segments and vice versa.
   * - ``kSmallSegmentSize``
     - bytes
     - ``2097152`` (2 MB)
     - Segment size NRT-allocates for small-pool requests.
   * - ``kMediumSegmentSize``
     - bytes
     - ``67108864`` (64 MB)
     - Segment size NRT-allocates for large-pool requests in ``[kPoolBoundary, kMediumThreshold)``.
   * - ``kMediumThreshold``
     - bytes
     - ``16777216`` (16 MB)
     - Above this, each request gets its own per-request segment sized to the request rounded up to ``kSegmentAlignment``.
   * - ``kSegmentAlignment``
     - bytes
     - ``2097152`` (2 MB)
     - Alignment for per-request segments above ``kMediumThreshold``. Must be a power of 2.
   * - ``kMinRoundedSize``
     - bytes
     - ``256``
     - Floor for ``RoundSize``. Requests smaller than this round up to it.
   * - ``kRoundupPower2Divisions``
     - int
     - ``4``
     - ``N`` in the power-of-two-with-N-divisions block rounding scheme — the number of equal-sized divisions between the lower and higher power-of-2 value of the requested size. ``N=1`` is pure pow2; ``N=4`` gives 25% max waste; higher ``N`` reduces waste at the cost of more distinct buckets, lower cache hit rate, and greater overhead of splitting/merging free blocks. Valid range: ``[1, 64]``.
   * - ``kSmallSplitThreshold``
     - bytes
     - ``kMinRoundedSize``
     - Small-pool minimum split remainder. If carving a chosen free block would leave a remainder smaller than this, the block is handed out whole instead.
   * - ``kLargeSplitThreshold``
     - bytes
     - ``kPoolBoundary``
     - Large-pool minimum split remainder. Same idea as above for the large pool.
   * - ``kMaxSplitSizeBytes``
     - bytes
     - ``SIZE_MAX``
     - Cap above which a free block cannot be split. Default unlimited (all blocks splittable). Setting a finite cap keeps large blocks intact for future large requests, at the risk of cache misses for small requests when only above-cap blocks are free. Pairs with ``kMaxNonSplitRoundingBytes``.
   * - ``kMaxNonSplitRoundingBytes``
     - bytes
     - ``20971520`` (20 MB)
     - When a free block is non-splittable (``block.size > kMaxSplitSizeBytes``), allow it to serve a request whose size is within this tolerance of the block's size. Increase to widen reuse at the cost of more internal fragmentation.

**Examples**

Cap large-block splitting at 512 MB; allow up to 64 MB rounding waste on
non-split reuse:

.. code-block:: bash

   TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG="kMaxSplitSizeBytes=536870912,kMaxNonSplitRoundingBytes=67108864"

Reduce internal fragmentation in the medium-size range by widening rounding
divisions to 8:

.. code-block:: bash

   TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG="kRoundupPower2Divisions=8"

Related reference
-----------------

- :doc:`/torch/reference/guidelines-memory-management` — Guidelines for memory management on Neuron.
- :doc:`/torch/compile/reference-torch-compile` — ``torch.compile`` configuration reference.
