.. meta::
   :description: torch.compile backend reference for TorchNeuron on AWS Neuron. Parameter and torch.compiler API support, environment variables, the neuron_dynamo_backend helper API, and caching configuration for backend="neuron".
   :keywords: TorchNeuron, torch.compile, torch.compiler, API support, environment variables, neuron_dynamo_backend, get_dynamo_metrics, NEFF cache, reference, AWS Neuron, Trainium
   :date-modified: 2026-10-14

.. _torch-compile-ref:

===================================
torch.compile backend reference
===================================

This topic provides reference material for the Neuron ``torch.compile`` backend.
Use this page to look up the support status of the ``torch.compile`` parameters
and the ``torch.compiler`` APIs, the backend's environment variables, the public
helper API, and caching configuration.

Applies to
----------

- ``torch.compile(model, backend="neuron")`` with ``torch-neuronx`` (TorchNeuron) on Trn2, Trn3.

.. _torch-compile-api-support:

torch.compile API and parameter support
---------------------------------------

These tables give the support status of each `torch.compile parameter
<https://docs.pytorch.org/docs/2.13/generated/torch.compile.html>`_ and each
`torch.compiler API
<https://docs.pytorch.org/docs/2.13/torch.compiler_api.html>`_ on the Neuron
backend. Status values come from end-to-end runs on a Neuron device.

torch.compile parameters
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Status
     - Notes
   * - ``model``
     - Supported
     - Module or callable.
   * - ``fullgraph``
     - Supported
     - ``True`` and ``False``. ``True`` raises an error if graph breaks occur, and is recommended for the Neuron path.
   * - ``dynamic``
     - Unsupported
     - ``dynamic=True`` is rejected during compilation. Shapes that vary between steps run, recompiling per new shape; native dynamic shapes (a single artifact across a range of shapes) is not yet available.
   * - ``backend``
     - Supported
     - ``'neuron'``. Explicitly set, or auto-selected when running on a Neuron device.
   * - ``mode``
     - Unsupported (default only)
     - Only ``'default'`` applies. Other modes log a warning and fall back to default, so ``reduce-overhead``, ``max-autotune``, and ``max-autotune-no-cudagraphs`` have no effect.
   * - ``options``
     - Partial
     - Accepts a ``dict`` or ``None``, but only a few keys are read internally. There is no documented Neuron options contract yet.
   * - ``name``
     - Unsupported
     - Accepted without an error, but the name is not propagated to compilation artifacts.
   * - ``disable``
     - Supported
     - ``True`` disables compilation.
   * - ``recompile_limit``
     - Supported
     - With ``fullgraph=False``, reaching the limit falls back to eager. With ``fullgraph=True``, it raises ``FailOnRecompileLimitHit``.
   * - ``isolate_recompiles``
     - Supported
     - Compiles and runs. Independent-tracking behavior is not stress-tested.
   * - ``shapes_spec``
     - Unsupported
     - A static shape spec runs without an error, but a dynamic one fails for the same reason as ``dynamic=True``.

torch.compiler APIs
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Status
   * - ``compile``
     - Supported
   * - ``reset``
     - Supported
   * - ``allow_in_graph``
     - Supported
   * - ``substitute_in_graph``
     - Supported
   * - ``assume_constant_result``
     - Supported
   * - ``list_backends``
     - Supported
   * - ``disable``
     - Supported
   * - ``set_default_backend``
     - Supported
   * - ``get_default_backend``
     - Supported
   * - ``set_stance``
     - Supported
   * - ``set_enable_guard_collectives``
     - Supported
   * - ``cudagraph_mark_step_begin``
     - Unsupported
   * - ``is_compiling``
     - Supported
   * - ``is_dynamo_compiling``
     - Supported
   * - ``is_exporting``
     - Supported
   * - ``keep_portable_guards_unsafe``
     - Supported
   * - ``skip_guard_on_inbuilt_nn_modules_unsafe``
     - Supported
   * - ``skip_guard_on_all_nn_modules_unsafe``
     - Supported
   * - ``keep_tensor_guards_unsafe``
     - Supported
   * - ``skip_guard_on_globals_unsafe``
     - Supported
   * - ``skip_all_guards_unsafe``
     - Supported
   * - ``nested_compile_region``
     - Unsupported
   * - ``load_cache_artifacts``
     - Supported
   * - ``load_compiled_function``
     - Supported
   * - ``save_cache_artifacts``
     - Supported
   * - ``wrap_numpy``
     - Supported

.. _torchneuron-graph-breaks:

Graph breaks
^^^^^^^^^^^^

A graph break splits a function into graphs that compile separately. When TorchDynamo
traces an operation that the Neuron backend cannot compile, the backend forces a graph
break at that call. The operation runs in eager mode, and the rest of the function still
compiles. This applies to the function form (for example, ``torch.sort(x)``) and to the
tensor method form (``x.sort()``).

A graph break occurs only with ``fullgraph=False``, which is the default. With
``fullgraph=True``, the same operation raises ``torch._dynamo.exc.Unsupported``. The
diagnostic has the type ``Op not supported by the Neuron backend`` and gives the
operation name and the reason.

The Neuron backend forces a graph break for these operations:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Operation
     - When it breaks
   * - ``torch.sort``, ``torch.argsort``
     - When the sorted dimension has more than 16,384 elements.
   * - ``torch.topk``
     - When ``k`` is 16,384 or larger.
   * - ``torch.nn.functional.grouped_mm``, ``torch._grouped_mm``
     - Always.
   * - Operations with no Neuron lowering
     - Always. See the paragraph that follows this table.

There is no fixed list of operations with no Neuron lowering. When you import
``torch_neuronx``, the backend builds this list from the installed versions of PyTorch,
torch-mlir, and the Neuron decompositions. The list can change when you upgrade
``torch-neuronx`` or PyTorch. To find the graph breaks in your model, compile with
``fullgraph=True`` during development, so that each break raises an error.

Backend behavior environment variables
--------------------------------------

These variables control compilation behavior, artifact output, and fallback for
``backend="neuron"``. Boolean variables accept ``1``/``true``/``yes`` for true.

.. list-table::
   :header-rows: 1
   :widths: 40 8 16 36

   * - Variable
     - Type
     - Default
     - Effect
   * - ``TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION``
     - bool
     - ``True``
     - When true (the default), disables fallback to eager execution on a
       compilation or execution failure, so the backend re-raises the error
       instead of silently running the original graph. Set to ``0`` to re-enable
       fallback.
   * - ``TORCH_NEURONX_DEBUG_DIR``
     - str
     - ``/tmp/neuron_backend_<uuid>``
     - Directory where compilation artifacts (FX graphs, StableHLO IR, NEFF files)
       are written.
   * - ``TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS``
     - bool
     - ``False``
     - When true, keeps the compilation-artifacts directory after execution
       completes. Useful for inspecting the IR.
   * - ``TORCH_NEURONX_COMPILE_ONLY``
     - bool
     - ``False``
     - Build the NEFF without executing on device; a runtime-init failure is
       non-fatal. Also set by ``torch_neuronx.compiler.compile_only()``.
   * - ``TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY``
     - bool
     - ``False``
     - When true, raises an error if CPU tensors are passed as inputs instead of
       auto-copying them to the Neuron device.
   * - ``TORCH_NEURONX_RETAIN_DEVICE_MODE``
     - bool
     - ``False``
     - When true, outputs retain their original device placement instead of the
       Neuron device.
   * - ``NEURON_PLATFORM_TARGET_OVERRIDE``
     - str
     - (query live instance)
     - Target platform (for example, ``trn2``, ``trn3``) for cross-compilation on a
       device-less host. Also set through ``CompileOptions(target=...)``.
   * - ``NEURON_CC_FLAGS``
     - str
     - ``""``
     - Extra flags passed to the ``neuronx-cc`` invocation.
   * - ``TORCH_NEURONX_ENABLE_NKI_SDPA``
     - bool
     - ``True``
     - Decompose ``scaled_dot_product_attention`` to the NKI flash-attention kernel
       when possible (currently when the query sequence length is a multiple of
       2048).

Caching environment variables
-----------------------------

Compiled NEFFs and lowered HLO are cached on disk so that repeated compilations
across runs and instances avoid recompilation. The NEFF cache is content-addressed;
an entry from one run is reused by another as long as the inputs (HLO bytes, compiler
configuration, compiler version) are unchanged.

.. list-table::
   :header-rows: 1
   :widths: 40 8 16 36

   * - Variable
     - Type
     - Default
     - Effect
   * - ``TORCH_NEURONX_NEFF_CACHE_DIR``
     - path
     - ``/tmp/neff_cache``
     - Directory for the persistent NEFF cache. Point at a shared mount (for
       example, Amazon FSx) to reuse compiled NEFFs across runs and instances.
   * - ``TORCH_NEURONX_HLO_CACHE_DIR``
     - path
     - ``/tmp/hlo_cache``
     - Directory for the persistent HLO cache, shared by the eager and
       ``torch.compile`` entry points.
   * - ``TORCH_NEURONX_NEFF_DISABLE_CACHE``
     - bool
     - ``False``
     - When true, disables the persistent NEFF cache entirely; every compile runs
       fresh.
   * - ``TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR``
     - path
     - ``/tmp/local_cache``
     - Local directory for file locks and scratch. Must stay on a local POSIX
       filesystem — NFS and Lustre lock semantics are unreliable.
   * - ``TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT``
     - int
     - ``1200``
     - Seconds to wait for the per-key compile lock before giving up.
   * - ``TORCH_CACHING_PRECOMPILE``
     - bool
     - ``False``
     - When true, enables the PyTorch Dynamo precompile cache (MegaCache). For the
       Neuron backend the StableHLO result is cached, saved under
       ``TORCHINDUCTOR_CACHE_DIR``.

Metrics environment variables
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 8 16 36

   * - Variable
     - Type
     - Default
     - Effect
   * - ``TORCH_NEURONX_ENABLED_METRIC_TABLES``
     - str
     - ``""``
     - Comma-separated metric tables to emit as CSV. The compilation table is
       ``graph_stats``.
   * - ``TORCH_NEURONX_METRICS_DIR``
     - str
     - ``/tmp``
     - Output directory for metric CSV files.

Backend helper API
------------------

All symbols below are importable from ``torch_neuronx.neuron_dynamo_backend``.
(src: ``__init__.py``, ``config.py``, ``metrics.py``, ``exceptions.py``)

.. note::

   Signatures, parameters, and field descriptions are transcribed from the
   ``torch-neuronx`` source and should be confirmed against the release you are
   running.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Symbol
     - Description
   * - :ref:`set_model_name() <torch-compile-ref-set-model-name>`
     - Set the model name used for artifact and graph naming.
   * - :ref:`get_model_name() <torch-compile-ref-get-model-name>`
     - Get the current model name.
   * - :ref:`get_rank() <torch-compile-ref-get-rank>`
     - Current distributed rank, or the Neuron device ID when not distributed.
   * - :ref:`get_dynamo_metrics() <torch-compile-ref-get-dynamo-metrics>`
     - Retrieve per-graph compilation metrics.
   * - :ref:`reset_dynamo_metrics() <torch-compile-ref-reset-dynamo-metrics>`
     - Reset recorded compilation metrics and counters.
   * - :ref:`NEFFCompilationError <torch-compile-ref-neffcompilationerror>`
     - Raised when NEFF compilation fails.
   * - :ref:`NEFFExecutionError <torch-compile-ref-neffexecutionerror>`
     - Raised when NEFF execution fails.

Model naming
^^^^^^^^^^^^

.. _torch-compile-ref-set-model-name:

.. py:function:: set_model_name(name: str) -> None
   :noindex:

   Set the model name used when naming compilation artifacts and NEFF files for
   graphs compiled after this call.

   :param name: Name to use for the model.
   :type name: str
   :rtype: None

.. _torch-compile-ref-get-model-name:

.. py:function:: get_model_name() -> str
   :noindex:

   Return the current model name. If no name has been set, returns the default
   ``"model_default"``.

   :returns: The current model name.
   :rtype: str

Distributed
^^^^^^^^^^^

.. _torch-compile-ref-get-rank:

.. py:function:: get_rank() -> int
   :noindex:

   Return the current process rank. Returns ``torch.distributed.get_rank()`` when
   ``torch.distributed`` is initialized; otherwise returns the current Neuron
   device ID (``torch.neuron.current_device()``).

   :returns: Process rank, or the Neuron device ID when not running under ``torch.distributed``.
   :rtype: int

Compilation metrics
^^^^^^^^^^^^^^^^^^^

.. _torch-compile-ref-get-dynamo-metrics:

.. py:function:: get_dynamo_metrics() -> list[NeuronCompilationMetrics]
   :noindex:

   Return per-graph compilation metrics collected during ``torch.compile``. Calls
   ``torch_neuronx.synchronize()`` first so that asynchronous compile timings are
   populated before the metrics are returned.

   :returns: One :py:class:`NeuronCompilationMetrics` record per compiled graph.
   :rtype: list[NeuronCompilationMetrics]

   Each record contains the following fields:

   .. list-table::
      :header-rows: 1
      :widths: 32 12 56

      * - Field
        - Type
        - Description
      * - ``graph_id``
        - str
        - Unique identifier (cache key) for the compiled graph.
      * - ``graph_name``
        - str
        - Human-readable name, ``{model_name}_{aot_id}_{graph_type}_{nth}``.
      * - ``model_name``
        - str
        - User-provided or auto-generated model name.
      * - ``has_collectives``
        - bool
        - Whether the graph contains collective operations (for example, all-reduce).
      * - ``graph_node_count``
        - int
        - Number of nodes in the FX graph before lowering.
      * - ``timestamp``
        - str
        - Compilation timestamp / segment ID used for ordering.
      * - ``torch_neuronx_fx_passes_us``
        - int
        - Time for FX graph-level passes, in microseconds.
      * - ``torch_neuronx_lower_us``
        - int
        - Time to lower to StableHLO (torch-mlir), in microseconds.
      * - ``torch_neuronx_compile_us``
        - int
        - Time to compile StableHLO to NEFF, in microseconds (populated lazily on read).

.. _torch-compile-ref-reset-dynamo-metrics:

.. py:function:: reset_dynamo_metrics() -> None
   :noindex:

   Clear all recorded compilation metrics and counters. Call this between test
   runs or when starting a new profiling session.

   :rtype: None

.. code-block:: python

   import torch
   from torch_neuronx.neuron_dynamo_backend import get_dynamo_metrics, reset_dynamo_metrics

   reset_dynamo_metrics()
   compiled = torch.compile(model, backend="neuron")
   compiled(*example_inputs)  # first call triggers compilation

   for m in get_dynamo_metrics():
       print(m.graph_name, m.graph_node_count, m.torch_neuronx_compile_us)

Exceptions
^^^^^^^^^^

.. _torch-compile-ref-neffcompilationerror:

.. py:exception:: NEFFCompilationError
   :noindex:

   Raised when NEFF compilation fails. Subclass of :py:class:`Exception`.

.. _torch-compile-ref-neffexecutionerror:

.. py:exception:: NEFFExecutionError
   :noindex:

   Raised when NEFF execution fails. Subclass of :py:class:`Exception`.

Caching configuration
---------------------

The backend caches at three layers: the content-addressed persistent NEFF cache
(``TORCH_NEURONX_NEFF_CACHE_DIR``), the HLO cache (``TORCH_NEURONX_HLO_CACHE_DIR``),
and the optional Dynamo precompile cache (``TORCH_CACHING_PRECOMPILE``). To reuse
compiled NEFFs across runs and across instances, point ``TORCH_NEURONX_NEFF_CACHE_DIR``
at a shared mount such as Amazon FSx, and keep ``TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR``
on a local POSIX filesystem for lock files.

Because the backend forces ``dynamic=False``, each input shape produces a separate
cache entry. Pad or bucket inputs to a fixed set of shapes to maximize reuse. Pin
the ``torch-neuronx`` and compiler versions in production, since the NEFF cache key
includes the compiler version and a version change invalidates prior entries. For
the caching mechanics, see :ref:`torch-compile-deep-dive`.

Related reference
-----------------

- `torch-neuronx operator compatibility table <https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md>`__ - Per-operator support status and CPU fallbacks for ``torch-neuronx``
- :ref:`torch-compile-best-practices` - torch.compile best practices

Learn more
----------

- :ref:`torch-compile-deep-dive` - How torch.compile works on Neuron
- :ref:`torch-compile-optimize-howto` - Optimize a model with ``torch.compile``
