.. meta::
   :description: nrtpy model API — load, execute, and benchmark compiled NEFF models on AWS Neuron NeuronCores from Python.
   :date-modified: 2026-07-30
   :keywords: Neuron, nrtpy, NrtpyModel, BenchmarkResult, NEFF, benchmark, NeuronCore

.. _nrtpy-model-ref:

===================
nrtpy model API
===================

This page documents :py:class:`~nrtpy.NrtpyModel` and
:py:class:`~nrtpy.BenchmarkResult` — the classes you use to load a compiled
NEFF file, run it on a NeuronCore, and benchmark its execution.

NrtpyModel
----------

.. py:class:: nrtpy.NrtpyModel

   A wrapper class for executing compiled kernels from NEFF files.

   .. py:attribute:: input_tensors_info
      :type: dict[str, TensorMetadata]

      Mapping of input tensor name to metadata (shape, size). Use this to
      discover the expected input names and sizes for a loaded NEFF:

      .. code-block:: python

         model = NrtpyModel.load_from_neff("model.neff")
         for name, info in model.input_tensors_info.items():
             print(f"{name}: shape={info.shape}, size={info.size}")

   .. py:attribute:: output_tensors_info
      :type: dict[str, TensorMetadata]

      Mapping of output tensor name to metadata (shape, size).

   .. py:classmethod:: load_from_neff(neff_path, name=None, core_id=0, cc_enabled=False, rank_id=0, world_size=1)

      Load a NEFF file and return an ``NrtpyModel`` instance.

      :param neff_path: Path to the NEFF file to load.
      :type neff_path: pathlib.Path or str
      :param name: Optional name for the model. If ``None``, the NEFF filename
         stem is used.
      :type name: str or None
      :param int core_id: Target NeuronCore for the model (default 0).
      :param bool cc_enabled: Enable collective communication.
      :param int rank_id: Rank of this model within the collective group.
      :param int world_size: Total number of ranks in the collective group.
      :returns: An ``NrtpyModel`` wrapping the loaded model.
      :rtype: nrtpy.NrtpyModel

   .. py:method:: __call__(inputs, outputs=None, save_trace=False, ntff_name=None)

      Execute the model. Invoke by calling the model instance directly:
      ``model(inputs={...})``.

      :param inputs: Mapping of input tensor name to
         :py:class:`~nrtpy.NrtpyTensor`. Keys must match the NEFF input names.
      :type inputs: dict[str, nrtpy.NrtpyTensor]
      :param outputs: Mapping of output tensor name to
         :py:class:`~nrtpy.NrtpyTensor`. Keys must match the NEFF output names.
         If ``None``, output tensors are allocated automatically and returned.
      :type outputs: dict[str, nrtpy.NrtpyTensor] or None
      :param bool save_trace: Whether to save an execution trace (``.ntff``
         file).
      :param ntff_name: Optional path for the trace file. If ``None``, the
         trace is written next to the NEFF file with an ``.ntff`` suffix.
      :type ntff_name: str or None
      :returns: The auto-allocated output tensors when ``outputs`` is ``None``;
         otherwise ``None``.
      :rtype: dict[str, nrtpy.NrtpyTensor] or None

   .. py:method:: benchmark(inputs, outputs=None, warmup_iter=5, benchmark_iter=5, mode="device")

      Benchmark model execution and return timing statistics.

      :param inputs: Mapping of input tensor name to
         :py:class:`~nrtpy.NrtpyTensor`.
      :type inputs: dict[str, nrtpy.NrtpyTensor]
      :param outputs: Mapping of output tensor name to
         :py:class:`~nrtpy.NrtpyTensor`. If ``None``, output tensors are
         allocated automatically.
      :type outputs: dict[str, nrtpy.NrtpyTensor] or None
      :param int warmup_iter: Number of warmup iterations before timing.
      :param int benchmark_iter: Number of timed iterations. Must be at least 1.
      :param str mode: Timing mode:

         - ``"device"`` — NeuronCore execution time via device-side tracing.
           Most accurate for kernel timing.
         - ``"host"`` — Host wall-clock time including host-device overhead.
           No tracing overhead.

      :returns: The benchmark statistics.
      :rtype: nrtpy.BenchmarkResult
      :raises ValueError: If ``mode`` is not ``"device"`` or ``"host"``, or if
         ``benchmark_iter`` is less than 1.
      :raises RuntimeError: In ``"device"`` mode, if no device execution events
         are captured.

BenchmarkResult
---------------

.. py:class:: nrtpy.BenchmarkResult

   Result of a model benchmark run, returned by
   :py:meth:`~nrtpy.NrtpyModel.benchmark`. All timing values are in
   milliseconds.

   .. py:attribute:: mean_ms
      :type: float

      Mean execution time across timed iterations.

   .. py:attribute:: min_ms
      :type: float

      Minimum execution time.

   .. py:attribute:: max_ms
      :type: float

      Maximum execution time.

   .. py:attribute:: std_dev_ms
      :type: float

      Standard deviation of execution time.

   .. py:attribute:: iterations
      :type: int

      Number of timed iterations reflected in the statistics.

   .. py:attribute:: warmup_iterations
      :type: int

      Number of warmup iterations performed before timing.

   .. py:attribute:: durations_ms
      :type: list[float]

      Per-iteration execution times.

Examples
--------

Basic execution
~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from nrtpy import NrtpyModel, NrtpyTensor

   # Load a compiled NEFF model
   model = NrtpyModel.load_from_neff("path/to/model.neff")

   # Build an input tensor from a NumPy array
   input_data = np.random.randn(1, 10).astype(np.float32)
   input_tensor = NrtpyTensor.from_numpy(input_data, name="input")

   # Execute — output tensors are allocated automatically and returned
   outputs = model(inputs={"input": input_tensor})
   result = outputs["output"].numpy()

Benchmarking
~~~~~~~~~~~~~

.. code-block:: python

   # Device-side timing (most accurate for kernel latency)
   stats = model.benchmark(
       inputs={"input": input_tensor},
       warmup_iter=5,
       benchmark_iter=100,
   )
   print(f"Mean: {stats.mean_ms:.2f} ms")
   print(f"Min: {stats.min_ms:.2f} ms, Max: {stats.max_ms:.2f} ms")
   print(f"Std dev: {stats.std_dev_ms:.4f} ms")

   # Host-side timing (includes host-device overhead)
   host_stats = model.benchmark(
       inputs={"input": input_tensor},
       warmup_iter=5,
       benchmark_iter=100,
       mode="host",
   )

Related reference
-----------------

- :ref:`nrtpy-tensor-ref` — build the input and output tensors this API
  consumes.
- :ref:`nrtpy-errors-ref` — exceptions raised during load and execute.
- :ref:`nrtpy-configuration-ref` — configure which NeuronCores a model runs on.
