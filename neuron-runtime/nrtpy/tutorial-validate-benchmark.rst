.. meta::
   :description: Tutorial — validate NKI kernel correctness, benchmark device-side latency, and capture execution traces with nrtpy.
   :keywords: Neuron, nrtpy, NKI, NEFF, benchmark, validate, trace, profiling, tutorial
   :date-modified: 2026-07-30

.. _nrtpy-tutorial-validate-benchmark:

================================================================
Tutorial: Validate, benchmark, and trace a kernel with nrtpy
================================================================

This topic guides you through using ``nrtpy`` to validate kernel correctness
against a NumPy reference, measure device-side execution latency, and capture
execution traces for performance analysis. When you have completed it, you will
have a repeatable test harness for iterating on NKI kernel quality.

Overview
--------

Once you have a compiled NEFF that executes without errors, the next steps are:

1. **Validate correctness** — compare kernel output against a known-good
   reference to catch numerical bugs.
2. **Benchmark performance** — measure NeuronCore execution time to establish
   a baseline and detect regressions.
3. **Capture traces** — save execution traces for profiling tools to analyze
   instruction scheduling, DMA overlap, and pipeline utilization.

This tutorial builds on :ref:`nrtpy-tutorial-debug-kernel`. You should have a
working NEFF before starting.

Before you start
----------------

This tutorial assumes that you have:

* Completed :ref:`nrtpy-tutorial-debug-kernel` (or have a known-good NEFF).
* A working nrtpy environment (see :ref:`nrtpy-getting-started`).

Before you begin, complete the :ref:`nrtpy-getting-started` setup. This
tutorial uses the addition kernel NEFF from
:ref:`nrtpy-tutorial-debug-kernel`. If you do not have it, any working NEFF
will do — adjust tensor names, shapes, and the reference function accordingly.

.. note::

   Auto-allocated output tensors may have a ``void`` dtype (for example,
   ``|V2`` for fp16) because ``libnrt`` does not always report the original
   dtype. Use ``.view(np.float16)`` or ``.view(np.float32)`` on the NumPy
   result to reinterpret the raw bytes into the correct type.

----

Step 1: Load the model and prepare test inputs
------------------------------------------------

In this step, you will load the NEFF and create multiple test input patterns
to exercise different code paths in the kernel.

.. code-block:: python

   import numpy as np
   from nrtpy import NrtpyModel, NrtpyTensor

   model = NrtpyModel.load_from_neff("./add_neff/kernel.neff")

   # Discover expected interface
   print("Inputs:", list(model.input_tensors_info.keys()))
   print("Outputs:", list(model.output_tensors_info.keys()))

   # Prepare diverse test inputs (addition kernel: a + b)
   test_cases = [
       ("ones+ones", np.ones((4, 3), dtype=np.float16), np.ones((4, 3), dtype=np.float16)),
       ("zeros+zeros", np.zeros((4, 3), dtype=np.float16), np.zeros((4, 3), dtype=np.float16)),
       ("neg+pos", np.full((4, 3), -3.0, dtype=np.float16), np.full((4, 3), 5.0, dtype=np.float16)),
       ("random", np.random.randn(4, 3).astype(np.float16), np.random.randn(4, 3).astype(np.float16)),
       ("large", np.full((4, 3), 1000.0, dtype=np.float16), np.full((4, 3), 2000.0, dtype=np.float16)),
   ]

Step 2: Validate correctness against NumPy
--------------------------------------------

In this step, you will execute the kernel for each test case and compare
against a NumPy reference implementation.

.. code-block:: python

   def numpy_reference(a, b):
       """Reference implementation: element-wise addition."""
       return a + b

   passed = 0
   failed = 0

   for name, a_data, b_data in test_cases:
       a_tensor = NrtpyTensor.from_numpy(a_data, name="a_input")
       b_tensor = NrtpyTensor.from_numpy(b_data, name="b_input")
       outputs = model(inputs={"a_input": a_tensor, "b_input": b_tensor})

       # Reinterpret void dtype to float16
       result = list(outputs.values())[0].numpy().view(np.float16)
       expected = numpy_reference(a_data, b_data)

       try:
           np.testing.assert_allclose(
               result.astype(np.float32), expected.astype(np.float32),
               rtol=1e-3, atol=1e-3
           )
           print(f"  PASS: {name}")
           passed += 1
       except AssertionError as e:
           print(f"  FAIL: {name}")
           print(f"    {e}")
           failed += 1

   print(f"\nResults: {passed} passed, {failed} failed")

.. tip::

   For kernels with floating-point accumulation (matmul, reductions), use
   looser tolerances (``rtol=1e-2``) and test with inputs that stress
   precision: denormals, values near overflow, and mixed-sign inputs.

Step 3: Benchmark device-side latency
---------------------------------------

In this step, you will measure the kernel's NeuronCore execution time using
device-side hardware tracing.

.. code-block:: python

   a_tensor = NrtpyTensor.from_numpy(np.ones((4, 3), dtype=np.float16), name="a_input")
   b_tensor = NrtpyTensor.from_numpy(np.ones((4, 3), dtype=np.float16), name="b_input")

   # Device-side timing: measures only NeuronCore execution
   device_stats = model.benchmark(
       inputs={"a_input": a_tensor, "b_input": b_tensor},
       warmup_iter=10,
       benchmark_iter=100,
       mode="device",
   )

   print("Device-side benchmark (NeuronCore execution only):")
   print(f"  Mean:    {device_stats.mean_ms:.4f} ms")
   print(f"  Min:     {device_stats.min_ms:.4f} ms")
   print(f"  Max:     {device_stats.max_ms:.4f} ms")
   print(f"  Std dev: {device_stats.std_dev_ms:.4f} ms")
   print(f"  Iterations: {device_stats.iterations}")

   # Host-side timing: includes host-device round-trip overhead
   host_stats = model.benchmark(
       inputs={"a_input": a_tensor, "b_input": b_tensor},
       warmup_iter=10,
       benchmark_iter=100,
       mode="host",
   )

   print(f"\nHost-side benchmark (total round-trip):")
   print(f"  Mean:    {host_stats.mean_ms:.4f} ms")
   print(f"  Min:     {host_stats.min_ms:.4f} ms")
   print(f"  Overhead vs device: {host_stats.mean_ms - device_stats.mean_ms:.4f} ms")

**When to use each mode:**

- **device** — Use for kernel optimization. Measures only the NeuronCore
  compute and DMA time. Most accurate for comparing kernel variants.
- **host** — Use when total latency matters (for example, in a serving path).
  Includes host-device communication overhead.

Step 4: Detect performance regressions
----------------------------------------

In this step, you will save a baseline and compare against it after kernel
changes.

.. code-block:: python

   import json

   # Save baseline
   baseline = {
       "mean_ms": device_stats.mean_ms,
       "min_ms": device_stats.min_ms,
       "max_ms": device_stats.max_ms,
       "std_dev_ms": device_stats.std_dev_ms,
   }
   with open("baseline_perf.json", "w") as f:
       json.dump(baseline, f, indent=2)
   print(f"Baseline saved: mean={baseline['mean_ms']:.4f} ms")

   # Later, after a kernel change: compare against baseline
   with open("baseline_perf.json") as f:
       baseline = json.load(f)

   new_stats = model.benchmark(
       inputs={"a_input": a_tensor, "b_input": b_tensor},
       warmup_iter=10,
       benchmark_iter=100,
       mode="device",
   )

   regression_threshold = 1.10  # 10% slower = regression
   ratio = new_stats.mean_ms / baseline["mean_ms"]
   if ratio > regression_threshold:
       print(f"REGRESSION: {ratio:.2f}x slower ({new_stats.mean_ms:.4f} ms "
             f"vs baseline {baseline['mean_ms']:.4f} ms)")
   else:
       print(f"OK: {ratio:.2f}x of baseline ({new_stats.mean_ms:.4f} ms)")

Step 5: Capture an execution trace
------------------------------------

In this step, you will save a ``.ntff`` execution trace file that can be
analyzed with Neuron profiling tools.

.. code-block:: python

   # Execute with trace capture
   model(
       inputs={"a_input": a_tensor, "b_input": b_tensor},
       save_trace=True,
       ntff_name="./kernel_trace.ntff",
   )
   print("Trace saved to ./kernel_trace.ntff")

The ``.ntff`` file captures device-side execution events including:

- DMA transfer start/end times
- Compute engine utilization
- Instruction scheduling on each engine

Use this trace when:

- Device-side benchmark shows unexpected latency and you need to identify
  the source of the bottleneck.
- You want to verify that DMA and compute overlap as intended in your kernel.
- You are comparing two kernel implementations and need instruction-level
  timing differences.

All complete! Now, let's confirm everything works.

Confirmation
------------

You should have:

1. Validated correctness across multiple input patterns with no failures.
2. Established a device-side latency baseline.
3. A saved ``.ntff`` trace file for profiling.

.. code-block:: text

   Results: 5 passed, 0 failed

   Device-side benchmark (NeuronCore execution only):
     Mean:    0.0032 ms
     Min:     0.0030 ms
     Max:     0.0041 ms
     Std dev: 0.0002 ms
     Iterations: 100

   Trace saved to ./kernel_trace.ntff

Congratulations! You now have a repeatable validation and benchmarking harness
for your NKI kernels. If you encounter issues, see :ref:`nrtpy-troubleshooting`.

----

Clean up
--------

.. code-block:: bash

   rm -f baseline_perf.json kernel_trace.ntff

Next steps
----------

* :ref:`nrtpy-model-ref` — full ``NrtpyModel`` API documentation.
* :ref:`nrtpy-tensor-ref` — advanced tensor operations including multi-core
  allocation and in-place writes.
* :ref:`nrtpy-tutorial-debug-kernel` — debug runtime errors from faulty
  kernels.

Further reading
---------------

* :ref:`nrtpy-guide` — nrtpy overview and architecture.
