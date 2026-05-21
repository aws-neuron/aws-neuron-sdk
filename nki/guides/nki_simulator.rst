.. meta::
    :description: Documentation for the nki.simulate API in the Neuron SDK
    :keywords: nki, simulate, nki.simulate, test, kernels, aws neuron sdk
    :date-modified: 05/11/2026

.. _nki-simulator:

NKI CPU Simulator
=================

.. warning::

   This API is experimental and may change in future releases.

``nki.simulate`` runs NKI kernels on your CPU using Python (and NumPy), with no Trainium hardware required.
It executes kernel code as regular Python, making it ideal for fast development, debugging, and correctness testing.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

``nki.simulate`` is a CPU-based functional simulator for NKI kernels. It executes every ``nki.isa``
and ``nki.language`` operation using Python and NumPy, producing results that approximate hardware behavior.
You write your kernel once and can run it on both the simulator and real Trainium devices. Some kernels
may require adjustments when moving to hardware — see :ref:`simulation-limitations` for details.

**Why use the simulator?**

- **No hardware required** — develop and test NKI kernels on any machine with Python.
- **Cost savings** — avoid the cost of developing on Trainium instances; iterate locally, then deploy to hardware when ready.
- **Same kernel code** — the same ``@nki.jit`` kernel can run on both hardware and the simulator. See :ref:`simulation-limitations` for cases where adjustments may be needed.
- **Full debugging support** — use ``breakpoint()``, PDB, or IDE debuggers to step through kernel execution and inspect tensor values.
- **Fast iteration** — test kernels instantly without compilation or deployment.
- **Hardware constraint validation** — catches invalid shapes, buffer misuse, dtype errors, and other constraint violations at runtime with clear error messages.
- **AI-assisted development** — ideal for GenAI coding agents authoring NKI kernels: instant local feedback, detailed error messages, and the ability to instrument every line of code with debug prints (including intermediate tensors) enable rapid autonomous iteration without hardware access.

Quick Start
-----------

.. nki_example:: /nki/examples/simulate/nki_simulate_example.py
   :language: python
   :marker: NKI_EXAMPLE_SIMULATE

.. nki_example:: /nki/examples/simulate/nki_simulate_example.py
   :language: python
   :marker: NKI_EXAMPLE_SIMULATE_RUN


Usage
-----

Running the Simulator
^^^^^^^^^^^^^^^^^^^^^

There are two ways to invoke the simulator:

**1. NKI_SIMULATOR=1 environment variable (recommended)**

The easiest approach. Set this environment variable and run your existing PyTorch framework code with NKI kernel calls
with **zero code changes** — every ``@nki.jit`` kernel automatically runs on the simulator:

.. code-block:: bash

   NKI_SIMULATOR=1 python my_script.py

This is the recommended approach for PyTorch development, testing, and running existing scripts
on the simulator without any code modifications.

.. note::

   ``NKI_SIMULATOR=1`` does not currently support JAX. For JAX workflows, use the
   ``nki.simulate()`` explicit API instead.

**2. nki.simulate() — Explicit API**

Use the explicit API to run a kernel on the simulator. This is also useful when you want
to run a kernel on *both* the simulator and hardware in the same script — for example,
to compare results:


.. code-block:: python

   # Run on simulator with NumPy
   sim_result_np = nki.simulate(my_kernel)(a_np, b_np)

   # Run on simulator with PyTorch
   sim_result = nki.simulate(my_kernel)(a_torch, b_torch)

   # Run on hardware (requires Trainium and neuronx-cc)
   hw_result = my_kernel(a_torch, b_torch)

   # Compare
   torch.testing.assert_close(sim_result, hw_result, rtol=1e-2, atol=1e-5)


Target Platform
^^^^^^^^^^^^^^^

The simulator models different NeuronCore generations. Set the target using the
``NEURON_PLATFORM_TARGET_OVERRIDE`` environment variable:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment variable value
     - Hardware
   * - ``trn1`` or ``gen2``
     - Trn1 (NeuronCore-v2)
   * - ``trn2`` or ``gen3``
     - Trn2 (NeuronCore-v3)
   * - ``trn3`` or ``gen4``
     - Trn3 (NeuronCore-v4)
   * - *(unset)*
     - Auto-detect (uses the Neuron chip detected on the running machine, otherwise defaults to ``trn3``)

Precise Floating-Point Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the simulator uses **precise floating-point mode** (``NKI_PRECISE_FP=1``). In this mode,
low-precision dtypes (``bfloat16``, ``float8_e4m3``, ``float8_e5m2``) are stored using ``ml_dtypes``
(``bfloat16``, ``float8``, etc.), producing numerical behavior similar to hardware.


**Disabling precise mode (high-precision mode):**

You can disable precise mode by setting ``NKI_PRECISE_FP=0``:

.. code-block:: bash

   NKI_PRECISE_FP=0 python my_script.py

When disabled, all low-precision tensors are stored as ``float32``, eliminating rounding and
truncation effects. This mode is useful for **isolating accuracy loss due to downcasts**:

1. Run your kernel with ``NKI_PRECISE_FP=0`` to see how the kernel logic performs in full ``float32``
   precision — compare this result against your torch/numpy reference to verify algorithmic correctness
   of your kernel.
2. Run with the default (``NKI_PRECISE_FP=1``) to see how accuracy is impacted by the lower-precision
   dtypes used in your kernel.

This workflow helps you distinguish between algorithmic bugs and precision-related numerical differences.

Debugging
^^^^^^^^^

Because the simulator runs kernels as regular Python, you have full access to Python's
debugging ecosystem.

**Using breakpoint():**

.. code-block:: python

   @nki.jit
   def my_kernel(a_ptr):
       tile = nl.load(a_ptr)
       breakpoint()  # Debugger stops here — inspect `tile`
       result = nl.add(tile, tile)
       return nl.store(result)

   nki.simulate(my_kernel)(data)

**Using device_print:**

``nl.device_print`` works in the simulator and prints tensor values to stdout:

.. code-block:: python

   @nki.jit
   def my_kernel(a_ptr):
       tile = nl.load(a_ptr)
       nl.device_print("my tile", tile)
       ...

**Using Python print:**

Since the simulator executes kernels as standard Python, you can use ``print()`` to inspect any
intermediate tensor or register value during execution. This is especially useful for both interactive
debugging and AI-assisted development workflows where agents iterate on kernels locally.

**IDE Debugging (VSCode / PyCharm):**

Set breakpoints in your kernel code and run your script normally. The simulator executes
kernel code in-process, so IDE debuggers work without any special configuration.


How It Works
------------

Execution
^^^^^^^^^

When you call ``nki.simulate(kernel)(a, b)``:

1. Each NumPy array argument is wrapped into an ``NkiTensor`` with ``buffer=nl.hbm``
   (or ``shared_hbm`` for LNC2). Non-array arguments pass through unchanged.
2. The simulator backend is activated, routing all ``nki.isa`` and ``nki.language``
   operations to NumPy-based implementations.
3. The kernel function runs as regular Python — each NKI API call executes eagerly
   and sequentially. There is no instruction scheduling or engine parallelism.
4. On return, ``NkiTensor`` results are converted back to NumPy arrays. Input arrays are
   updated in-place if the kernel modified the corresponding HBM tensors.

For **LNC2 kernels** (``kernel[2]``), the simulator spawns two Python threads that execute the
kernel concurrently, each with its own ``program_id``. Input arrays use ``shared_hbm`` buffers,
so both threads can access shared memory. ``nki.isa.sendrecv`` and ``nki.isa.core_barrier``
use thread-safe synchronization primitives.

Uninitialized Memory Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simulator automatically fills all newly allocated tensors with **sentinel values** — ``NaN`` for
floating-point types and ``4`` for integer types. This makes it easy to detect bugs where a kernel
reads from memory that was never written to.

Because ``NaN`` propagates through arithmetic (any operation involving ``NaN`` produces ``NaN``), if
your kernel accidentally computes on uninitialized memory, the resulting output will contain ``NaN``
values. You can check for this in your test:

.. code-block:: python

   result = nki.simulate(my_kernel)(inputs)
   assert not np.any(np.isnan(result)), "Kernel computed on uninitialized memory!"

**Why this matters:**

On real hardware, uninitialized memory contains arbitrary leftover values from previous operations.
A kernel that reads uninitialized data may appear to produce correct results on hardware by coincidence —
making these bugs extremely difficult to track down. The simulator's sentinel values turn these silent
correctness hazards into immediately visible ``NaN`` values in the output.

.. tip::

   If you see unexpected ``NaN`` values in your simulation output, check that all tensors are properly
   initialized before use. Common causes include:

   - Allocating a tensor with ``nl.ndarray`` but not writing to all elements before reading
   - Off-by-one errors in tile loop bounds that leave some elements unwritten
   - Conditional writes that skip certain partitions or indices


Hardware Constraint Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each ``nki.isa`` operation validates hardware constraints at runtime — shape limits, dtype
compatibility, buffer types, engine restrictions, and architecture version requirements.
Invalid operations raise clear Python exceptions with descriptive error messages.

.. warning::

   Hardware constraint validation is actively being developed. Some constraints may not yet
   be checked by the simulator. If your kernel passes simulation but fails on hardware,
   report it to the Neuron team as an issue.


**Example:**

.. code-block:: python

   @nki.jit
   def bad_kernel(a_ptr):
       tile = nl.ndarray((256, 512), dtype=nl.float32, buffer=nl.sbuf)  # exceeds 128
       ...

   nki.simulate(bad_kernel)(data)
   # AssertionError: tensor_tensor data1 partition dimension 256 exceeds maximum 128



.. _simulation-limitations:

Simulation Limitations
----------------------

The simulator approximates hardware behavior but is not identical. Understanding these
limitations helps you write kernels that work on both the simulator and real Trainium hardware.

NKI Meta-Programming Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simulator executes kernel code directly as Python — there is no compilation step. As a result,
the simulator accepts any valid Python in the kernel body, including arbitrary classes, closures,
and dynamic control flow. The NKI compiler, however, only supports a restricted subset of Python
for meta-programming, see :ref:`NKI Language Guide<nki-language-guide>`. Kernels that use
unsupported Python constructs will execute successfully on the simulator but fail to compile for hardware.

Numerical Precision
^^^^^^^^^^^^^^^^^^^

The simulator uses precise floating-point mode by default (``NKI_PRECISE_FP=1``), storing
low-precision types using ``ml_dtypes`` for numerical behavior close to hardware. However,
simulation results may still differ slightly from hardware due to differences in how the
simulator computes operations versus the hardware engines, as well as
floating-point non-associativity (different accumulation ordering can produce different
rounding results).
See `Precise Floating-Point Mode`_ for details on how to use ``NKI_PRECISE_FP=0`` to isolate
precision effects from algorithmic correctness.

Performance
^^^^^^^^^^^

The simulator runs on the CPU using Python and NumPy. It does not model instruction latency,
engine parallelism, or hardware scheduling. Since kernels are interpreted rather than compiled
and optimized for Trainium NeuronCores, the simulator is significantly slower than hardware
execution and is not suitable for performance benchmarking.

Memory Model
^^^^^^^^^^^^

The simulator allocates each tensor independently without simulating overlapping memory regions
or validating against SBUF/PSUM capacity limits. Kernels with memory conflicts may run
successfully on the simulator but fail or produce incorrect results on real hardware, where
SBUF and PSUM are shared physical memory with capacity constraints.

Known Gaps
^^^^^^^^^^

- Some hardware constraint checks are missing — see `Hardware Constraint Validation`_ for details.
- ``nki.collectives`` APIs are not implemented in the simulator.
- ``oob_mode.skip`` with ``_x4`` dtypes will incorrectly raise an out-of-bounds error instead of skipping the oob access.
- ``nki.isa.quantize_mx()`` scale packing fails when ``dst_scale`` number of partitions doesn't match ``src`` partitions.