.. meta::
   :description: Tutorial — compile an NKI kernel to a NEFF and use nrtpy to execute, debug, and benchmark it on AWS Neuron hardware.
   :keywords: Neuron, nrtpy, NKI, NEFF, debug, runtime error, tutorial, NeuronCore
   :date-modified: 2026-07-30

.. _nrtpy-tutorial-debug-kernel:

================================================================
Tutorial: Test and debug a NKI kernel end-to-end with nrtpy
================================================================

This topic guides you through generating a NEFF that contains a known bug,
then using ``nrtpy`` to load it, execute it, and interpret the runtime error.
When you have completed it, you will understand how to use ``nrtpy`` to isolate
and debug NKI kernel runtime failures without a framework in the loop.

Overview
--------

Not all kernel bugs are caught at compile time. Some produce valid NEFFs that
only fail at execution — for example, out-of-bounds memory access, invalid DMA
patterns, or incorrect tensor layouts. These failures surface as
``NrtError`` exceptions from ``libnrt`` with specific error codes.

This tutorial demonstrates:

1. Writing an NKI kernel with a deliberate out-of-bounds DMA access.
2. Compiling it to a NEFF using the standalone NKI path.
3. Loading the NEFF with ``nrtpy`` and observing the runtime error.
4. Interpreting the error and iterating on a fix.

Before you start
----------------

This tutorial assumes that you have experience in the following areas:

* Writing NKI kernels (see
  :ref:`Quickstart: Implement and run your first kernel <quickstart-run-nki-kernel>`).
* Basic ``nrtpy`` usage (see :ref:`nrtpy-getting-started`).

Before you begin, complete the :ref:`nrtpy-getting-started` setup. You will
also need an NKI environment for kernel compilation (for example, the PyTorch
Neuron venv on your DLAMI).

----

Step 1: Write a kernel with a known bug
-----------------------------------------

In this step, you will write an NKI kernel that performs an indirect DMA gather
with out-of-bounds indices. The kernel compiles successfully (the compiler does
not validate index values), but fails at runtime when the DMA engine attempts
to read from an invalid address.

Create ``oob_kernel.py``:

.. code-block:: python

   import nki
   import nki.language as nl
   import nki.isa as nisa


   @nki.jit
   def oob_dma_kernel(src_tensor):
       """Indirect DMA gather with out-of-bounds indices.

       src_tensor has 128 rows, but the indices point to row 99999.
       This compiles fine but triggers NRT_EXEC_OOB at runtime.
       """
       P, F = 128, 512

       # Create an index tensor filled with 99999 (far beyond src's 128 rows)
       idx = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
       nisa.memset(dst=idx, value=99999)

       # Destination buffer
       dst = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
       nisa.memset(dst=dst, value=0.0)

       # Indirect DMA gather — indices are out of bounds
       src_ap = src_tensor.ap(
           pattern=[[F, P], [1, 1]], offset=0, vector_offset=idx, indirect_dim=0
       )
       dst_ap = dst.ap(pattern=[[F, P], [1, 1]], offset=0)
       nisa.dma_copy(dst=dst_ap, src=src_ap)

       # Write result to HBM
       out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
       nl.store(out, dst)
       return out

Step 2: Compile the kernel to a NEFF
--------------------------------------

In this step, you will compile the kernel and preserve the NEFF using the
``NKI_ARTIFACTS_DIR`` environment variable. Run this step in your **NKI
environment** (for example, the PyTorch Neuron venv on your DLAMI).

Create ``compile_oob.py``:

.. code-block:: python

   import os
   import shutil
   import numpy as np

   # Set artifacts directory BEFORE importing nki
   ARTIFACTS_DIR = "./oob_neff"
   if os.path.exists(ARTIFACTS_DIR):
       shutil.rmtree(ARTIFACTS_DIR)
   os.environ["NKI_ARTIFACTS_DIR"] = ARTIFACTS_DIR

   from oob_kernel import oob_dma_kernel

   # Compile and execute (will fail on hardware, but NEFF is preserved)
   src = np.ones((128, 512), dtype=np.float32)

   try:
       oob_dma_kernel(src)
   except Exception as e:
       print(f"Expected error: {type(e).__name__}: {e}")

   # Confirm the NEFF was saved
   neff_path = os.path.join(ARTIFACTS_DIR, "kernel.neff")
   if os.path.exists(neff_path):
       print(f"\nNEFF saved: {neff_path} ({os.path.getsize(neff_path)} bytes)")
   else:
       for root, dirs, files in os.walk(ARTIFACTS_DIR):
           for f in files:
               if f.endswith(".neff"):
                   print(f"\nNEFF saved: {os.path.join(root, f)}")

Run:

.. code-block:: bash

   python compile_oob.py

Expected output:

.. code-block:: text

   Expected error: NrtError: NRT Error NRT_EXEC_OOB(1006): ...

   NEFF saved: ./oob_neff/kernel.neff (XXXXX bytes)

.. note::

   The ``NKI_ARTIFACTS_DIR`` environment variable must be set **before**
   importing ``nki``. The compile options are resolved at import time.
   Even though execution fails, the NEFF is preserved because compilation
   succeeded — the error occurs at runtime, not compile time.

Step 3: Load the NEFF with nrtpy and reproduce the error
----------------------------------------------------------

In this step, you will load the saved NEFF with ``nrtpy`` and execute it to
observe the runtime error in isolation. Run this step in your **nrtpy
environment** (see :ref:`nrtpy-getting-started`).

Create ``debug_with_nrtpy.py``:

.. code-block:: python

   import numpy as np
   from nrtpy import NrtpyModel, NrtpyTensor, NrtError

   # Load the buggy NEFF
   model = NrtpyModel.load_from_neff("./oob_neff/kernel.neff")

   # Inspect the model interface
   print("Model inputs:")
   for name, info in model.input_tensors_info.items():
       print(f"  {name}: shape={info.shape}, size={info.size}")
   print("Model outputs:")
   for name, info in model.output_tensors_info.items():
       print(f"  {name}: shape={info.shape}, size={info.size}")
   print()

   # Prepare input
   src_data = np.ones((128, 512), dtype=np.float32)
   src_tensor = NrtpyTensor.from_numpy(src_data, name="src_tensor")

   # Execute and catch the runtime error
   try:
       outputs = model(inputs={"src_tensor": src_tensor})
       print("Execution succeeded (unexpected)")
   except NrtError as e:
       print(f"Caught NrtError:")
       print(f"  {e}")
       print()
       print("This confirms the kernel has an out-of-bounds DMA access.")
       print("The indices (99999) exceed the source tensor's row count (128).")

Run:

.. code-block:: bash

   python debug_with_nrtpy.py

Expected output:

.. code-block:: text

   Model inputs:
     src_tensor: shape=[128, 512], size=262144
   Model outputs:
     output_0: shape=[128, 512], size=262144

   Caught NrtError:
     NRT Error NRT_EXEC_OOB(1006): Failed to execute model

   This confirms the kernel has an out-of-bounds DMA access.
   The indices (99999) exceed the source tensor's row count (128).

.. note::

   The NRT runtime also emits error log lines (``TDRV``, ``NMGR``, ``NRT``
   prefixed) to stderr before the Python exception is raised. These provide
   additional context (affected NeuronCore, model path) and are normal
   diagnostic output.

Step 4: Fix the kernel and validate
-------------------------------------

In this step, you will fix the kernel by using valid indices, recompile, and
confirm the fix with ``nrtpy``.

Create ``fixed_kernel.py``:

.. code-block:: python

   import nki
   import nki.language as nl
   import nki.isa as nisa


   @nki.jit
   def fixed_dma_kernel(src_tensor):
       """Same kernel but with valid indices (row 0 instead of 99999)."""
       P, F = 128, 512

       # Valid index: row 0 (within the 128-row source)
       idx = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
       nisa.memset(dst=idx, value=0)  # Fixed: 0 instead of 99999

       dst = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
       nisa.memset(dst=dst, value=0.0)

       src_ap = src_tensor.ap(
           pattern=[[F, P], [1, 1]], offset=0, vector_offset=idx, indirect_dim=0
       )
       dst_ap = dst.ap(pattern=[[F, P], [1, 1]], offset=0)
       nisa.dma_copy(dst=dst_ap, src=src_ap)

       out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
       nl.store(out, dst)
       return out

Recompile and validate:

.. code-block:: python

   # validate_fix.py
   import os
   import shutil
   import numpy as np

   ARTIFACTS_DIR = "./fixed_neff"
   if os.path.exists(ARTIFACTS_DIR):
       shutil.rmtree(ARTIFACTS_DIR)
   os.environ["NKI_ARTIFACTS_DIR"] = ARTIFACTS_DIR

   from fixed_kernel import fixed_dma_kernel

   src = np.ones((128, 512), dtype=np.float32)
   result = fixed_dma_kernel(src)
   print(f"Kernel executed successfully. Output shape: {result.shape}")

Then confirm with ``nrtpy``:

.. code-block:: python

   # confirm_with_nrtpy.py
   import numpy as np
   from nrtpy import NrtpyModel, NrtpyTensor, NrtError

   model = NrtpyModel.load_from_neff("./fixed_neff/kernel.neff")
   src_tensor = NrtpyTensor.from_numpy(
       np.ones((128, 512), dtype=np.float32), name="src_tensor"
   )

   try:
       outputs = model(inputs={"src_tensor": src_tensor})
       result = list(outputs.values())[0].numpy()
       print(f"Execution succeeded. Output shape: {result.shape}")
       print(f"Output sample (first row): {result[0, :5]}")
   except NrtError as e:
       print(f"Still failing: {e}")

Expected output:

.. code-block:: text

   Execution succeeded. Output shape: (128, 512)
   Output sample (first row): [1. 1. 1. 1. 1.]

All complete! Now, let's confirm everything works.

Confirmation
------------

You have successfully:

1. Written a kernel with a deliberate runtime bug (out-of-bounds DMA).
2. Compiled it to a NEFF preserved via ``NKI_ARTIFACTS_DIR``.
3. Used ``nrtpy`` to load and execute the NEFF in isolation, catching
   the ``NrtError`` with its diagnostic error code.
4. Fixed the bug and validated the fix using the same ``nrtpy`` workflow.

Congratulations! You have now debugged a faulty NKI kernel end-to-end using
``nrtpy``. If you encountered any issues, see :ref:`nrtpy-troubleshooting`.

----

Clean up
--------

Remove the generated NEFF directories:

.. code-block:: bash

   rm -rf oob_neff/ fixed_neff/

Next steps
----------

Now that you've completed this tutorial, take your work and dive into other
topics that build off of it.

* :ref:`nrtpy-tutorial-validate-benchmark` — validate correctness, benchmark
  performance, and capture execution traces with ``nrtpy``.
* :ref:`nrtpy-model-ref` — full ``NrtpyModel`` API including trace capture for
  deeper debugging.
* :ref:`nrtpy-errors-ref` — complete ``NrtError`` and ``NrtpyError``
  exception reference.

Further reading
---------------

* :ref:`nrtpy-guide` — nrtpy overview and architecture.
* :ref:`quickstart-run-nki-kernel` — write and compile NKI kernels.
