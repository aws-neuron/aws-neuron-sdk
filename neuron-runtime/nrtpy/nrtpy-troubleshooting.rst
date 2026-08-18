.. meta::
   :description: Troubleshooting common issues with nrtpy on AWS Neuron.
   :date-modified: 2026-07-31
   :keywords: Neuron, nrtpy, troubleshooting, errors, libnrt, NrtError

.. _nrtpy-troubleshooting:

========================
nrtpy troubleshooting
========================

This page lists common issues encountered when using ``nrtpy`` and their
solutions.

Import and setup issues
-----------------------

``import nrtpy`` fails with "libnrt.so.1: cannot open shared object file"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Neuron Runtime library is not on the library path. Set it before running
Python:

.. code-block:: bash

   export LD_LIBRARY_PATH=/opt/aws/neuron/lib:$LD_LIBRARY_PATH

Verify the library exists:

.. code-block:: bash

   ls /opt/aws/neuron/lib/libnrt.so*

If missing, install ``aws-neuronx-runtime-lib``.

Import conflict between nrtpy and nki
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standalone ``nrtpy`` wheel and the ``nki`` wheel cannot be installed in the
same Python environment due to a namespace conflict. In a future release, this
limitation will be resolved. Until then, install only one per environment:

- Use your NKI environment to compile kernels to NEFFs.
- Use a separate ``nrtpy`` environment to load and execute NEFFs.

Model loading issues
--------------------

``NrtError: NRT_FAILURE(6)`` on load_from_neff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NEFF was compiled for a different target architecture than the current
instance. Recompile with the correct ``--target`` (for example, ``trn2`` for
Trn2 instances).

NEFF file not found after compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using ``NKI_ARTIFACTS_DIR``, the NEFF is saved as ``kernel.neff`` in the
specified directory. When using ``torch_neuronx.trace`` with
``compiler_workdir``, the NEFF filename includes a hash. Use:

.. code-block:: bash

   find <artifacts_dir> -name "*.neff"

``NKI_ARTIFACTS_DIR`` must be set **before** importing ``nki``. If set after,
the compile options will not pick it up and the NEFF goes to a temp directory
that is deleted after execution.

Execution issues
----------------

Input name mismatch error
~~~~~~~~~~~~~~~~~~~~~~~~~~

The dictionary keys passed to ``model()`` must exactly match the NEFF's
expected input names. Use ``model.input_tensors_info.keys()`` to discover the
correct names:

.. code-block:: python

   print(model.input_tensors_info.keys())

Output tensor has void dtype (``|V2``, ``|V4``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auto-allocated output tensors may report a ``void`` dtype because ``libnrt``
does not always report the original dtype. Reinterpret the raw bytes with
NumPy:

.. code-block:: python

   result = output_tensor.numpy().view(np.float16)   # for fp16 kernels
   result = output_tensor.numpy().view(np.float32)   # for fp32 kernels

``NrtError: NRT_EXEC_OOB(1006)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The kernel attempted an out-of-bounds memory access at runtime. This typically
means DMA indices point outside the source or destination tensor bounds. Check
your kernel's indirect addressing logic.

NRT runtime also emits diagnostic lines to stderr (``TDRV``, ``NMGR``, ``NRT``
prefixed) before the Python exception is raised. These provide the affected
NeuronCore and model path.

Benchmarking issues
-------------------

Benchmark variance is high (std_dev > 10% of mean)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Increase ``warmup_iter`` to ensure the kernel is fully warmed up. Also check
that no other workload is sharing the NeuronCore.

Device benchmark shows more iterations than ``benchmark_iter``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some kernels emit multiple execution events per call (multi-subgraph NEFFs).
The reported ``iterations`` in ``BenchmarkResult`` reflects actual device
events, not Python calls.

Benchmark raises RuntimeError in device mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``device`` mode, the NeuronCore must support system tracing. Ensure your SDK
version is 2.32 or later and that no other process is holding the NeuronCore.

Trace file is empty or very small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure ``save_trace=True`` is passed and that execution completes without
error. Traces are only written on successful execution.
