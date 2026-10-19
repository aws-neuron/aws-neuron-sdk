.. meta::
   :description: Troubleshooting guide for TorchNeuron on AWS Trainium. Covers common issues with installation, eager mode, torch.compile, TorchTitan, and distributed training.
   :keywords: TorchNeuron, troubleshooting, errors, debugging, torch.compile, eager mode, TorchTitan, distributed training, PyTorch, Trainium, AWS Neuron
   :date-modified: 2026-04-21

.. _torch-troubleshooting:

================================
Troubleshooting: TorchNeuron
================================

Overview
---------

This page covers common issues and diagnostic steps for TorchNeuron on AWS Trainium. Use the
sections below to find symptoms and solutions organized by area. If your issue spans multiple
areas, start with the **General** section for environment and installation diagnostics.

----

General TorchNeuron troubleshooting
===================================

Installation, environment setup, and device availability issues that apply regardless of
execution mode.

Neuron device not found
------------------------

- **Symptom:** Error referencing ``neuron`` device not being available, or ``RuntimeError`` when
  calling ``.to('neuron')``.

  **Possible Causes:**
    - Running on an instance without Trainium hardware
    - The ``torch-neuronx`` package is not installed or is an older XLA-based version
    - Neuron driver is not loaded

  **Solutions:**
    - Verify you are on a Trn2 or Trn3 instance
    - Run ``pip show torch-neuronx`` to check the installed version. ``torch-neuronx`` 2.14.x
      is the native backend; earlier versions are the legacy XLA-based library
    - Run ``neuron-ls`` to verify Neuron devices are visible

Package version conflict
-------------------------

- **Symptom:** Unexpected behavior or import errors after installing ``torch-neuronx``.

  **Possible Causes:**
    - The older XLA-based ``torch-neuronx`` is installed instead of the native version
    - Multiple versions coexist in the same environment

  **Solutions:**
    - Check the installed version: ``pip show torch-neuronx``. ``torch-neuronx`` 2.14.x is the
      native backend; earlier versions are the legacy XLA-based library.
    - Use a clean virtual environment to avoid conflicts
    - Uninstall and reinstall the package:

      .. code-block:: bash

         python -m pip uninstall torch-neuronx
         python -m pip install "torch-neuronx==2.14.*" \
           --extra-index-url https://pip.repos.neuron.amazonaws.com \
           --extra-index-url https://download.pytorch.org/whl/cpu

Python or PyTorch version mismatch
------------------------------------

- **Symptom:** Import errors or compatibility warnings on startup.

  **Possible Causes:**
    - Python version outside the supported range (3.12, 3.13, 3.14)
    - PyTorch version other than 2.14

  **Solutions:**
    - Check Python version: ``python --version``
    - Check PyTorch version: ``python -c "import torch; print(torch.__version__)"``
    - See :doc:`/torch/get-started/quickstart-torch-neuron` for supported version combinations

----

PyTorch eager mode troubleshooting
====================================

Issues specific to running TorchNeuron in eager mode, including operator support and
Adaptive Eager Execution behavior.

Unsupported operator error
---------------------------

- **Symptom:** A model runs much slower than expected in eager mode, or raises an error for a
  specific PyTorch operation on the ``neuron`` device.

  **Possible Causes:**
    - The operator is not implemented in the Neuron backend. By default, it falls back to
      CPU, which adds device-to-host copies.

  **Solutions:**
    - List the operators that fell back with ``torch_neuronx.get_fallback_ops()``
    - Check the supported operations reference for the current operator list
    - Set ``TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1`` so that execution errors are
      raised instead of hidden by a fallback
    - Consider implementing the operation as an NKI kernel
      (see :doc:`/nki/index`)

Unexpected numerical results with Adaptive Eager Execution
-----------------------------------------------------------

- **Symptom:** Model output differs slightly from expected results, or differs from the same
  model running on CPU/CUDA.

  **Possible Causes:**
    - Adaptive Eager Execution (off by default) was enabled with
      ``TORCH_NEURONX_ENABLE_CONCATENATION=1``. Fusing operators can change intermediate
      precision in edge cases

  **Solutions:**
    - Unset ``TORCH_NEURONX_ENABLE_CONCATENATION`` and rerun to see whether results match
    - Insert ``print()`` statements on intermediate tensors to create synchronization points
      and narrow down the discrepancy
    - Compare output against CPU execution with the same input to isolate the source

Tensor synchronization and debugging
--------------------------------------

- **Symptom:** Printed tensor values appear stale or debugging is difficult in eager mode.

  **Possible Causes:**
    - Neuron operations execute asynchronously; tensor values may not be available until
      a synchronization point

  **Solutions:**
    - Printing a tensor automatically triggers synchronization — use ``print(tensor)`` to
      force a device-to-host copy
    - Set ``NEURON_LAUNCH_BLOCKING=1`` so each operation runs synchronously and errors point
      at the operation that caused them
    - If you enabled Adaptive Eager Execution, unset ``TORCH_NEURONX_ENABLE_CONCATENATION`` while debugging

----

torch.compile troubleshooting
===============================

Issues specific to using ``torch.compile(backend="neuron")`` on Trainium, including compilation
failures, graph breaks, and caching.

Compilation failure or timeout
-------------------------------

- **Symptom:** Compilation hangs, times out, or raises an error when ``torch.compile`` is invoked.

  **Possible Causes:**
    - Model contains operations not supported by the Neuron backend
    - Model is very large and compilation exceeds available memory

  **Solutions:**
    - Check compilation logs for unsupported operator messages
    - Try running the model in eager mode first to isolate the issue
    - For large models, check available memory with ``neuron-top``

Graph break warnings
---------------------

- **Symptom:** TorchDynamo emits warnings about graph breaks during compilation.

  **Possible Causes:**
    - Certain Python constructs or operations cause TorchDynamo to break the graph
    - Data-dependent control flow (e.g., ``if tensor.item() > 0``)
    - Unsupported operations within the compiled region

  **Solutions:**
    - Review the :doc:`torch.compile backend reference </torch/compile/reference-torch-compile>` for
      supported ``torch.compile`` parameters and known graph-break patterns
    - The Neuron backend targets full-graph compilation; use ``fullgraph=True``
      for best performance. Graph breaks still run correctly but reduce
      optimization because the compiler has less context per segment.
    - Restructure code to minimize graph breaks for better performance
    - See :doc:`/torch/reference/best-practices-pytorch-neuron` for optimization guidance

Slow first run (compilation overhead)
--------------------------------------

- **Symptom:** The first execution of a compiled model is significantly slower than subsequent runs.

  **Possible Causes:**
    - This is expected behavior — the first run includes compilation time
    - The NEFF cache is empty or was invalidated

  **Solutions:**
    - Later runs reuse compiled NEFFs from the on-disk cache (``TORCH_NEURONX_NEFF_CACHE_DIR``,
      default ``/tmp/neff_cache``) and start faster
    - Point the cache at persistent or shared storage, because ``/tmp`` does not survive a reboot
    - Avoid changing model structure or input shapes between runs, which produces new cache keys

Compiling without a Neuron device (compile-only)
-------------------------------------------------

- **Symptom:** You want to compile a model to a NEFF on a host that has no
  attached Trainium device (for example, a build or CI environment), or
  compilation errors out trying to reach the runtime.

  **Possible Causes:**
    - By default, the backend expects a Neuron device to execute the compiled
      graph.

  **Solutions:**
    - Use **compile-only (device-free) mode** to produce a NEFF without
      executing it: set ``TORCH_NEURONX_COMPILE_ONLY=1`` or use
      ``torch_neuronx.compiler.compile_only()``. Both work with
      ``torch.compile(backend="neuron")``. In this mode, ops compile but
      never run, outputs are placeholders, and a Neuron runtime-initialization
      failure is non-fatal.
    - On a host with no Neuron device, also set ``NEURON_RT_FAKE_INSTANCE_TYPE``
      to the target instance type.
    - See :ref:`about-torchneuron-torch-compile` for how compilation and
      execution are separated.

----

TorchTitan troubleshooting
============================

Issues specific to running TorchTitan on Trainium with TorchNeuron.

TorchTitan does not detect Trainium hardware
----------------------------------------------

- **Symptom:** TorchTitan falls back to CPU or raises a device error when launched on a
  Trainium instance.

  **Possible Causes:**
    - ``torch-neuronx`` is not installed or is the older XLA-based version
    - TorchTitan version does not support ``torch.accelerator`` device detection

  **Solutions:**
    - Verify ``torch-neuronx`` (native version) is installed: ``pip show torch-neuronx``
    - Verify Neuron devices are visible: ``neuron-ls``
    - Ensure you are using a TorchTitan version compatible with PyTorch 2.14

torch.compile backend not recognized in TorchTitan config
-----------------------------------------------------------

- **Symptom:** Error when specifying ``backend="neuron"`` in TorchTitan's configuration.

  **Possible Causes:**
    - CUDA-specific parameters are still present in the config
    - TorchTitan config syntax is incorrect

  **Solutions:**
    - Remove CUDA-specific parameters (e.g., ``mode="max-autotune-no-cudagraphs"``)
    - Specify only ``backend="neuron"`` in the torch.compile configuration
    - See :doc:`/training/torchtitan/about-torchtitan-neuron` for configuration guidance

Parallelism configuration mismatch
------------------------------------

- **Symptom:** Training fails or hangs when launching TorchTitan with a specific parallelism
  configuration.

  **Possible Causes:**
    - NeuronCore count does not match the parallelism configuration
    - ``NGPU`` or equivalent setting does not reflect the actual number of NeuronCores

  **Solutions:**
    - On Trainium, the unit of distribution is the NeuronCore, not the GPU
    - Set the rank count to the number of logical NeuronCores. For example, a ``trn2.48xlarge``
      with logical NeuronCore configuration 2 has 16 devices × 4 logical NeuronCores = 64, so
      use ``NGPU=64``. Run ``neuron-ls`` to verify the number of available NeuronCores
    - See :doc:`/torch/guides/torch-distributed` for NeuronCore topology guidance

----

Distributed training troubleshooting
======================================

Issues specific to distributed training with TorchNeuron using FSDP, DDP, DTensor, and
Tensor Parallelism.

NeuronCore allocation failure
------------------------------

- **Symptom:** Error indicating NeuronCores cannot be allocated, or training fails to start
  across the expected number of devices.

  **Possible Causes:**
    - Requested more NeuronCores than available on the instance
    - Another process is using NeuronCores

  **Solutions:**
    - Run ``neuron-ls`` to check available devices
    - Run ``neuron-top`` to check for other processes using NeuronCores
    - Terminate other Neuron processes or reduce the requested NeuronCore count

Collective communication timeout
----------------------------------

- **Symptom:** Training hangs or times out during collective operations (e.g., ``all_reduce``,
  ``all_gather``).

  **Possible Causes:**
    - Network misconfiguration in multi-node setups
    - EFA driver not installed or not functioning
    - Firewall or security group blocking EFA traffic

  **Solutions:**
    - Verify EFA driver installation: ``fi_info -p efa``
    - Check security group rules allow EFA traffic between nodes
    - For single-node issues, verify NeuronCore count matches the distributed configuration
    - Increase the timeout value if the operation is expected to take longer

FSDP configuration issues
---------------------------

- **Symptom:** FSDP initialization fails or produces incorrect sharding behavior.

  **Possible Causes:**
    - Using FSDPv1 or v2 with ``torch.compile`` (SimpleFSDP is recommended for compiled workloads)
    - Sharding configuration does not match NeuronCore topology

  **Solutions:**
    - For ``torch.compile`` workloads, use SimpleFSDP (more compiler-friendly)
    - For eager mode, FSDPv1, v2, and SimpleFSDP are all supported
    - Verify sharding configuration matches available NeuronCores
    - See :doc:`/torch/guides/torch-distributed` for configuration guidance

----

Diagnostic steps
=================

Use these general steps to diagnose TorchNeuron issues:

.. code-block:: bash

   # Check Neuron devices are visible
   neuron-ls

   # Check running Neuron processes and NeuronCore utilization
   neuron-top

   # Verify torch-neuronx installation and version
   pip show torch-neuronx

   # Check PyTorch version
   python -c "import torch; print(torch.__version__)"

   # Verify the Neuron device is available in PyTorch
   python -c "import torch; print(torch.neuron.is_available(), torch.neuron.device_count())"

   # Run a simple tensor operation on Neuron
   python -c "import torch; t = torch.randn(4, device='neuron'); print(t)"

   # Check EFA driver (for multi-node distributed training)
   fi_info -p efa

----

Collecting information for support
====================================

When contacting AWS support or filing an issue, gather the following:

- Output of ``neuron-ls``
- Output of ``pip show torch-neuronx``
- PyTorch version: ``python -c "import torch; print(torch.__version__)"``
- Full error traceback
- Minimal reproducing script if possible
- Instance type and region
- For distributed training: number of nodes, NeuronCores per node, launch command
- For torch.compile: whether the issue reproduces in eager mode

.. include:: /includes/torch/trn3-sizing-note.txt

----

Related topics
===============

- :doc:`/torch/compile/reference-torch-compile` — ``torch.compile`` supported parameters and graph-break details
- :doc:`/torch/guides/torch-distributed` — Distributed training configuration
- :doc:`/training/torchtitan/about-torchtitan-neuron` — TorchTitan Neuron overview
- :doc:`/torch/reference/best-practices-pytorch-neuron` — Optimization best practices
- :doc:`/torch/reference/torchneuron-resources` — Links to issue trackers and community resources
