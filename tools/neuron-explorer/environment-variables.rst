.. meta::
    :description: Complete reference of environment variables for Neuron Explorer profiling, capture, and configuration.
    :date-modified: 10/26/2026

.. _neuron-explorer-env-vars:

Neuron Explorer Environment Variables
=====================================

This page documents all environment variables that affect Neuron Explorer
profiling, profile capture, and profile processing. For the full list of Neuron
Runtime variables, see :ref:`nrt-env-profiling` in the Neuron Runtime reference.

.. contents:: On this page
   :local:
   :depth: 2

Environment variables are read at two different times:

* **Compile time.** ``NKI_DEBUG_INFO``, ``XLA_IR_DEBUG``, and ``XLA_HLO_DEBUG`` change what
  the compiler writes into the NEFF. See :ref:`neuron-explorer-env-vars-debug-info`.
* **Run time.** ``NEURON_RT_*`` variables control whether the Neuron Runtime captures a
  profile and what it captures. Set them before the workload starts.

.. _neuron-explorer-env-vars-two-mechanisms:

Which mechanism turns profiling on
----------------------------------

Each capture method turns profiling on in a different way.

.. warning::

   Set ``NEURON_RT_INSPECT_ENABLE=1`` only for environment-variable capture. Don't combine it
   with ``torch.profiler`` or ``jax.profiler.trace``. The two mechanisms conflict.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Capture method
     - What turns profiling on
     - Run-time variables to set
   * - Environment variables (any framework, containers, EKS)
     - ``NEURON_RT_INSPECT_ENABLE=1``
     - ``NEURON_RT_INSPECT_ENABLE=1``. Add ``NEURON_RT_INSPECT_DEVICE_PROFILE=1`` or
       ``NEURON_RT_INSPECT_DEVICE_PROFILE=session`` for device traces, and
       ``NEURON_RT_INSPECT_OUTPUT_DIR=<dir>`` to choose the output directory.
   * - PyTorch (``torch.profiler`` with ``NeuronConfig``)
     - ``torch.profiler.profile``
     - None required. Configure capture through ``NeuronConfig``.
   * - JAX (``jax.profiler.trace``)
     - ``jax.profiler.trace``
     - ``NEURON_RT_INSPECT_DEVICE_PROFILE=1`` for device traces, and
       ``NEURON_RT_INSPECT_OUTPUT_DIR=<dir>``.
   * - ``neuron-explorer capture``
     - The ``capture`` command (device profile by default)
     - None. Configure capture with command-line flags.
   * - ``neuron-explorer inspect``
     - The ``inspect`` command
     - None. Configure capture with command-line flags.

Profiling control
-----------------

These run-time variables enable and configure environment-variable capture. Except where
noted, they have no effect unless ``NEURON_RT_INSPECT_ENABLE=1``.

.. list-table::
   :header-rows: 1
   :widths: 35 45 10 10

   * - Variable
     - Description
     - Default
     - Values
   * - ``NEURON_RT_INSPECT_ENABLE``
     - Master switch for environment-variable capture. Required when you
       capture with environment variables alone.
     - ``0`` (off)
     - ``0``, ``1``
   * - ``NEURON_RT_INSPECT_OUTPUT_DIR``
     - Directory where profile output is written.
     - ``./output``
     - Any valid path
   * - ``NEURON_RT_INSPECT_SYSTEM_PROFILE``
     - Capture system-level runtime events and operations.
     - ``1`` (on when INSPECT_ENABLE=1)
     - ``0``, ``1``
   * - ``NEURON_RT_INSPECT_DEVICE_PROFILE``
     - Capture device-level hardware traces (``.ntff`` files). Without it,
       you get a system profile only. With ``jax.profiler.trace``, set this
       variable without ``NEURON_RT_INSPECT_ENABLE``.

       * ``1`` or ``model``: one NTFF for the first execution of each NEFF,
         per core (synchronous). Best for compiled-graph workloads.
       * ``session``: one NTFF for the entire capture, per core. Use for
         async workloads, continuous traces, or inference serving (for
         example, vLLM).
     - ``0`` (off)
     - ``0``, ``1``/``model``, ``session``
   * - ``NEURON_RT_INSPECT_CPU_UTIL``
     - Add host CPU utilization sampling (``cpu_util.pb``) to the profile.
     - ``0`` (off)
     - ``0``, ``1``
   * - ``NEURON_RT_INSPECT_HOST_MEMORY``
     - Add host memory usage sampling (``host_mem.pb``) to the profile.
     - ``0`` (off)
     - ``0``, ``1``
   * - ``NEURON_RT_INSPECT_DEVICE_PROFILE_NUM_WARMUP``
     - Number of executions to skip before device profiling begins. When set
       to N, the first N executions of each model are skipped and only the
       (N+1)-th execution is profiled. Only supported with model mode
       (``NEURON_RT_INSPECT_DEVICE_PROFILE=1``). Ignored for other modes.
     - ``0``
     - ``0`` – ``4294967295``

**NEURON_RT_INSPECT_DEVICE_PROFILE_NUM_WARMUP usage:**

Use this variable to bypass warmup noise and profile steady-state behavior.
For example, setting ``NEURON_RT_INSPECT_DEVICE_PROFILE_NUM_WARMUP=5`` skips the
first 5 executions of each model and profiles the 6th. Each model's execution
count is tracked independently — if two different models are loaded, their
warmup counters are separate.


.. _neuron-explorer-env-vars-debug-info:

Source code and debug info
--------------------------

Set these variables **before compiling**. They change what the compiler writes into
the NEFF, so recompile after setting them. A NEFF loaded from the compilation cache
doesn't have the debug info.

.. list-table::
   :header-rows: 1
   :widths: 35 50 15

   * - Variable
     - Description
     - Default
   * - ``NEURON_FRAMEWORK_DEBUG``
     - Also writes the compiled NEFF and the HLO module to the working
       directory. Applies to XLA-based workloads. Does not affect what the
       profile contains. Use it to get a NEFF file to pass to
       ``neuron-explorer capture``.
     - ``0`` (off)
   * - ``NKI_DEBUG_INFO``
     - Enables NKI kernel debug info capture. Produces
       ``kernel_debug_info.json`` in the NEFF. This file provides both the
       instruction-to-NKI-source mapping and the kernel-level NKI Call Stack
       shown in Event Details. Without it, the Source Code Viewer, NKI Call
       Stack, and NKI regions show nothing for your kernels. Off by default
       because it increases compile time and memory consumption.
     - ``False``
   * - ``XLA_IR_DEBUG``
     - Enables XLA IR debug info for richer operation names in profiles.
     - ``0`` (off)
   * - ``XLA_HLO_DEBUG``
     - Enables XLA HLO debug info for descriptive layer names in the
       hierarchy view.
     - ``0`` (off)


DGE and DMA configuration
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - Variable
     - Description
     - Default
   * - ``NEURON_RT_ENABLE_DGE_NOTIFICATIONS``
     - Enables Descriptor Generation Engine notifications so DMA transfers are labeled
       with their tensor variable name (instead of ``unknown``).
     - ``0`` (off)
   * - ``NEURON_RT_PROFILE_BUF_H2D_DMA_MB``
     - Device memory, in MB, reserved for recording host-device DMA events
       in device profiles. Events after the buffer fills are dropped. Set to
       ``0`` to turn off host-device DMA capture. See
       :ref:`Host-device DMA tracks <neuron-explorer-host-device-dma>`.
     - ``64``

.. warning::

   Enabling DGE notifications adds overhead and can cause timeouts for very
   large NEFFs. If profiling hangs or times out, disable this flag and
   re-capture. The profile will be complete but DMA variable names will
   show ``unknown``.


Capture filtering (capture-time)
--------------------------------

These reduce memory and file size by filtering events **at capture time**.
Data not captured is permanently lost.

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - Variable
     - Description
     - Default
   * - ``NEURON_RT_INSPECT_EVENT_FILTER_NC``
     - Only capture events from specified NeuronCores.
       Accepts comma-separated indices or ranges.
     - All cores
   * - ``NEURON_RT_INSPECT_EVENT_FILTER_TYPE``
     - Only capture specified event types or categories.
     - All events
   * - ``NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC``
     - Maximum system trace events per NeuronCore before ring buffer
       overwrites oldest. Increase if you see dropped events.
     - 1,000,000

For examples and the C API, see :ref:`Capture-time filtering <capture-filtering>`.


NEFF cache and artifact control
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - Variable
     - Description
     - Default
   * - ``TORCH_NEURONX_NEFF_CACHE_DIR``
     - Directory for the NEFF compilation cache. When set, cached NEFFs
       are automatically available in the profile output directory.
     - System default
   * - ``NEURON_RT_INSPECT_PRECACHE_ENABLE``
     - Enable profiling before graph cache is populated. Useful for
       capturing the first execution of a model.
     - ``0`` (off)


PyTorch-specific
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - Variable
     - Description
     - Default
   * - ``TORCH_NEURONX_ENABLE_HOST_CC``
     - Enable host-mediated collective communication for distributed
       profiling.
     - ``0`` (off)
   * - ``TORCH_NEURONX_ENABLE_ASYNC_NRT``
     - Enable asynchronous Neuron Runtime for overlapping compute and
       communication.
     - ``0`` (off)


Runtime configuration
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - Variable
     - Description
     - Default
   * - ``NEURON_RT_NUM_CORES``
     - Number of NeuronCores to use for the workload.
     - All available
   * - ``NEURON_RT_VIRTUAL_CORE_SIZE``
     - Number of physical NeuronCores per virtual NeuronCore (VNC).
     - 1


.. _neuron-explorer-env-vars-quick-reference:

Quick reference: recommended combinations
-----------------------------------------

**NKI kernel (compile-time only):**

.. code-block:: bash

   # Set BEFORE compiling the kernel. Capturing with neuron-explorer capture
   # afterward needs no run-time variables.
   export NKI_DEBUG_INFO=True
   export XLA_IR_DEBUG=1
   export XLA_HLO_DEBUG=1

For the full workflow, see :doc:`/nki/guides/use-neuron-profile`.

**System profile only (lightweight):**

.. code-block:: bash

   export NEURON_RT_INSPECT_ENABLE=1
   export NEURON_RT_INSPECT_OUTPUT_DIR=./output

**System + device profile (full picture):**

.. code-block:: bash

   export NEURON_RT_INSPECT_ENABLE=1
   export NEURON_RT_INSPECT_DEVICE_PROFILE=1
   export NEURON_RT_INSPECT_OUTPUT_DIR=./output

**Full debug info (system + device + source code + NKI + DMA names):**

.. code-block:: bash

   # Set BEFORE compilation
   export NEURON_FRAMEWORK_DEBUG=1
   export NKI_DEBUG_INFO=True
   export XLA_IR_DEBUG=1
   export XLA_HLO_DEBUG=1

   # Set BEFORE execution
   export NEURON_RT_INSPECT_ENABLE=1
   export NEURON_RT_INSPECT_DEVICE_PROFILE=1
   export NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1
   export NEURON_RT_INSPECT_OUTPUT_DIR=./output

**Inference / vLLM (session-based device profile):**

.. code-block:: bash

   export NEURON_RT_INSPECT_ENABLE=1
   export NEURON_RT_INSPECT_DEVICE_PROFILE=session
   export NEURON_RT_INSPECT_OUTPUT_DIR=./output

**Large distributed workload (filtered to reduce size):**

.. code-block:: bash

   export NEURON_RT_INSPECT_ENABLE=1
   export NEURON_RT_INSPECT_DEVICE_PROFILE=1
   export NEURON_RT_INSPECT_EVENT_FILTER_NC=0,1
   export NEURON_RT_INSPECT_OUTPUT_DIR=./output
