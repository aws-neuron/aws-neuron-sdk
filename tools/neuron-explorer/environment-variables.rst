.. meta::
    :description: Complete reference of environment variables for Neuron Explorer profiling, capture, and configuration.
    :date-modified: 06/25/2026

.. _neuron-explorer-env-vars:

Neuron Explorer Environment Variables
=====================================

This page documents all environment variables that affect Neuron Explorer
profiling, profile capture, and profile processing.

.. contents:: On this page
   :local:
   :depth: 2


Profiling control
-----------------

These variables enable and configure profiling at runtime.

.. list-table::
   :header-rows: 1
   :widths: 35 45 10 10

   * - Variable
     - Description
     - Default
     - Values
   * - ``NEURON_RT_INSPECT_ENABLE``
     - Master switch to enable runtime profiling. Required for all
       environment-variable-based capture.
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
     - Capture device-level hardware traces.
     - ``0`` (off)
     - ``0``, ``1``/``model``, ``session``

**NEURON_RT_INSPECT_DEVICE_PROFILE modes:**

- ``0`` — Disabled (system profile only)
- ``1`` or ``model`` — Captures first execution per NEFF per core (synchronous). Best for compiled-graph workloads.
- ``session`` — Captures all device activity in a single NTFF file. Use for async workloads, continuous traces, or inference serving (e.g., vLLM).


Source code and debug info
--------------------------

These variables must be set **before compilation** to enable source-level
mapping in the profiler.

.. list-table::
   :header-rows: 1
   :widths: 35 50 15

   * - Variable
     - Description
     - Default
   * - ``NEURON_FRAMEWORK_DEBUG``
     - Enables HLO-level stack trace capture. Required for the Source Code
       Viewer to map hardware instructions to Python model code.
     - ``0`` (off)
   * - ``NKI_DEBUG_INFO``
     - Enables NKI kernel source location capture. Produces
       ``kernel_debug_info.json`` in the NEFF for instruction-to-NKI-source
       mapping.
     - ``False``
   * - ``XLA_IR_DEBUG``
     - Enables XLA IR debug info for richer operation names in profiles.
     - ``0`` (off)
   * - ``XLA_HLO_DEBUG``
     - Enables XLA HLO debug info for descriptive layer names in the
       hierarchy view.
     - ``0`` (off)

.. note::

   ``NKI_DEBUG_INFO=True`` is not enabled by default because it increases
   compile time and memory consumption. Only enable when you need NKI
   source code linking in the profiler.

.. warning::

   These variables affect compilation output. You must **recompile** after
   setting them — existing cached NEFFs won't have the debug info.


DGE and DMA configuration
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - Variable
     - Description
     - Default
   * - ``NEURON_RT_ENABLE_DGE_NOTIFICATIONS``
     - Enables Data Gather Engine notifications so DMA transfers are labeled
       with their tensor variable name (instead of ``unknown``).
     - ``0`` (off)

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

**NeuronCore filter examples:**

.. code-block:: bash

   # Only NeuronCore 0
   export NEURON_RT_INSPECT_EVENT_FILTER_NC=0

   # NeuronCores 0, 2, and 4
   export NEURON_RT_INSPECT_EVENT_FILTER_NC=0,2,4

   # Range: NeuronCores 0 through 3
   export NEURON_RT_INSPECT_EVENT_FILTER_NC=0-3

**Event type filter examples:**

.. code-block:: bash

   # Only hardware events
   export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=hardware

   # Only software events
   export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=software

   # Hardware events except cc_exec
   export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=hardware,^cc_exec

   # Specific event types
   export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=model_load,nrt_execute,runtime_execute

**Event type groups:**

- ``hardware``: ``nc_exec_running``, ``cc_running``, ``cc_exec_barrier``, ``numerical_err``, ``nrt_model_switch``, ``timestamp_sync_point``, ``hw_notify``
- ``software``: All other events


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

.. warning::

   Do not set ``NEURON_RT_INSPECT_ENABLE=1`` and use the PyTorch profiling APIs
   (for example, ``torch.profiler``) simultaneously. They conflict. Use one or the other.


JAX-specific
------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - Variable
     - Description
     - Default
   * - ``NEURON_RT_INSPECT_DEVICE_PROFILE``
     - For JAX, set to ``1`` with ``jax.profiler.trace`` for device captures.
       Do **not** combine with ``NEURON_RT_INSPECT_ENABLE=1``.
     - ``0`` (off)

.. warning::

   Do not set both ``NEURON_RT_INSPECT_ENABLE=1`` and use ``jax.profiler.trace``
   simultaneously. They conflict. Use one or the other.


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


Quick reference: recommended combinations
-----------------------------------------

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
