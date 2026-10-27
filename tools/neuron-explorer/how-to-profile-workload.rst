.. meta::
    :description: Learn how to capture system and device profiles from your Neuron workloads using JAX, environment variables, or the CLI.
    :date-modified: 10/26/2026

Capture profiles with Neuron Explorer
======================================

Overview
--------

This guide covers every way to capture profiling data from your Neuron workloads. Once you have a profile, see the :ref:`Getting Started Guide <new-neuron-profiler-setup>` for how to launch Neuron Explorer, upload, and view results.

**What you'll learn:**

* How to capture system and device profiles using JAX, environment variables, or the CLI. For PyTorch, see :doc:`how-to-profile-pytorch`.
* What output files to expect and how to verify a successful capture
* How to filter captures to reduce memory usage and file size

Prerequisites
--------------

* A working model on a Trainium or Inferentia instance
* The Neuron SDK installed (``torch-neuronx`` or ``jax-neuronx``, ``aws-neuronx-tools``)

.. _capture-how-profiling-works:

What happens during profiling
------------------------------

When you profile a workload, the Neuron Runtime instruments your execution and writes trace data to disk:

* **When** each operation started and how long it took
* **Where** it ran (CPU, Neuron Runtime, or NeuronCore hardware)
* **How much** memory and bandwidth was used

.. _capture-profile-types:

Profile types: system vs device
--------------------------------

Neuron Explorer supports two complementary profile types. You can capture one or both
depending on what you need to analyze:

**System profile**
   Software execution data: framework operations, Neuron Runtime API calls, CPU utilization,
   and memory usage. Use system-level profiling to analyze framework overhead, identify CPU
   bottlenecks, and debug runtime issues.

**Device profile**
   Hardware execution data from NeuronCores: compute engine instructions, DMA operations,
   and hardware utilization. Use device-level profiling to analyze hardware performance,
   identify compute or memory bottlenecks, and optimize kernel implementations.

For most initial investigations, start with a system profile. Add device profiling when you
need instruction-level hardware analysis. See the :doc:`System Trace Viewer <overview-system-profiles>`
and :doc:`Device Trace Viewer <overview-device-profiles>` for details on each viewer.

.. _capture-setup:

Setup (all methods)
--------------------

Set these environment variables **before compiling** to get descriptive layer names and source mapping in your profiles:

.. code-block:: bash

   export XLA_IR_DEBUG=1
   export XLA_HLO_DEBUG=1

They are recommended for all capture methods below.

For NKI kernel developers, also set:

.. code-block:: bash

   export NKI_DEBUG_INFO=True

This adds the debug info that the Source Code Viewer and the NKI Call Stack read. Without it, neither shows data for your kernels.

Recompile after setting them. The variables that turn profiling on depend on the capture method and are covered in each section below. For every variable, see :doc:`environment-variables`.

Choose your capture method
----------------------------

Choose the method that matches your framework and use case:

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Method
     - Best for
   * - :doc:`Native PyTorch profiling <how-to-profile-pytorch>`
     - PyTorch models
   * - JAX Profiling
     - JAX models
   * - Environment Variables
     - Any framework, containerized workloads, EKS
   * - CLI (``neuron-explorer capture``)
     - Profiling compiled NEFFs directly, multi-worker distributed jobs, NKI kernels
   * - CLI (``neuron-explorer inspect``)
     - Quick system profiles without code changes

PyTorch profiling
--------------------

PyTorch workloads use ``torch.profiler`` with a ``NeuronConfig`` object. See
:doc:`how-to-profile-pytorch`.

.. _neuron-explorer-jax-profiling:

JAX profiling
-------------

JAX profiling uses the context-managed ``jax.profiler.trace`` API. Set two environment
variables to enable device profile capture:

.. code-block:: python

   import os
   import jax

   os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
   os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "./output"

   with jax.profiler.trace(os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]):
       # Your JAX workload here
       out = my_jax_function(inputs)

Full JAX example (distributed matmul with AllGather)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from functools import partial
   import os
   import jax
   import jax.numpy as jnp
   from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
   from jax.experimental.shard_map import shard_map
   from time import sleep

   os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
   os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "./output"

   jax.config.update("jax_default_prng_impl", "rbg")
   mesh = Mesh(jax.devices(), ('i',))

   def device_put(x, pspec):
       return jax.device_put(x, NamedSharding(mesh, pspec))

   lhs_spec = P('i', None)
   lhs = device_put(jax.random.normal(jax.random.key(0), (128, 128)), lhs_spec)
   rhs_spec = P('i', None)
   rhs = device_put(jax.random.normal(jax.random.key(1), (128, 16)), rhs_spec)

   @jax.jit
   @partial(shard_map, mesh=mesh, in_specs=(lhs_spec, rhs_spec), out_specs=rhs_spec)
   def matmul_allgather(lhs_block, rhs_block):
       rhs = jax.lax.all_gather(rhs_block, 'i', tiled=True)
       return lhs_block @ rhs

   with jax.profiler.trace(os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]):
       out = matmul_allgather(lhs, rhs)
       for i in range(10):
           with jax.profiler.TraceAnnotation("my_label" + str(i)):
               out = matmul_allgather(lhs, rhs)
           sleep(0.001)

   expected = lhs @ rhs
   with jax.default_device(jax.devices('cpu')[0]):
       equal = jnp.allclose(jax.device_get(out), jax.device_get(expected), atol=1e-3, rtol=1e-3)
   print("Tensors are the same") if equal else print("Tensors are different")

Key differences from PyTorch profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* JAX uses ``jax.profiler.trace`` context manager instead of ``torch.profiler.profile``
* Device profiling is controlled via environment variables (``NEURON_RT_INSPECT_DEVICE_PROFILE``),
  not a NeuronConfig object
* ``jax.profiler.TraceAnnotation`` lets you label regions in the timeline (similar to
  ``record_function`` in PyTorch)

.. important::

   Do NOT set ``NEURON_RT_INSPECT_ENABLE=1`` when using ``jax.profiler``. These two mechanisms conflict: use one or the other.

.. _neuron-explorer-non-framework-user-experience:

Environment variable profiling (framework-agnostic)
----------------------------------------------------

Use environment variables when you cannot modify application code in containerized workloads, EKS deployments, or any executable that calls the Neuron Runtime.

.. note::

   When capturing with environment variables, profiling is enabled for the entire lifetime of the application. For finer-grained control over specific code regions, use the PyTorch or JAX APIs instead.

.. _neuron-explorer-core-control-variables:

Basic usage
~~~~~~~~~~~

.. code-block:: bash

   export NEURON_RT_INSPECT_ENABLE=1
   export NEURON_RT_INSPECT_DEVICE_PROFILE=1
   export NEURON_RT_INSPECT_OUTPUT_DIR=./output
   python train.py

``NEURON_RT_INSPECT_ENABLE=1`` turns on profiling and captures a system profile. ``NEURON_RT_INSPECT_DEVICE_PROFILE=1`` adds device traces; use ``session`` instead for async workloads or inference serving. For defaults and the other capture variables, see :doc:`environment-variables`. If your profile is
missing early events, see :ref:`Handling dropped events and partial profiles <neuron-explorer-dropped-events>`.

EKS deployment
~~~~~~~~~~~~~~

Update your deployment YAML to include profiling environment variables. For distributed
workloads, ``NEURON_RT_INSPECT_OUTPUT_DIR`` must point to a shared volume accessible by
all workers.

.. code-block:: yaml

   apiVersion: v1
   kind: Pod
   metadata:
     name: trn1-mlp
   spec:
     restartPolicy: Never
     schedulerName: default-scheduler
     nodeSelector:
       beta.kubernetes.io/instance-type: trn1.32xlarge
     containers:
       - name: trn1-mlp
         env:
           - name: NEURON_RT_INSPECT_ENABLE
             value: "1"
           - name: NEURON_RT_INSPECT_OUTPUT_DIR
             value: "/shared/output"
         command: ['torchrun']
         args:
           - '--nnodes=1'
           - '--nproc_per_node=32'
           - 'train_torchrun.py'
         image: ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:mlp
         imagePullPolicy: IfNotPresent
         resources:
           limits:
             aws.amazon.com/neuron: 16

.. _neuron-explorer-multinode-env-vars:

Multi-node profiling with environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multi-node workloads outside of EKS (for example, ``torchrun`` or custom launcher
scripts), set the same environment variables on all nodes. The runtime saves profile data
organized by instance ID and process ID, so there are no filename collisions when
``NEURON_RT_INSPECT_OUTPUT_DIR`` points to a shared filesystem.

.. code-block:: bash

   # Set on all nodes before launching
   export NEURON_RT_INSPECT_ENABLE=1
   export NEURON_RT_INSPECT_DEVICE_PROFILE=session
   export NEURON_RT_INSPECT_SYSTEM_PROFILE=1
   export NEURON_RT_INSPECT_CPU_UTIL=1
   export NEURON_RT_INSPECT_HOST_MEMORY=1
   export NEURON_RT_INSPECT_OUTPUT_DIR=/shared/profiles/my-run

   # Launch your distributed workload as usual

The runtime writes ``NEFF + NTFF + system-profile .pb`` files into
``NEURON_RT_INSPECT_OUTPUT_DIR``. Setting ``NEURON_RT_INSPECT_DEVICE_PROFILE=session``
is required on PJRT runtimes (PyTorch native, JAX) for the runtime to emit the device
trace (``.ntff``). The ``CPU_UTIL`` and ``HOST_MEMORY`` variables add host-side time
series that keep the system profile complete.

View the combined multi-node profile:

.. code-block:: bash

   neuron-explorer view -d /shared/profiles/my-run

.. _neuron-explorer-capture-cli:

CLI profiling with neuron-explorer capture
-------------------------------------------

The ``neuron-explorer capture`` command takes a compiled NEFF, executes it with profiling
enabled, and saves the results. This is the most direct way to profile a compiled model
without modifying application code.

.. note::

   ``neuron-explorer capture`` replaces the deprecated ``neuron-profile capture`` command.
   The interface is identical — only the executable name changed.

Basic usage
~~~~~~~~~~~

.. code-block:: bash

   neuron-explorer capture -n file.neff -s profile.ntff

.. _neuron-explorer-capture-default-inputs:

When no inputs are provided, Neuron Explorer reads each input name and size from the NEFF,
allocates the corresponding buffers, and fills them with zero bytes before executing. These
buffers have the expected sizes, but the zero-filled input tensors may not be semantically
valid for the model. This may result in execution errors or a profile that does not represent
the real workload. Use ``--ignore-exec-errors`` to continue profiling after execution errors.

To profile with representative inputs, pass ``<NAME> <FILE_PATH>`` pairs as positional
arguments:

.. code-block:: bash

   neuron-explorer capture -n file.neff -s profile.ntff IN1 x.npy IN2 y.npy

By default, ``neuron-explorer capture`` records a device profile (``--capture-type profile``). It
turns on profiling itself, so you don't set any ``NEURON_RT_INSPECT_*`` variables. To record a
runtime-level system trace instead, use ``--capture-type systrace``. The compile-time variables in
:ref:`capture-setup` still apply, because they control what the NEFF contains.

The NEFF you passed to ``-n`` and the NTFF that ``capture`` wrote together form a device profile.
To view it, add both files to the Neuron Explorer server as described in
:ref:`Step 5 of Get Started <neuron-explorer-step5-upload-profile>`.

CLI reference
~~~~~~~~~~~~~

.. code-block:: text

   neuron-explorer capture [parameters] [inputs...]

**Core parameters**

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Flag
     - Description
     - Default
   * - ``-n, --neff``
     - Compiled NEFF file
     - ``file.neff``
   * - ``-s, --session-file``
     - File to store profile session information
     - ``profile.ntff``
   * - ``--num-exec``
     - Number of executions to run
     - ``1``
   * - ``--profile-nth-exec``
     - Profile every Nth execution rather than all executions. Passing 0 disables profiling.
     - ``1``
   * - ``-t, --capture-type``
     - Determines what to capture. ``profile`` collects device-level information. ``systrace`` collects runtime-level information.
     - ``profile``
   * - ``--exec-mode``
     - Execution mode: ``sync`` (blocking nrt_execute), ``async`` (queues all then waits), ``async-wait`` (schedules and waits one at a time)
     - ``sync``
   * - ``--ignore-exec-errors``
     - Ignore errors during execution
     -
   * - ``--disable-profile``
     - Load and execute a NEFF without capturing a profile. Useful for running a minimal execution when profiling-related runtime calls fail due to insufficient device memory.
     -
   * - ``--enable-dge-notifs``
     - Enable DGE notifications in Runtime for tracing dynamic DMAs
     -

**I/O parameters**

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Flag
     - Description
     - Default
   * - ``inputs`` (positional)
     - List of inputs as ``<NAME> <FILE_PATH>`` pairs separated by space (for example, ``IN1 x.npy IN2 y.npy``)
     -
   * - ``--single-io``
     - Create a single tensor for all I/O equaling the size of the largest I/O tensor. Individual tensors point to a slice of this tensor. Lowers device memory usage, but outputs are not correct.
     -
   * - ``--single-io-threshold``
     - When using ``--single-io``, give any I/O tensor whose size in bytes is at or below this value its own buffer instead of aliasing it onto the shared buffer. Useful to keep small index tensors correct. ``0`` aliases all tensors.
     - ``0``
   * - ``--save-output``
     - Save raw tensor outputs to a plain binary file (filename matches the output name)
     -
   * - ``--save-output-npy``
     - Save tensor outputs to a ``.npy`` file (filename matches the output name)
     -
   * - ``--check-output-determinism``
     - Check output determinism using CRC32 checksums. Compares CRC of all executions against the first and fails if any mismatch is detected.
     -

**Multi-worker parameters (collectives)**

These parameters enable profiling NEFFs that use collective communication across
multiple NeuronCores, NeuronDevices, or nodes. Providing ``--collectives-worker-count``
or ``--collectives-workers-per-node`` automatically enables collectives mode.

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Flag
     - Description
     - Default
   * - ``-r, --collectives-workers-per-node``
     - The number of workers on the current node. The global worker ID (rank) of worker *n* on the current node is ``collectives-worker-start-id + n``.
     -
   * - ``--collectives-worker-count``
     - Total number of Neuron workers across all nodes for this collectives run
     -
   * - ``--collectives-worker-start-id``
     - The rank offset for the first worker on the current node. For example, if node 0 has workers 0,1 and node 1 has workers 2,3 then the value for node 0 is 0 and for node 1 is 2.
     - ``0``
   * - ``-i, --collectives-profile-id``
     - Worker ID to profile. Pass ``all`` to profile all workers.
     - ``all``
   * - ``-m, --multi-input``
     - Path to a file that describes the input list for each requested worker. Each line corresponds to one worker and follows the same ``<NAME> <FILE_PATH>`` format as the positional inputs. Cannot be used together with positional inputs.
     -

**Multi-worker output files**

When profiling multiple workers, the session file is renamed with a ``_rank_{id}`` suffix
for each worker. For example, if ``-s output/profile.ntff`` is used with
``--collectives-workers-per-node 4``, the output files are:

.. code-block:: text

   output/profile_rank_0.ntff
   output/profile_rank_1.ntff
   output/profile_rank_2.ntff
   output/profile_rank_3.ntff

To profile only a specific worker, pass its rank to ``--collectives-profile-id``:

.. code-block:: bash

   neuron-explorer capture -n file.neff -r 4 -i 1 -s output/profile.ntff
   # Output: output/profile_rank_1.ntff

**Multi-node usage**

For multi-node jobs, run ``neuron-explorer capture`` on each node. Use
``--collectives-worker-start-id`` to offset the rank for each node and
``--collectives-worker-count`` to specify the total number of workers across all nodes:

.. code-block:: bash

   # On node 0 (workers 0, 1)
   neuron-explorer capture -n file.neff \
       --collectives-worker-start-id 0 \
       --collectives-workers-per-node 2 \
       --collectives-worker-count 4 \
       -s output/profile.ntff

   # On node 1 (workers 2, 3)
   neuron-explorer capture -n file.neff \
       --collectives-worker-start-id 2 \
       --collectives-workers-per-node 2 \
       --collectives-worker-count 4 \
       -s output/profile.ntff

**Systrace parameters** (used when ``--capture-type systrace``)

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Flag
     - Description
   * - ``--systrace-get-event-types``
     - Query runtime for list of available event types and exit
   * - ``--systrace-event-filter``
     - Events to enable in the system trace. All events are enabled if no filters are specified. Can be passed multiple times.

.. tip::

   For NKI kernel profiling workflows — including source code linking, hierarchy navigation,
   and instruction-level analysis — see :doc:`/nki/guides/use-neuron-profile`.

.. _neuron-explorer-inspect-cli:

CLI profiling with neuron-explorer inspect
------------------------------------------

The ``neuron-explorer inspect`` command wraps your workload and captures a system profile
without any code changes. It provides the same functionality as environment variables but
with argument validation and a ``--help`` command.

.. code-block:: bash

   neuron-explorer inspect -o ./inspect-output nccom-test allg -b 512kb -e 512kb -r 32 -n 10 -d fp32 -w 1 -f 512

.. note::

   If using a trn1.2xlarge instance, change ``-r 32`` to ``-r 2`` to use fewer NeuronCores.

CLI reference
~~~~~~~~~~~~~

.. code-block:: text

   Usage:
     neuron-explorer [OPTIONS] inspect [inspect-OPTIONS] [userscript...]

   Application Options:
     -v, --version               Show version and exit

   Help Options:
     -h, --help                  Show this help message

   [inspect command options]
         -o, --output-dir=       Output directory for inspection results (default: .)
         -n, --num-trace-events= Maximum number of trace events before old ones are dropped

   [inspect command arguments]
     userscript:                 Command that launches a Neuron workload

.. _neuron-explorer-inspect-output:

neuron-explorer inspect output example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The command traces a Neuron workload and saves output to the specified directory. You'll
see NEFF files, device profiles (NTFF) for all NeuronCores that executed that NEFF, and
system profile data:

.. code-block:: bash

   ubuntu@ip-172-31-63-210:~$ tree inspect-output-nccom-test
   inspect-output-nccom-test
   ├── i-012590440bb9fd263_pid_98399
   │   ├── 14382885777943380728_instid_0_vnc_0.ntff
   │   ├── 14382885777943380728_instid_0_vnc_1.ntff
   │   ├── 14382885777943380728_instid_0_vnc_10.ntff
   │   ├── 14382885777943380728_instid_0_vnc_11.ntff
   ...
   │   ├── 14382885777943380728_instid_0_vnc_8.ntff
   │   ├── 14382885777943380728_instid_0_vnc_9.ntff
   │   ├── cpu_util.pb
   │   ├── host_mem.pb
   │   ├── neff_14382885777943380728.neff
   │   ├── ntrace.pb
   │   └── trace_info.pb
   └──

   2 directories, 74 files

To view a summary of the captured profile data:

.. code-block:: bash

   neuron-explorer view -d inspect-output-nccom-test --output-format summary-text


.. _neuron-explorer-filtering-system-profiles:

.. _neuron-explorer-verify-capture:

Verifying your capture
----------------------

After profiling, check that your output directory is not empty and contains the expected files:

.. code-block:: bash

   ls -la ./profile_output/

Common signs of a failed capture:

* **Empty output directory** — profiling wasn't enabled or the workload didn't execute on Neuron.
* **NTFF files are 0 bytes** — device profiling was enabled but no execution was captured (likely a warm-up issue).
* **No .neff files** — for PyTorch, see :ref:`neuron-explorer-pytorch-neffs`.
* **Only .pb files, no .ntff** — device profiling wasn't enabled; you captured a system profile only (may be intentional).

For troubleshooting help, see :doc:`Troubleshooting & FAQs </tools/neuron-explorer/troubleshooting>`.

Filtering
----------

You can filter at two points: **capture time** (before data is written) or **processing time** (when viewing). Capture-time filtering reduces memory and file size but discards data permanently.

When to use capture-time vs processing-time filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Situation
     - Use
   * - You know exactly which NeuronCores to profile (for example, only core 0)
     - Capture-time filtering saves host memory
   * - You're profiling a large distributed workload and running out of host memory
     - Capture-time filtering reduces buffer allocation
   * - You want to explore the same trace from different angles
     - Processing-time filtering preserves all data
   * - You're not sure what you need yet
     - Don't filter at capture time — filter later when viewing

**Rule of thumb:** If in doubt, capture everything. You can always filter when viewing. You can never recover data you didn't capture.

.. _capture-filtering:

Capture-time filtering
^^^^^^^^^^^^^^^^^^^^^^^^

Reduces memory usage by only collecting specific events. Useful for large distributed workloads or when you know exactly which cores to profile.

**NeuronCore filtering:**

.. _neuron-explorer-core-filter-env:

Using environment variables:

.. code-block:: bash

   # Only NeuronCore 0
   export NEURON_RT_INSPECT_EVENT_FILTER_NC=0

   # NeuronCores 0, 2, and 4
   export NEURON_RT_INSPECT_EVENT_FILTER_NC=0,2,4

   # Range: NeuronCores 0 through 3
   export NEURON_RT_INSPECT_EVENT_FILTER_NC=0-3

   # Reset to all cores
   unset NEURON_RT_INSPECT_EVENT_FILTER_NC

Using C API:

.. code-block:: c

   #include <nrt/nrt_sys_trace.h>

   nrt_sys_trace_config_t *config;
   nrt_sys_trace_config_allocate(&config);
   nrt_sys_trace_config_set_defaults(config);

   // Disable all cores, then enable specific ones
   int num_cores = 128;
   for (int i = 0; i < num_cores; i++) {
       nrt_sys_trace_config_set_capture_enabled_for_nc(config, i, false);
   }
   nrt_sys_trace_config_set_capture_enabled_for_nc(config, 0, true);
   nrt_sys_trace_config_set_capture_enabled_for_nc(config, 2, true);

   nrt_sys_trace_start(config);
   // ... your workload ...
   nrt_sys_trace_stop();
   nrt_sys_trace_config_free(config);

**Event type filtering**

Using environment variables:

.. code-block:: bash

   # Specific event types
   export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=model_load,nrt_execute,runtime_execute

   # All hardware events
   export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=hardware

   # All software events
   export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=software

   # Hardware events EXCEPT cc_exec
   export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=hardware,^cc_exec

   # Software events EXCEPT model_load
   export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=software,^model_load

   # Mix categories and specific events
   export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=hardware,tensor_read,tensor_write

   # Reset to all event types
   unset NEURON_RT_INSPECT_EVENT_FILTER_TYPE

Event groups: ``hardware``: ``nc_exec_running``, ``cc_running``, ``cc_exec_barrier``, ``numerical_err``, ``nrt_model_switch``, ``timestamp_sync_point``, ``hw_notify``. ``software``: all other events.

Using C API:

.. code-block:: c

   #include <nrt/nrt_sys_trace.h>

   nrt_sys_trace_config_t *config;
   nrt_sys_trace_config_allocate(&config);
   nrt_sys_trace_config_set_defaults(config);

   // Discover available event types
   const char **event_types = nullptr;
   size_t count = 0;
   nrt_sys_trace_get_event_types(&event_types, &count);

   // Disable all event types first
   for (size_t i = 0; i < count; ++i) {
       nrt_sys_trace_config_set_capture_enabled_for_event_type(config, event_types[i], false);
   }

   // Enable only specific event types
   nrt_sys_trace_config_set_capture_enabled_for_event_type(config, "model_load", true);
   nrt_sys_trace_config_set_capture_enabled_for_event_type(config, "nrt_execute", true);

   // Verify which event types are enabled
   const char **enabled_types = nullptr;
   size_t enabled_count = 0;
   nrt_sys_trace_config_get_enabled_event_types(config, &enabled_types, &enabled_count);
   printf("Enabled event types: %zu\n", enabled_count);
   for (size_t i = 0; i < enabled_count; ++i) {
       printf("  %s\n", enabled_types[i]);
   }

   // Clean up
   for (size_t i = 0; i < enabled_count; ++i) free((void*)enabled_types[i]);
   free((void*)enabled_types);
   for (size_t i = 0; i < count; ++i) free((void*)event_types[i]);
   free((void*)event_types);

   nrt_sys_trace_start(config);
   // ... your workload ...
   nrt_sys_trace_stop();
   nrt_sys_trace_config_free(config);

Processing-time filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Processing-time filtering preserves the complete trace and lets you analyze the same data with different filters without recapturing. Apply these when viewing or exporting profiles.

**Filter by NeuronCore**

.. code-block:: bash

   neuron-explorer view -d ./output --system-trace-filter-neuron-core "0"
   neuron-explorer view -d ./output --system-trace-filter-neuron-core "0,1,2,3"

**Filter by event type**

.. code-block:: bash

   neuron-explorer view -d ./output --system-trace-filter-event-type "nrt_execute"
   neuron-explorer view -d ./output --system-trace-filter-event-type "nrt_execute,nrt_load"

**Filter by instance ID**

.. code-block:: bash

   neuron-explorer view -d ./output --system-trace-filter-instance-id "i-abc123"
   neuron-explorer view -d ./output --system-trace-filter-instance-id "i-abc123,i-def456"

**Skip system or device profiles**

Reduce processing time by skipping one profile type:

.. code-block:: bash

   # Skip device profiles (faster, system-only view)
   neuron-explorer view -d ./output --ignore-device-profile

   # Skip system profiles (device-only view)
   neuron-explorer view -d ./output --ignore-system-profile

These work with ``--output-format parquet`` (default) or ``json``. We intend to end support for the ``json`` output format in Neuron 2.34.

.. _neuron-explorer-dropped-events:

Handling dropped events and partial profiles in system trace
------------------------------------------------------------

On long or high-throughput workloads, the system trace can appear to cover only a short
window near the **end** of the profiled region, leaving earlier events missing. This is a
known limitation, not a data-processing bug.

**Why it happens.** The Neuron Runtime stores system trace events in a fixed-size, per-NeuronCore
ring buffer while your workload runs, and only flushes to disk when profiling stops. If a workload
generates more events than the buffer can hold, the oldest events are silently overwritten, so the
saved profile keeps only the most recent events.

**How to detect it.** When events are dropped, the runtime emits a warning like:

.. code-block:: text

   WARN: System profile events were dropped due to full ring buffers (max <N> dropped on a
   single NC). The profile will have incomplete system trace data.

The capture metadata (``trace_info.pb``) also records whether the system trace overflowed and how
many events were dropped, which Neuron Explorer uses to flag affected profiles.

**How to capture longer profiles.** Increase the per-NeuronCore event buffer so it can hold the full run, then
re-capture. Size it above the dropped-event count reported in the warning (a safe starting point is
roughly the total events reported plus headroom):

* **PyTorch profiling API** — raise ``max_events_per_nc`` in ``NeuronConfig``:

  .. code-block:: python

     neuron_config = NeuronConfig(
         modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
         profile_output_dir="./profile_output",
         max_events_per_nc=1_000_000,  # increase until the drop warning disappears
     )

* **Environment variables / CLI capture** — set the equivalent variable before running:

  .. code-block:: bash

     export NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC=1000000

.. note::

   A larger event buffer increases host memory usage during capture. Increase it only as much as
   needed to clear the drop warning. Profiling fewer active steps (for example, one instead of two)
   also reduces the number of events generated.

For **device profiles** (instruction-level traces), the on-device (HBM) notification buffers can
saturate separately. See :ref:`Profile Buffers <nd-profile-buffers>` in the Neuron Runtime device
memory guide for tuning the ``NEURON_RT_PROFILE_BUF_<buffer type>_MB`` buffers.

Next steps
----------

* :doc:`Get Started with Neuron Explorer <get-started>` — Launch Explorer, upload profiles, and view results
* :doc:`Neuron Explorer Full Documentation <index>` — Complete viewer and feature reference
* :doc:`Profile Parquet Schema Reference <profile-schema-reference>` — Complete profile data reference, matching ``neuron-explorer view --show-profile-schema``

