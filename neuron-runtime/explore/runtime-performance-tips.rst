.. meta::
   :description: Performance optimization tips for AWS Neuron Runtime
   :keywords: AWS Neuron, performance, optimization, runtime, asynchronous execution, NUMA, CPU affinity

.. _runtime-performance-tips:

==========================================
Best Practices: Neuron Runtime Performance
==========================================

This topic provides best practices and performance optimization tips for applications using the AWS Neuron Runtime (NRT). Following these guidelines can help you achieve optimal performance when running workloads on AWS Neuron devices.

Best Practice: Enable asynchronous execution
---------------------------------------------

Background
^^^^^^^^^^

The Neuron runtime's main submission interface, ``nrt_execute()``, is synchronous by default. It's typically required to
enable asynchronous mode to achieve high on-device utilization. The asynchronous interface allows the application's call
to ``nrt_execute()`` to return immediately after preparing and enqueuing a request. A callback can be registered (see
``nrt_register_async_exec_callback()``) to receive completion notifications.

Enabling this feature improves performance by allowing the application thread to proceed with host-side processing,
which creates a pipeline between host and device work in the critical path.

Instructions
^^^^^^^^^^^^^

Enable this feature by setting the following environment variable to the required queue depth::

    NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=<queue-depth>

Additional Notes
^^^^^^^^^^^^^^^^^

* The queue depth can be arbitrarily large. However, each execution submission typically has pre-allocated reserved
  tensors for output buffers, which means limiting the number of requests in the queue is necessary to manage memory
  usage.

Best Practice: Isolate latency-sensitive threads
-------------------------------------------------

Background
^^^^^^^^^^

The proxy thread is a per-neuron-core thread in the runtime that drives network communication over EFA. While this
thread doesn't perform heavy computation, it is latency-sensitive and directly impacts on-device execution. This thread
needs to be isolated for consistent performance.

Instructions
^^^^^^^^^^^^^

Neuron Runtime provides an environment variable that allows you to specify the CPU affinity of proxy threads. This
enables you to isolate a set of CPUs and place the proxy threads on them. Here's a simple way to achieve this::

    NEURON_RT_LOW_LATENCY_TASKS_CPU_AFFINITY=40-47,88-95,136-143,184-191
    taskset --cpu-list 0-39,48-87,96-135,144-183 my_workload.py

Additional Notes
^^^^^^^^^^^^^^^^^

* Using fewer cores than threads for proxy threads naturally results in slightly higher P0 latency. However, isolating
  to a small set of cores is practically preferred because the performance is predictable and consistent, while the
  impact remains negligible.

* The configuration suggested above is specific to the trn2.48xlarge instance. It allocates 32 out of 192 cores to
  latency-sensitive threads, split across 2 NUMA nodes. If you choose a custom configuration, it's important to balance
  the allocated cores across NUMA nodes. Use ``lscpu | grep -i numa`` if needed to check your system's NUMA topology.

* The approach above provides a simple baseline configuration. If your application involves multiple processes, you'll
  want to adjust their affinities away from critical path threads, as demonstrated using taskset above. You can also
  enable system-wide isolation using the kernel parameter
  `isolcpus <https://wiki.linuxfoundation.org/realtime/documentation/howto/tools/cpu-partitioning/isolcpus>`_.

Understanding Neuron Runtime CPU Usage
---------------------------------------

During typical operation, there can be many polling threads in the runtime. For example, in a trn2.48xlarge instance
used with an ``lnc=1`` configuration, there will be 128 threads polling for execution completions. Additionally, the
application can perform three operations in parallel per core: read, write, and execute. Between NRT and upper layers
(PJRT, etc.), this is handled with three different threads that busy-loop while polling for the completion of these
events. This results in a total of 384 threads.

This activity appears as busy CPUs but is typically harmless for the following reasons:

1. **Thread yielding**: The threads simply poll and yield, so other threads on the system will not be starved of CPU
   resources.

2. **Non-blocking execution**: These threads do not block on-device executions. Since the execution queue is managed on
   the device, as long as there are queued executions, no performance impact should be observed from host jitter.

Best Practice: Respect the NUMA node layout
--------------------------------------------

Background
^^^^^^^^^^

Each Neuron Device is connected to a specific NUMA node on the host instance. Data movements between host <-> device are
affected by the NUMA node layout.

Instructions
^^^^^^^^^^^^^^

While the Neuron Runtime internally takes the NUMA layout into account, configuring application threads to respect the
NUMA node layout may also lead to performance benefits. As a general rule of thumb, threads that interact with a specific
Neuron Core might see latency improvements if the CPU affinity for that thread places it on the same NUMA Node as the
Neuron Core it interacts with. The NUMA node layout can be obtained from the ``neuron-ls`` tool and is also listed below.

Layout
^^^^^^

trn1.32xlarge
"""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 10 10 12 10 15 15 15 8

   * - NEURON DEVICE
     - NEURON CORES
     - NEURON CORE IDS
     - NEURON MEMORY
     - CONNECTED DEVICES
     - PCI BDF
     - CPU AFFINITY
     - NUMA NODE
   * - 0
     - 2
     - 0-1
     - 32 GB
     - 12, 3, 4, 1
     - 0000:10:1c.0
     - 0-31,64-95
     - 0
   * - 1
     - 2
     - 2-3
     - 32 GB
     - 13, 0, 5, 2
     - 0000:10:1d.0
     - 0-31,64-95
     - 0
   * - 2
     - 2
     - 4-5
     - 32 GB
     - 14, 1, 6, 3
     - 0000:a0:1c.0
     - 32-63,96-127
     - 1
   * - 3
     - 2
     - 6-7
     - 32 GB
     - 15, 2, 7, 0
     - 0000:a0:1d.0
     - 32-63,96-127
     - 1
   * - 4
     - 2
     - 8-9
     - 32 GB
     - 0, 7, 8, 5
     - 0000:20:1b.0
     - 0-31,64-95
     - 0
   * - 5
     - 2
     - 10-11
     - 32 GB
     - 1, 4, 9, 6
     - 0000:20:1c.0
     - 0-31,64-95
     - 0
   * - 6
     - 2
     - 12-13
     - 32 GB
     - 2, 5, 10, 7
     - 0000:90:1b.0
     - 32-63,96-127
     - 1
   * - 7
     - 2
     - 14-15
     - 32 GB
     - 3, 6, 11, 4
     - 0000:90:1c.0
     - 32-63,96-127
     - 1
   * - 8
     - 2
     - 16-17
     - 32 GB
     - 4, 11, 12, 9
     - 0000:20:1d.0
     - 0-31,64-95
     - 0
   * - 9
     - 2
     - 18-19
     - 32 GB
     - 5, 8, 13, 10
     - 0000:20:1e.0
     - 0-31,64-95
     - 0
   * - 10
     - 2
     - 20-21
     - 32 GB
     - 6, 9, 14, 11
     - 0000:90:1d.0
     - 32-63,96-127
     - 1
   * - 11
     - 2
     - 22-23
     - 32 GB
     - 7, 10, 15, 8
     - 0000:90:1e.0
     - 32-63,96-127
     - 1
   * - 12
     - 2
     - 24-25
     - 32 GB
     - 8, 15, 0, 13
     - 0000:10:1e.0
     - 0-31,64-95
     - 0
   * - 13
     - 2
     - 26-27
     - 32 GB
     - 9, 12, 1, 14
     - 0000:10:1b.0
     - 0-31,64-95
     - 0
   * - 14
     - 2
     - 28-29
     - 32 GB
     - 10, 13, 2, 15
     - 0000:a0:1e.0
     - 32-63,96-127
     - 1
   * - 15
     - 2
     - 30-31
     - 32 GB
     - 11, 14, 3, 12
     - 0000:a0:1b.0
     - 32-63,96-127
     - 1

inf2.48xlarge
"""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 10 10 12 10 12 15 15 8

   * - NEURON DEVICE
     - NEURON CORES
     - NEURON CORE IDS
     - NEURON MEMORY
     - CONNECTED DEVICES
     - PCI BDF
     - CPU AFFINITY
     - NUMA NODE
   * - 0
     - 2
     - 0-1
     - 32 GB
     - 11, 1
     - 0000:80:1e.0
     - 48-71,144-167
     - 2
   * - 1
     - 2
     - 2-3
     - 32 GB
     - 0, 2
     - 0000:90:1e.0
     - 72-95,168-191
     - 3
   * - 2
     - 2
     - 4-5
     - 32 GB
     - 1, 3
     - 0000:80:1d.0
     - 48-71,144-167
     - 2
   * - 3
     - 2
     - 6-7
     - 32 GB
     - 2, 4
     - 0000:90:1f.0
     - 72-95,168-191
     - 3
   * - 4
     - 2
     - 8-9
     - 32 GB
     - 3, 5
     - 0000:80:1f.0
     - 48-71,144-167
     - 2
   * - 5
     - 2
     - 10-11
     - 32 GB
     - 4, 6
     - 0000:90:1d.0
     - 72-95,168-191
     - 3
   * - 6
     - 2
     - 12-13
     - 32 GB
     - 5, 7
     - 0000:20:1e.0
     - 24-47,120-143
     - 1
   * - 7
     - 2
     - 14-15
     - 32 GB
     - 6, 8
     - 0000:20:1f.0
     - 24-47,120-143
     - 1
   * - 8
     - 2
     - 16-17
     - 32 GB
     - 7, 9
     - 0000:10:1e.0
     - 0-23,96-119
     - 0
   * - 9
     - 2
     - 18-19
     - 32 GB
     - 8, 10
     - 0000:10:1f.0
     - 0-23,96-119
     - 0
   * - 10
     - 2
     - 20-21
     - 32 GB
     - 9, 11
     - 0000:10:1d.0
     - 0-23,96-119
     - 0
   * - 11
     - 2
     - 22-23
     - 32 GB
     - 10, 0
     - 0000:20:1d.0
     - 24-47,120-143
     - 1

trn2.48xlarge
"""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 10 10 12 10 15 15 15 8

   * - NEURON DEVICE
     - NEURON CORES
     - NEURON CORE IDS
     - NEURON MEMORY
     - CONNECTED DEVICES
     - PCI BDF
     - CPU AFFINITY
     - NUMA NODE
   * - 0
     - 4
     - 0-3
     - 96 GB
     - 12, 3, 4, 1
     - 0000:cc:00.0
     - 48-95,144-191
     - 1
   * - 1
     - 4
     - 4-7
     - 96 GB
     - 13, 0, 5, 2
     - 0000:b5:00.0
     - 48-95,144-191
     - 1
   * - 2
     - 4
     - 8-11
     - 96 GB
     - 14, 1, 6, 3
     - 0000:b6:00.0
     - 48-95,144-191
     - 1
   * - 3
     - 4
     - 12-15
     - 96 GB
     - 15, 2, 7, 0
     - 0000:cb:00.0
     - 48-95,144-191
     - 1
   * - 4
     - 4
     - 16-19
     - 96 GB
     - 0, 7, 8, 5
     - 0000:6f:00.0
     - 0-47,96-143
     - 0
   * - 5
     - 4
     - 20-23
     - 96 GB
     - 1, 4, 9, 6
     - 0000:58:00.0
     - 0-47,96-143
     - 0
   * - 6
     - 4
     - 24-27
     - 96 GB
     - 2, 5, 10, 7
     - 0000:59:00.0
     - 0-47,96-143
     - 0
   * - 7
     - 4
     - 28-31
     - 96 GB
     - 3, 6, 11, 4
     - 0000:6e:00.0
     - 0-47,96-143
     - 0
   * - 8
     - 4
     - 32-35
     - 96 GB
     - 4, 11, 12, 9
     - 0000:9b:00.0
     - 0-47,96-143
     - 0
   * - 9
     - 4
     - 36-39
     - 96 GB
     - 5, 8, 13, 10
     - 0000:84:00.0
     - 0-47,96-143
     - 0
   * - 10
     - 4
     - 40-43
     - 96 GB
     - 6, 9, 14, 11
     - 0000:85:00.0
     - 0-47,96-143
     - 0
   * - 11
     - 4
     - 44-47
     - 96 GB
     - 7, 10, 15, 8
     - 0000:9a:00.0
     - 0-47,96-143
     - 0
   * - 12
     - 4
     - 48-51
     - 96 GB
     - 8, 15, 0, 13
     - 0000:f8:00.0
     - 48-95,144-191
     - 1
   * - 13
     - 4
     - 52-55
     - 96 GB
     - 9, 12, 1, 14
     - 0000:e1:00.0
     - 48-95,144-191
     - 1
   * - 14
     - 4
     - 56-59
     - 96 GB
     - 10, 13, 2, 15
     - 0000:e2:00.0
     - 48-95,144-191
     - 1
   * - 15
     - 4
     - 60-63
     - 96 GB
     - 11, 14, 3, 12
     - 0000:f7:00.0
     - 48-95,144-191
     - 1
