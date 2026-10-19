.. meta::
   :description: Reference for the metrics and device health signals Neuron Observer reports, organized by use case, with metric names, units, and sources.
   :keywords: Neuron Observer, metrics, reference, ECC, health, memory, latency, utilization, EFA, faults, Trainium
   :date-modified: 10/15/2026
   :content-type-template: _content-types/reference-general.rst

.. _neuron-observer-metrics-fault-reference:

=================================================
Reference: Neuron Observer metrics and faults
=================================================

Use this page to look up default metric names, meanings, sources, units, and kinds. It also defines the API-only device health states and fault actions.

Overview
--------

Neuron Observer exposes the same metric series through the native Prometheus ``/metrics`` endpoint on the daemon's API port and through OpenTelemetry Protocol (OTLP) push to a collector. Unless a row identifies a specific aggregation, the tables use fully qualified base metric names with their ``neuron`` scope prefixes and omit aggregation suffixes.

Metric naming
-------------

The tables use fully qualified base names such as ``neuron.device.mem_ecc_uncorrected``. The default export configuration appends an aggregation suffix when it aggregates a metric.

Exported OpenTelemetry metrics use ``neuron.{scope}.{name}`` for native exports and ``neuron.{scope}.{name}.{strategy}`` for aggregated exports:

- ``{scope}`` identifies the source: ``device``, ``core``, ``vcpu``, ``efa``, or ``system``.
- ``{name}`` is the metric name.
- ``{strategy}`` is the aggregation applied (for example, ``avg``, ``sum``, or ``p99``).

The default export configuration uses these suffixes:

- Most aggregated device- and core-scoped metrics use ``sum``.
- ``infer_completed`` also uses ``avg``.
- ``utilization`` and ``time_in_use`` use ``sum``, ``avg``, and ``max``.
- Node-level latency aggregations use ``avg``, ``p50``, and ``p99``.
- Per-vCPU percentage metrics use ``avg``.
- Native system and EFA metrics, per-process latency percentiles, and device-health gauges have no aggregation suffix.

Device-health gauges use ``neuron.{name}``.

The ``/metrics`` endpoint renders the same series under the Prometheus data model: characters outside ``[a-zA-Z0-9_:]`` become ``_`` (so ``.`` becomes ``_``), a real unit becomes a name suffix (``By`` becomes ``_bytes``, ``s`` becomes ``_seconds``, ``%`` becomes ``_percent``), and counters get a trailing ``_total``. For example, ``neuron.device.mem_ecc_uncorrected.sum`` renders as ``neuron_device_mem_ecc_uncorrected_sum_total``.

Metric attributes depend on the configured export scope. Device-scoped metrics use ``neuron_device_index``; core-scoped metrics use ``neuron_device_index`` and ``neuroncore``. Process-attributed metrics carry a ``runtime_tag`` attribute (the value of the workload's ``NEURON_PROCESS_TAG``, defaulting to the PID). Every data point carries ``instance_id``.

Device health and suggested actions
-----------------------------------

Device health
~~~~~~~~~~~~~~

Neuron Observer derives a per-device health verdict from the kernel driver's ``unhealthy_components`` sysfs node. The verdict is visible through ``GET /api/v1/devices``, ``GET /api/v1/node/snapshot``, and ``nobs devices``.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - State
     - Meaning
   * - ``healthy``
     - ``unhealthy_components`` is empty after validation against the corresponding uncorrected ECC counters.
   * - ``unhealthy``
     - A non-empty validated token list. ``GET /api/v1/devices`` and ``nobs devices`` report the list as ``unhealthy_components``; ``GET /api/v1/node/snapshot`` reports it as ``components``.
   * - ``unknown``
     - The ``health_status/`` tree exists but ``unhealthy_components`` is missing or unreadable.

The daemon emits two info-style gauges for per-device health:

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - Observer metric name
     - Kind
     - Description
   * - ``neuron.device_health``
     - gauge
     - One row per device with an authoritative verdict, with a ``state`` attribute (``healthy`` or ``unhealthy``), value 1. Devices whose state is ``unknown`` and devices on legacy drivers without the health sysfs tree emit no row.
   * - ``neuron.device_unhealthy_component``
     - gauge
     - One sparse row per (device, failing component) with a ``component`` attribute, value 1. Healthy and unknown devices emit no row.

Observer preserves driver component tokens after validation, except that it removes corrected-only ``hbm`` and ``sram`` tokens when the corresponding uncorrected ECC counters are confirmed as zero. Observer can also report ``healthcheck_heartbeat_stalled`` and ``device_unreachable``. Clients must accept previously unseen component values.

Hardware error metrics
----------------------

Hardware ECC counters (device)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 44 12 10 34

   * - Observer metric name
     - Kind
     - Unit
     - Source
   * - ``neuron.device.mem_ecc_corrected``
     - counter
     - errors
     - sysfs ``stats/hardware/mem_ecc_corrected``
   * - ``neuron.device.mem_ecc_corrected_delta``
     - gauge
     - errors
     - Derived: new corrected DRAM errors since the previous collection cycle.
   * - ``neuron.device.mem_ecc_uncorrected``
     - counter
     - errors
     - Derived: unrepairable + repairable.
   * - ``neuron.device.mem_ecc_unrepairable_uncorrected``
     - counter
     - errors
     - sysfs ``stats/hardware/mem_ecc_uncorrected``
   * - ``neuron.device.mem_ecc_repairable_uncorrected``
     - counter
     - errors
     - sysfs ``stats/hardware/mem_ecc_repairable_uncorrected``
   * - ``neuron.device.sram_ecc_corrected``
     - counter
     - errors
     - sysfs ``stats/hardware/sram_ecc_corrected``
   * - ``neuron.device.sram_ecc_corrected_delta``
     - gauge
     - errors
     - Derived: new corrected SRAM errors since the previous collection cycle.
   * - ``neuron.device.sram_ecc_uncorrected``
     - counter
     - errors
     - sysfs ``stats/hardware/sram_ecc_uncorrected``

The counters are cumulative. The ``*_corrected_delta`` gauges report new corrected errors since the previous collection cycle. Observer does not emit a delta on the first collection cycle, when the source is unreadable, or when the source counter decreases.

Per-NeuronCore hardware and software error counters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These cumulative counters are read from sysfs per core and summed across the node by the default export configuration. The sysfs source paths are relative to ``status/``.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Observer metric name
     - Meaning and source
   * - ``neuron.core.err_generic``
     - Generic errors reported by ``generic_error``.
   * - ``neuron.core.err_numerical``
     - Deprecated numerical-error series. Current runtimes do not write this value, so Observer exports 0.
   * - ``neuron.core.err_model``
     - Deprecated model-error series. Current runtimes do not write this value, so Observer exports 0.
   * - ``neuron.core.err_transient``
     - Deprecated transient-error series. Current runtimes do not write this value, so Observer exports 0.
   * - ``neuron.core.err_hw``
     - Hardware errors reported by ``hw_error``.
   * - ``neuron.core.err_rt``
     - Deprecated runtime-error series. Current runtimes do not write this value, so Observer exports 0.
   * - ``neuron.core.generic_fail``
     - Generic failures reported by ``failure``.
   * - ``neuron.core.err_resource``
     - Resource errors reported by ``resource_error``.
   * - ``neuron.core.err_resource_nc``
     - NeuronCore resource errors reported by ``resource_nc_error``.
   * - ``neuron.core.err_invalid``
     - Invalid operation errors reported by ``invalid_error``.
   * - ``neuron.core.err_unsupported_neff_version``
     - Unsupported NEFF version errors reported by ``unsupported_neff_version``.
   * - ``neuron.core.hw_err_collectives``
     - Collective communication hardware errors reported by ``hw_collectives_error``.
   * - ``neuron.core.hw_err_hbm_ue``
     - Uncorrectable HBM errors reported by ``hw_hbm_ue_error``.
   * - ``neuron.core.hw_err_nc_ue``
     - Uncorrectable NeuronCore errors reported by ``hw_nc_ue_error``.
   * - ``neuron.core.hw_err_dma_abort``
     - DMA abort errors reported by ``hw_dma_abort_error``.
   * - ``neuron.core.hw_err_repairable_hbm_ue``
     - Repairable uncorrectable HBM errors reported by ``hw_repairable_hbm_ue_error``.
   * - ``neuron.core.err_sw_nq_overflow``
     - Notification queue overflows reported by ``execute_sw_nq_overflow``.
   * - ``neuron.core.err_sw_semaphore_error``
     - Semaphore errors reported by ``execute_sw_semaphore_error``.
   * - ``neuron.core.err_sw_event_error``
     - Event errors reported by ``execute_sw_event_error``.
   * - ``neuron.core.err_sw_psum_collision``
     - Partial-sum collisions reported by ``execute_sw_psum_collision``.
   * - ``neuron.core.err_sw_sequencer_fatal``
     - Fatal sequencer errors reported by ``execute_sw_sequencer_fatal``.

NeuronCore resets
~~~~~~~~~~~~~~~~~

Reset request and failure metrics are cumulative counters.

.. list-table::
   :header-rows: 1
   :widths: 42 12 46

   * - Observer metric name
     - Unit
     - Meaning and source
   * - ``neuron.core.reset_req_count``
     - resets
     - NeuronCore reset requests from ``reset_req_count/total``.
   * - ``neuron.core.reset_fail_count``
     - resets
     - Failed NeuronCore reset attempts from ``reset_fail_count/total``.

Memory
------

Host memory (device)
~~~~~~~~~~~~~~~~~~~~~~

Each gauge reports current per-device host-memory use from the driver's counters. The sysfs source paths are relative to ``host_mem/``.

.. list-table::
   :header-rows: 1
   :widths: 50 10 40

   * - Observer metric name
     - Unit
     - Sysfs source (``host_mem/…``)
   * - ``neuron.device.host_mem_dma_buffers``
     - bytes
     - ``dma_buffers/total``
   * - ``neuron.device.host_mem_tensors``
     - bytes
     - ``tensors/total``
   * - ``neuron.device.host_mem_constants``
     - bytes
     - ``constants/total``
   * - ``neuron.device.host_mem_application_memory``
     - bytes
     - ``application_memory/total``
   * - ``neuron.device.host_mem_driver_memory``
     - bytes
     - ``driver_memory/total``
   * - ``neuron.device.host_mem_notifications``
     - bytes
     - ``notifications/total``
   * - ``neuron.device.host_mem_dma_rings``
     - bytes
     - ``dma_rings/total``
   * - ``neuron.device.host_mem_uncategorized``
     - bytes
     - ``uncategorized/total``

Device memory (core)
~~~~~~~~~~~~~~~~~~~~~~

Each gauge reports current per-NeuronCore device-memory use from the driver's counters. The sysfs source paths are relative to ``device_mem/``.

.. list-table::
   :header-rows: 1
   :widths: 54 10 36

   * - Observer metric name
     - Unit
     - Sysfs source (``device_mem/…``)
   * - ``neuron.core.device_mem_model_code``
     - bytes
     - ``model_code/total``
   * - ``neuron.core.device_mem_tensors``
     - bytes
     - ``tensors/total``
   * - ``neuron.core.device_mem_constants``
     - bytes
     - ``constants/total``
   * - ``neuron.core.device_mem_model_shared_scratchpad``
     - bytes
     - ``model_shared_scratchpad/total``
   * - ``neuron.core.device_mem_nonshared_scratchpad``
     - bytes
     - ``nonshared_scratchpad/total``
   * - ``neuron.core.device_mem_runtime_memory``
     - bytes
     - ``runtime_memory/total``
   * - ``neuron.core.device_mem_driver_memory``
     - bytes
     - ``driver_memory/total``
   * - ``neuron.core.device_mem_collectives``
     - bytes
     - ``collectives/total``
   * - ``neuron.core.device_mem_notifications``
     - bytes
     - ``notifications/total``
   * - ``neuron.core.device_mem_dma_rings``
     - bytes
     - ``dma_rings/total``
   * - ``neuron.core.device_mem_uncategorized``
     - bytes
     - ``uncategorized/total``

Runtime and derived memory metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runtime-attributed metrics carry the ``runtime_tag`` attribute. The default value is the process ID unless the workload sets ``NEURON_PROCESS_TAG``.

.. list-table::
   :header-rows: 1
   :widths: 43 10 47

   * - Observer metric name
     - Unit
     - Meaning and source
   * - ``neuron.device.mem_usage_code_host``
     - bytes
     - Host code memory reported by the runtime device statistics.
   * - ``neuron.device.mem_usage_tensors_host``
     - bytes
     - Host tensor memory reported by the runtime device statistics.
   * - ``neuron.device.mem_usage_constants_host``
     - bytes
     - Host constant memory reported by the runtime device statistics.
   * - ``neuron.device.mem_usage_scratchpad_host``
     - bytes
     - Host scratchpad memory reported by the runtime device statistics.
   * - ``neuron.device.mem_usage_misc_host``
     - bytes
     - Other host memory reported by the runtime device statistics.
   * - ``neuron.device.mem_usage_total_host``
     - bytes
     - Sum of the five runtime host-memory categories, per device attachment.
   * - ``neuron.core.mem_usage_total_device``
     - bytes
     - Sum of the runtime device-memory categories on one NeuronCore.
   * - ``neuron.device.process_rss``
     - bytes
     - Resident memory for the runtime process from ``/proc/<pid>/statm``.
   * - ``neuron.device.device_memory_capacity``
     - bytes
     - Static device capacity by architecture: V3 is 96 GiB and V4 is 144 GiB. This metric does not carry ``runtime_tag``.

Utilization
-----------

NeuronCore activity
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 12 38

   * - Observer metric name
     - Kind
     - Unit
     - Meaning and source
   * - ``neuron.core.utilization``
     - gauge
     - —
     - Per-NeuronCore utilization ratio, calculated as ``delta(time_in_use) / elapsed_wall_clock``. Observer requires two collection cycles. The default configuration exports core aggregations as ``neuron.core.utilization.{strategy}`` and node aggregations as ``neuron.core.utilization_system.{strategy}``.
   * - ``neuron.core.effective_flops``
     - gauge
     - —
     - Floating-point operations per second, calculated as ``delta(flop_count) / elapsed_seconds``. Observer requires two collection cycles.
   * - ``neuron.core.time_in_use``
     - gauge
     - s
     - Total NeuronCore use time from sysfs ``other_info/nc_time_in_use/total``. Observer converts the source value from picoseconds to seconds.
   * - ``neuron.core.flop_count``
     - counter
     - operations
     - Cumulative floating-point operations from sysfs ``other_info/flop_count/total``. The driver converts the runtime MAC count to FLOPs.
   * - ``neuron.core.model_load_count``
     - gauge
     - model_loads
     - Increment-only model load count for one NeuronCore from sysfs ``other_info/model_load_count/total``. The default export sums the values across the node. The Observer schema classifies this metric as a gauge.
   * - ``neuron.core.loaded_model_count``
     - gauge
     - models
     - Model subgraphs currently placed on one NeuronCore, from the runtime model store. The default export sums the values across the node. The metric is 0 when no model is loaded.

Inference and latency
---------------------

Inference counters (core)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sysfs provides cumulative inference counters for each core. The default export configuration publishes ``neuron.core.infer_completed.avg`` as a gauge.

.. list-table::
   :header-rows: 1
   :widths: 42 12 46

   * - Observer metric name
     - Unit
     - Meaning and source
   * - ``neuron.core.infer_completed``
     - inferences
     - Successful inferences from ``status/success``.
   * - ``neuron.core.infer_completed_with_err``
     - inferences
     - Inferences that completed with an error, from ``status/execute_completed_with_error``.
   * - ``neuron.core.infer_completed_with_num_err``
     - inferences
     - Inferences that completed with a numerical error, from ``status/execute_completed_with_num_error``.
   * - ``neuron.core.infer_timed_out``
     - inferences
     - Inference timeouts from ``status/timeout``.
   * - ``neuron.core.infer_incorrect_input``
     - inferences
     - Inferences rejected because of incorrect input, from ``status/exec_bad_input``.
   * - ``neuron.core.infer_failed_to_queue``
     - inferences
     - Inferences that failed to enter the execution queue, from ``status/execute_failed_to_queue``.
   * - ``neuron.core.inference_count``
     - inferences
     - Total inference count from ``other_info/inference_count/total``.

Latency (core, per process)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All latency metrics in this section are gauges in seconds. Per-process metrics carry ``runtime_tag``. Observer samples the runtime latency stream and calculates ``p0``, ``p1``, ``p25``, ``p50``, ``p75``, ``p99``, and ``p100``. Idle collection windows report 0.

.. list-table::
   :header-rows: 1
   :widths: 45 12 43

   * - Observer metric name or pattern
     - Unit
     - Meaning
   * - ``neuron.core.latency_device``
     - s
     - Latest device-side inference latency from the runtime latency stream.
   * - ``neuron.core.latency_total``
     - s
     - Latest total inference latency from the runtime latency stream.
   * - ``neuron.core.latency_device_p{0,1,25,50,75,99,100}``
     - s
     - Device-side latency percentiles for one runtime on one NeuronCore.
   * - ``neuron.core.latency_total_p{0,1,25,50,75,99,100}``
     - s
     - Total latency percentiles for one runtime on one NeuronCore.
   * - ``neuron.core.latency_device_system.{avg,p50,p99}``
     - s
     - Node-level aggregations of ``latency_device``.
   * - ``neuron.core.latency_total_system.{avg,p50,p99}``
     - s
     - Node-level aggregations of ``latency_total``.
   * - ``neuron.core.nc_time.sum``
     - s
     - NeuronCore execution time, summed across the node.
   * - ``neuron.core.cc_time.sum``
     - s
     - Collective communication time, summed across the node.

EFA network counters
--------------------

Neuron Observer exports EFA driver counters at the ``efa`` scope, with one data point for each interface. Each data point carries ``efa_device`` and ``efa_port``. Counters are cumulative since instance launch or the last driver reset. Calculate rates from the raw values.

.. list-table::
   :header-rows: 1
   :widths: 44 12 44

   * - Observer metric name
     - Unit
     - Sysfs counter (``hw_counters/…``)
   * - ``neuron.efa.tx_bytes``
     - bytes
     - ``tx_bytes``
   * - ``neuron.efa.rx_bytes``
     - bytes
     - ``rx_bytes``
   * - ``neuron.efa.tx_pkts``
     - packets
     - ``tx_pkts``
   * - ``neuron.efa.rx_pkts``
     - packets
     - ``rx_pkts``
   * - ``neuron.efa.rx_drops``
     - packets
     - ``rx_drops``
   * - ``neuron.efa.send_bytes``
     - bytes
     - ``send_bytes``
   * - ``neuron.efa.recv_bytes``
     - bytes
     - ``recv_bytes``
   * - ``neuron.efa.send_wrs``
     - operations
     - ``send_wrs``
   * - ``neuron.efa.recv_wrs``
     - operations
     - ``recv_wrs``
   * - ``neuron.efa.rdma_write_wrs``
     - operations
     - ``rdma_write_wrs``
   * - ``neuron.efa.rdma_read_wrs``
     - operations
     - ``rdma_read_wrs``
   * - ``neuron.efa.rdma_write_bytes``
     - bytes
     - ``rdma_write_bytes``
   * - ``neuron.efa.rdma_read_bytes``
     - bytes
     - ``rdma_read_bytes``
   * - ``neuron.efa.rdma_write_wr_err``
     - errors
     - ``rdma_write_wr_err``
   * - ``neuron.efa.rdma_read_wr_err``
     - errors
     - ``rdma_read_wr_err``
   * - ``neuron.efa.rdma_read_resp_bytes``
     - bytes
     - ``rdma_read_resp_bytes``
   * - ``neuron.efa.rdma_write_recv_bytes``
     - bytes
     - ``rdma_write_recv_bytes``
   * - ``neuron.efa.retrans_bytes``
     - bytes
     - ``retrans_bytes`` (Nitro v4+)
   * - ``neuron.efa.retrans_pkts``
     - packets
     - ``retrans_pkts`` (Nitro v4+)
   * - ``neuron.efa.retrans_timeout_events``
     - events
     - ``retrans_timeout_events`` (Nitro v4+)
   * - ``neuron.efa.impaired_remote_conn_events``
     - events
     - ``impaired_remote_conn_events`` (Nitro v4+)
   * - ``neuron.efa.unresponsive_remote_events``
     - events
     - ``unresponsive_remote_events`` (Nitro v4+)

.. note::

   The SRD retransmit counters (``retrans_*``, ``impaired_remote_conn_events``, ``unresponsive_remote_events``) exist only on Nitro v4+ instances; on older instances, they are absent rather than reported as 0.

System and instance metrics
---------------------------

Observer reads host memory and CPU data from ``/proc`` and calculates device and core counts from the devices discovered on the node. The default configuration exports the gauges as ``neuron.system.*``.

.. list-table::
   :header-rows: 1
   :widths: 38 12 50

   * - Observer metric name
     - Unit
     - Meaning and source
   * - ``neuron.system.memory_total_bytes``
     - bytes
     - Host memory reported by ``MemTotal`` in ``/proc/meminfo``.
   * - ``neuron.system.memory_used_bytes``
     - bytes
     - ``MemTotal`` minus ``MemFree`` from ``/proc/meminfo``.
   * - ``neuron.system.swap_total_bytes``
     - bytes
     - Swap space reported by ``SwapTotal`` in ``/proc/meminfo``.
   * - ``neuron.system.swap_used_bytes``
     - bytes
     - ``SwapTotal`` minus ``SwapFree`` from ``/proc/meminfo``.
   * - ``neuron.system.cpu_average_usage``
     - percent
     - Change in busy CPU time divided by the change in total CPU time between collection cycles. Observer emits no value on the first collection cycle.
   * - ``neuron.system.context_switch_count``
     - —
     - Change in the ``ctxt`` value from ``/proc/stat`` since the previous collection cycle. Observer emits no value on the first collection cycle.
   * - ``neuron.system.vcpu_count``
     - —
     - Number of CPU entries in ``/proc/stat``.
   * - ``neuron.system.neuron_device_count``
     - —
     - Number of Neuron devices discovered on the node.
   * - ``neuron.system.physical_neuroncore_count``
     - —
     - Number of physical NeuronCores across the discovered devices.
   * - ``neuron.system.logical_neuroncore_count``
     - —
     - Number of logical NeuronCores, calculated from the physical count and LNC configuration.

Metadata gauges
~~~~~~~~~~~~~~~

The default configuration exports the metadata gauges with value 1. Backends join their attributes to metric data by ``instance_id``.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Observer metric name
     - Attributes
   * - ``neuron.system_info``
     - Instance name and type, Region, Availability Zone, subnet, AMI, and UltraServer information, as set under ``info_metrics.system_info``.
   * - ``neuron.device_info``
     - Device and core architecture, device name, physical and logical NeuronCore counts, serial number, driver version, connected devices, and LNC configuration, as set under ``info_metrics.device_info``. Observer emits one row for each device.

Per-vCPU and per-process CPU utilization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-process CPU usage is read from ``/proc/<pid>/stat`` for each process attached to a device. Observer divides the change in process CPU ticks by the change in total CPU ticks. The default export sums values across the node while retaining ``runtime_tag``. Observer emits 0 for a process on its first collection cycle.

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - Observer metric name
     - Unit
     - Meaning and source
   * - ``neuron.device.vcpu_usage_user``
     - percent
     - User-mode process time from ``utime`` and ``cutime`` as a percentage of total CPU time.
   * - ``neuron.device.vcpu_usage_system``
     - percent
     - Kernel-mode process time from ``stime`` and ``cstime`` as a percentage of total CPU time.

Per-vCPU utilization is calculated from changes between ``/proc/stat`` samples. The default configuration exports an average across all vCPUs and does not export one data point for each vCPU. Observer emits no value on the first collection cycle or for an offline vCPU.

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - Observer metric name
     - Unit
     - Meaning
   * - ``neuron.vcpu.vcpu_user_percent``
     - percent
     - CPU time spent in user mode, excluding niced processes.
   * - ``neuron.vcpu.vcpu_nice_percent``
     - percent
     - CPU time spent in user mode by processes that have an adjusted nice value.
   * - ``neuron.vcpu.vcpu_system_percent``
     - percent
     - CPU time spent in kernel mode.
   * - ``neuron.vcpu.vcpu_idle_percent``
     - percent
     - CPU idle time.
   * - ``neuron.vcpu.vcpu_iowait_percent``
     - percent
     - CPU time spent waiting for I/O.
   * - ``neuron.vcpu.vcpu_irq_percent``
     - percent
     - CPU time spent servicing hardware interrupts.
   * - ``neuron.vcpu.vcpu_softirq_percent``
     - percent
     - CPU time spent servicing software interrupts.

Learn more
----------

- :doc:`/tools/neuron-observer/get-started` — install Neuron Observer and view these metrics.
- :doc:`/tools/neuron-observer/install-configure` — installation and export pipeline options.
- :doc:`/tools/neuron-observer/configuration-reference` — the Helm chart configuration values, their types, and defaults.
- :doc:`/tools/neuron-observer/troubleshooting-faq` — common issues, FAQs, and beta limitations.
