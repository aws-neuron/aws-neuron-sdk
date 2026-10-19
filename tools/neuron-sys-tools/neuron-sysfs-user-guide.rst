.. _neuron-sysfs-ug:

Neuron Sysfs User Guide
=======================

.. contents:: Table of contents
    :local:
    :depth: 3

Introduction
------------
The kernel provides a few ways in which userspace programs can get system information from the kernel space. Sysfs is one common way to do so. It is a virtual filesystem typically mounted on the ``/sys`` directory and contains information about hardware devices attached to the system and about drivers handling those devices. By navigating the hierarchical structure of the sysfs filesystem and viewing the information provided by its files and directories, you can gather valuable information that can help diagnose and resolve a wide range of hardware and system issues.

Thus a sysfs filesystem is set up per Neuron Device under ``/sys/devices/virtual/neuron_device`` to give you an insight into the Neuron Driver and Runtime at system level. By performing several simple CLIs such as reading or writing to a sysfs file, you can get information such as Runtime status, memory usage, Driver info etc. You can even create your own shell scripts to query Runtime and Driver statistics from sysfs and generate customized reports.

This user guide will first explain the Neuron sysfs structure and then introduce many ways where you can perform diagnostic works with Neuron sysfs.


Neuron Sysfs Filesystem Structure
---------------------------------

.. important::

   Neuron Driver 2.31 reorganized the Neuron sysfs tree and moved several nodes to new paths. This change is not backwards compatible: scripts and tools that read a moved node must be updated to use the new path. For the full list of moved nodes, see :ref:`neuron-sysfs-migration`.

High Level Overview
^^^^^^^^^^^^^^^^^^^

Here is the high level structure of the Neuron sysfs filesystem, where the total, present, and peak values of counters are not shown:

.. code-block:: bash

  /sys/devices/virtual/neuron_device/
  ├── neuron0/
  │   ├── subsystem
  │   ├── uevent
  │   ├── core_count
  │   ├── power/
  │   │   ├── async
  │   │   ├── control
  │   │   ├── runtime_active_time
  │   │   ├── runtime_active_kids
  │   │   └── ...
  │   ├── info/
  │   │   ├── connected_devices
  │   │   ├── fw_api_version
  │   │   ├── fw_build
  │   │   ├── serial_number
  │   │   └── architecture/
  │   │       ├── arch_type
  │   │       ├── device_name
  │   │       └── instance_type
  │   ├── stats/
  │   │   ├── hardware/
  │   │   │   ├── power/
  │   │   │   │   ├── utilization
  │   │   │   │   └── utilization_raw
  │   │   │   ├── mem_ecc_corrected
  │   │   │   ├── mem_ecc_uncorrected
  │   │   │   ├── mem_ecc_repairable_uncorrected
  │   │   │   ├── sram_ecc_corrected
  │   │   │   └── sram_ecc_uncorrected
  │   │   └── memory_usage/
  │   │       └── host_mem/
  │   │           ├── application_memory
  │   │           ├── constants
  │   │           ├── dma_buffers
  │   │           ├── dma_rings
  │   │           ├── driver_memory
  │   │           ├── notifications
  │   │           ├── tensors
  │   │           └── uncategorized
  │   ├── health_status/
  │   │   ├── ecc_counters/
  │   │   │   ├── mem_ecc_corrected
  │   │   │   ├── mem_ecc_uncorrected
  │   │   │   ├── mem_ecc_repairable_uncorrected
  │   │   │   ├── sram_ecc_corrected
  │   │   │   └── sram_ecc_uncorrected
  │   │   ├── unhealthy_components
  │   │   ├── healthcheck_heartbeat
  │   │   └── hw_error_event
  │   ├── ctrl/
  │   │   ├── reset
  │   │   └── notify_delay
  │   ├── neuron_core0/
  │   │   ├── info/
  │   │   │   ├── efa_device
  │   │   │   └── architecture/
  │   │   │       └── arch_type
  │   │   └── stats/
  │   │       ├── hardware/
  │   │       │   ├── tensor_engine/
  │   │       │   │   └── pe_cntrs
  │   │       │   └── ham_throttle/
  │   │       │       └── throttle
  │   │       ├── application/
  │   │       │   ├── status/
  │   │       │   │   ├── success
  │   │       │   │   ├── failure
  │   │       │   │   ├── timeout
  │   │       │   │   ├── exec_bad_input
  │   │       │   │   ├── hw_error
  │   │       │   │   ├── execute_completed_with_error
  │   │       │   │   ├── execute_completed_with_num_error
  │   │       │   │   ├── generic_error
  │   │       │   │   ├── resource_error
  │   │       │   │   ├── resource_nc_error
  │   │       │   │   ├── execute_failed_to_queue
  │   │       │   │   ├── invalid_error
  │   │       │   │   ├── unsupported_neff_version
  │   │       │   │   ├── oob_error
  │   │       │   │   ├── hw_collectives_error
  │   │       │   │   ├── hw_hbm_ue_error
  │   │       │   │   ├── hw_nc_ue_error
  │   │       │   │   ├── hw_dma_abort_error
  │   │       │   │   ├── execute_sw_nq_overflow
  │   │       │   │   ├── execute_sw_semaphore_error
  │   │       │   │   ├── execute_sw_event_error
  │   │       │   │   ├── execute_sw_psum_collision
  │   │       │   │   ├── execute_sw_sequencer_fatal
  │   │       │   │   └── hw_repairable_hbm_ue_error
  │   │       │   └── other_info/
  │   │       │       ├── flop_count
  │   │       │       ├── inference_count
  │   │       │       ├── model_load_count
  │   │       │       ├── reset_fail_count
  │   │       │       ├── reset_req_count
  │   │       │       └── nc_time_in_use
  │   │       └── memory_usage/
  │   │           ├── device_mem/
  │   │           │   ├── collectives
  │   │           │   ├── constants
  │   │           │   ├── dma_rings
  │   │           │   ├── driver_memory
  │   │           │   ├── model_code
  │   │           │   ├── model_shared_scratchpad
  │   │           │   ├── nonshared_scratchpad
  │   │           │   ├── notifications
  │   │           │   ├── runtime_memory
  │   │           │   ├── tensors
  │   │           │   └── uncategorized
  │   │           └── host_mem
  │   ├── neuron_core1/
  │   │   ├── info/
  │   │   │   └── ...
  │   │   └── stats/
  │   │       └── ...
  │   └── ...
  ├── neuron1
  ├── neuron2
  ├── neuron3
  └── ...


Each Neuron Device is represented as a directory under ``/sys/devices/virtual/neuron_device/``, where ``neuron0/`` represents the Neuron Device 0, ``neuron1/`` represents the Neuron Device 1, etc. Each NeuronCore is represented as a directory under a Neuron Device directory, represented as ``neuron_core{0,1,2,...}``. Metrics such as Runtime and Driver info and statistics are collected per NeuronCore in two directories under the NeuronCore directory, i.e. ``info/`` and ``stats/``.

Most of the metrics belong to a category called “counter.”
Each counter is represented as a directory, which holds two numerical values as two files: total and present. Each memory usage counter has an additional value called peak.
The total value starts accumulating metrics when the Driver is loaded. The present value records the last changed metric value. The peak value records the max value so far.
Each counter has the same filesystem structure like this:

.. code-block:: text

    /sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/application/status/
    ├── exec_bad_input/
    │   ├── total
    │   └── present
    ├── hw_error/
    │   ├── total
    │   └── present
    ├── execute_failed_to_queue/
    │   ├── total
    │   └── present
    └── ...



Description for Each Field
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron Device fields
""""""""""""""""""""

``info/``: This directory stores information about the Neuron Device hardware and software that does not change with workloads. None of them are counter types.

* ``connected_devices``: The list of connected devices' ids. You should see the same output as neuron-ls's CONNECTED DEVICES.

* ``fw_api_version``: The API version of the Neuron Device firmware.

* ``fw_build``: The build number of the Neuron Device firmware.

* ``serial_number``: The unique device identifier.

* ``architecture/``: This directory stores hardware architecture information.

  * ``arch_type``: The architecture type of the Neuron Device. Sample architecture types are v1, v2, and v3. You can only read the value. You cannot change it.

  * ``instance_type``: The instance type of the Neuron Device. Sample instance types are Inf1, Inf2, and Trn1. You can only read the value. You cannot change it.

  * ``device_name``: The Neuron Device type. Sample Neuron Device types are Inferentia, Inferentia2, and Trainium1. You can only read the value. You cannot change it.

``stats/``: This directory stores Neuron Device statistics. It contains two subdirectories: ``hardware/`` and ``memory_usage/``.

* ``hardware/``: Hardware statistics.

  * ``power/``: Power statistics.

    * ``utilization``: Reports per-minute power usage statistics as a percentage of max power in the following format:

        <status>,<timestamp>,<min_power>,<max_power>,<avg_power>

        **Field descriptions:**

        status
            Indicates the sampling state in a string.  Valid values are:

              ``POWER_STATUS_VALID`` - Sampling successful

              ``POWER_STATUS_NO_DATA`` - No samples available

              ``POWER_STATUS_INVALID`` - An internal sampling error occurred

        timestamp
            Time when the sample was collected in Unix epoch seconds (integer)

        min_power
            Minimum power utilization during the sampling period (0.00-100.00%)

        max_power
            Maximum power utilization during the sampling period (0.00-100.00%)

        avg_power
            Average power utilization during the sampling period (0.00-100.00%)

      The interface updates these statistics every minute based on continuous power sampling.

    * ``utilization_raw``: Reports the raw power data that is used to compute ``utilization``.

  * ``mem_ecc_corrected``, ``mem_ecc_uncorrected``, ``mem_ecc_repairable_uncorrected``, ``sram_ecc_corrected``, and ``sram_ecc_uncorrected``: ECC counters. These are retained in this location for compatibility with existing tools. New tools should read them from ``health_status/ecc_counters/`` instead. These copies will be removed in a future release.

* ``memory_usage/``: This directory contains memory usage statistics for the Neuron Device, represented as counters. In this directory, the total counters indicate the current memory usage, present counters represent the memory allocation or deallocation amount in the previous operation, and peak counters indicate the maximum memory usage observed. These memory breakdown details correspond to the :ref:`Memory Usage Summary <neuron_top_mem_usage>` section displayed in Neuron Monitor.

  * ``host_mem/``: The amount of host memory that Neuron Runtime uses for input and output tensors.

    * The host memory per Neuron Device is further categorized into eight types: ``application_memory/``, ``constants/``, ``dma_buffers/``, ``dma_rings/``, ``driver_memory/``, ``notifications/``, ``tensors/``, ``uncategorized/``.  These categories provide more granular host memory classification compared to :ref:`Host Used Memory <neuron_top_host_mem_usage>` section. Each of these categories has total, present, and peak.

``health_status/``: This directory stores critical health signals for the Neuron Device. The Neuron Driver refreshes these values every 50 ms.

* ``ecc_counters/``: The canonical location for ECC counters.

  * ``mem_ecc_corrected``: The number of corrected ECC events in the Neuron Device's DRAM.

  * ``mem_ecc_uncorrected``: The number of unrepairable uncorrected ECC events in the Neuron Device's DRAM.

  * ``mem_ecc_repairable_uncorrected``: The number of repairable uncorrected ECC events in the Neuron Device's DRAM.

  * ``sram_ecc_corrected``: The number of corrected ECC events in the Neuron Device's SRAM.

  * ``sram_ecc_uncorrected``: The number of uncorrected ECC events in the Neuron Device's SRAM.

* ``unhealthy_components``: A comma-separated list of the device components that the firmware reports as unhealthy, for example ``hbm,link_efa``. The value is empty when all components are healthy, and is ``device_unreachable`` when the device cannot be read. Possible components are ``hbm``, ``sram``, ``firmware_service_api``, ``firmware_service_processor``, ``link_intraserver``, ``link_host``, ``link_efa``, ``link_interserver``, ``link_switch``, and ``axi_fabric``.

* ``healthcheck_heartbeat``: An 8-bit heartbeat counter that the device firmware increments every 0.5 seconds. A heartbeat that stops changing indicates that the firmware is busy (for example, during a firmware update) or that a hardware component the firmware relies on is unhealthy.

* ``hw_error_event``: A counter that increments whenever the device health status changes. You can wait on this node with ``poll()`` to be notified of new failures, then read ``unhealthy_components`` to find out which components are affected. For a sample program, see :ref:`troubleshoot_via_sysfs`.

``ctrl/``: This directory stores nodes where writing to them triggers an action in the Neuron Driver, or nodes that report device configuration set by the Neuron SDK.

* ``reset``: Writing to this file resets the corresponding Neuron Device.

* ``notify_delay``: Controls whether the Neuron Driver notifies user space when a sysfs counter changes. When notifications are enabled, the Neuron Driver calls ``sysfs_notify()`` on the ``total``, ``present``, and ``peak`` files of a counter each time it updates them, so a program can wait for a change with ``poll()`` instead of reading the file in a loop. See :ref:`troubleshoot_via_sysfs` for a sample program. Write ``0`` to enable notifications and ``-1`` to disable them. Reading the file reports the current setting. Despite the node's name, it is an on/off switch rather than a duration. Notifications are disabled by default, and enabling them can impact performance. The setting is global: writing to one Neuron Device's ``ctrl/notify_delay`` enables or disables notifications for every Neuron Device on the instance. Notifications on ``health_status/hw_error_event`` are always issued, regardless of this setting.

Other fields:

* ``core_count``: The number of NeuronCores on the Neuron Device. This node is retained at its current path for compatibility. New tools should read ``/sys/class/neuron_device/platform/num_cores_per_chip`` instead.

NeuronCore fields
"""""""""""""""""

``info/``: This directory stores information about the NeuronCore that does not change with workloads. None of them are counter types.

* ``efa_device``: A symbolic link to the EFA PCI device that is paired with this NeuronCore.

* ``architecture/arch_type``: The architecture type of the NeuronCore, for example ``NCv2``.

``stats/``: This directory stores Neuron Runtime and Driver statistics for the NeuronCore. It contains three subdirectories: ``hardware/``, ``application/``, and ``memory_usage/``.

* ``hardware/``: Hardware activity statistics.

  * ``tensor_engine/pe_cntrs``: Processing-element activity counters for the Tensor Engine.

  * ``ham_throttle/throttle``: HAM throttle status. Available on Trn2 and later.

* ``application/``: Application-level statistics.

  * ``status/``: This directory stores the number of each return status of API calls. As explained in :ref:`The LIBNRT API Return Codes <nrt_api>`, every API call returns an NRT_STATUS value, which represents the return status of that API call. Our sysfs filesystem stores all ``NRT_STATUS`` as subdirectories under the ``status/`` directory. They all have the counter structure. Thus each ``NRT_STATUS`` subdirectory holds two values (total and present) and records the number of times you receive a certain ``NRT_STATUS``. You should see the subdirectories align with what is described in :ref:`The LIBNRT API Return Codes <nrt_api>`.

  * ``other_info/``: This directory contains statistics that are not included by ``status/`` and ``memory_usage/``. Each of them is a counter with total and present values.

    * ``flop_count``: The number of flops. You can use it to calculate the TFLOP/s by ``flop_count`` / time interval

    * ``inference_count``: The number of successful inferences

    * ``model_load_count``:  The number of successful model loads

    * ``reset_fail_count``: The number of failed device resets

    * ``reset_req_count``:  The number of device resets requests

    * ``nc_time_in_use``:  The time interval in microseconds between the start and the end of the current execution on hardware

* ``memory_usage/``: This directory contains memory usage statistics for both device and host, represented as counters. In this directory, the total counters indicate the current memory usage, present counters represent the memory allocation or deallocation amount in the previous operation, and peak counters indicate the maximum memory usage observed. Additionally, this directory provides detailed breakdown statistics for device and host memory usage. These memory breakdown details correspond to the :ref:`Memory Usage Summary <neuron_top_mem_usage>` section displayed in Neuron Monitor.

  * ``device_mem/``: The amount of memory that Neuron Runtime uses for weights, instructions and DMA rings.

    * This device memory per NeuronCore is further categorized into eleven types: ``collectives/``, ``constants/``, ``dma_rings/``, ``driver_memory/``, ``model_code/``, ``model_shared_scratchpad/``, ``nonshared_scratchpad/``, ``notifications/``, ``runtime_memory/``, ``tensors/``, and ``uncategorized/``. Each of these categories has total, present, and peak.

        * ``collectives`` - amount of device memory used for collective communication between workers
        * ``constants`` - amount of device memory used for constants (for applications running training) or weights (for applications running inferences)
        * ``dma_rings`` - amount of device memory used for storing model executable code used for data movements
        * ``driver_memory`` - amount of device memory used by the Neuron Driver
        * ``model_code`` - amount of device memory used for storing model executable code
        * ``model_shared_scratchpad`` - amount of device memory used for the shared model scratchpad, a buffer shared between models on the same Neuron Core used for internal model variables and other auxiliary buffers
        * ``nonshared_scratchpad`` - amount of device memory used for non-shared model scratchpad, a buffer used by a single model for internal model variables and other auxiliary buffers
        * ``notifications`` - amount of device memory used to store instruction level trace information used to profile workloads ran on the device
        * ``runtime_memory`` - amount of device memory used by the Neuron Runtime (outside of the previous categories)
        * ``tensors`` - amount of device memory used for tensors
        * ``uncategorized`` - amount of device memory that does not belong in any other category in this list

  * ``host_mem/``: The amount of host memory that Neuron Runtime uses for this NeuronCore. It has total, present, and peak.


.. _neuron-sysfs-platform:

Platform Information Under ``/sys/class/neuron_device``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting with Neuron Driver 2.31, the Neuron Driver describes the instance's hardware platform through three directories under ``/sys/class/neuron_device/``. Unlike the per-device tree under ``/sys/devices/virtual/neuron_device/``, these nodes describe the system as a whole, so they are not repeated per Neuron Device. All of them are read-only and their values do not change while the Neuron Driver is loaded.

.. code-block:: bash

  /sys/class/neuron_device/
  ├── neuron_soc/
  │   ├── version
  │   ├── sub_version
  │   └── hbm_capacity
  ├── efa/
  │   └── version
  └── platform/
      ├── family
      ├── num_chips
      ├── num_sleds_per_server
      ├── num_chips_per_sled
      ├── num_cores_per_chip
      ├── supported_lnc_degrees
      ├── intra_rack_topology
      ├── intra_rack_switch_type
      ├── inter_rack_topology
      └── inter_rack_switch_type

``neuron_soc/``: Information about the Neuron SoC.

* ``version``: The architecture version of the Neuron SoC, as an integer. For example, ``2`` for Trn1 and Inf2, ``3`` for Trn2, and ``4`` for Trn3.

* ``sub_version``: The silicon revision of the Neuron SoC, as an integer.

* ``hbm_capacity``: The usable HBM capacity per Neuron Device, in GiB.

``efa/``: Information about the Elastic Fabric Adapter (EFA) devices paired with the Neuron Devices.

* ``version``: The EFA device generation, as an integer.

``platform/``: Information about the instance's platform and its interconnect topology.

* ``family``: The platform family. For details, see `Neuron AWS EC2 Platform Architecture <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/index.html#neuron-aws-ec2-platform-architecture>`__.

* ``num_chips``: The number of Neuron Devices in the instance.

* ``num_sleds_per_server``: The number of sleds in the instance. A sled is a physical board that holds a group of the instance's Neuron Devices.

* ``num_chips_per_sled``: The number of Neuron Devices on each sled. ``num_sleds_per_server`` × ``num_chips_per_sled`` equals ``num_chips``.

* ``num_cores_per_chip``: The number of physical NeuronCores on each Neuron Device. This is the same value as ``neuron{N}/core_count`` in the per-device tree.

* ``supported_lnc_degrees``: The :ref:`logical NeuronCore (LNC) <logical-neuroncore-config>` configurations that the platform supports, as a space-separated list of integers. For example, ``1 2`` means the platform supports both ``LNC=1`` and ``LNC=2``.

* ``intra_rack_topology``: How the Neuron Devices within a rack are connected. Possible values are ``none``, ``ring``, ``torus``, and ``switch``.

* ``intra_rack_switch_type``: The switch generation used for intra-rack connections, when ``intra_rack_topology`` is ``switch``. Possible values are ``none``, ``v1``, and ``v2``.

* ``inter_rack_topology``: How racks are connected to each other. Possible values are ``none`` and ``p2p``.

* ``inter_rack_switch_type``: The switch generation used for inter-rack connections. Possible values are ``none``, ``v1``, and ``v2``.

For example, on a Trn2 UltraServer:

.. code-block:: bash

  ubuntu@ip-xxx-xx-xx-xxx:~$ cat /sys/class/neuron_device/platform/family
  trn2_ultraserver
  ubuntu@ip-xxx-xx-xx-xxx:~$ cat /sys/class/neuron_device/platform/num_cores_per_chip
  8
  ubuntu@ip-xxx-xx-xx-xxx:~$ cat /sys/class/neuron_device/neuron_soc/version
  3


.. _neuron-sysfs-migration:

Migrating from Neuron Driver 2.30 and Earlier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron Driver 2.31 moved the following nodes. Paths are relative to ``/sys/devices/virtual/neuron_device/``. Update any scripts or tools that read these nodes.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Node(s)
     - Previous path
     - New path
   * - ``connected_devices``, ``fw_api_version``, ``fw_build``
     - ``neuron{N}/``
     - ``neuron{N}/info/``
   * - ``reset``
     - ``neuron{N}/reset``
     - ``neuron{N}/ctrl/reset``
   * - ``notify_delay``
     - ``neuron{N}/info/notify_delay``
     - ``neuron{N}/ctrl/notify_delay``
   * - ``power/``
     - ``neuron{N}/stats/power/``
     - ``neuron{N}/stats/hardware/power/``
   * - ``unhealthy_components``, ``healthcheck_heartbeat``, ``hw_error_event``
     - ``neuron{N}/stats/hardware/health_status/``
     - ``neuron{N}/health_status/``
   * - ECC counters (copied, not moved)
     - ``neuron{N}/stats/hardware/``
     - ``neuron{N}/health_status/ecc_counters/``
   * - ``status/``, ``other_info/``
     - ``neuron{N}/neuron_core{M}/stats/``
     - ``neuron{N}/neuron_core{M}/stats/application/``
   * - ``tensor_engine/``
     - ``neuron{N}/neuron_core{M}/stats/``
     - ``neuron{N}/neuron_core{M}/stats/hardware/``

Compatibility notes:

* The ECC counters are available in two locations. ``health_status/ecc_counters/`` is the canonical location. The copies under ``stats/hardware/`` are retained for existing tools and will be removed in a future release.
* ``core_count`` remains at ``neuron{N}/core_count``. New tools should read ``/sys/class/neuron_device/platform/num_cores_per_chip`` instead.
* All other moved nodes are available only at their new paths.


Read and Write to Sysfs
^^^^^^^^^^^^^^^^^^^^^^^^^

Reading a sysfs file gives the value for the corresponding metric. You can use the cat command to view the contents of the sysfs files.:

.. code-block:: bash

  ubuntu@ip-xxx-xx-xx-xxx:~$ sudo cat /sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/application/status/failure/total
  0
  ubuntu@ip-xxx-xx-xx-xxx:~$ sudo cat /sys/devices/virtual/neuron_device/neuron0/neuron_core0/info/architecture/arch_type
  NCv2

Sysfs metrics of counter type are write to clear. You can write any value to the file, and the metric will be set to 0:

.. code-block:: bash

  ubuntu@ip-xxx-xx-xx-xxx:~$ echo 1 | sudo tee /sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/application/status/failure/total
  1


Writing to ``ctrl/reset`` resets the corresponding Neuron Device. E.g. the below resets Neuron Device 0:

.. code-block:: bash

  ubuntu@ip-xxx-xx-xx-xxx:~$ echo 1 | sudo tee /sys/devices/virtual/neuron_device/neuron0/ctrl/reset
  1

Note
^^^^

All files under ``/sys/devices/virtual/neuron_device/neuron0/power`` such as ``runtime_active_kids`` or ``runtime_status`` are related to generic device power management. They are not created or controlled by our sysfs metrics. The word ``runtime`` in these files does not refer to Neuron Runtime.

.. _troubleshoot_via_sysfs:

How to Troubleshoot via Sysfs
-----------------------------
You can perform simple and easy tasks to troubleshoot your ML jobs with one or a few CLIs to read or write the sysfs filesystem.
You can do aggregations across all the NeuronCores and all the Neuron Device to get a summarized view using your scripts.


You can also use the Sysfs notification feature to wait passively (without wasting CPU cycles) for changes to the values of Sysfs files. To use this feature, you need to implement a user-space program that calls the poll() function on the Sysfs file that you want to wait on. 
The poll() function has the following signature: ``unsigned int (*poll) (struct file *, struct poll_table_struct *)``.
By default, the Sysfs notification feature is turned off when the driver is loaded. To enable notifications, you can set the value of ``/sys/devices/virtual/neuron_device/neuron0/ctrl/notify_delay`` to 0. To disable notifications, you can set it to -1. Please note that enabling this feature can impact performance.

Here is a sample user space program using poll():

.. code-block:: c

	#include <fcntl.h>
	#include <poll.h>
	#include <unistd.h>
	#include <stdio.h>
	#include <stdlib.h>

	int main(int argc, char * argv[])
	{
		char readbuf[128];
		int attr_fd = -1; 
		struct pollfd pfd;
		int retval = 0;
		ssize_t read_bytes;

		if (argc < 2) {
			fprintf(stderr, "Error: Please specify sysfs file path\n");
			exit(1);
		}   
		attr_fd = open(argv[1], O_RDONLY, 0); 
		if (attr_fd < 0) {
			perror(argv[1]);
			exit(2);
		}   

		read_bytes = read(attr_fd, readbuf, sizeof(readbuf));
		if (read_bytes < 0) {
			perror(argv[1]);
			exit(3);
		}   
		printf("%.*s", (int)read_bytes, readbuf);

		pfd.fd = attr_fd;
		pfd.events = POLLERR | POLLPRI;
		pfd.revents = 0;
		while ((retval = poll(&pfd, 1, 100)) >= 0) {
			if (pfd.revents & (POLLERR | POLLPRI)) {
				pfd.revents = 0;

				lseek(attr_fd, 0, SEEK_SET);
				read_bytes = read(attr_fd, readbuf, sizeof(readbuf));
				if (read_bytes < 0) {
					perror(argv[1]);
					exit(4);
				}
				printf("%.*s", (int)read_bytes, readbuf);
			}
		}
		return 0;
	}


