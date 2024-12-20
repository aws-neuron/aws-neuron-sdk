.. _neuron-driver-release-notes:

Neuron Driver Release Notes
===========================

.. contents:: Table of contents
   :local:
   :depth: 1


Known issues
------------

Updated : 04/29/2022

- In rare cases of multi-process applications running under heavy stress a model load failure my occur. This may require reloading of the Neuron Driver as a workaround.

Neuron Driver release [2.19.64.0]
--------------------------------
Date: 12/20/2024


New in this release
^^^^^^^^^^^^^^^^^^^
* Added Trainium2 support

Improvements
^^^^^^^^^^^^
* Optimized HBM Memory allocation to reduce fragmentation. See :ref:`here <small_allocations_mempool>` for more details.

Neuron Driver release [2.18.20.0]
--------------------------------
Date: 11/20/2024

Bug Fixes
^^^^^^^^^
* This release addresses an issue with Neuron Driver that can lead to a user-space application either gaining access to kernel addresses or providing the driver with spoofed memory handles (kernel addresses) that can be potentially used to gain elevated privileges. We would like to thank `Cossack9989 <https://github.com/Cossack9989>`_ for reporting and collaborating on this issue.

Neuron Driver release [2.18.12.0]
--------------------------------

Date: 09/16/2024

New in this release
^^^^^^^^^^^^^^^^^^^
* Introduced a sysfs memory usage counter for DMA rings (:ref:`reference <neuron-sysfs-ug>`)

Bug Fixes
^^^^^^^^^
* Resolved an issue where a memory allocation failure caused a hang due to the memory allocation lock not being released
* Resolved an issue where the driver was allocating more memory than needed for aligned device allocations

Neuron Driver release [2.17.17.0]
--------------------------------

Date: 07/03/2024

New in this release
^^^^^^^^^^^^^^^^^^^
* Improved detection and reporting of DMA errors
* Added more fine grained sysfs metrics to track memory allocation types
* Logging improvements

Bug Fixes
^^^^^^^^^
* Fixed compatibility issues for the Linux 6.3 kernel
* Resolved issue where device reset handling code was not properly checking the failure metric


Neuron Driver release [2.16.7.0]
--------------------------------

Date: 04/01/2024

Bug Fixes
^^^^^^^^^

* Fixed installation issues caused by API changes in Linux 6.3 and 6.4 kernel distributions.
* Fixed an installation build failure when fault-injection is enabled in the kernel.
* Fixed an issue where sysfs total peak memory usage metrics can underflow
* Removed usage of sysfs_emit which is not supported on Linux kernels <= v5.10-rc1


Neuron Driver release [2.15.9.0]
--------------------------------

Date: 12/21/2023

Bug Fixes
^^^^^^^^^

* Release PCIe BAR4 on driver startup failure
* Fix container BDF indexing issues to support relative device ordering used by containers
* Remove incorrect error message in neuron_p2p_unregister_va and harden P2P error checking


Neuron Driver release [2.14.5.0]
--------------------------------

Date: 10/26/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Show uncorrectable SRAM and HBM ECC errors on TRN1 and INF2
* Fixed double free on error path during driver startup


Neuron Driver release [2.13.4.0]
--------------------------------

Date: 9/14/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Added sysfs support for showing connected devices on trn1.32xl, inf2.24xl, and inf2.48xl instances.


Neuron Driver release [2.12.18.0]
--------------------------------

Date: 9/01/2023

Bug Fixes
^^^^^^^^^
* Added fixes required by Neuron K8 components for improving reliability of pod failures (see :ref:`Neuron K8 release notes <neuron-k8-rn>` for more details).
* Added fixes required by Neuron K8 components to support zero-based indexing of Neuron Devices in Kubernetes deployments.


Neuron Driver release [2.12.11.0]
--------------------------------

Date: 8/28/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Added FLOP count to sysfs (flop_count)
* Added connected Neuron Device ids to sysfs (connected_devices)
* Added async DMA copy support
* Suppressed benign timeout/retry messages


Bug Fixes
^^^^^^^^^
* Allocated CC-Core to correct NeuronCore; splitting CC-Cores evenly between NeuronCores.



Neuron Driver release [2.11.9.0]
--------------------------------

Date: 7/19/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Added support for creating batch DMA queues.

Bug Fixes
^^^^^^^^^

* Error message, "ncdev is not NULL", was being printed unnecessarily.  Fixed.
* Fix DMA timeouts during NeuronCore reset of neighboring core caused by incorrect nc_id (NeuronCore ID) assigned to reserved memory


Neuron Driver release [2.10.11.0]
--------------------------------

Date: 6/14/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Added memory usage breakdown by category to the Neuron Sysfs nodes.  New categories are code, misc, tensors, constants, and scratchpad.  Please see the Sysfs page under Neuron Tools for more detailed description of each. 
* Improved NeuronCore initialization (nrt_init) performance by approximately 1 second. 

Bug Fixes
^^^^^^^^^

* Fixed small timing window during NeuronCore resets, which previously would timeout during memcpy
* Removed potential double free of memory when terminating the Neuron Driver.
* Fixed sysfs race condition, which was leading to Neuron Driver crash during termination.


Neuron Driver release [2.9.4.0]
--------------------------------

Date: 05/01/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Added dma_buf support, which is needed for future EFA implementations in the Linux kernel. 
* Added new IOCTL to get Neuron Device BDF (used by Neuron Runtime)
* Added optional support for sysfs notify (off by default). See Neuron Sysfs documentation (under Neuron System Tools) for more details. 


Bug Fixes
^^^^^^^^^

* Fixed max DMA queue size constant to be the correct size - previous incorrect sizing had potential to lead to DMA aborts (execution timeout). 


Neuron Driver release [2.8.4.0]
--------------------------------

Date: 03/28/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Supports both Trn1n and Inf2 instance types.
* Renamed NEURON_ARCH_INFERENTIA=>NEURON_ARCH_V1 and NEURON_ARCH_TRN=>NEURON_ARCH_V2
* Under sysfs nodes, the following changes were made:

  * Changed “infer” metrics to “execute” metrics
  * Added peak memory usage metric
  * Removed empty dynamic metrics directory
  * Removed refresh rate metric
  * Fixed arch type names in sysfs


Bug Fixes
^^^^^^^^^

* Fixed minor memory leak when closing the Neuron Runtime. 
* Fixed memory leaks on error paths in Neuron Driver. 
* Added a workaround to resolve hangs when NeuronCore reset is ran while another core is performing DMA operations. 



Neuron Driver release [2.7.33.0]
--------------------------------

Date: 02/24/2023

Bug Fixes
^^^^^^^^^

* Added a retry mechanism to mitigate possible data copy failures during reset of a NeuronCore.  An info log message will be emitted when this occurs indicating that the retry was attempted.  An example::


   kernel: [726415.485022] neuron:ndma_memcpy_wait_for_completion: DMA completion timeout for UDMA_ENG_33 q0
   kernel: [726415.491744] neuron:ndma_memcpy_offset_move: Failed to copy memory during a NeuronCore reset: nd 0, src 0x100154480000, dst 0x100154500000, size 523264. Retrying the copy.
::


Neuron Driver release [2.7.15.0]
--------------------------------

Date: 02/08/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Added Neuron sysfs metrics under ``/sys/devices/virtual/neuron_device/neuron{0,1, ...}/metrics/``



Neuron Driver release [2.6.26.0]
--------------------------------

Date: 11/07/2022

New in this release
^^^^^^^^^^^^^^^^^^^

* Minor bug fixes and improvements.



Neuron Driver release [2.5.38.0]
--------------------------------

Neuron Driver now supports INF1 and TRN1 EC2 instance types.  Name of the driver package changed from aws-neuron-dkms to aws-neuronx-dkms.  Please remove the older driver package before installing the newest one.

Date: 10/10/2022

New in this release
^^^^^^^^^^^^^^^^^^^

* Support added for EC2 Trn1 instance types and ML training workloads.
* Added missing GPL2 LICENSE file. 
* Changed package name to aws-neuronx-dkms (was previously minus the 'x'). 
* Security Update -- blocked user space access to control registers and DMA control queues intended to be used by the Neuron Driver only.
* Added support for DMA Aborts to avoid hangs.
* Added support for TPB Reset.
* Added sysfs entries for triggering resets and reading core counts.  
* Added write combining on BAR4.  
* Added PCI Device ID update as part of install.
* Added handling for known duplicate device id error.


Bug Fixes
^^^^^^^^^

* Fixed a null pointer free scenario.
* Fixed installation issue related to install without internet connectivity.


Neuron Driver release [2.3.26.0]
--------------------------------

Date: 08/02/2022

Bug Fixes
^^^^^^^^^

- Security Update: Blocked user space access to control registers and DMA control queues intended to be used by the Neuron Driver only.  Recommending upgrade to all customers.


Neuron Driver release [2.3.11.0]
--------------------------------

Date: 05/27/2022

New in this release
^^^^^^^^^^^^^^^^^^^

- This driver is required to support future releases of the Neuron Runtime.  Included in the release is both a bug fix to avoid a kernel crash scenario and an increased compatibility range to ensure compatibility with future versions of Neuron Runtime.

Bug Fixes
^^^^^^^^^

- Correction to huge aligned memory allocation/freeing logic that was previously susceptible to crashes in the kernel.  The crash would bring down the OS.  Recommending upgrade to all customers.



Neuron Driver release [2.3.3.0]
--------------------------------

Date: 04/29/2022

New in this release
^^^^^^^^^^^^^^^^^^^

- Minor performance improvements on inference and loading of models.

Bug Fixes
^^^^^^^^^

- Reduced Host CPU usage when reading ``hw_counters`` metric from neuron-monitor
- Minor bug fixes. 



Neuron Driver release [2.2.14.0]
--------------------------------

Date: 03/25/2022

New in this release
^^^^^^^^^^^^^^^^^^^

- Minor updates


Neuron Driver release [2.2.13.0]
--------------------------------

Date: 01/20/2022

New in this release
^^^^^^^^^^^^^^^^^^^

- Minor updates


Neuron Driver release [2.2.6.0]
-------------------------------

Date: 10/27/2021

New in this release
^^^^^^^^^^^^^^^^^^^

-  Memory improvements made to ensure all allocations are made with 4K
   alignments.


Resolved issues
^^^^^^^^^^^^^^^

-  No longer delays 1s per NeuronDevice when closing Neuron Tools
   applications.
-  Fixes a Ubuntu 20 build issue


Neuron Driver release [2.1]
---------------------------

-  Support is added for Neuron Runtime 2.x (``libnrt.so``).
-  Support for previous releases of Neuron Runtime 1.x is continued with
   Driver 2.x releases.
