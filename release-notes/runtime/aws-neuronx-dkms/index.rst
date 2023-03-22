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

Neuron Driver release [2.8.4.0]
--------------------------------

Date: 03/24/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Supports both TRN1N and INF2 instance types.
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
