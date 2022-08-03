.. _neuron-driver-release-notes:

Neuron Driver 2.x Release Notes
===============================

.. contents::
   :local:
   :depth: 1


Known issues
------------

Updated : 04/29/2022

- In rare cases of multi-process applications running under heavy stress a model load failure my occur. This may require reloading of the Neuron Driver as a workaround.

.. _ndriver_2_3_26_0:

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
