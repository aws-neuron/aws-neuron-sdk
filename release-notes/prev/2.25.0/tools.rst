.. _neuron-2-25-0-tools:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Developer Tools component, version 2.25.0. Release date: 7/31/2025.

AWS Neuron SDK 2.25.0: Developer Tools release notes
====================================================

**Date of release**: July 31, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.25.0 release notes home <neuron-2-25-0-whatsnew>`

Improvements
------------

*Improvements are significant new or improved features and solutions introduced this release of the AWS Neuron SDK. Read on to learn about them!*

neuron-ls now shows NeuronCore IDs and CPU affinity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each Neuron device, ``neuron-ls`` will now show the corresponding NeuronCore IDs as well as CPU and NUMA node affinity in both the text and JSON outputs.
These can be used as reference when setting certain Neuron runtime environment variables such as ``NEURON_RT_VISIBLE_CORES``.
See :ref:`neuron-ls-ug` for an example.

System profiles now show sync point events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

System profiles now show the sync point events that are used to approximate CPU and Neuron device timestamp alignment.
This can be used as a reference point if any inconsistencies are detected between the runtime and hardware trace timestamps.
See :ref:`neuron-profile-system-timestamp-adjustment` for more details.


Behavioral changes
------------------

*Behavioral changes are small, user-facing changes that you may notice after upgrading to this version.*

* Added a summary metric to device profiles for ``total_active_time`` to help determine if the device was unnecessarily idle during execution.
* Removed metrics for defunct processes from Neuron Monitor's Prometheus output to more accurately reflect the current utilization of NeuronCores.
  Only processes that are currently active at the time of reporting will be included in the output.


Bug fixes
---------

*We're always fixing bugs. It's developer's life!* Here's what we fixed in 2.25.0:

* Fixed issue in Neuron Profiler summary metrics where ``dma_active_time`` was larger than expected.
* Fixed type inconsistency for certain event types and attributes in the system profile data that could result in a crash.

Known issues
------------

*Something doesn't work. Check here to find out if we already knew about it. We hope to fix these soon!*

* System profile hardware events may be misaligned due to sync point imprecision.  In Perfetto, this may cause events to be interleaved.
* System profile events shown in the Neuron Profiler UI for multiprocess workloads are grouped together.  Please try the Perfetto output if you encounter this issue.
* Currently, only a Neuron Runtime trace can be shown when capturing a system profile for a PyTorch workload. (Full framework traces can be shown for JAX workloads, though.) We are working to bring PyTorch traces into parity in a future release.