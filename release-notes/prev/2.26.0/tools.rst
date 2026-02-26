.. _neuron-2-26-0-tools:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Developer Tools component, version 2.26.0. Release date: 9/18/2025.

AWS Neuron SDK 2.26.0: Developer Tools release notes
====================================================

**Date of release**:  September 18, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.26.0 release notes home <neuron-2-26-0-whatsnew>`

Improvements
------------

View multiple semaphores simultaneously
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Neuron Profiler UI now allows you to select multiple semaphore values to display simultaneously for a more comprehensive view of activity.

``nccom-test`` new State Buffer support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``nccom-test`` support on Trn2 for State Buffer to State Buffer collectives benchmarking for all-reduce, all-gather, and reduce-scatter operations.

Behavioral changes
------------------

* System profile grouping default in Perfetto now uses global NeuronCore ID instead of process local NeuronCore ID for better display of multi-process workloads.
* Added warning when system profile events are dropped due to limited buffer space, and added suggestion of how configure more buffer space if desired.
* ``nccom-test`` will show helpful error message when invalid sizes are used with all-to-all collectives.

Bug fixes
---------

Here's what we fixed in 2.26.0:

* Fixed device memory usage type table and improvement made to stay in sync between runtime and tools versions.
* Fixed system profile crash when processing long-running workloads.
* Fixed display of system profiles in Perfetto to correctly separate rows within the same Logical NeuronCore when using ``NEURON_LOGICAL_NC_CONFIG=2`` on Trn2.

Known issues
------------

* System profile hardware events may be misaligned due to sync point imprecision. In Perfetto, this may cause events to be interleaved.
* System profile events shown in the Neuron Profiler UI for multiprocess workloads are grouped together. Please try the Perfetto output if you encounter this issue.
* Currently, only a Neuron Runtime trace can be shown when capturing a system profile for a PyTorch workload. (Full framework traces can be shown for JAX workloads, though.) We are working to bring PyTorch traces into parity in a future release.

Previous versions
-----------------

* :ref:`neuron-2-25-0-tools`
* :ref:`dev-tools_rn`
