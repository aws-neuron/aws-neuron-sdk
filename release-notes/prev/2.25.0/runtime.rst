.. _neuron-2-25-0-runtime:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Runtime component, version 2.25.0. Release date: 7/31/2025.

AWS Neuron SDK 2.25.0: Neuron Runtime release notes
===================================================

**Date of release**: July 31, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.25.0 release notes home <neuron-2-25-0-whatsnew>`

Released versions
-----------------

- Neuron Collectives: ``2.27.34.0``
- Neuron Driver: ``2.23.9.0``
- Neuron Runtime Library: ``2.27.23.0``

Behavioral changes
------------------

*Behavioral changes are small, user-facing changes that you may notice after upgrading to this version.*

Neuron Collectives 2.27.34.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Improved the interface with the Neuron Runtime for minor stability improvements.

Neuron Driver 2.23.9.0
^^^^^^^^^^^^^^^^^^^^^^
* Exposed Tensor Engine activity counters in `sysfs`.

Neuron Runtime Library 2.27.23.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Introduced ``nrt_get_vnc_memory_stats`` API to retrieve device memory usage.
* Added support for State-Buffer to State-Buffer collective support for ``all_reduce``, ``reduce_scatter``, and ``all_gather`` for LNC2, which helps reduce HBM memory pressure.
* Added support for coalescing of Collectives operations for internode RDH.
* Introduced a new DGE priority class feature to select preferred packet size for memory transfers.
* Improved ``nrt_init`` time by up to ~3 seconds on AWS Trainium and Inferentia instances.
* Added a warning message along with a recommended scratchpad configuration when a loaded NEFF has non-optimial scratchpad usage.

Breaking changes
----------------

*Sometimes we have to break something now to make the experience better in the longer term. Breaking changes are changes that may require you to update your own code, tools, and configurations.*

Neuron Runtime Library 2.27.23.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Due to a hardware bug that can cause numerical errors to be falsely reported (see the **Known Issues** section below), the runtime has disabled numerical errors by default. Users can re-enable numerical errors by setting ``NEURON_RT_NUMERICAL_ERRORS_VERBOSITY=critical`` or ``NEURON_FAIL_ON_NAN=1`` to enable debug flows and to prevent numerical errors from blowing up a training run.

Bug fixes
---------

*We're always fixing bugs. It's developer's life!* Here's what we fixed in 2.25.0:

Neuron Runtime Library 2.27.23.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Fixed profiling APIs to report execution duration from explicit notifications.
* Fixed race condition which can cause a crash when starting inspect traces.


Known issues
------------

*Something doesn't work. Check here to find out if we already knew about it. We hope to fix these soon!*


* A hardware bug affecting **Trainium** and **Inferentia2** devices causes numerical errors to become "sticky" within the Neuron Core hardware. When a legitimate numerical error occurs during execution, the error state persists in the hardware, causing all subsequent executions to incorrectly report numerical errors even when the computations are valid. This sticky error state can only be resolved by restarting the application to clear the hardware.
