.. meta::
    :description: Complete release notes for the Neuron Runtime component across all AWS Neuron SDK versions.
    :keywords: neuron runtime, neuron driver, neuron collectives, release notes, aws neuron sdk
    :date-modified: 05/21/2026

.. _runtime_rn:

Component Release Notes for Neuron Runtime
==========================================

The release notes for the Neuron Runtime Neuron component, including Neuron collectives, the Runtime driver, and the Runtime library. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _runtime-2-30-0-rn:

Neuron Runtime (Neuron 2.30.0 Release)
------------------------------------------------------------------------

**Date of release**: 05/21/2026

Neuron Runtime Library
~~~~~~~~~~~~~~~~~~~~~~

**Version:** 2.32.31.0

Announcement
^^^^^^^^^^^^

* Implicit async execution mode (``NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS``) will be removed in an upcoming release. A warning is now emitted when used. Users should migrate to the explicit async APIs.

New Features
^^^^^^^^^^^^

* Added new :doc:`async event APIs </neuron-runtime/api/nrt_async>` (``nrta_event_register_seq_id_completion`` and ``nrta_event_register_xu_completion``) which allow applications to register user-managed eventfds for completion notifications, simplifying framework integration.

Updates
^^^^^^^^^^^^

* Mesh collective proxy buffers are now reused across NEFFs per-stream, reducing memory pressure loading many models that use collective operations.
* UltraServer topology information is now included in ``nrt_infodump`` diagnostic output.
* Profiling overhead reduced by caching device protobuf setup, with support for pre-populating the cache during warmup.
* NEFF revision ID is now validated against the target device during loading, providing clear error messages for incompatible NEFFs.
* Improved LNC mismatch error messages with actionable guidance.
* Zero-copy for HOST <-> DEVICE data transfers are now enabled by default, improving data transfer performance.
* ``nrt_tensor_read`` and ``nrt_tensor_write`` APIs now use separate DMA queues, eliminating head-of-line blocking between read and write operations improving bidirectional throughput.
* Added support for context caching for Host-based collective communication, significantly reducing prepare-phase latency for repeated collective patterns.

Breaking changes
^^^^^^^^^^^^^^^^

* Support for VIRTUAL tensor type has been removed as part of Inf1 platform deprecation. Applications using virtual tensors should migrate to standard tensor allocation.

Bug Fixes
^^^^^^^^^

* Fixed memory leak in continuous profiling session cleanup where map keys and pending serialization entries were not freed.
* Fixed multiple memory leaks totaling 84+ MB in device driver shutdown and model error paths, including allocator destruction, scratchpad cleanup, and NEFF loader error paths.
* Fixed additional memory leaks in profiling re-entry, ring buffer cleanup, config string parsing, and barrier descriptor preparation.
* Fixed a Hybrid Ring deadlock in one-rank-per-chip configurations when running concurrently with Mesh algorithms in 2D sharding groups.
* Fixed a bug where the NCCL initialization error tracker was cleared after first logging, causing subsequent calls to segfault.
* Fixed ring algorithm selection bug with cached communicators that could cause incorrect algorithm selection in workloads with both intra-node and multi-node replica groups.
* Fixed a double-free bug in DMA ring cleanup that could cause crashes during device teardown.
* Fixed a crash during DMA queue set cleanup when initialization failed mid-allocation.
* Fixed Hybrid Ring data corruption for coalesced operations when disjoint buffers span multiple page-table entries.
* Fixed permute operations on Trn3 Gen2 UltraServer to fail immediately with a clear error when ring algorithm cannot be built, instead of segfaulting.
* Fixed single-channel permute operations having their channel count incorrectly overridden by hybrid ring infrastructure.
* Fixed 2-device allreduce regression where reduce-scatter dispatchers were missing rank count cases after tiling was added.
* Fixed inconsistent safety checks for early recv/send optimizations in collective operations that could cause data corruption in edge cases.
* Fixed correctness issue in variable-length AllToAllV inter-axis transfers where uniform recv_block_size caused stale data with variable send counts.
* Fixed assertion failure when using inter-RDH algorithm on Trn3 Gen2 UltraServer platforms.

Known Issues
^^^^^^^^^^^^

* Neuron Runtime Library 2.32 is not backward-compatible with Neuron Driver 2.26 or earlier.
  Applications that upgrade the Runtime without also upgrading the Driver will fail to initialize.
  To work around this issue, upgrade the Neuron Driver to the latest version.
  A fix to restore backward compatibility is planned for a future patch release.

Compatibility Support Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Neuron runtime was tested for the following EC2 instances and configurations:

=========================== ============= ============== ================= ===============
Instance Family               OS Type       OS Version     Kernel Version    GLIBC Version
=========================== ============= ============== ================= ===============
``Inf2``                    Ubuntu        U24            6.17              2.39
``Inf2``                    Ubuntu        U22            6.8               2.35
``Inf2``                    Rocky Linux   RL9            5.14              2.34
``Inf2``                    Debian        D12            6.1               2.36
``Inf2``                    Amazon Linux  AL2023         6.12              2.34
``Inf2``                    Amazon Linux  AL2023         6.1               2.34
``Trn1``                    Ubuntu        U24            6.17              2.39
``Trn1``                    Ubuntu        U22            6.8               2.35
``Trn1``                    Rocky Linux   RL9            5.14              2.34
``Trn1``                    Debian        D12            6.1               2.36
``Trn1``                    Amazon Linux  AL2023         6.12              2.34
``Trn1``                    Amazon Linux  AL2023         6.1               2.34
``Trn2``                    Ubuntu        U24            6.17              2.39
``Trn2``                    Ubuntu        U22            6.8               2.35
``Trn2``                    Debian        D12            6.1               2.36
``Trn2``                    Amazon Linux  AL2023         6.12              2.34
``Trn2``                    Amazon Linux  AL2023         6.1               2.34
=========================== ============= ============== ================= ===============

Neuron Driver
~~~~~~~~~~~~~

**Version:** 2.28.0.0

New Features
^^^^^^^^^^^^

* Added new sysfs nodes under ``stats/hardware/health_status/`` exposing cached device health metrics including HBM ECC error counts and hardware error event counter, with sysfs_notify support for event-driven monitoring.

Improvements
^^^^^^^^^^^^

* The DKMS RPM package now uses SHA-512 file digests and xz payload compression, unblocking installation on Bottlerocket and other RPM v6 environments.
* ``core_count`` and ``connected_devices`` sysfs attributes are now world-readable (0444), allowing non-root monitoring tools to query device information.

Bug Fixes
^^^^^^^^^

* DMA page list pool allocation changed from kcalloc to kvcalloc, fixing allocation failures when host memory is fragmented.
* Fixed a race condition in back-to-back NeuronCore resets where DMA ring teardown from a second reset could be overwritten by initialization from the first reset.
* Fixed a bug where DMA rings were destroyed and recreated for all NeuronCores instead of only those in the NC map during no_reset path, causing DMA failures.
* Fixed HBM ECC error reporting to correctly distinguish between repairable and unrepairable uncorrectable errors.
* Fixed an off-by-one error in physical address to BAR4 mapping that rejected valid mappings at the end of the BAR region.
* Fixed potential out-of-bounds array access in sysfs show functions.

Compatibility Support Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Neuron driver was tested for the following EC2 instances and configurations:

=========================== ============= ============== ================= ===============
Instance Family               OS Type       OS Version     Kernel Version    GLIBC Version
=========================== ============= ============== ================= ===============
``Inf2``                    Ubuntu        U24            6.17              2.39
``Inf2``                    Ubuntu        U22            6.8               2.35
``Inf2``                    Rocky Linux   RL9            5.14              2.34
``Inf2``                    Red Hat       RHEL10         6.12              2.39
``Inf2``                    Debian        D12            6.1               2.36
``Inf2``                    Amazon Linux  AL2023         6.12              2.34
``Inf2``                    Amazon Linux  AL2023         6.1               2.34
``Inf2``                    Amazon Linux  AL2            5.10              2.26
``Trn1``                    Ubuntu        U24            6.17              2.39
``Trn1``                    Ubuntu        U22            6.8               2.35
``Trn1``                    Rocky Linux   RL9            5.14              2.34
``Trn1``                    Red Hat       RHEL10         6.12              2.39
``Trn1``                    Debian        D12            6.1               2.36
``Trn1``                    Amazon Linux  AL2023         6.12              2.34
``Trn1``                    Amazon Linux  AL2023         6.1               2.34
``Trn1``                    Amazon Linux  AL2            5.10              2.26
``Trn2``                    Ubuntu        U24            6.17              2.39
``Trn2``                    Ubuntu        U22            6.8               2.35
``Trn2``                    Red Hat       RHEL10         6.12              2.39
``Trn2``                    Debian        D12            6.1               2.36
``Trn2``                    Amazon Linux  AL2023         6.12              2.34
``Trn2``                    Amazon Linux  AL2023         6.1               2.34
``Trn2``                    Amazon Linux  AL2            5.10              2.26
=========================== ============= ============== ================= ===============

Neuron Collectives
~~~~~~~~~~~~~~~~~~

**Version:** 2.32.28.0 

New Features
^^^^^^^^^^^^

* Added one-rank-per-chip and one-rank-per-JBOG graph support for Trn3 Gen2 UltraServer, constructing ring paths that maximize intra-JBOG link utilization.

Bug Fixes
^^^^^^^^^

* Fixed memory leaks of net properties and duplicate channel allocation that accumulated over repeated communicator creation/destruction cycles.
* Fixed spurious TIMEOUT warnings in mesh proxy due to variable_peer, where the timeout check compared against a zero-initialized timestamp instead of the configured 10-second timeout.


.. _runtime-2-29-0-rn:

Neuron Runtime (Neuron 2.29.0 Release)
------------------------------------------------------------------------

Date of Release: 04/09/2026


Neuron Runtime Library
~~~~~~~~~~~~~~~~~~~~~~

**Version:** 2.31.24.0

New Features
^^^^^^^^^^^^
* Added new |nrt_cc_create_stream|_ API for programmatic host-driven collective stream creation, replacing the previous environment variable approach.
* Added |nrt_get_attached_efa_bdf|_ API that returns the BDF string of the EFA device attached to a specified Neuron device index, enabling optimal network interface selection.
* Added |lnc_idx|_ parameter to async tensor APIs, allowing users to select the specific DMA engine for data transfers.
* New environment variables:

  * ``NEURON_RT_ONE_THREAD_PER_CORE``: Pins each network proxy thread to a dedicated CPU core, providing up to 2x improvement in p50/p99 collective communication latency.
  * ``NEURON_RT_RANKS_PER_NETWORK_PROXY``: Controls how many ranks share the same network proxy thread, enabling proxy thread consolidation for improved latency in large-scale distributed workloads.

.. |nrt_cc_create_stream| replace:: ``nrt_cc_create_stream``
.. _nrt_cc_create_stream: https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h
.. |nrt_get_attached_efa_bdf| replace:: ``nrt_get_attached_efa_bdf``
.. _nrt_get_attached_efa_bdf: https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h
.. |lnc_idx| replace:: ``lnc_idx``
.. _lnc_idx: https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h

Improvements
^^^^^^^^^^^^
* Added EFA collectives support for Trn3(previously only available on Trn2), enabling cross-instance data transfers.
* Added profiling support for the :doc:`standalone collectives </neuron-runtime/api/nrt-async-api-overview>`, allowing :doc:`standalone collectives </neuron-runtime/api/nrt-async-api-overview>` traces to appear in the profiler timeline.
* Added context caching for :doc:`standalone collectives </neuron-runtime/api/nrt-async-api-overview>` operations (all-gather, reduce-scatter, all-reduce) that are run outside of a compiled model/kernel, significantly improving schedule performance by up to 90% for repeated calls.
* Removed unnecessary memset operations during :doc:`standalone collectives </neuron-runtime/api/nrt-async-api-overview>` request processing flow, eliminating tens of milliseconds of overhead.
* Removed limit of 512 queue set instances per NEFF which lead to ``NRT_RESOURCE`` errors when loading NEFF with too many queue set instnaces. The Neuron Runtime now supports an unbounded number of queue set instance enabling it to load NEFFs that are further optimized for code size reduction.
* Added Physical Neuron Core ID and Global Rank ID fields to debug tensor read payload for better multi-core/multi-rank debugging.
* Added async sequence IDs (nrta_seq_t) to system trace events for correlation between async operations and hardware execution events.

Breaking changes
^^^^^^^^^^^^^^^^
* Error tracker removed from async API in favor of a simpler status pointer pass-in model. Applications now pass a status pointer directly. See :ref:`this section <nrta-error-handling>` for an example.
* |nrta_get_completion_handle|_ API removed.
* Due to the breaking changes, we have performed a version bump from 2.x to 3.0 for the :doc:`NRT Async APIs </neuron-runtime/api/nrt-async-api-overview>` (APIs prefixed with ``nrta``). Applications using the async API will need to be recompiled against the new version.

.. |nrta_get_completion_handle| replace:: ``nrta_get_completion_handle``
.. _nrta_get_completion_handle: https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h

Bug Fixes
^^^^^^^^^
* Fixed profile data loss on SIGTERM shutdown (e.g., vLLM worker processes); the signal handler now covers both SIGTERM and SIGINT.
* Increased :doc:`Collectives XU </neuron-runtime/api/nrt-async-api-overview>` communication wait timeout (increased from 100ms to 30s) to prevent false timeout failures when ranks have timing drift.
* Fixed a double-free crash during ``nrt_close`` when NEFF execution fails and model unload is not called.
* Fixed DMA ring allocation in :doc:`Collectives XU </neuron-runtime/api/nrt-async-api-overview>` context caching that caused hangs and invalid cache hits.
* Fixed :doc:`Collectives XU </neuron-runtime/api/nrt-async-api-overview>` to properly support in-place operations where send and receive buffers are identical.
* Fixed device memory leak during repeated NEFF load/unload cycles.
* Fixed crash when ``proxy_queue`` is destroyed before ``start()`` due to ``ncclSetAffinity`` failure.
* Fixed system trace event correlation by passing correct execution ID to wait events.
* Fixed NEFF output during inspect profiling to use UUID for distinguishing NEFFs with same hash.
* Fixed BranchPrefetchHint addressing mode bug where backwards-relative branch hints computed incorrect target addresses on trn2 and later.
* Fixed dynamically loaded kernel code carveout size on trn2 (16KB → 32KB) to support migrated operations.

Compatibility Support Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Neuron runtime was tested for the following EC2 instances and configurations:

=========================== ============= ============== ================= ===============
Instance Family               OS Type       OS Version     Kernel Version    GLIBC Version
=========================== ============= ============== ================= ===============
``Inf2``                    Ubuntu        U24            6.17              2.39
``Inf2``                    Ubuntu        U22            6.8               2.35
``Inf2``                    Rocky Linux   RL9            5.14              2.34
``Inf2``                    Amazon Linux  AL2023         6.12              2.34
``Inf2``                    Amazon Linux  AL2023         6.1               2.34
``Trn1``                    Ubuntu        U24            6.17              2.39
``Trn1``                    Ubuntu        U22            6.8               2.35
``Trn1``                    Rocky Linux   RL9            5.14              2.34
``Trn1``                    Amazon Linux  AL2023         6.12              2.34
``Trn1``                    Amazon Linux  AL2023         6.1               2.34
``Trn2``                    Ubuntu        U24            6.17              2.39
``Trn2``                    Ubuntu        U22            6.8               2.35
``Trn2``                    Amazon Linux  AL2023         6.12              2.34
``Trn2``                    Amazon Linux  AL2023         6.1               2.34
=========================== ============= ============== ================= ===============

Neuron Driver
~~~~~~~~~~~~~

**Version:** 2.27.4.0

New Features
^^^^^^^^^^^^
* Added support for new :ref:`TRN3 Gen2 Ultraserver <aws-trn3-arch>` configurations: US3 (2-node), US4 (4-node), US16 (4-node), and US18 (4-node).

Improvements
^^^^^^^^^^^^
* Added top-level DMA reset support during TPB reset on trn3 and later platforms, improving reset reliability.

Compatibility Support Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Neuron driver was tested for the following EC2 instances and configurations:

=========================== ============= ============== ================= ===============
Instance Family               OS Type       OS Version     Kernel Version    GLIBC Version
=========================== ============= ============== ================= ===============
``Inf2``                    Ubuntu        U24            6.17              2.39
``Inf2``                    Ubuntu        U22            6.8               2.35
``Inf2``                    Rocky Linux   RL9            5.14              2.34
``Inf2``                    Red Hat       RHEL10         6.12              2.39
``Inf2``                    Amazon Linux  AL2023         6.12              2.34
``Inf2``                    Amazon Linux  AL2023         6.1               2.34
``Inf2``                    Amazon Linux  AL2            5.10              2.26
``Trn1``                    Ubuntu        U24            6.17              2.39
``Trn1``                    Ubuntu        U22            6.8               2.35
``Trn1``                    Rocky Linux   RL9            5.14              2.34
``Trn1``                    Red Hat       RHEL10         6.12              2.39
``Trn1``                    Amazon Linux  AL2023         6.12              2.34
``Trn1``                    Amazon Linux  AL2023         6.1               2.34
``Trn1``                    Amazon Linux  AL2            5.10              2.26
``Trn2``                    Ubuntu        U24            6.17              2.39
``Trn2``                    Ubuntu        U22            6.8               2.35
``Trn2``                    Red Hat       RHEL10         6.12              2.39
``Trn2``                    Amazon Linux  AL2023         6.12              2.34
``Trn2``                    Amazon Linux  AL2023         6.1               2.34
``Trn2``                    Amazon Linux  AL2            5.10              2.26
=========================== ============= ============== ================= ===============

Neuron Collectives
~~~~~~~~~~~~~~~~~~~

**Version:** 2.31.24.0

Improvements
^^^^^^^^^^^^
* Restructured EFA device processing to per-stream granularity, simplifying resource management for concurrent streams improving stability.
* Improved bootstrap error messages with actionable troubleshooting guidance when ranks fail to receive root parameters.

Bug Fixes
^^^^^^^^^
* Fixed incorrect interface selection in multi-ultraserver collectives where EFA was used instead of Ultraserver interfaces, by adding explicit cross-rack-to-rack flag detection.
* Fixed crash and undefined behavior when channels fail to initialize due to EFA Device plugin errors, now returning clear error messages instead of accessing uninitialized data.

----

.. _runtime-2-28-0-rn:

Neuron Runtime (Neuron 2.28.0 Release)
------------------------------------------------------------------------

Date of Release: 02/26/2026


Neuron Runtime Library
~~~~~~~~~~~~~~~~~~~~~~

**Version:** 2.30.50.0

Improvements
^^^^^^^^^^^^

* Added support for :ref:`TRN3 Gen1 Ultraserver <aws-trn3-arch>` instance type with full system topology
* Added support for tensors larger than 4GB with 64-bit addressing
* Introduced experimental async APIs (see :doc:`NRT Async APIs Overview </neuron-runtime/api/nrt-async-api-overview>`)
* Optimized mesh AllGather on TP8 configurations using destination routing
* Added bound check support for ``dma_direct2d_xpose`` operations

Bug Fixes
^^^^^^^^^

* Fixed proxy thread signaling condition in topsp barrier
* Fixed segfaults in NEFF load cleanup and error paths
* Fixed incompatible network/ultraserver interface selection for inter-node mesh
* Fixed RDH buffer reservation and AllGather bugs
* Fixed corrupted memory logs in multi-threaded model loads
* Improved error handling to return a clear error instead of asserting during ``nrt_init``

Compatibility Support Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Neuron runtime was tested for the following EC2 instances and configurations:

=========================== ============= ============== ================= ===============
Instance Family               OS Type       OS Version     Kernel Version    GLIBC Version
=========================== ============= ============== ================= ===============
``Inf2``                    Ubuntu        U24            6.14              2.39
``Inf2``                    Ubuntu        U22            6.8               2.35
``Inf2``                    Rocky Linux   RL9            5.14              2.34
``Inf2``                    Amazon Linux  AL2023         6.12              2.34
``Inf2``                    Amazon Linux  AL2023         6.1               2.34
``Trn1``                    Ubuntu        U24            6.14              2.39
``Trn1``                    Ubuntu        U22            6.8               2.35
``Trn1``                    Rocky Linux   RL9            5.14              2.34
``Trn1``                    Amazon Linux  AL2023         6.12              2.34
``Trn1``                    Amazon Linux  AL2023         6.1               2.34
``Trn2``                    Ubuntu        U24            6.14              2.39
``Trn2``                    Ubuntu        U22            6.8               2.35
``Trn2``                    Amazon Linux  AL2023         6.12              2.34
``Trn2``                    Amazon Linux  AL2023         6.1               2.34
=========================== ============= ============== ================= ===============


Neuron Driver
~~~~~~~~~~~~~

**Version:** 2.26.10.0

Date of Release: 03/12/2026

Bug Fixes
^^^^^^^^^

* Compatibility fixes for Linux kernel 6.18.

.. _neuron-driver-2-26-5-0:

**Version:** 2.26.5.0

Improvements
^^^^^^^^^^^^^^^

* Added support for detecting :ref:`TRN3 Gen1 Ultraserver <aws-trn3-arch>` platforms
* Added IOCTL to lookup both the Neuron device and the HBM for a given virtual address, enabling frameworks to identify which device holds a tensor
* Updated driver uninstall behavior to fail gracefully without uninstalling the driver if driver is in use

Bug Fixes
^^^^^^^^^

* Fixed kernel crash where non-validated input to DMA-related IOCTLs could trigger BUG_ON requiring an instance reboot to recover
* Added BAR bounds validation during ``ncdev_bar_read`` to prevent out-of-bounds register access through IOCTLs
* Fixed bounds checks on memory accesses where u64 wraparound attacks can lead to out-of-bounds memory access

We would like to thank Shaul Ben Hai from SentinelOne Security Research for reporting the above three issues.

* Fixed use-after-free issues in sysfs cleanup flow that caused kernel crashes
* Fixed race condition in sysfs access during driver initialization

Compatibility Support Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Neuron driver was tested for the following EC2 instances and configurations:

=========================== ============= ============== ================= ===============
Instance Family               OS Type       OS Version     Kernel Version    GLIBC Version
=========================== ============= ============== ================= ===============
``Inf2``                    Ubuntu        U24            6.14              2.39
``Inf2``                    Ubuntu        U22            6.8               2.35
``Inf2``                    Rocky Linux   RL9            5.14              2.34
``Inf2``                    Amazon Linux  AL2023         6.12              2.34
``Inf2``                    Amazon Linux  AL2023         6.1               2.34
``Inf2``                    Amazon Linux  AL2            5.10              2.26
``Trn1``                    Ubuntu        U24            6.14              2.39
``Trn1``                    Ubuntu        U22            6.8               2.35
``Trn1``                    Rocky Linux   RL9            5.14              2.34
``Trn1``                    Amazon Linux  AL2023         6.12              2.34
``Trn1``                    Amazon Linux  AL2023         6.1               2.34
``Trn1``                    Amazon Linux  AL2            5.10              2.26
``Trn2``                    Ubuntu        U24            6.14              2.39
``Trn2``                    Ubuntu        U22            6.8               2.35
``Trn2``                    Amazon Linux  AL2023         6.12              2.34
``Trn2``                    Amazon Linux  AL2023         6.1               2.34
``Trn2``                    Amazon Linux  AL2            5.10              2.26
=========================== ============= ============== ================= ===============

Neuron Collectives
~~~~~~~~~~~~~~~~~~~

**Version:** 2.30.58.0

Improvements
^^^^^^^^^^^^

* Added support for :ref:`TRN3 Gen1 Ultraserver <aws-trn3-arch>` instance types with optimized topology configurations
* Added support for Neuron-Switch-v1 topology and proper network interface selection

Bug Fixes
^^^^^^^^^

* Fixed bug where uninitialized socket file descriptors were incorrectly closed during bootstrap, preventing connection errors in multi-context scenarios
* Improved error handling for channel creation failures due to plugin initialization errors, preventing crashes with misconfigured plugins
* Initialized file descriptor arrays to -1 in bootstrap code to prevent accidental use of uninitialized descriptors

----

.. _runtime-2-27-0-rn:

Neuron Runtime [2.29.40.0] (Neuron 2.27.0 Release)
---------------------------------------------------

Date of Release: 12/19/2025

Improvements
~~~~~~~~~~~~~~~

* Added support for Trainium3 (single node mode)
* Reduced the overhead of reprogramming the Collectives Engine by up to 100x for NEFFs compiled with the ``-O1`` flag. This improves end-to-end performance of these NEFFs by up to 15%.
* Reduced NeuronCore branch overhead by up to 3x, decreasing the overhead of starting a NEFF program by up to 5%.
* Reduced the overhead of starting a NEFF program by up to 50% with an on-device hardware barrier between ranks.
* Improved all-gather latency by up to 35% for messages greater than 1MB in TP8 (LNC2) and TP16 (LNC1) collectives.
* Added support for NRT Debug Stream APIs.

Bug Fixes
~~~~~~~~~

* Fixed scratchpad page allocation bug that caused excessive page allocations due to page rounding error.
* Fixed segfault that occurred when freeing an empty tensor.

Known Issues
~~~~~~~~~~~~

* The ``nrt_tensor_allocate`` APIs do not support more then 4 GB (>= 4GB) sizes. Passing in a size larger than or equal to 4GB will result in datatype overflow leading to undefined behavior.
* A hardware bug affecting **Trainium** and **Inferentia2** devices causes numerical errors to become "sticky" within the Neuron Core hardware.


----

.. _runtime-2-26-0-rn:

Neuron Runtime [2.28.19.0] (Neuron 2.26.0 Release)
---------------------------------------------------

Date of Release: 09/18/2025

Improvements
~~~~~~~~~~~~~~~

* Added rank ID to all events emitted from the Profiler 2.0 system trace.
* Improved timestamp alignment of Profiler 2.0 NeuronCore and CPU system trace events enhancing the accuracy of the trace timeline.

Bug Fixes
~~~~~~~~~

* Fixed bug where `nrt_unload` returned `NRT_SUCCESS` even when model stop fails due to Neuron Core lockups.
* Fixed bug where `model_name` was empty in Profiler 2.0 system trace events.
* Fixed bug where error messages were incorrectly being displayed on machines with no EFA devices.

Known Issues
~~~~~~~~~~~~

* The ``nrt_tensor_allocate`` APIs do not support more then 4 GB (>= 4GB) sizes.
* A hardware bug affecting **Trainium** and **Inferentia2** devices causes numerical errors to become "sticky" within the Neuron Core hardware.


----

.. _runtime-2-25-0-rn:

Neuron Runtime (Neuron 2.25.0 Release)
---------------------------------------

Date of Release: 07/31/2025

Neuron Runtime Library
~~~~~~~~~~~~~~~~~~~~~~

**Version:** 2.27.23.0

Improvements
^^^^^^^^^^^^^^

* Introduced ``nrt_get_vnc_memory_stats`` API to retrieve device memory usage.
* Added support for State-Buffer to State-Buffer collective support for ``all_reduce``, ``reduce_scatter``, and ``all_gather`` for LNC2, which helps reduce HBM memory pressure.
* Added support for coalescing of Collectives operations for internode RDH.
* Introduced a new DGE priority class feature to select preferred packet size for memory transfers.
* Improved ``nrt_init`` time by up to ~3 seconds on AWS Trainium and Inferentia instances.
* Added a warning message along with a recommended scratchpad configuration when a loaded NEFF has non-optimial scratchpad usage.

Breaking Changes
^^^^^^^^^^^^^^^^

* Due to a hardware bug that can cause numerical errors to be falsely reported, the runtime has disabled numerical errors by default. Users can re-enable numerical errors by setting ``NEURON_RT_NUMERICAL_ERRORS_VERBOSITY=critical`` or ``NEURON_FAIL_ON_NAN=1``.

Bug Fixes
^^^^^^^^^

* Fixed profiling APIs to report execution duration from explicit notifications.
* Fixed race condition which can cause a crash when starting inspect traces.

Known Issues
^^^^^^^^^^^^

* A hardware bug affecting **Trainium** and **Inferentia2** devices causes numerical errors to become "sticky" within the Neuron Core hardware.

Neuron Collectives
~~~~~~~~~~~~~~~~~~

**Version:** 2.25.65.0

Improvements
^^^^^^^^^^^^^^^

* Added multinode collectives support for Trainium2 instances without EFA devices
* Minor performance improvement to network proxy handshake

Bug Fixes
^^^^^^^^^

* Fixed memory leak clearing up communication devices during ``nrt_close``

Neuron Driver
~~~~~~~~~~~~~

**Version:** 2.21.37.0

Improvements
^^^^^^^^^^^^^^^

* Added the ability for users to read power utilization for each neuron device via a sysfs interface. This interface shows the minimum, maximum and average power consumed by the device over the past minute, expressed as a percentage of the device's maximum power.
* Added the ability for users to read the device utilization. This shows up as the microseconds between the start and end of the current execution on hardware.


----

.. _runtime-2-24-0-rn:

Neuron Runtime (Neuron 2.24.0 Release)
---------------------------------------

Date of Release: 06/24/2025

Neuron Runtime Library
~~~~~~~~~~~~~~~~~~~~~~

**Version:** 2.26.42.0

Improvements
^^^^^^^^^^^^^^^

* Added support for 8x8 collective groups (TP8 + CP8) on **TRN2** for **LNC=2**
* Added support for direct `State-Buffer` to `State-Buffer` collective ops for **LNC=1**
* Introduce RDH algorithm for inter-node collective communication
* Added support for loading NEFF with different world sizes in the same NRT process
* Reduced the average latency of 32x2 collective groups by 65%
* Reduced latency for intra-chip reduce scatter operations on **TRN2** instances by up to 20% for small transfers and 60% for medium to large transfers
* Improved latency for medium message sizes for intra-chip All Gather operations on **TRN2** by up to 60%
* Improved the debugging experience by adding logs which print out the value of timed-out, non-zero semaphores on **Trainium2** platforms
* Improved timeout error messages by displaying the NEFF program counters for the stuck Neuron Core
* Refined out-of-memory error messages to report a NEFF level memory breakdown table

Breaking Changes
^^^^^^^^^^^^^^^^

* This version of the Neuron runtime requires `aws-neuron-dkms` version `2.22` or later on **Trainium2** instances.

Bug Fixes
^^^^^^^^^

* Fixed crash caused by race condition during the capture of system profiles
* Fixed various memory leaks that occur during `nrt_close`

Known Issues
^^^^^^^^^^^^

* The ``nrt_tensor_allocate`` APIs do not support more then 4 GB (>= 4GB) sizes.
* A hardware bug affecting **Trainium** and **Inferentia2** devices causes numerical errors to become "sticky" within the Neuron Core hardware.

Neuron Collectives
~~~~~~~~~~~~~~~~~~

**Version:** 2.24.59.0

Improvements
^^^^^^^^^^^^^^^

* Improved interface between ``libnccom`` and ``libnrt`` resulting stability improvements

Neuron Driver
~~~~~~~~~~~~~

**Version:** 2.20.28.0

Improvements
^^^^^^^^^^^^^^^

* This driver is required to run with Neuron Runtime 2.24 or later on Trainium2 machines. Included in the release is a bug fix to avoid device memory corruption issues leading to undefined Neuron Device behavior.
* Improved interface between ``libnrt`` and the Driver resulting in stability improvements.


----

.. _runtime-2-23-0-rn:

Neuron Runtime (Neuron 2.23.0 Release)
---------------------------------------

Date of Release: 05/19/2025

Neuron Runtime Library
~~~~~~~~~~~~~~~~~~~~~~

**Version:** 2.25.57.0

Improvements
^^^^^^^^^^^^^^^

* Added ``NEURON_RT_LOW_LATENCY_TASKS_CPU_AFFINITY`` environment variable to allow users to set the thread affinity of low latency tasks that run on host cpu
* Refined software notification queue overflow detection flow and improved error message
* Reduced latency for All-Reduce intra-chip collective (TP 4) by 50% for medium message sizes
* Improved error message when an execution request is passed a tensor allocated on an incorrect HBM
* Improved NEFF switch latency by up to 95% when using async mode
* Increased the number of different replica groups supported in the same NEFF on TRN2
* Explicitly limit the max number of in-flight async requests to the hard limit of 63
* Added traces for Host <-> device data transfer events in system profiles (Neuron Profiler 2.0 Beta)
* Added pre/post execution hooks to system profiles (Neuron Profiler 2.0 Beta)
* Significant performance improvements in time taken by calls to nrt_sys_trace_fetch_events() (Neuron Profiler 2.0 Beta)

Bug Fixes
^^^^^^^^^

* Fixed segfault that can occur when applications attempt to load a NEFF with an unsupported number of FMA source descriptors

Known Issues
^^^^^^^^^^^^

* The ``nrt_tensor_allocate`` APIs do not support more then 4 GB (>= 4GB) sizes.

Neuron Collectives
~~~~~~~~~~~~~~~~~~

**Version:** 2.23.135.0 / 2.23.133.0

Improvements
^^^^^^^^^^^^^^^

* Added Trainium2 support
* Improved startup times for large scale training jobs by up to 5 seconds
* Enhanced error logging for bootstrap failures
* Aws-ofi-nccl: minor performance improvement

Bug Fixes
^^^^^^^^^

* Fixed various memory leaks which occur during process cleanup


----

.. _runtime-2-22-0-rn:

Neuron Runtime (Neuron 2.22.0 Release)
---------------------------------------

Date of Release: 04/03/2025

Neuron Runtime Library
~~~~~~~~~~~~~~~~~~~~~~

**Version:** 2.24.53.0

Improvements
^^^^^^^^^^^^^^^

* Improved dynamic DMA descriptor generation performance by up to 3% for certain workloads
* Reduced collectives device memory footprint for large Neffs
* Improved device latency for memory bound workloads on TRN2
* Added support for profiling executions when NRT is launched in Async Execution Mode
* Added check to detect execution completion queue overflows
* Reduced overhead of Neuron Profiler 2.0 to <1% of overall latency (Neuron Profiler 2.0 Beta)
* Added new ``nrt_sys_trace_fetch_events`` API to retrieve system trace events (Neuron Profiler 2.0 Beta)
* Added out of bound error events to system trace (Neuron Profiler 2.0 Beta)
* Removed the ``NEURON_RT_INSPECT_DURATION_NSEC`` and ``NEURON_RT_INSPECT_START_OFFSET_NSEC`` configuration options (Neuron Profiler 2.0 Beta)
* Added dynamic DMA support for block scatter ops (NKI)
* Added RangeSelect instruction Support for the Vector engine (NKI)

Breaking Changes
^^^^^^^^^^^^^^^^

* Removed support for Neuron Distributed Event Tracing

Bug Fixes
^^^^^^^^^

* Fixed bug introduced in NRT 2.23 where the runtime was incorrectly reporting executions that hit "Out of Bound" errors as successful executions
* Fixed segfault when encountering "out of memory" errors when starting profiles

Known Issues
^^^^^^^^^^^^

* The ``nrt_tensor_allocate`` APIs do not support more then 4 GB (>= 4GB) sizes.

Neuron Collectives
~~~~~~~~~~~~~~~~~~

**Version:** 2.22.26.0

Improvements
^^^^^^^^^^^^^^^

* Added check to print out an error message on invalid ``NEURON_RT_ROOT_COMM_ID`` configurations

Bug Fixes
^^^^^^^^^

* Resolved an issue where the ``libnccom.so`` filename was versioned incorrectly as ``libnccom.so.2.y.y``. Will be correctly versioned as ``libnccom.so.2.22.26`` in this release.

Neuron Driver
~~~~~~~~~~~~~

**Version:** 2.22.2.0

Improvements
^^^^^^^^^^^^^^^

* Added workaround for HW DGE descriptor fetching bug

Bug Fixes
^^^^^^^^^

* Fixed typos in certain error log messages

Breaking Changes
^^^^^^^^^^^^^^^^

* Starting with Neuron Release 2.26, Neuron driver versions above 2.24 will only support non-Inf1 instances (such as ``Trn1``, ``Inf2``, or other instance types).
* ``Inf1`` instance users, only Neuron driver version 2.24 will remain supported with regular security patches.
* ``Inf1`` instance users are advised to pin the Neuron driver version to ``2.24.*`` in their installation script.


----

.. _runtime-2-21-0-rn:

Neuron Runtime (Neuron 2.21.0 Release)
---------------------------------------

Date of Release: 12/20/2024

Neuron Runtime Library
~~~~~~~~~~~~~~~~~~~~~~

**Version:** 2.23.110.0 / 2.23.112.0

Improvements
^^^^^^^^^^^^^^^

* Added Trainium2 support
* Added runtime support to detect and fail on out-of-bound memory access in DMA operations
* Added support for 4-rank replica group on adjacent Neuron cores on TRN1/TRN1N
* Added new profiling API for capturing system and device profiles (Neuron Profiler 2.0 Beta)
* Reduced runtime host RAM utilization
* Improved Neff context switch overhead reducing latency by up to 500us
* Split hardware errors into more granular categories:
   * ``NRT_EXEC_HW_ERR_HBM_UE`` (1201)
   * ``NRT_EXEC_HW_ERR_NC_UE`` (1202)
   * ``NRT_EXEC_HW_ERR_DMA_ABORT`` (1203)
* Updated runtime to breakdown DMA ring memory usage into more detailed categories:
   * dma rings io
   * dma rings spill
   * dma rings collectives
   * dma rings runtime
* Updated the ``nrt_load`` error path to print a clear error message when failing to load a collectives Neff instead of aborting

Breaking Changes
^^^^^^^^^^^^^^^^

* Removed INF1 Support from Runtime library

Bug Fixes
^^^^^^^^^

* Fixed multiple memory corruptions and exhaustions on the collectives failure path
* Fixed bug where incorrect execution status was passed to the async execution callback
* Fixed DMA abort errors on TRN2

Known Issues
^^^^^^^^^^^^

* The ``nrt_tensor_allocate`` APIs do not support more then 4 GB (>= 4GB) sizes.

Neuron Collectives
~~~~~~~~~~~~~~~~~~

**Version:** 2.21.46.0

Improvements
^^^^^^^^^^^^^^^

* Bootstrap changes to improve application startup latency for large-scale workloads
* Logging improvements
