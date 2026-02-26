.. meta::
    :description: Complete release notes for the Neuron Developer Tools component across all AWS Neuron SDK versions.
    :keywords: neuron tools, developer tools, profiler, release notes, neuron explorer, aws neuron sdk
    :date-modified: 02/26/2026

.. _dev-tools_rn:

Component Release Notes for Neuron Developer Tools
==================================================

The release notes for the Neuron Developer Tools. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.


.. _dev-tools-2-28-0-rn:   

Neuron Developer Tools & Neuron Explorer (Neuron 2.28.0 Release)
--------------------------------------------------------------------------------------

Date of Release: 02/26/2026

Improvements
~~~~~~~~~~~~~~~

* Added system profiling support in Neuron Explorer, enabling you to capture and analyze system-level performance data with drill-down navigation to device profiles. See :doc:`Neuron Explorer system profiling </tools/neuron-explorer/overview-system-profiles>`.
* Added migration guide from Neuron Profiler/Profiler 2.0 to Neuron Explorer. See :ref:`neuron-profiler-migration-guide`.
* Added ability to save and search by tags in Neuron Explorer Profile Manager, allowing you to organize and quickly locate profiles across multiple profiling sessions. See :ref:`neuron-explorer-profile-manager`.
* Added help pop-up for the ``Device Trace Viewer`` in Neuron Explorer to see shortcuts and dependency color legend. See :doc:`Device Trace Viewer </tools/neuron-explorer/overview-device-profiles>`.
* Introduced ``Tensor Viewer`` in Neuron Explorer, enabling you to quickly identify memory bottlenecks by viewing tensor names, shapes, sizes, and memory usage in a single interface. See :ref:`tensor-viewer-overview`.
* Introduced ``Database Viewer`` in Neuron Explorer as an interactive interface for querying and exploring profiling data using SQL or natural language, allowing you to perform custom analysis without writing code. See :ref:`database-viewer-overview`.
* Enhanced data integrity checks in ``nccom-test`` by using pseudo-random data patterns instead of fixed patterns, improving detection of data corruption during collective operations. See `Data Integrity`_ in the nccom-test documentation.
* Added support for ``alltoallv`` collective operation in ``nccom-test``, enabling benchmarking of variable-sized all-to-all communication patterns. See `AlltoAllV Example`_ in the nccom-test documentation.

Breaking Changes
~~~~~~~~~~~~~~~~

* The ``neuron-profile analyze`` subcommand is no longer supported.  We recommend migrating to Neuron Explorer.  See :doc:`Get Started with Neuron Explorer </tools/neuron-explorer/get-started>`.

Bug Fixes
~~~~~~~~~

* ``neuron-ls`` now handles concurrent queries correctly. Previously, when multiple processes queried Neuron devices simultaneously, ``neuron-ls`` would fail with a driver error, preventing you from viewing device status.
* Neuron Explorer now correctly calculates PSUM usage for operations spanning multiple partitions. Previously, PSUM usage was underreported, which could lead to incorrect performance optimization decisions.

.. _dev-tools-2-27-0-rn:

Neuron Developer Tools & Neuron Explorer [2.29.0] (Neuron 2.27.0 Release)
-------------------------------------------------------------------------

Date of Release: 12/19/2025

Improvements
~~~~~~~~~~~~~~~

* Introduced Neuron Explorer - A unified profiling suite that replaces Neuron Profiler and Profiler 2.0.
* Four core viewers provide insights into model performance: Hierarchy Viewer, AI Recommendation Viewer, Source Code Viewer, and Summary Viewer.
* Neuron Explorer is available through UI, CLI, and VSCode IDE integration.
* Added fine-grained collective communication support to nccom-test utility.
* New tutorials cover profiling NKI kernels, multi-node training jobs, and vLLM inference workloads.
* Added Trn3 support for neuron-monitor, neuron-top, neuron-ls, and nccom-test.

Breaking Changes
~~~~~~~~~~~~~~~~

* Neuron Profiler and Profiler 2.0 support ends after Neuron 2.28.

Bug Fixes
~~~~~~~~~

* Improved profiling accuracy and reduced overhead.
* Fixed visualization issues in multi-process scenarios.

Known Issues
~~~~~~~~~~~~

* Existing NTFF files are compatible but require reprocessing for new features.
* Neuron Explorer does not support system level profiling at this time.


----

.. _dev-tools-2-26-0-rn:

Neuron Developer Tools [2.26.7.0] (Neuron 2.26.0 Release)
----------------------------------------------------------

Date of Release: 09/18/2025

Improvements
~~~~~~~~~~~~~~~

* Profiler UI now allows selecting multiple semaphore values to display simultaneously for a more comprehensive view of activity.
* System profile grouping default in Perfetto now uses global NeuronCore ID instead of process local NeuronCore ID for better display of multi-process workloads.
* Added warning when system profile events are dropped due to limited buffer space.
* nccom-test support on Trn2 for State Buffer to State Buffer collectives benchmarking for all-reduce, all-gather, and reduce-scatter operations.
* nccom-test will show helpful error message when invalid sizes are used with all-to-all collectives.

Bug Fixes
~~~~~~~~~

* Fixed device memory usage type table and improvement made to stay in sync between runtime and tools versions.
* Fixed system profile crash when processing long-running workloads.
* Fixed display of system profiles in Perfetto to correctly separate rows within the same Logical NeuronCore when using NEURON_LOGICAL_NC_CONFIG=2 on Trn2.


----

.. _dev-tools-2-25-0-rn:

Neuron Developer Tools [2.25.100.0] (Neuron 2.25.0 Release)
------------------------------------------------------------

Date of Release: 07/31/2025

Improvements
~~~~~~~~~~~~~~~

* neuron-ls now shows NeuronCore IDs associated with each Neuron device as well as CPU and NUMA node affinity in both the text and JSON outputs.
* Added a summary metric to device profiles for total_active_time (the amount of time the device was not idle during execution).
* System profiles now show the sync point events that are used to approximate CPU and Neuron device timestamp alignment.
* Removed metrics for defunct processes from Neuron Monitor's Prometheus output to more accurately reflect the current utilization of NeuronCores.

Bug Fixes
~~~~~~~~~

* Fixed issue in Neuron Profiler summary metrics where dma_active_time was larger than expected.
* Fixed type inconsistency for certain event types and attributes in the system profile data that could result in a crash.

Known Issues
~~~~~~~~~~~~

* System profile hardware events may be misaligned due to sync point imprecision.
* System profile events shown in the Neuron Profiler UI for multiprocess workloads are grouped together.


----

.. _dev-tools-2-24-0-rn:

Neuron Developer Tools [2.24.54.0] (Neuron 2.24.0 Release)
-----------------------------------------------------------

Date of Release: 06/24/2025

Improvements
~~~~~~~~~~~~~~~

* Scratchpad memory usage visualization is now available in the Neuron Profiler UI.
* Framework stack traces are now available in the Neuron Profiler UI.
* On-device collectives barriers are now shown in the Neuron Profiler UI.
* HBM throughput visualization over time is now shown in the Neuron Profiler UI.
* Added option to filter the Neuron Cores to capture trace events on.
* Added option to filter the event types recorded when capturing system traces.
* Added a flag to nccom-test to get results in JSON (--report-to-json-file <filename>).
* Added a flag to nccom-test to explicitly show input and output sizes based on the operation (--show-input-output-size).

Bug Fixes
~~~~~~~~~

* Fixed instance id labeling in system profile view for framework events.
* Fixed issue in Neuron Profiler UI where the full data was not shown in the NEFF Nodes tab.


----

.. _dev-tools-2-23-0-rn:

Neuron Developer Tools [2.23.16.0] (Neuron 2.23.0 Release)
-----------------------------------------------------------

Date of Release: 05/19/2025

Improvements
~~~~~~~~~~~~~~~

* Improved Neuron Profiler performance, allowing users to view profile results 5x times faster on average.
* Improved error reporting with timeline support for error signatures via custom notifications in the Neuron Profiler UI.
* Added execution and out-of-bounds (OOB) error tracking in Neuron Profiler JSON outputs.
* Updated the default grouping for system profiles to include process ID.
* Added neuron-monitor companion script for collecting Kubernetes info in EKS.

Bug Fixes
~~~~~~~~~

* Fixed hang during data collection when running nccom-test across multiple instances.
* Fixed certain cases in Neuron Profiler where DMA sizes were always reported as 0 bytes.


----

.. _dev-tools-2-22-0-rn:

Neuron Developer Tools [2.22.66.0] (Neuron 2.22.0 Release)
-----------------------------------------------------------

Date of Release: 04/03/2025

Improvements
~~~~~~~~~~~~~~~

* Added several enhancements to the Neuron Profiler UI, including NeuronCore barrier annotations, a minimal default view to improve initial load performance, usability of updating markers, and better organization of view settings.
* Added new event types in the system profile for Neuron Profiler 2.0 (Beta) related to out-of-bounds execution errors, execution request submission, and model switch overhead.
* Updated system trace output format for Neuron Profiler 2.0 (Beta).

Breaking Changes
~~~~~~~~~~~~~~~~

* neuron-det is no longer supported starting with this release. We recommend customers transition to Neuron Profiler 2.0 (Beta) for debugging runtime hangs and issues in large-scale settings.

Bug Fixes
~~~~~~~~~

* Fixed an issue in the Neuron Profiler UI where dependencies were misaligned in the timeline when highlighted.
* Fixed an issue where instruction dependency IDs were truncated in the Neuron Profiler JSON output.


----

.. _dev-tools-2-21-0-rn:

Neuron Developer Tools [2.20.204.0] (Neuron 2.21.0 Release)
------------------------------------------------------------

Date of Release: 12/20/2024

Improvements
~~~~~~~~~~~~~~~

* Added support for Trn2 instance types.
* Added support for Logical Neuroncores. neuron-top, neuron-monitor, and neuron-ls now display and aggregate information per Logical Neuroncore based on LNC configuration.
* Added Neuron Profile 2.0 (Beta) with system profiles featuring Neuron Runtime API trace and ML framework trace.
* Option to view system and device profiles using the Perfetto UI.
* Support for native JAX and PyTorch profilers.
* Support for distributed workloads in environments such as EKS and ParallelCluster.
* Ability to drill down from high-level system profiles to low-level device profiles.
* Simplified experience for capturing profiles.


----

.. _dev-tools-2-20-0-rn:

Neuron Developer Tools [2.19.0.0] (Neuron 2.20.0 Release)
----------------------------------------------------------

Date of Release: 09/16/2024

Improvements
~~~~~~~~~~~~~~~

* Added support for Neuron Kernel Interface (NKI).
* Updated neuron-profile JSON output to include information regarding instruction dependencies, DMA throughput, and SRAM usage.
* Updated Neuron Profiler UI to display transpose information for DMAs (when applicable).

Bug Fixes
~~~~~~~~~

* Fixed error handling in neuron-top to exit gracefully when passing an unknown argument.

Known Issues
~~~~~~~~~~~~

* None reported for this release.

----

.. _dev-tools-2-19-0-rn:

Neuron Developer Tools [2.18.3.0] (Neuron 2.19.0 Release)
----------------------------------------------------------

Date of Release: 07/03/2024

Improvements
~~~~~~~~~~~~~~~

* Profile captured with Neuron Runtime 2.20+ now includes annotations with additional information such as duration, size, and replica groups around collective operations.
* Running neuron-profile capture for workloads with collectives will now attempt to use the required number of workers if --collectives-workers-per-node or --collectives-worker-count is not set.
* Profiler UI now persists searched information in the URL and provides a summary of the search results.
* Updating sampling approach to show more representative data in the profiler UI when zoomed out.
* Updated groupings for displayed info on click in the profiler UI.
* Added neuron_device_type and neuron_device_memory_size to neuron-monitor's hardware information output.

Bug Fixes
~~~~~~~~~

* Resolved issue where NaN would be seen in the JSON output of neuron-profile and result in parsing errors.
* Resolved inconsistent timeline display issues in profiler UI that depended on when the profile was processed.
* neuron-profile view --output-format summary-text will now display in a fixed order.
* Updated accuracy of pending DMA count in the profiler UI.
* Removed unnecessary calls to exec when capturing memory utilization metrics in neuron-monitor.


----

.. _dev-tools-2-18-0-rn:

Neuron Developer Tools [2.17.1.0] (Neuron 2.18.0 Release)
----------------------------------------------------------

Date of Release: 04/01/2024

Improvements
~~~~~~~~~~~~~~~

* NeuronPerf 1.8.55.0: Minor updates.

Bug Fixes
~~~~~~~~~

* Fixed potential hang during synchronization step in nccom-test.


----

.. _dev-tools-2-17-0-rn:

Neuron Developer Tools [2.17.0.0] (Neuron 2.17.0 Release)
----------------------------------------------------------

Date of Release: 02/13/2024

Improvements
~~~~~~~~~~~~~~~

* Added support to neuron-profile for collective communication operator improvements in Neuron SDK 2.17.
* Optimized count query for sampling in neuron-profile UI for up to 3x faster load performance.
* Introduced warning annotations in neuron-profile UI to automatically highlight potential performance issues.

Bug Fixes
~~~~~~~~~

* Resolved issue of inaccurate execution time reported by neuron-profile as mentioned in Neuron Tools 2.16.1.0 release notes.
* Fixed NaN display errors in the neuron-profile UI.
* Fixed file naming issue when capturing collectives profiles with neuron-profile.


----

.. _dev-tools-2-16-0-rn:

Neuron Developer Tools [2.16.1.0] (Neuron 2.16.0 Release)
----------------------------------------------------------

Date of Release: 12/21/2023

Improvements
~~~~~~~~~~~~~~~

* First release of the Neuron Distributed Event Tracing tool neuron-det to visualize execution for multi-node workloads.
* neuron-profile now has the ability to capture multi-worker jobs.
* Added terminology descriptions to neuron-profile summary statistics.
* Added optional flags to neuron-profile view to change the InfluxDB bucket name (--db-bucket <bucket name>) and profile display name (--display-name <name>).
* NeuronPerf 1.8.15.0: Minor updates.

Bug Fixes
~~~~~~~~~

* Fixed bug where GPSimd summary values were missing in the profile summary.
* Fixed issue in nccom-test to no longer expect Neuron Device 0 in a container environment.
* Fixed issue in nccom-test to no longer require the instance launching nccom-test to be participating in the workload.

Known Issues
~~~~~~~~~~~~

* Execution time reported in neuron-profile is sometimes inaccurate due to a bug in how the time is captured. The bug will be addressed in upcoming Neuron releases.


----

.. _dev-tools-2-15-0-rn:

Neuron Developer Tools [2.15.4.0] (Neuron 2.15.0 Release)
----------------------------------------------------------

Date of Release: 10/26/2023

Improvements
~~~~~~~~~~~~~~~

* Improved visibility of summary stats in the profiler UI with added groupings.
* Added support for alltoall CC operation in nccom-test.

Bug Fixes
~~~~~~~~~

* Fixed bug in neuron-profile that may result in a crash when using the NeuronCore Pipeline feature on Inf1.


----

.. _dev-tools-2-14-0-rn:

Neuron Developer Tools [2.14.6.0] (Neuron 2.14.0 Release)
----------------------------------------------------------

Date of Release: 09/15/2023

Improvements
~~~~~~~~~~~~~~~

* Added legend in neuron-ls to clarify wrap around edges for topology view.
* Improved error messaging when passing invalid arguments to neuron-profile view.
* Profiler output now includes HLO name in addition to framework layer names.
* neuron-profile view now has --output-format json option which will write to a file specified by --output-file <name> (default is ntff.json) instead of writing data to InfluxDB.

Bug Fixes
~~~~~~~~~

* Fixed bug in neuron-profile that incorrectly calculated buffer utilization for more recently compiled NEFFs.
* Fixed bug in neuron-profile where the profile would sometimes include additional idle time while waiting for execution to start.


----

.. _dev-tools-2-13-0-rn:

Neuron Developer Tools [2.13.4.0] (Neuron 2.13.0 Release)
----------------------------------------------------------

Date of Release: 08/28/2023

Improvements
~~~~~~~~~~~~~~~

* --check option of nccom-test now supports more data types (fp16, bf16, (u)int8, (u)int16, and (u)int32 are now supported in addition to fp32).
* NeuronPerf 1.8.7.0: Minor updates.

Bug Fixes
~~~~~~~~~

* Fixed bug in nccom-test that would wait indefinitely for execution to end when running on multiple instances (-N 2 and higher).
* Fixed bug in neuron-profile to prevent a crash during utilization calculation.


----

.. _dev-tools-2-12-0-rn:

Neuron Developer Tools [2.12.2.0] (Neuron 2.12.0 Release)
----------------------------------------------------------

Date of Release: 07/19/2023

Improvements
~~~~~~~~~~~~~~~

* Bumped the max supported profiling NTFF version to version 2 to resolve crashes when postprocessing NTFFs captured with newer versions of the Neuron Runtime Library.
* When viewing profiles captured using Neuron Runtime Library 2.15 or above, please upgrade tools to 2.12.
* This version of Neuron tools remains compatible with NTFF version 1.

Bug Fixes
~~~~~~~~~

* Bug fixes for neuron-profile related to the calculation of some summary stats.


----

.. _dev-tools-2-11-0-rn:

Neuron Developer Tools [2.11.10.0] (Neuron 2.11.0 Release)
-----------------------------------------------------------

Date of Release: 06/14/2023

Improvements
~~~~~~~~~~~~~~~

* nccom-test can now show multiple latency stats in the results table, such as average or percentiles, by specifying the -s option (for example: -s p10 p99 avg p50).
* First public support for neuron-profile as a standalone tool that can be used to profile executions on Neuron Devices.


----

.. _dev-tools-2-10-0-rn:

Neuron Developer Tools [2.10.1.0] (Neuron 2.10.0 Release)
----------------------------------------------------------

Date of Release: 05/01/2023

Improvements
~~~~~~~~~~~~~~~

* Added new Neuron Collectives benchmarking tool, nccom-test, to enable benchmarking sweeps on various Neuron Collective Communication operations.
* Expanded support for Neuron profiling to include runtime setup/teardown times and collapsed execution of NeuronCore engines and DMA.


----

.. _dev-tools-2-9-0-rn:

Neuron Developer Tools [2.9.5.0] (Neuron 2.9.0 Release)
--------------------------------------------------------

Date of Release: 03/28/2023

Improvements
~~~~~~~~~~~~~~~

* Updated neuron-top to show effective FLOPs across all NeuronCores.
* NeuronPerf 1.7.0.0: Adds trn1/inf2 support for PyTorch and TensorFlow 2.x. Uses new IMDSv2 for obtaining instance types.


----

.. _dev-tools-2-8-0-rn:

Neuron Developer Tools [2.8.2.0] (Neuron 2.8.0 Release)
--------------------------------------------------------

Date of Release: 02/24/2023

Improvements
~~~~~~~~~~~~~~~

* Updated neuron-top to show aggregated utilization/FLOPs across all NeuronCores.


----

.. _dev-tools-2-7-0-rn:

Neuron Developer Tools [2.7.2.0] (Neuron 2.7.0 Release)
--------------------------------------------------------

Date of Release: 02/08/2023

Improvements
~~~~~~~~~~~~~~~

* Added support for model FLOPS metrics in both neuron-monitor and neuron-top.


----

.. _dev-tools-2-6-0-rn:

Neuron Developer Tools [2.6.0.0] (Neuron 2.6.0 Release)
--------------------------------------------------------

Date of Release: 12/09/2022

Improvements
~~~~~~~~~~~~~~~

* Added support for profiling with the Neuron Plugin for TensorBoard on TRN1.
* Updated profile post-processing for workloads executed on TRN1.


----

.. _dev-tools-2-5-0-rn:

Neuron Developer Tools [2.5.19.0] (Neuron 2.5.0 Release)
---------------------------------------------------------

Date of Release: 11/07/2022

Improvements
~~~~~~~~~~~~~~~

* Minor bug fixes and improvements.

Bug Fixes
~~~~~~~~~

* Minor bug fixes and improvements.


----

.. _dev-tools-2-5-0-2-rn:

Neuron Developer Tools [2.5.16.0] (Neuron 2.5.0 Release)
---------------------------------------------------------

Date of Release: 10/26/2022

Improvements
~~~~~~~~~~~~~~~

* New neuron-monitor and neuron-top feature: memory utilization breakdown. This new feature provides more details on how memory is being currently used on the Neuron Devices as well as on the host instance.
* neuron-top's UI layout has been updated to accommodate the new memory utilization breakdown feature.
* neuron-monitor's inference_stats metric group was renamed to execution_stats. While the previous release still supported inference_stats, starting this release the name inference_stats is considered deprecated and can't be used anymore.
* NeuronPerf 1.6.0.0: New Evaluation + metrics API. Support map and iterable-type torch datasets. Support custom torch DataLoader args via dataloader_kwargs. New get_report_by_tag utility to identify specific configurations. Python 3.7+ now default from 3.6. Pricing and sizing info updated for inf1 + trn1.

Breaking Changes
~~~~~~~~~~~~~~~~

* neuron-monitor's inference_stats metric group was renamed to execution_stats.

Bug Fixes
~~~~~~~~~

* Fix a rare crash in neuron-top when the instance is under heavy CPU load.
* Fix process names on the bottom tab bar of neuron-top sometimes disappearing for smaller terminal window sizes.
* NeuronPerf: GPU inputs are now moved correctly.


----

.. _dev-tools-2-4-0-rn:

Neuron Developer Tools [2.4.6.0] (Neuron 2.4.0 Release)
--------------------------------------------------------

Date of Release: 10/10/2022

Improvements
~~~~~~~~~~~~~~~

* Added support for both EC2 INF1 and TRN1 platforms. Name of the package changed from aws-neuron-tools to aws-neuronx-tools.
* Added support for ECC counters on Trn1.
* Added version number output to neuron-top.
* Expanded support for longer process tags in neuron-monitor.
* Removed hardware counters from the default neuron-monitor config to avoid sending repeated errors - will add back in future release.
* neuron-ls - Added option neuron-ls --topology with ASCII graphics output showing the connectivity between Neuron Devices on an instance.

Breaking Changes
~~~~~~~~~~~~~~~~

* Package name changed from aws-neuron-tools to aws-neuronx-tools.

Bug Fixes
~~~~~~~~~

* Fix neuron-monitor and neuron-top to show the correct Neuron Device when running in a container where not all devices are present.


----

.. _dev-tools-2-1-0-rn:

Neuron Developer Tools [2.1.4.0] (Neuron 2.1.0 Release)
--------------------------------------------------------

Date of Release: 04/29/2022

Improvements
~~~~~~~~~~~~~~~

* Minor updates.
* NeuronPerf 1.3.0.0: Minor updates.


----

.. _dev-tools-2-0-790-0-rn:

Neuron Developer Tools [2.0.790.0] (Neuron 2.0.0 Release)
----------------------------------------------------------

Date of Release: 03/25/2022

Improvements
~~~~~~~~~~~~~~~

* NeuronPerf 1.2.0.0: Initial release of NeuronPerf. Supports PyTorch, TensorFlow, and Apache MXNet. Supports customizable JSON and CSV reports.

Bug Fixes
~~~~~~~~~

* neuron-monitor: fixed a floating point error when calculating CPU utilization.


----

.. _dev-tools-2-0-623-0-rn:

Neuron Developer Tools [2.0.623.0] (Neuron 2.0.0 Release)
----------------------------------------------------------

Date of Release: 01/20/2022

Improvements
~~~~~~~~~~~~~~~

* neuron-top - Added "all" tab that aggregates all running Neuron processes into a single view.
* neuron-top - Improved startup time to approximately 1.5 seconds in most cases.
* neuron-ls - Removed header message about updating tools from neuron-ls output.

Bug Fixes
~~~~~~~~~

* neuron-top - Reduced single CPU core usage down to 0.7% from 80% on inf1.xlarge when running neuron-top by switching to an event-driven approach for screen updates.


----

.. _dev-tools-2-0-494-0-rn:

Neuron Developer Tools [2.0.494.0] (Neuron 2.0.0 Release)
----------------------------------------------------------

Date of Release: 12/27/2021

Improvements
~~~~~~~~~~~~~~~

* Security related updates related to log4j vulnerabilities.


----

.. _dev-tools-2-0-327-0-rn:

Neuron Developer Tools [2.0.327.0] (Neuron 2.0.0 Release)
----------------------------------------------------------

Date of Release: 11/05/2021

Improvements
~~~~~~~~~~~~~~~

* Updated Neuron Runtime (which is integrated within this package) to libnrt 2.2.18.0 to fix a container issue that was preventing the use of containers when /dev/neuron0 was not present.

Bug Fixes
~~~~~~~~~

* Fixed container issue preventing use of containers when /dev/neuron0 was not present.


----

.. _dev-tools-2-0-277-0-rn:

Neuron Developer Tools [2.0.277.0] (Neuron 2.0.0 Release)
----------------------------------------------------------

Date of Release: 10/27/2021

Improvements
~~~~~~~~~~~~~~~

* Tools now support applications built with Neuron Runtime 2.x (libnrt.so).
* Updates have been made to neuron-ls and neuron-top to significantly improve the interface and utility of information provided.
* Expands neuron-monitor to include additional information when used to monitor latest Frameworks released with Neuron 1.16.0.
* neuron-cli entering maintenance mode as its use is no longer relevant when using ML Frameworks with an integrated Neuron Runtime (libnrt.so).

Breaking Changes
~~~~~~~~~~~~~~~~

* You must update to the latest Neuron Driver (aws-neuron-dkms version 2.1 or newer) for proper functionality of the new runtime library.
* neuron-cli entering maintenance mode.

Known Issues
~~~~~~~~~~~~

* None reported for this release.
