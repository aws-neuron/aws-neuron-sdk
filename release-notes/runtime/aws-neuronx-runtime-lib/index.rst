.. _neuron-runtime-rn:

Neuron Runtime Release Notes
============================

Neuron Runtime consists of a kernel mode driver and C/C++ libraries which provides APIs to access Neuron Devices. The runtime itself (libnrt.so) is integrated into the ML frameworks for simplicity of deployment. The Neuron Runtime supports training models and executing inference on the Neuron Cores.

.. contents:: Table of contents
   :local:
   :depth: 1

Known issues
------------

Updated : 07/03/2024

- The ``nrt_tensor_allocate`` APIs do not support more then 4 GB (>= 4GB) sizes. Passing in a size larger than or equal to 4GB will result in datatype overflow leading to undefined behavior.

NEFF Support Table:
-------------------

Use this table to determine the version of Runtime that will support the
version of NEFF you are using. NEFF version is determined by the version
of the Neuron Compiler.

============ ===================== ===================================
NEFF Version Runtime Version Range Notes
============ ===================== ===================================
0.6          \*                    All versions of RT support NEFF 0.6
1.0          >= 1.0.6905.0         Starting support for 1.0 NEFFs 
2.0          >= 1.6.5.0            Starting support for 2.0 NEFFs 
============ ===================== ===================================

Neuron Runtime Library [2.22.14.0]
---------------------------------
Date: 09/16/2024

New in this release
^^^^^^^^^^^^^^^^^^^
* Improved the inter-node mesh algorithm to scales better for larger number of nodes and larger allreduce problem sizes

Bug fixes
^^^^^^^^^
* Implemented a fix that differentiate between out-of-memory (OOM) conditions occurring on the host system versus the device when an OOM event occurs
* Resolved a performance issue with transpose operations, which was caused by an uneven distribution of work across DMA engines

Neuron Runtime Library [2.21.41.0]
---------------------------------
Date: 07/03/2024

New in this release
^^^^^^^^^^^^^^^^^^^
* Improved collectives performance on small buffers
* Improved memory utilization by reducing the size of collective buffers
* Logging improvements including improvements for HW errors, out of bounds issues, and collectives
* Added fine grained NRT error return codes for execution errors (`reference <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-troubleshoot.html?highlight=hardware%20errors#hardware-errors>`_)

Bug fixes
^^^^^^^^^
* Fixed bug where runtime was incorrectly reporting instruction offsets to the profiler

Neuron Runtime Library [PATCH 2.20.22.0]
----------------------------------------
Date: 04/01/2024

Bug fixes
^^^^^^^^^
* Fixed a bug where setting `NEURON_SCRATCHPAD_PAGE_SIZE` to a non-power of two value could lead to unnecessary Neuron memory allocations.
* Fixed messaging so that logs of benign numerical errors do not include a full dump of runtime state.
* Fixed a bug that was causing Neuron Collectives to consume excessive amount of Neuron memory, causing out of memory errors during model load.
* Fixed a bug where the Runtime would fail to report a hardware error while the status API reported instance retirement.
* Fixed a hang in Neuron Collectives that could occur when subgraphs running on different workers had a different number of replicas.


Neuron Runtime Library [2.20.11.0]
---------------------------------
Date: 02/13/2024

New in this release
^^^^^^^^^^^^^^^^^^^
* Improved performance of collective communication operators (CC ops) by up to 30% for problem sizes smaller than 16MB. This is a typical size of CC ops when executing LLM inference.
* Added support for inter-node alltoall which is a MoE use case.
* Added NRT version check across all ranks to make sure all ranks are using the same runtime.
* Improved logging on collectives timeout during model execution.
    * "(FATAL-RT-UNDEFINED-STATE) missing collectives status on Neuron Device 0 NC 0, model model.neff - suspected hang in collectives operation 0 out of 32"
* Log HBM uncorrectable errors on timeout if any occurred during model execution.
    * "(FATAL-RT-UNDEFINED-STATE) encountered uncorrectable memory error on Neuron Device 0, execution results may be invalid.  Please terminate or start/stop this instance to recover from bad hardware."

Bug fixes
^^^^^^^^^
* Fixed bug where metrics were undercounting the amount of memory used for a loaded model.
* Fixed bug which prevented the runtime from reporting more than 32 loaded models to metrics.
* Fixed replica group signature calculation check.


Neuron Runtime Library [2.19.5.0]
---------------------------------
Date: 12/21/2023

New in this release
^^^^^^^^^^^^^^^^^^^
* Added Out-of-bound error detection logic for Gather/Scatter operations
   * Out-of-bound error message "failed to run scatter/gather (indirect memory copy), due to out-of-bound access" will be displayed on an OOB error
   * The runtime execution will return an “Out of Bound” error return code in the case an OOB error occurs
      * NRT_EXEC_OOB = 1006
* Improved Neff not supported error message to list out runtime supported features vs features used by the Neff
   * Example output: "NEFF version 2.0 uses unsupported features: [0x100000]. Neuron Runtime NEFF supported features map: [0x1ff]. Please update the aws-neuronx-runtime-lib package"
* Increased limit of multicore custom ops functions
   * Total number of CustomOps in a model has been increased to 10.
   * Note: these 10 ops have to reside in one .so, as a result, they either have to be all single core op or all multicore op.


Neuron Runtime Library [2.18.15.0]
---------------------------------
Date: 11/09/2023

Bug fixes
^^^^^^^^^
* Removed unnecessary data collection during execution logging which could impact performance.


Neuron Runtime Library [2.18.14.0]
---------------------------------
Date: 10/26/2023

New in this release
^^^^^^^^^^^^^^^^^^^
* Add beta Collectives barrier API (nrt_barrier) to nrt_experimental.h
* Improved error handling and logging for NaNs produced by intermediate calculations that do not affect output.
* Improved logging by surfacing model id on load and execution errors.
* Output a better error message when Neff fails to load due to JSON size issues, e.g. “File sg00/def.json size (8589934592) exceeds json parser maximum (4294967295)”

Bug fixes
^^^^^^^^^
* Fixed logging error message to specify Neuron Cores instead of Neuron Devices when loading unsupported collectives topology.
* Fixed segfault on error path when Neuron Device fails to initialize.


Neuron Runtime Library [2.17.7.0]
---------------------------------
Date: 9/14/2023

New in this release
^^^^^^^^^^^^^^^^^^^
* Improved logging by printing out NEFF name in debug logs of nrt_execute

Bug fixes
^^^^^^^^^
* Fixed hang that would occur when running a NEFF which contains embedding update instructions in multiple functions.
* Fixed issue where the Neuron Runtime registered the same memory multiple times to an EFA device causing applications to exceed the number of physical pages that could be registered.
* Fixed assert (``void tvm::runtime::GraphRuntime::PatchDltDataPtr(DLTensor*, uint32_t*, size_t): Assertion `tensor_get_mem_type(grt->io_tensor) == NRT_TENSOR_MEM_TYPE_MALLOC' failed.``) that occured on INF1 caused by an uninitialized pointer.
* Fixed potential hang that can occur when partial replica groups for collectives are present in a NEFF.



Neuron Runtime Library [2.16.14.0]
---------------------------------
Date: 9/01/2023

Bug fixes
^^^^^^^^^
* Fixed a segfault on failure to complete Neuron Device initialization.  New behavior will avoid the failure and escalate a fixed Neuron Runtime error code (NERR_FAIL, code 0x1)
* Improved error messages around Neuron Device initialization failures.



Neuron Runtime Library [2.16.8.0]
---------------------------------
Date: 8/09/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Add runtime version and capture time to NTFF
* Improved Neuron Device copy times for all instance types via async DMA copies
* Improved error messages for unsupported topologies (example below)

   global comm ([COMM ID]) has less channels than this replica group ([REPLICA GROUP ID]) :

   likely not enough EFA devices found if running on multiple nodes or CC not permitted on this group [[TOPOLOGY]]

* Improved logging message for collectives timeouts by adding rank id to trace logs (example below)

   [gid: [RANK ID]] exchange proxy tokens

* Improved error messages when loading NEFFs with unsupported instructions (example below)

   Unsupported hardware operator code [OPCODE] found in neff.

   Please make sure to upgrade to latest aws-neuronx-runtime-lib and aws-neuronx-collective; for detailed installation instructions visit Neuron documentation.


Bug fixes
^^^^^^^^^
* Fixed “failed to get neighbor input/output addr” error when loading collectives NEFF compiled with callgraph flow and NEFF without callgraph flow.





Neuron Runtime Library [2.15.14.0]
---------------------------------
Date: 8/09/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Reduced the contiguous memory size requirement for initializing Neuron Runtime on trn1/inf2 instance families by shrinking some of the notification buffers.  A particularly large decrease was the reduction of a 4MB error notification buffer down to 64K.  Expectation is that under memory constrained or highly fragmented memory systems, the Neuron Runtime would come up more reliably than previous versions.  



Neuron Runtime Library [2.15.11.0]
---------------------------------
Date: 7/19/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Added beta asynchronous execution feature which can reduce latency by roughly 12% for training workloads.  See Runtime Configuration guide for details on how to use the feature.
* AllReduce with All-to-all communication pattern enabled for 16 ranks on TRN1/TRN1N within the instance (intranode); choice of 16 ranks is limited to NeuronCores 0-15 or 16-31.
* Minor improvement in end-to-end execution latency after reducing the processing time required for benign error notifications.
* Reduced notification overhead by using descriptor packing improving DMA performance for memory bound workloads by up to 25%.
* Improved load speed by removing extraneous checks that were previously being performed during loads.  
* Minor performance boost to CC Ops by removing the need to sort execution end notifications.
* Bumped profiling NTFF version to version 2 to remove duplicate information which may result in hitting protobuf limits, and avoid crashing when using an older version of Neuron tools to postprocess the profile.
  Please upgrade to Neuron tools 2.12 or above to view profiles captured using this version of the Neuron runtime.



Neuron Runtime Library [2.14.8.0]
---------------------------------
Date: 6/14/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Added All-to-All All-Reduce support for Neuron Collective operations, which is expected to improve All-Reduce performance by 3-7x in most cases.
* Added more descriptive NEURON_SCRATCHPAD_PAGE_SIZE to eventually replace NEURON_RT_ONE_TMPBUF_PAGE_SIZE_MB
* Neuron Runtime is now getting the device BDF from Neuron Driver for internal use.

Bug fixes
^^^^^^^^^

* Fixed rare race condition caused by DMA memory barrier not being set for certain data transfers leading to non-determinism in outputs
* Fixed NeuronCore latency not being counted properly in Neuron metrics
* Removed stack allocation of error notifications buffer when parsing error notifications, which may lead to stack overflows on smaller stack sizes. 



Neuron Runtime Library [2.13.6.0]
---------------------------------
Date: 05/01/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Added support for internal Neuron Compiler change, Queue Set Instances, which leads to reduced NEFF footprints on Neuron Devices.  In some cases, the reduction is as much as 60% smaller DMA ring size. 

Bug fixes
^^^^^^^^^

* Fixed a rare fabric deadlock scenario (hang) in NeuronCore v2 related to notification events.
* Ensure tensor store writes are complete before synchronization event is set. 


Neuron Runtime Library [2.12.23.0]
---------------------------------
Date: 04/19/2023

Bug fixes
^^^^^^^^^

* Minor internal bug fixes. 


Neuron Runtime Library [2.12.14.0]
---------------------------------
Date: 03/28/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Added support for 16 channels and 16 EFA devices, which is required for enabling EC2 TRN1N instances with Neuron.
* Added support for hierarchical All-Reduce and Reduce-Scatter. These implementations are now used by default and provides up to 75% reduction in latency for 2MB buffers across 256 ranks.
* Added support for loading more than one Neuron Custom Operator library. 
* Added support for loading multicore Neuron Custom Operators.
* Updated INF2 to support rank 1 topology. 
* Minor improvement in model load time for small models (below 100MB).



Neuron Runtime Library [2.11.43.0]
---------------------------------
Date: 02/08/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Added support for Neuron Custom C++ operators as a beta feature. As of this release, usage of Custom C++ operators requires a reset of the Neuron Runtime after running a model which invoked a Neuron Custom C++ operator.
* Added support for a counter that enable measuring FLOPS on neuron-top and neuron-monitor. 
* Added support for LRU cache for DMA rings. 


Bug fixes
^^^^^^^^^

* Fixed load failures due to memory bounds checking for Neuron Collective Compute operations in Runtime during model load.
* Fixed an internal bug that was preventing Neuron Runtime metrics from posting.
* Fixed a bug that caused segfaults as a result of double frees and stack overflows.



Neuron Runtime Library [2.10.18.0]
---------------------------------
Date: 11/07/2022

New in this release
^^^^^^^^^^^^^^^^^^^

* Minor bug fixes and enhancements. 



Neuron Runtime Library [2.10.15.0]
---------------------------------
Date: 10/26/2022

.. _note::

   Neuron Driver version 2.5 or newer is required for this version of Neuron Runtime Library

New in this release
^^^^^^^^^^^^^^^^^^^

* Changed default runtime behavior to reset NeuronCores when initializing applications.  With this change, the reseting of the Neuron Driver after application crash is no longer necessary. The new reset functionality is controled by setting environment variable: ``NEURON_RT_RESET_CORES``, see :ref:`nrt-configuration` for more information.

Bug fixes
^^^^^^^^^

* Fixed a bug where Stochastic Rounding was not being set for collective communication operators
* Fixed an issue with triggering DMA for large tensors
* Increased default execution timeout to 30 seconds
* Fixed IOQ resetting queue to incorrect ring id value
* Updated the Neuron driver for more reliable behavior of driver device reset.  Driver no longer busy waits on reset or gets stuck waiting on reset, which caused kernel taints or caused driver unload attempts to fail.
* Fixed a bug the prevented collective communication over tensors larger than 2GB
* Fixed a bug that caused intermittent memory corruption when unloading a model
* Fixed a bug that caused the exhausting of EFA memory registration pool after multiple model reloads.




Neuron Runtime Library [2.9.64.0]
---------------------------------
Date: 10/10/2022


This release specifically adds support for training workloads on one or more EC2 TRN1 instances.

Required Neuron Driver Version: 2.5 or newer

New in this release
^^^^^^^^^^^^^^^^^^^

* Broke out runtime into a separate package called aws-neuronx-runtime-lib. 
* Added RUNPATH for discovery of libnrt.so, can be overridden with LD_LIBRARY_PATH.
* Added support for multiple collective compute operations, e.g. All-Reduce, Reduce-Scatter, All-Gather.
* Added Send/Recv operation support
* Added support for using multiple DMA engines with single pseudo embedding update instruction.
* Changed instruction buffer alignment to 32K.
* Reduced memory required during NEFF swapping.
* Enabled notifications for send/recv collectives operations.
* Added trace apis in support of execution profiling.
* Added support for TPB reset (default: off).  
* Added version checking for libnccom (aws-neuronx-collectives). 
* Added new runtime version API.
* Added 8-channel support for Trn1.
* Improved debug outputs.
* Added support for write combining on BAR4.
* Increased default execution timeout from 2 seconds to 30 seconds.
* Improved handling of zero-sized tensors









Neuron Runtime 2.x (``libnrt.so``) release [2.2.51.0]
-----------------------------------------------------

Date: 03/25/2022

* Fixed an invalid memory access that could occur when unloading models.
* Reduced severity of logging for numerical errors from ERROR to WARN.
* Improved handling of models with numerous CPU operations to avoid inference failure due to memory exhaustion.


Neuron Runtime 2.x (``libnrt.so``) release [2.2.31.0]
-----------------------------------------------------

Date: 01/20/2022

New in the release
^^^^^^^^^^^^^^^^^^

* Changed error notifications from ``WARN`` to ``ERROR`` in cases when the causing problem is non-recoverable.
* Changed handling of inference timeouts (``NERR_TIMEOUT``) to avoid failure when the timeout is related to a software thread scheduling conflict.

Bug fixes
^^^^^^^^^

* Increased the number of data queues in Neuron Runtime 2.x to match what was previously used in Neuron Runtime 1.x.  The use 
  of fewer number of data queues in Neuron Runtime 2.x was leading to crashes in a limited number of models.
* Fixed the way Neuron Runtime 2.x updates the inference end timestamp.  Previously, Neuron Runtime 2.x update of the inference 
  end timestamp would have lead to a negative latency statistics in neuron-monitor with certain models.




Neuron Runtime 2.x (``libnrt.so``) release [2.2.18.0]
-----------------------------------------------------

Date: 11/05/2021

-  Resolved an issue that affect the use of Neuron within container. In previous Neuron Runtime release (libnrt.so.2.2.15.0), when /dev/neuron0
   was not used by the application, Neuron Runtime attempted and failed to initialize /dev/neuron0 because user didn't pass /dev/neuron0 to the 
   container. this Neuron Runtime release (``libnrt.so.2.2.18.0``) allows customers to launch containers with specific NeuronDevices other 
   than /dev/neuron0.
   
   

Neuron Runtime 2.x (``libnrt.so``) release [2.2.15.0]
-----------------------------------------------------

Date: 10/27/2021

New in this release
^^^^^^^^^^^^^^^^^^^

-   :ref:`First release of Neuron Runtime 2.x <introduce-libnrt>` - In this release we are
    introducing Neuron Runtime 2.x which is a shared library named
    (``libnrt.so``) and replacing Neuron Runtime 1.x server
    (``neruon-rtd``) . Upgrading to ``libnrt.so`` improves throughput and
    latency, simplifies Neuron installation and upgrade process,
    introduces new capabilities for allocating NeuronCores to
    applications, streamlines container creation, and deprecates tools
    that are no longer needed. The new library-based runtime
    (``libnrt.so``) is integrated into Neuron’s ML Frameworks (with the exception of MXNet 1.5) and Neuron
    Tools packages directly - users no longer need to install/deploy the
    ``aws-neuron-runtime``\ package. 

    .. important::

        -  You must update to the latest Neuron Driver (``aws-neuron-dkms`` version 2.1 or newer) 
           for proper functionality of the new runtime library.
        -  Read :ref:`introduce-libnrt`
           application note that describes :ref:`why are we making this
           change <introduce-libnrt-why>` and
           how :ref:`this change will affect the Neuron
           SDK <introduce-libnrt-how-sdk>` in detail.
        -  Read :ref:`neuron-migrating-apps-neuron-to-libnrt` for detailed information of how to
           migrate your application.

