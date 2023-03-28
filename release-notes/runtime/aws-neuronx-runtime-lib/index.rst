.. _neuron-runtime-rn:

Neuron Runtime Release Notes
============================

Neuron Runtime consists of a kernel mode driver and C/C++ libraries which provides APIs to access Neuron Devices. The runtime itself (libnrt.so) is integrated into the ML frameworks for simplicity of deployment. The Neuron Runtime supports training models and executing inference on the Neuron Cores.

.. contents:: Table of contents
   :local:
   :depth: 1

Known issues
------------

Updated : 04/29/2022

- In rare cases of multi-process applications running under heavy stress a model load failure my occur. This may require reloading of the Neuron Driver as a workaround.


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

* Added support for Neuron Custom C++ operators as an experimental feature. As of this release, usage of Custom C++ operators requires a reset of the Neuron Runtime after running a model which invoked a Neuron Custom C++ operator.
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

