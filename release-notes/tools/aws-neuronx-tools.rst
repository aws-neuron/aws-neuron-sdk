.. _neuron-tools-rn:

Neuron System Tools
===================

.. contents:: Table of Contents
   :local:
   :depth: 2


Neuron Tools  [2.13.4.0]
------------------------
Date: 08/28/2023

New in the release:

* ``--check`` option of ``nccom-test`` now supports more data types (``fp16``, ``bf16``, ``(u)int8``, ``(u)int16``, and ``(u)int32`` are now supported in addition to ``fp32``)
* Fixed bug in ``nccom-test`` that would wait indefinitely for execution to end when running on multiple instances (``-N 2`` and higher).
* Fixed bug in ``neuron-profile`` to prevent a crash during utilization calculation


Neuron Tools  [2.12.2.0]
-------------------------
Date: 7/19/2023

New in the release:

* Bumped the max supported profiling NTFF version to version 2 to resolve crashes when postprocessing NTFFs captured with newer versions of the Neuron Runtime Library.
  When viewing profiles captured using Neuron Runtime Library 2.15 or above, please upgrade tools to 2.12.
  This version of Neuron tools remains compatible with NTFF version 1.
* Bug fixes for ``neuron-profile`` related to the calculation of some summary stats.


Neuron Tools  [2.11.10.0]
-------------------------
Date: 6/14/2023

New in the release:

* ``nccom-test`` can now show multiple latency stats in the results table, such as average or percentiles, by specifying the ``-s`` option (for example: ``-s p10 p99 avg p50``).
* First public support for ``neuron-profile`` as a standalone tool that can be used to profile executions on Neuron Devices.  Visit the Neuron Tools documentation page for more details on how to use the Neuron Profiler.


Neuron Tools  [2.10.1.0]
-------------------------

Date: 05/01/2023

New in the release:

* Added new Neuron Collectives benchmarking tool, ``nccom-test``, to enable benchmarking sweeps on various Neuron Collective Communication operations.  See new nccom-test documentation under System Tools for more details.

* Expanded support for Neuron profiling to include runtime setup/teardown times and collapsed execution of NeuronCore engines and DMA.  See Tensorboard release notes and tutorial for more details. 


Neuron Tools  [2.9.5.0]
-------------------------

Date: 03/28/2023

New in the release:

* Updated neuron-top to show effective FLOPs across all NeuronCores.


Neuron Tools  [2.8.2.0]
-------------------------
Date: 02/24/2023

New in the release:

* Updated neuron-top to show aggregated utilization/FLOPs across all NeuronCores.


Neuron Tools  [2.7.2.0]
-------------------------
Date: 02/08/2023

New in the release:

* Added support for model FLOPS metrics in both neuron-monitor and neuron-top. More details can be found in the Neuron Tools documentation.



Neuron Tools  [2.6.0.0]
-------------------------
Date: 12/09/2022

This release adds support for profiling with the Neuron Plugin for TensorBoard on TRN1.  Please check out the documentation :ref:`neuronx-plugin-tensorboard`.

New in the release:

* Updated profile post-processing for workloads executed on TRN1 


Neuron Tools  [2.5.19.0]
-------------------------
Date: 11/07/2022

New in the release:

* Minor bug fixes and improvements.


Neuron Tools  [2.5.16.0]
-------------------------
Date: 10/26/2022

New in the release:

* New ``neuron-monitor`` and ``neuron-top`` feature: **memory utilization breakdown**. This new feature provides more details on how memory is being currently used on the Neuron Devices as well as on the host instance.
* ``neuron-top``'s UI layout has been updated to accommodate the new **memory utilization breakdown** feature.
* ``neuron-monitor``'s ``inference_stats`` metric group was renamed to ``execution_stats``. While the previous release still supported ``inference_stats``, starting this release the name ``inference_stats`` is considered deprecated and can't be used anymore.

.. note ::
  For more details on the new **memory utilization breakdown** feature in ``neuron-monitor`` and ``neuron-top`` check out the full user guides: :ref:`neuron-monitor-ug` and :ref:`neuron-top-ug`.

Bug Fixes:

* Fix a rare crash in ``neuron-top`` when the instance is under heavy CPU load.
* Fix process names on the bottom tab bar of ``neuron-top`` sometimes disappearing for smaller terminal window sizes.


Neuron Tools  [2.4.6.0]
-------------------------
Date: 10/10/2022

This release adds support for both EC2 INF1 and TRN1 platforms.  Name of the package changed from aws-neuron-tools to aws-neuronx-tools.  Please remove the old package before installing the new one.

New in the release:

* Added support for ECC counters on Trn1
* Added version number output to neuron-top
* Expanded support for longer process tags in neuron-monitor.
* Removed hardware counters from the default neuron-monitor config to avoid sending repeated errors - will add back in future release.
* ``neuron-ls``  - Added option ``neuron-ls --topology`` with ASCII graphics output showing the connectivity between Neuron Devices on an instance. This feature aims to help in understanding pathways between Neuron Devices and in exploiting code or data locality.


Bug Fixes:

* Fix neuron-monitor and neuron-top to show the correct Neuron Device when running in a container where not all devices are present.


Neuron Tools [2.1.4.0]
-------------------------------

Date: 04/29/2022

* Minor updates 


Neuron Tools [2.0.790.0]
--------------------------------

Date: 03/25/2022

* ``neuron-monitor``: fixed a floating point error when calculating CPU utilization.   


Neuron Tools  [2.0.623.0]
--------------------------------

Date: 01/20/2022

New in the release:

* ``neuron-top`` - Added “all” tab that aggregates all aggregate all running Neuron processes into a single view.  
* ``neuron-top`` - Improved startup time to approximately 1.5 seconds in most cases.
* ``neuron-ls``  - Removed header message about updating tools from neuron-ls output


Bug fixes:

* ``neuron-top`` - Reduced single CPU core usage down to 0.7% from 80% on inf1.xlarge when running ``neuron-top`` by switching to an event-driven 
  approach for screen updates.  


Neuron Tools [2.0.494.0]
------------------------

Date: 12/27/2021

* Security related updates related to log4j vulnerabilities.


Neuron Tools [2.0.327.0]
------------------------

Date: 11/05/2021

* Updated Neuron Runtime (which is integrated within this package) to ``libnrt 2.2.18.0`` to fix a container issue that was preventing 
  the use of containers when /dev/neuron0 was not present. See details here :ref:`neuron-runtime-release-notes`.


Neuron Tools [2.0.277.0]
------------------------

Date: 10/27/2021

New in this release:

   -  Tools now support applications built with Neuron Runtime 2.x (``libnrt.so``).

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

   -  Updates have been made to ``neuron-ls`` and ``neuron-top`` to
      significantly improve the interface and utility of information
      provided.      
   -  Expands ``neuron-monitor`` to include additional information when
      used to monitor latest Frameworks released with Neuron 1.16.0.

         **neuron_hardware_info**
         Contains basic information about the Neuron hardware.
         ::

            "neuron_hardware_info": {
               "neuron_device_count": 16,
               "neuroncore_per_device_count": 4,
               "error": ""
            }

         -  ``neuron_device_count`` : number of available Neuron Devices
         -  ``neuroncore_per_device_count`` : number of NeuronCores present on each Neuron Device
         -  ``error`` : will contain an error string if any occurred when getting this information
            (usually due to the Neuron Driver not being installed or not running).

   -  ``neuron-cli`` entering maintenance mode as it’s use is no longer
      relevant when using ML Frameworks with an integrated Neuron
      Runtime (libnrt.so). see :ref:`maintenance_mxnet_1_5` for more information.
   -  For more information visit :ref:`neuron-tools`

