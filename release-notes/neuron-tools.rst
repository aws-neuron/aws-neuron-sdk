.. _neuron-tools-rn:

Neuron Tools 2.x Release Notes
==============================

This documents lists the release notes for AWS Neuron tools. Neuron
tools are used for debugging, profiling and gathering inferentia system
information.

.. contents:: Table of Contents
   :local:
   :depth: 1



Neuron Tools release [2.0.327.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 01/20/2022

New in the release
------------------

* ``neuron-top`` - Added “all” tab that aggregates all aggregate all running Neuron processes into a single view.  
* ``neuron-top`` - Improved startup time to approximately 1.5 seconds in most cases.
* ``neuron-ls``  - Removed header message about updating tools from neuron-ls output

Bug fixes
---------

* ``neuron-top`` - Reduced single CPU core usage down to 0.7% from 80% on inf1.xlarge when running ``neuron-top`` by switching to an event-driven 
  approach for screen updates.  




Neuron Tools release [2.0.327.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 11/05/2021

* Updated Neuron Runtime (which is integrated within this package) to ``libnrt 2.2.18.0`` to fix a container issue that was preventing 
  the use of containers when /dev/neuron0 was not present. See details here :ref:`neuron-runtime-release-notes`.


Neuron Tools release [2.0.277.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 10/27/2021

New in this release
-------------------

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

