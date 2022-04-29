.. _neuron-runtime-release-notes:

Neuron Runtime 2.x Release Notes
================================

.. contents::
   :local:
   :depth: 1

.. _neff-support-table:

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


