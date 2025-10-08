.. _introduce-libnrt:

Introducing Neuron Runtime 2.x (libnrt.so)  
==========================================

.. contents:: Table of contents
   :local:
   :depth: 2


What are we changing?
---------------------

Starting with the *Neuron 1.16.0* release, *Neuron Runtime 1.x* (``neuron-rtd``) is entering maintenance mode and is being replaced by *Neuron Runtime 2.x*, a shared library named (``libnrt.so``). For more information on Runtime 1.x see :ref:`maintenance_rtd`.

Upgrading to ``libnrt.so`` simplifies the Neuron installation and upgrade process, introduces new capabilities for allocating NeuronCores 
to applications, streamlines container creation, and deprecates tools that are no longer needed.

This document describes the capabilities of *Neuron Runtime 2.x* in detail, provides information needed for successful installation and upgrade, 
and provides information needed for successful upgrade of Neuron applications using *Neuron Runtime 1.x* (included in releases before *Neuron 1.16.0*)
to *Neuron Runtime 2.x* (included in releases *Neuron 1.16.0* or newer).

.. _introduce-libnrt-why:

Why are we making this change?
------------------------------

Before *Neuron 1.16.0*, Neuron Runtime was delivered as a daemon (``neuron-rtd``), and communicated with Neuron framework extensions through a ``gRPC`` interface. 
``neuron-rtd`` was packaged as an ``rpm`` or ``debian`` package (``aws-neuron-runtime``) and required a separate installation step.

Starting with *Neuron 1.16.0*, *Neuron Runtime 2.x* is delivered as a shared
library (``libnrt.so``) and is directly linked to Neuron framework extensions.
``libnrt.so`` is packaged and installed as part of the Neuron framework extensions
(e.g. TensorFlow Neuron, PyTorch Neuron or MXNet Neuron), and does not require a
separate installation step. Installing Neuron Runtime as part of the Neuron
framework extensions simplifies installation and improves the user experience.
In addition, since ``libnrt.so`` is directly linked to the Neuron framework
extensions, faster communication between the Neuron Runtime and
Neuron Frameworks is enabled by eliminating the ``gRPC`` interface overhead.

For more information see :ref:`introduce-libnrt-how-sdk` and :ref:`neuron-migrating-apps-neuron-to-libnrt`.


.. _libnrt-neuron-cmponents:

.. _introduce-libnrt-how-sdk:

How will this change affect the Neuron SDK?
-------------------------------------------

Neuron Driver
^^^^^^^^^^^^^

Use the latest Neuron Driver. For successful installation and upgrade to *Neuron 1.16.0* or newer, 
you must install or upgrade to Neuron Driver (``aws-neuron-dkms``) *version 2.1.5.0* or newer. Neuron applications using *Neuron 1.16.0* will fail if 
they do not detect *Neuron Driver version 2.1.5.0* or newer. For installation and upgrade instructions see :ref:`install-guide-index`.


.. include:: ./important-neuronx-dkms.txt

To see details of Neuron component versions please see :ref:`neuron-release-content`.

.. important ::

   For successful installation or update to Neuron 1.16.0 and newer from previous releases:
      * Stop Neuron Runtime 1.x daemon (``neuron-rtd``) by running: ``sudo systemctl stop neuron-rtd``
      * Uninstall ``neuron-rtd`` by running: ``sudo apt remove aws-neuron-runtime`` or ``sudo dnf remove aws-neuron-runtime``
      * Install or upgrade to the latest Neuron Driver (``aws-neuron-dkms``) by following the :ref:`install-guide-index` instructions.
      * Starting with Neuron version 2.3, ``aws-neuron-dkms`` the package name has been changed to ``aws-neuronx-dkms``, see :ref:`neuron2-intro`


Neuron Runtime
^^^^^^^^^^^^^^

* Installation
  Starting from *Neuron 1.16.0*, Neuron releases will no longer include the ``aws-neuron-runtime packages`` and Neuron Runtime will be part of the Neuron 
  framework extension of choice (TensorFlow Neuron, PyTorch Neuron or MXNet Neuron). Installing any Neuron framework package will install the Neuron Runtime library 
  (``libnrt.so``).

      * For installation and upgrade instructions see :ref:`install-guide-index`.

* Configuring *Neuron Runtime*
   Before *Neuron 1.16.0*, *Neuron Runtime 1.x* was configured in configuration files (e.g. /opt/aws/neuron/config/neuron-rtd.config).
   Starting from *Neuron 1.16.0*, *Neuron Runtime 2.x* can be configured through environment variables. See :ref:`nrt-configuration` for details. 

* Starting and Stopping *Neuron Runtime*
   Before introducing ``libnrt.so``, ``neuron-rtd`` ran as a daemon that communicated through a ``gRPC`` interface. Whenever ``neuron-rtd`` took ownership of a Neuron device, 
   it continued owning that device until it was stopped. This created the need to stop ``neuron-rtd`` in certain cases. With the introduction of ``libnrt.so``, *Neuron Runtime* as it runs inside the context of the application. With *Neuron Runtime 2.x*, the act of starting and stopping a Neuron application causes ``libnrt.so`` to automatically claim or release ownership of the required Neuron devices.
   

* NeuronCore Groups (NCG) end-of-support
   Before the introduction of *Neuron Runtime 2.x*, NeuronCore Group (NCG) was used to define an execution group of one or more NeuronCores 
   where models could be loaded and executed. It also provided separation between processes.
   
   With the introduction of *Neuron Runtime 2.x*, strict separation of NeuronCores into groups is no longer necessary and NeuronCore Groups (NCG) has been 
   deprecated. See :ref:`eol-ncg` for more information.

* Running multiple *Neuron Runtimes*
   Before the introduction of ``libnrt.so``, it was necessary to run multiple ``neuron-rtd`` daemons to allocate Neuron devices for each ``neuron-rtd``, 
   using configuration files.
   After the introduction of ``libnrt.so``, it will no longer necessary to run multiple ``neuron-rtd`` daemons to allocate Neuron devices to a specific Neuron application. 
   With ``libnrt.so`` NeuronCores (A Neuron device includes multiple NeuronCores) are allocated to a particular application by using ``NEURON_RT_VISIBLE_CORES`` or ``NEURON_RT_NUM_CORES``
   environment variables, for example:

   .. code ::

      NEURON_RT_VISIBLE_CORES=0-3 myapp1.py
      NEURON_RT_VISIBLE_CORES=4-11 myapp2.py

   Or

   .. code ::

      NEURON_RT_NUM_CORES=3 myapp1.py &
      NEURON_RT_NUM_CORES=4 myapp2.py &



   See :ref:`nrt-configuration` for details. 

* Logging
   Similar to *Neuron Runtime 1.x*, *Neuron Runtime 2.x* logs into syslog (verbose logging). To make debugging easier, *Neuron Runtime 2.x* also logs into the console (error-only logging). Refer to :ref:`nrt-configuration` to see how to increase or decrease logging verbosity.

* Multi-process access to NeuronCores
    With the introduction of ``libnrt.so``, it is no longer possible to load models from multiple processes on the same NeuronCore.  
    A NeuronCore can only be accessed from the same process. Instead you can load models on a specific NeuronCore, using multiple threads from the same process.

    .. note::

      For optimal performance of multi-model execution, each NeuronCore executes a single model.


* Neuron Runtime architecture
    *Neuron Runtime 2.x* is delivered as a shared library (``libnrt.so``) and is directly linked to Neuron framework extensions.
    ``libnrt.so`` is packaged and installed as part of Neuron framework extensions 
    (e.g. TensorFlow Neuron, PyTorch Neuron, or MXNet Neuron), and does not require a 
    separate installation step. Installing Neuron Runtime as part of the Neuron 
    framework extensions simplifies installation and improves the user experience. 
    In addition, since ``libnrt.so`` is directly linked to Neuron framework 
    extensions, it enables faster communication between Neuron Runtime and 
    Neuron Frameworks by eliminating ``gRPC`` interface overhead.


Neuron framework extensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting from *Neuron 1.16.0*, Neuron framework extensions (TensorFlow Neuron, PyTorch Neuron, or MXNet Neuron) are packaged together with 
``libnrt.so``. It is required to install the ``aws-neuron-dkms`` Driver version 2.1.5.0 or newer for proper operation. The ``neuron-rtd`` daemon 
that was installed in previous releases no longer works starting with Neuron 1.16.0.

To see details of Neuron component versions see :ref:`neuron-release-content`.

.. :important:

   Starting Neuron version 2.3, the ``aws-neuron-dkms`` package name is changed to ``aws-neuronx-dkms``, see :ref:`neuron2-intro`

TensorFlow model server
^^^^^^^^^^^^^^^^^^^^^^^

Starting from *Neuron 1.16.0*, the TensorFlow Neuron model server is packaged together with ``libnrt.so`` and expects ``aws-neuron-dkms`` 
*version 2.1.5.0* or newer for proper operation.


.. note::

   The TensorFlow Neuron model server included in *Neuron 1.16.0* runs from the directory in which it was installed and will not run properly if copied to a different location, due to its dependency on ``libnrt.so``.

.. include:: ./important-neuronx-dkms.txt



Neuron tools
^^^^^^^^^^^^

* ``neuron-cli`` - Starting from *Neuron 1.16.0*, ``neuron-cli``  enters maintenance mode. See :ref:`maintenance_neuron-cli` for more information.
* ``neuron-top`` - Starting from *Neuron 1.16.0*, ``neuron-top`` has a new user interface. See :ref:`neuron-top-ug` for more information.
* ``neuron-monitor`` - ``neuron-monitor`` was updated to support Neuron Runtime 2.x (``libnrt.so``)

  * See :ref:`neuron-monitor-ug` for an updated user guide of ``neuron-monitor``.
  * See :ref:`neuron-monitor-upg` for a list of changes between *Neuron Monitor 2.x* and *Neuron Monitor 1.0*
  * See :ref:`neuron-monitor-bwc` for instructions for using *Neuron Monitor 2.x* with *Neuron Runtime 1.x* (``neuron-rtd``) .



.. _introduce-libnrt-how-user:

How will this change affect me?
-------------------------------

Neuron installation and upgrade
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


As explained in ":ref:`libnrt-neuron-cmponents`", starting from *Neuron 1.16.0*, ``libnrt.so`` requires the latest Neuron Driver (``aws-neuron-dkms``). 
In addition, it is no longer necessary to install ``aws-neuron-runtime``. To install Neuron or to upgrade to latest Neuron version, follow the 
installation and upgrade instructions below:

* PyTorch Neuron
   * :ref:`install-neuron-pytorch`.
   * :ref:`update-neuron-pytorch`.

* TensorFlow Neuron
   * :ref:`install-neuron-tensorflow`.
   * :ref:`update-neuron-tensorflow`.

* MXNet Neuron
   * :ref:`install-neuron-mxnet`.
   * :ref:`update-neuron-mxnet`.


.. include:: ./important-neuronx-dkms.txt


.. _neuron-migrating-apps-neuron-to-libnrt:

Migrate your application to Neuron Runtime 2.x (libnrt.so) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a successful migration from previous releases of your application to *Neuron 1.16.0* or newer, make sure you perform the following:

#. Prerequisite
    Read  ":ref:`libnrt-neuron-cmponents`".

#. Make sure you are not using *Neuron Runtime 1.x* (``aws-neuron-runtime``)   
    * Remove any code that installs ``aws-neuron-runtime`` from any CI/CD scripts.
    * Stop ``neuron-rtd`` by running ``sudo systemctl stop neuron-rtd``
    * Uninstall ``neuron-rtd`` by running ``sudo apt remove aws-neuron-runtime`` or ``sudo dnf remove aws-neuron-runtime``


#. Upgrade to your Neuron Framework of choice:
    * :ref:`update-neuron-pytorch`.
    * :ref:`update-neuron-tensorflow`.
    * :ref:`update-neuron-mxnet`.


#. If you have code that starts and/or stops ``neuron-rtd``
    Remove any code that starts or stops ``neuron-rtd`` from any CI/CD scripts.
       


#. Application running multiple ``neuron-rtd``
    If your application runs multiple processes and requires running multiple ``neuron-rtd`` daemons:

    * Remove the code that runs multiple ``neuron-rtd`` daemons.
    * Instead of allocating Neuron devices to ``neuron-rtd`` through configuration files, use ``NEURON_RT_VISIBLE_CORES`` or ``NEURON_RT_NUM_CORES`` environment variables to
      allocate NeuronCores. See :ref:`nrt-configuration` for details.

    If you application uses ``NEURONCORE_GROUP_SIZES``, see the next item.


    .. note::

      ``NEURON_RT_VISIBLE_CORES`` and ``NEURON_RT_NUM_CORES`` environment variables enable you to allocate NeuronCores to an application. Allocating NeuronCores improves application granularity, because Neuron devices include multiple NeuronCores.

#. Application running multiple processes using ``NEURONCORE_GROUP_SIZES``
    * Consider using ``NEURON_RT_VISIBLE_CORES`` or ``NEURON_RT_NUM_CORES`` environment variables instead of ``NEURONCORE_GROUP_SIZES``, which is being deprecated.  See :ref:`nrt-configuration` for details.

    * If you are using TensorFlow Neuron (``tensorflow-neuron (TF2.x)``) and you are replacing ``NEURONCORE_GROUP_SIZES=AxB`` which enables auto multicore replication, see the new API :ref:`tensorflow-ref-auto-replication-python-api` for usage and documentation.
   
    * The behavior of your application will remain the same as before if you do not set ``NEURON_RT_VISIBLE_CORES`` and do not set ``NEURON_RT_NUM_CORES``.

    * If you are considering migrating to ``NEURON_RT_VISIBLE_CORES`` or ``NEURON_RT_NUM_CORES``:

      * ``NEURON_RT_VISIBLE_CORES`` takes precedence over ``NEURON_RT_NUM_CORES``.

      * If you are migrating to ``NEURON_RT_VISIBLE_CORES``:

         * For TensorFlow applications or PyTorch applications make sure that ``NEURONCORE_GROUP_SIZES`` is unset, or that ``NEURONCORE_GROUP_SIZES`` allocates the same or smaller number of NeuronCores as allocated by ``NEURON_RT_VISIBLE_CORES``.
         * For MXNet applications, setting ``NEURONCORE_GROUP_SIZES`` and ``NEURON_RT_VISIBLE_CORES`` environment variables at the same time is not supported. Use ``NEURON_RT_VISIBLE_CORES`` only.
         * See :ref:`nrt-configuration` for more details on how to use ``NEURON_RT_VISIBLE_CORES``.


      * If you are migrating to ``NEURON_RT_NUM_CORES``:

         * Make sure that ``NEURONCORE_GROUP_SIZES`` is unset.
         * See :ref:`nrt-configuration` for more details on how to use ``NEURON_RT_NUM_CORES``.


#. Application running multiple processes accessing the same NeuronCore
    If  your application accesses the same NeuronCore from multiple processes, this is no longer possible with ``libnrt.so``.
    Instead, modify your application to access the same NeuronCore from multiple threads.

    .. note::

      Optimal performance of multi-model execution is achieved when each NeuronCore executes a single model.


#. Neuron Tools
    * If you are using Neuron Monitor, see :ref:`neuron-monitor-upg` for details.
    * If you are using ``neuron-cli`` remove any call to ``neuron-cli``. For more information, see :ref:`maintenance_neuron-cli`.



#. Containers
    If your application is running within a container, and it previously executed ``neuron-rtd`` within the container, you need
    to re-build your container, so it will not include or install ``aws-neuron-runtime``. See :ref:`neuron-containers` and :ref:`containers-migration-to-runtime2` for details.



Troubleshooting
---------------

Application fails to start
^^^^^^^^^^^^^^^^^^^^^^^^^^

Description
~~~~~~~~~~~

Starting with the *Neuron 1.16.0* release, Neuron Runtime (``libnrt.so``) requires *Neuron Driver 2.0* or greater (``aws-neuron-dkms``). Neuron Runtime requires the Neuron Driver (``aws-neuron-dkms`` package) to access Neuron devices. 

If ``aws-neuron-dkms`` is not installed, the application will fail with an error message on the console and syslog similar to the following:

.. code::

   NRT:nrt_init      Unable to determine Neuron Driver version. Please check aws-neuron-dkms package is installed.

If an old ``aws-neuron-dkms`` is installed, the application will fail with an error message on the console and syslog similar to the following:

.. code::

   NRT:nrt_init      This runtime requires Neuron Driver version 2.0 or greater. Please upgrade aws-neuron-dkms package.




Solution
~~~~~~~~

Follow the installation steps in :ref:`install-guide-index` to install ``aws-neuron-dkms``.

.. include:: ./important-neuronx-dkms.txt


Application fails to start although I installed latest ``aws-neuron-dkms``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description
~~~~~~~~~~~

Starting from the *Neuron 1.16.0* release, Neuron Runtime (``libnrt.so``) requires *Neuron Driver 2.0* or greater (``aws-neuron-dkms``). If an old ``aws-neuron-dkms`` is installed,  the application will fail. You may try to install ``aws-neuron-dkms`` and still face application failure, because the ``aws-neuron-dkms`` installation failed as a result of ``neuron-rtd`` daemon that was still running.


Solution
~~~~~~~~

* Stop ``neuron-rtd`` by running: ``sudo systemctl stop neuron-rtd``
* Uninstall ``neuron-rtd`` by running: ``sudo apt remove aws-neuron-runtime`` or sudo ``dnf remove aws-neuron-runtime``
* Install ``aws-neuron-dkms`` by following steps in :ref:`install-guide-index`

.. include:: ./important-neuronx-dkms.txt


Application unexpected behavior when upgrading to release *Neuron 1.16.0* or newer 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description
~~~~~~~~~~~

When upgrading to release *Neuron 1.16.0* or newer from previous releases, the OS may include two different versions of 
*Neuron Runtime*: the ``libnrt.so`` shared library and ``neuron-rtd`` daemon. This can happen if the user did not stop ``neuron-rtd`` daemon
or did not make sure to uninstall the existing Neuron version before upgrade. 
In this case the user application may behave unexpectedly.

Solution
~~~~~~~~

If the OS includes two different versions of *Neuron Runtime*, ``libnrt.so`` shared library and ``neuron-rtd`` daemon:

   * Before running applications that use ``neuron-rtd``, restart ``neuron-rtd`` by calling ``sudo systemctl restart neuron-rtd``.
   * Before running applications linked with ``libnrt.so``, stop ``neuron-rtd`` by calling ``sudo systemctl stop neuron-rtd``.


Application unexpected behavior when downgrading to releases before *Neuron 1.6.0* (from *Neuron 1.16.0* or newer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description
~~~~~~~~~~~

When upgrading to release *Neuron 1.16.0* or newer from previous releases, and then downgrading back to releases before *Neuron 1.6.0*, 
the OS may include two different versions of *Neuron Runtime*: the ``libnrt.so`` shared library and ``neuron-rtd`` daemon. This can happen 
if the user did not make sure to uninstall the existing Neuron version before the upgrade or downgrade.
In this case the user application may behave unexpectedly.

Solution
~~~~~~~~

If the OS include two different versions of *Neuron Runtime*, ``libnrt.so`` shared library and ``neuron-rtd`` daemon:

   * Before running applications that use ``neuron-rtd``, restart ``neuron-rtd`` by calling ``sudo systemctl restart neuron-rtd``.
   * Before running applications linked with ``libnrt.so``, stop ``neuron-rtd`` by calling ``sudo systemctl stop neuron-rtd``.



Neuron Core is in use
^^^^^^^^^^^^^^^^^^^^^

Description
~~~~~~~~~~~

A Neuron Core cannot be shared between two applications. If an application
started using a Neuron Core all other applications trying to use the
NeuronCore will fail during runtime initialization with the following
message in the console and in syslog:

.. code:: bash

   ERROR   NRT:nrt_allocate_neuron_cores               NeuronCore(s) not available - Requested:nc1-nc1 Available:0

Solution
~~~~~~~~

Terminate the the process using NeuronCore and then try launching the application.

Frequently Asked Questions (FAQ)
--------------------------------

Do I need to recompile my model to run it with Neuron Runtime 2.x (``libnrt.so``)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No. 

Do I need to change my application launch command?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No.


Can ``libnrt.so`` and ``neuron-rtd`` co-exist in the same environment?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although we recommend upgrading to the latest Neuron release, we understand that for a transition period you may continue using ``neuron-rtd`` for old releases. If you are using Neuron Framework (PyTorch,TensorFlow or MXNet) from releases before *Neuron 1.16.0*: 

* Install the latest Neuron Driver (``aws-neuron-dkms``) 

.. include:: ./important-neuronx-dkms.txt

* For development, we recommend using different environments for Neuron Framework (PyTorch,TensorFlow or MXNet) from releases before *Neuron 1.16.0* and for Neuron 
  Framework (PyTorch,TensorFlow or MXNet) from *Neuron 1.16.0* and newer. If that is not possible, make sure to stop ``neuron-rtd`` before executing models using
  Neuron Framework (PyTorch,TensorFlow or MXNet) from *Neuron 1.16.0* and newer.

* For deployment, when you are ready to upgrade, upgrade to Neuron Framework (PyTorch,TensorFlow or MXNet) from *Neuron 1.16.0* and newer. 
  See :ref:`neuron-migrating-apps-neuron-to-libnrt` for more information.


.. warning ::

   Executing models using Neuron Framework (PyTorch,TensorFlow or MXNet) from *Neuron 1.16.0* and newer in an environment where ``neuron-rtd`` is running may cause
   undefined behavior. Make sure to stop ``neuron-rtd`` before executing models using Neuron Framework (PyTorch,TensorFlow or MXNet) from *Neuron 1.16.0* and newer.

Are there Neuron framework versions that will not support Neuron Runtime 2.x (``libnrt.so``)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All supported PyTorch Neuron and TensorFlow framework extensions, in addition to Neuron MXnet 1.8.0 framework extensions support Neuron Runtime 2.x.

Neuron MxNet 1.5.1 does not support Neuron Runtime 2.x (``libnrt.so``) and has now entered maintenance mode. See :ref:`maintenance_mxnet_1_5` for details.
