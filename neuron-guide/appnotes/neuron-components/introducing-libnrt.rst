.. _introduce-libnrt:

Introducing Neuron Runtime 2.x (libnrt.so)  
==========================================

.. contents::
   :local:
   :depth: 2


What are we changing?
---------------------

Starting with *Neuron 1.16.0* release, *Neuron Runtime 1.x* (``neuron-rtd``) is entering maintenance mode and is replaced by *Neuron Runtime 2.x*, a shared library named (``libnrt.so``). For more information on Runtime 1.x see :ref:`maintenance_rtd`.

Upgrading to ``libnrt.so`` simplifies Neuron installation and upgrade process, introduces new capabilities for allocating NeuronCores 
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
``libnrt.so`` is packaged and installed as part of Neuron framework extensions
(e.g. Neuron TensorFlow, Neuron PyTorch or Neuron MXNet), and does not require a
separate installation step. Installing Neuron Runtime as part of the Neuron
framework extensions simplifies installation and improves the user experience.
In addition, since ``libnrt.so`` is directly linked to Neuron framework
extensions, it enables faster communication between the Neuron Runtime and
Neuron Frameworks by eliminating the ``gRPC`` interface overhead.

For more information please see :ref:`introduce-libnrt-how-sdk` and :ref:`neuron-migrating-apps-neuron-to-libnrt`.


.. _libnrt-neuron-cmponents:

.. _introduce-libnrt-how-sdk:

How will this change affect the Neuron SDK?
-------------------------------------------

Neuron Driver
^^^^^^^^^^^^^

You need to use latest Neuron Driver. For successful installation and upgrade to *Neuron 1.16.0* or newer, 
you must install or upgrade to Neuron Driver (``aws-neuron-dkms``) *version 2.1.5.0* or newer. Neuron applications using *Neuron 1.16.0* will fail if 
they do not detect *Neuron Driver version 2.1.5.0* or newer. For installation and upgrade instructions see :ref:`neuron-install-guide`.

To see details of Neuron component versions please see :ref:`neuron-release-content`.

.. important ::

   For successful installation or update to Neuron 1.16.0 and newer from previous releases:
      * Stop Neuron Runtime 1.x daemon (``neuron-rtd``) by running: ``sudo systemctl stop neuron-rtd``
      * Uninstall ``neuron-rtd`` by running: ``sudo apt remove aws-neuron-runtime`` or ``sudo yum remove aws-neuron-runtime``
      * Install or upgrade to latest Neuron Driver (``aws-neuron-dkms``) by following the :ref:`neuron-install-guide` instructions.


Neuron Runtime
^^^^^^^^^^^^^^

* Installation
   Starting from *Neuron 1.16.0*, Neuron releases will no longer include the ``aws-neuron-runtime packages``, and the Neuron Runtime will be part of the Neuron 
   framework extension of choice (Neuron TensorFlow, Neuron PyTorch or Neuron MXNet). Installing any Neuron framework package will install the Neuron Runtime library 
   (``libnrt.so``).
      * For installation and upgrade instructions see :ref:`neuron-install-guide`.

* Configuring *Neuron Runtime*
   Before *Neuron 1.16.0*, configuring *Neuron Runtime 1.x* was performed through configuration files (e.g. /opt/aws/neuron/config/neuron-rtd.config).
   Starting from *Neuron 1.16.0*, configuring *Neuron Runtime 2.x* can be done through environment variables, see :ref:`nrt-configuration` for details. 

* Starting and Stopping *Neuron Runtime*
   Before introducing ``libnrt.so``, ``neuron-rtd`` ran as a daemon that communicated through a ``gRPC`` interface. Whenever ``neuron-rtd`` took ownership of a Neuron device, 
   it continued owning that device until it was stopped. This created the need to stop ``neuron-rtd`` in certain cases. With the introduction of ``libnrt.so``, stopping 
   and starting the *Neuron Runtime* is no longer needed as it runs inside the context of the application. With *Neuron Runtime 2.x*, the act of starting and stopping a Neuron application will cause ``libnrt.so`` to automatically claim or release the ownership of the required Neuron devices.
   

* NeuronCore Groups (NCG) deprecation
   Before the introduction of *Neuron Runtime 2.x*, NeuronCore Group (NCG) has been used to define an execution group of one or more NeuronCores 
   where models can be loaded and executed. It also provided separation between processes.
   
   With the introduction of *Neuron Runtime 2.x*, the strict separation of NeuronCores into groups is no longer needed and NeuronCore Groups (NCG) is 
   deprecated. see :ref:`eol-ncg` for more information.

* Running multiple *Neuron Runtimes*
   Before the introduction of ``libnrt.so``, you needed to run multiple ``neuron-rtd`` daemons to allocate Neuron devices for each ``neuron-rtd`` 
   using configuration files.
   After the introduction of ``libnrt.so``, you will no longer need to run multiple ``neuron-rtd`` daemons to allocate Neuron devices to specific Neuron application . 
   With ``libnrt.so`` allocation of NeuronCores (Neuron device include multiple NeuronCores) to a particular application is done by using ``NEURON_RT_VISIBLE_CORES`` 
   environment variable, for example:

   .. code ::

      NEURON_RT_VISIBLE_CORES=0-3 myapp1.py
      NEURON_RT_VISIBLE_CORES=4-11 myapp2.py

   See :ref:`nrt-configuration` for details. 

* Logging
   Similar to *Neuron Runtime 1.x*, *Neuron Runtime 2.x* logs to syslog (verbose logging). To make debugging easier, *Neuron Runtime 2.x* also logs to the console (error-only logging). Refer to :ref:`nrt-configuration` to see how to increase or decrease logging verbosity.

* Multi-process access to NeuronCores
    With the introduction of ``libnrt.so``, it's no longer possible to load models on the same NeuronCore from multiple processes. 
    Access to the same NeuronCore should be done from the same process. 
    Instead you can load models on the same NeuronCore using multiple threads from the same process.

    .. note ::

      For optimal performance of multi-model execution, each NeuronCore should execute single model.


* Neuron Runtime architecture
    *Neuron Runtime 2.x* is delivered as a shared library (``libnrt.so``) and is directly linked to Neuron framework extensions.
    ``libnrt.so`` is packaged and installed as part of Neuron framework extensions 
    (e.g. Neuron TensorFlow, Neuron PyTorch or Neuron MXNet), and does not require a 
    separate installation step. Installing Neuron Runtime as part of the Neuron 
    framework extensions simplifies installation and improves the user experience. 
    In addition, since ``libnrt.so`` is directly linked to Neuron framework 
    extensions, it enables faster communication between the Neuron Runtime and 
    Neuron Frameworks by eliminating the ``gRPC`` interface overhead.


Neuron framework extensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting from *Neuron 1.16.0*, Neuron framework extensions (Neuron TensorFlow, Neuron PyTorch or Neuron MXNet) will be packaged together with 
``libnrt.so``. It is required to install the ``aws-neuron-dkms`` Driver version 2.1.5.0 or newer for proper operation. The ``neuron-rtd`` daemon 
that was installed in previous releases no longer works starting with Neuron 1.16.0.

To see details of Neuron component versions please see :ref:`neuron-release-content`.


TensorFlow model server
^^^^^^^^^^^^^^^^^^^^^^^

Starting from *Neuron 1.16.0*, Neuron TensorFlow model server will be packaged together with ``libnrt.so`` and will expect ``aws-neuron-dkms`` 
*version 2.1.5.0* or newer for proper operation.

.. note ::

   The Neuron TensorFlow model server included in *Neuron 1.16.0* should run from the directory in which it was installed, as it  will not run properly if copied to a different location due to its dependency on ``libnrt.so``.

Neuron tools
^^^^^^^^^^^^

* ``neuron-cli`` - Starting from *Neuron 1.16.0*, ``neuron-cli``  enters maintenance mode, see :ref:`maintenance_neuron-cli` for more information.
* ``neuron-top`` - Starting from *Neuron 1.16.0*, ``neuron-top`` has a new user interface, see :ref:`neuron-top-ug` for more information.
* ``neuron-monitor`` - ``neuron-monitor`` was updated to support Neuron Runtime 2.x (``libnrt.so``)

  * See :ref:`neuron-monitor-ug` for a updated user guide of ``neuron-monitor``.
  * See :ref:`neuron-monitor-upg` for a list of changes between *Neuron Monitor 2.x* and *Neuron Monitor 1.0*
  * See :ref:`neuron-monitor-bwc` for how you can use *Neuron Monitor 2.x* with *Neuron Runtime 1.x* (``neuron-rtd``) .



.. _introduce-libnrt-how-user:

How will this change affect me?
-------------------------------

Neuron installation and upgrade
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


As explained in ":ref:`libnrt-neuron-cmponents`", starting from *Neuron 1.16.0*, ``libnrt.so`` requires the latest Neuron Driver (``aws-neuron-dkms``), 
in addition there is no longer the need to install ``aws-neuron-runtime``. To install Neuron or upgrade to latest Neuron version, please follow the 
installation and upgrade instructions below:

* Neuron PyTorch
   * :ref:`install-neuron-pytorch`.
   * :ref:`update-neuron-pytorch`.

* Neuron TensorFlow
   * :ref:`install-neuron-tensorflow`.
   * :ref:`update-neuron-tensorflow`.

* Neuron MXNet
   * :ref:`install-neuron-mxnet`.
   * :ref:`update-neuron-mxnet`.


.. _neuron-migrating-apps-neuron-to-libnrt:

Migrate your application to Neuron Runtime 2.x (libnrt.so) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a successful migration of your application to *Neuron 1.16.0* or newer from previous releases,  please make sure you perform the following:

#. Prerequisite
    Please read  ":ref:`libnrt-neuron-cmponents`" section.

#. Make sure you are not using *Neuron Runtime 1.x* (``aws-neuron-runtime``)   
    * Remove any code that install ``aws-neuron-runtime`` from any CI/CD scripts.
    * Stop ``neuron-rtd`` by running: ``sudo systemctl stop neuron-rtd``
    * Uninstall ``neuron-rtd`` by running: ``sudo apt remove aws-neuron-runtime`` or ``sudo yum remove aws-neuron-runtime``


#. Upgrade to your Neuron Framework of choice:
    * :ref:`update-neuron-pytorch`.
    * :ref:`update-neuron-tensorflow`.
    * :ref:`update-neuron-mxnet`.


#. If you have a code that start and/or stop ``neuron-rtd``
    Remove any code that start or stop ``neuron-rtd`` from any CI/CD scripts.
       


#. Application running multiple ``neuron-rtd``
    If your application runs multiple processes and required running multiple ``neuron-rtd`` daemons:

    * Remove the code that runs multiple ``neuron-rtd`` daemons.
    * Instead of allocating Neuron devices to ``neuron-rtd`` through configuration files, use ``NEURON_RT_VISIBLE_CORES`` environment variable to
      allocate NeuronCores. See :ref:`nrt-configuration` for details.

    If you application uses ``NEURONCORE_GROUP_SIZES``, see next item.


    .. note ::

      ``NEURON_RT_VISIBLE_CORES`` environment variable enables you to allocate NeuronCores to an application. Allocating NeuronCores improves application granularity because Neuron device include multiple NeuronCores.

#. Application running multiple processes using ``NEURONCORE_GROUP_SIZES``
    * Please consider using ``NEURON_RT_VISIBLE_CORES`` introduced in *Neuron 1.16.0* release instead of ``NEURONCORE_GROUP_SIZES`` as it is being deprecated, 
    see :ref:`nrt-configuration` for details.
   
    * Your application behavior will remain the same as before if you do not set ``NEURON_RT_VISIBLE_CORES``.

    * If you are considering migrating to ``NEURON_RT_VISIBLE_CORES``, please use the following guidelines:

      * For TensorFlow applications or PyTorch applications make sure that ``NEURONCORE_GROUP_SIZES`` is unset, or that ``NEURONCORE_GROUP_SIZES`` allocate the same or less number of NeuronCores allocated by ``NEURON_RT_VISIBLE_CORES``.
      * For MXNet applications, setting ``NEURONCORE_GROUP_SIZES`` and ``NEURON_RT_VISIBLE_CORES`` environment variables at the same time is not supported. Please use ``NEURON_RT_VISIBLE_CORES`` only.


#. Application running multiple processes accessing same NeuronCore
    If  your application accesses the same NeuronCore from multiple processes, this is no longer possible with ``libnrt.so``.
    Instead, please modify your application to access the same NeuronCore from multiple threads.

    .. note ::

      For optimal performance of multi-model execution, each NeuronCore should execute a single model.


#. Neuron Tools
    * If you are using Neuron Monitor, see :ref:`neuron-monitor-upg` for details.
    * If you are using ``neuron-cli`` please remove any call to ``neuron-cli``. For more information, see :ref:`maintenance_neuron-cli`.



#. Containers
    If your application is running within a container, and it previously executed ``neuron-rtd`` within the container, you need
    to re-build your container so it will not include or install ``aws-neuron-runtime``. See :ref:`neuron-containers` and :ref:`containers-migration-to-runtime2` for details.



Troubleshooting
---------------

Application fails to start
^^^^^^^^^^^^^^^^^^^^^^^^^^

Description
~~~~~~~~~~~

Starting from *Neuron 1.16.0* release, Neuron Runtime (``libnrt.so``) requires *Neuron Driver 2.0* or greater (``aws-neuron-dkms``). Neuron Runtime requires Neuron Driver(``aws-neuron-dkms`` package) to access Neuron devices. 

If ``aws-neuron-dkms`` is not installed then the application will fail with an error message on console and syslog that look like the following:

.. code::

   NRT:nrt_init      Unable to determine Neuron Driver version. Please check aws-neuron-dkms package is installed.

If an old ``aws-neuron-dkms`` is installed then the application will fail with an error message on console and syslog that look like the following:

.. code::

   NRT:nrt_init      This runtime requires Neuron Driver version 2.0 or greater. Please upgrade aws-neuron-dkms package.


Solution
~~~~~~~~

Please follow the installation steps in :ref:`neuron-install-guide` to install ``aws-neuron-dkms``.

Application fails to start although I installed latest ``aws-neuron-dkms``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description
~~~~~~~~~~~

Starting from *Neuron 1.16.0* release, Neuron Runtime (``libnrt.so``) require *Neuron Driver 2.0* or greater (``aws-neuron-dkms``). If an old ``aws-neuron-dkms`` is installed,  the application will fail. You may try to install ``aws-neuron-dkms`` and still face application failure, this may happen because the ``aws-neuron-dkms`` installation failed as a result of ``neuron-rtd`` daemon that is still running .


Solution
~~~~~~~~

* Stop ``neuron-rtd`` by running: ``sudo systemctl stop neuron-rtd``
* Uninstall ``neuron-rtd`` by running: ``sudo apt remove aws-neuron-runtime`` or sudo ``yum remove aws-neuron-runtime``
* Install ``aws-neuron-dkms`` by following steps in :ref:`neuron-install-guide`


Application unexpected behavior when upgrading to release *Neuron 1.16.0* or newer 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Description
~~~~~~~~~~~

When upgrading to release *Neuron 1.16.0* or newer from previous releases, the OS may include two different versions of 
*Neuron Runtime*: the ``libnrt.so`` shared library and ``neuron-rtd`` daemon. This can happen if the user didn't stop ``neuron-rtd`` daemon
or didn't make sure to uninstall the existing Neuron version before upgrade. 
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
if the user didn't make sure to uninstall the existing Neuron version before upgrade or downgrade.
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

A Neuron Core can't be shared between two applications. If an application
started using a Neuron Core all other applications trying to use the
NeuronCore would fail during runtime initialization with the following
message in the console and in syslog:

.. code:: bash

   ERROR   NRT:nrt_allocate_neuron_cores               NeuronCore(s) not available - Requested:nc1-nc1 Available:0

Solution
~~~~~~~~

Terminate the the process using NeuronCore and then try launching the application again.

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

* For development, we recommend using different environments for Neuron Framework (PyTorch,TensorFlow or MXNet) from releases before *Neuron 1.16.0* and for Neuron 
  Framework (PyTorch,TensorFlow or MXNet) from *Neuron 1.16.0* and newer, if that is not possible, please make sure to stop ``neuron-rtd`` before executing models using
  Neuron Framework (PyTorch,TensorFlow or MXNet) from *Neuron 1.16.0* and newer.

* For deployment, when you are ready to upgrade, please upgrade to Neuron Framework (PyTorch,TensorFlow or MXNet) from *Neuron 1.16.0* and newer. 
  see :ref:`neuron-migrating-apps-neuron-to-libnrt` for more information.


.. warning ::

   Executing models using Neuron Framework (PyTorch,TensorFlow or MXNet) from *Neuron 1.16.0* and newer in an environment where ``neuron-rtd`` is running may cause
   undefined behavior. Please make sure to stop ``neuron-rtd`` before executing models using Neuron Framework (PyTorch,TensorFlow or MXNet) from *Neuron 1.16.0* and newer.

Are there Neuron framework versions that will not support Neuron Runtime 2.x (``libnrt.so``)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All supported Neuron PyTorch and TensorFlow framework extensions in addition to Neuron MXnet 1.8.0 framework extensions support Neuron Runtime 2.x.

Neuron MxNet 1.5.1 does not support Neuron Runtime 2.x (``libnrt.so``) and has now entered maintenance mode. Please see :ref:`maintenance_mxnet_1_5` for details.
