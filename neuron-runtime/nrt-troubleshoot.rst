.. _nrt-troubleshooting:

Neuron Runtime Troubleshooting on Inf1, Inf2 and Trn1
=====================================================

This document aims to provide more information on how to fix issues you
might encounter while using the Neuron Runtime 2.x or above. For each
issue we will provide an explanation of what happened and what can
potentially correct the issue.


If your issue is not listed below or you have a more nuanced problem, contact
us via `issues <https://github.com/aws/aws-neuron-sdk/issues>`__ posted
to this repo, the `AWS Neuron developer
forum <https://forums.aws.amazon.com/forum.jspa?forumID=355>`__, or
through AWS support.


.. contents::  Table of contents
   :local:
   :depth: 2


Generic Errors
$$$$$$$$$$$$$$


Neuron Driver installation fails
--------------------------------

aws-neuron-dkms is a driver package which needs to be compiled during
installation. The compilation requires kernel headers for the instance's
kernel. ``uname -r`` can be used to find kernel version in the instance.
In some cases, the installed kernel headers might be newer than the
instance's kernel itself.

Please look at the aws-neuron-dkms installation log for message like the
following:

::

   Building for 4.14.193-149.317.amzn2.x86_64
   Module build for kernel 4.14.193-149.317.amzn2.x86_64 was skipped since the
   kernel headers for this kernel does not seem to be installed.

If installation log is not available, check whether the module is
loaded.

::

   $ lsmod | grep neuron

If the above has no output then that means ``aws-neuron-dkms``
installation is failed.

Solution
''''''''

1. Stop all applications using the NeuronCores.

2. Uninstall aws-neuron-dkms ``sudo apt remove aws-neuron-dkms`` or
   ``sudo yum remove aws-neuron-dkms``

3. Install kernel headers for the current kernel
   ``sudo apt install -y linux-headers-$(uname -r)`` or
   ``sudo yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)``

4. Install aws-neuron-dkms ``sudo apt install aws-neuron-dkms`` or
   ``sudo yum install aws-neuron-dkms``

------------

Application fails to start
--------------------------

Neuron Runtime requires Neuron Driver(aws-neuron-dkms package) to access Neuron
devices. If the driver is not installed then Neuron Runtime wont able to access the
Neuron devices and will fail with an error message in console and syslog.

If ``aws-neuron-dkms`` is not installed then the error message will be like the following::

 2021-Aug-11 18:38:27.0917 13713:13713 ERROR   NRT:nrt_init      Unable to determine Neuron Driver version. Please check aws-neuron-dkms package is installed.

If ``aws-neuron-dkms`` is installed but does not support the latest runtime then the error message will be like the following::

 2021-Aug-11 19:18:21.0661 24616:24616 ERROR   NRT:nrt_init      This runtime requires Neuron Driver version 2.0 or greater. Please upgrade aws-neuron-dkms package.

When using any supported framework from Neuron SDK version 2.5.0 and Neuron Driver (aws-neuron-dkms) versions 2.4 or older, Neuron Runtime will return the following error message::

  2022-Dec-01 09:34:12.0559   138:138   ERROR   HAL:aws_hal_tpb_pooling_write_profile       failed programming the engine

Solution
''''''''

Please follow the installation steps in :ref:`setup-guide-index` to install ``aws-neuronx-dkms``.

------------


This Neuron Runtime (compatibility id: X) is not compatible with the installed aws-neuron-dkms package
------------------------------------------------------------------------------------------------------

This error is caused by incompatibility between the Neuron Driver (dkms package) and the Runtime Library (runtime-lib package).  The driver remains backwards compatible with older versions of Neuron Runtime, but newer versions of the Runtime might rely on the functionality that is only provided by a newer driver.  In that case, an update to the newer driver is required.

In some cases the compatibility error persists even after the driver has been updated.  That happens when the update process fails to reload the driver at the end of the update.  Note that ``$ modinfo neuron``  will misleadingly show the new version because modinfo reads the version information for neuron.ko file that’s been successfully replaced.

Reload failure happens because one of the processes is still using Neuron Devices and thus the driver cannot be reloaded.  

Solution
''''''''

Check for any process that is still using the Neuron driver by running lsmod:

.. code:: bash

   ubuntu@ip-10-1-200-50:~$ lsmod | grep neuron
   neuron                237568  0
   ubuntu@ip-10-1-200-50:~$ 
   
“Used by” counter, the second number, should be 0.  If it is not, there is still a running process that is using Neuron.  Terminate that process and either:

.. code:: bash

   $ sudo rmmod neuron
   $ sudo modprobe neuron

Or simply rerun the installation one more time.  The driver logs its version in dmesg:

.. code:: base

   $ sudo dmesg
   ...
   [21531.105295] Neuron Driver Started with Version:2.9.4.0-8a6fdf292607dccc3b7059ebbe2fb24c60dfc7c4

A common culprit is a Jupyter process.  If you are using Jupyter on the instance, make sure to terminate Jupyter process before updating the driver.

------------


Neuron Core is in use
---------------------

A Neuron Core cant be shared between two applications. If an application
started using a Neuron Core all other applications trying to use the
NeuronCore would fail during runtime initialization with the following
message in the console and in syslog:

.. code:: bash

   2021-Aug-27 23:22:12.0323 28078:28078 ERROR   NRT:nrt_allocate_neuron_cores               NeuronCore(s) not available - Requested:nc1-nc1 Available:0

Solution
''''''''

Terminate any other processes that are using NeuronCore devices and then try launching the application again. If you are using Jupyter, ensure that you only have a single Jupyter kernel attempting to access the NeuronCores by restarting or shutting-down any other kernels, which will release any NeuronCores that might be in use.

------------

Unsupported NEFF Version
------------------------

While loading a model(NEFF), Neuron Runtime checks the version compatibility.
If the version the NEFF is incompatible with Runtime then it would fail the
model load with following error message:

::

   NEFF version mismatch supported: 1.1 received: 2.0

Solution
''''''''

Use compatible versions of Neuron Compiler and Runtime. Updating to the
latest version of both Neuron Compiler and Neuron Runtime is the
simplest solution. If updating one of the two is not an option, please
refer to the :ref:`neuron-runtime-release-notes`
of the Neuron Runtime to determine NEFF version support.

------------

Unsupported Hardware Operator Code
----------------------------------

While loading a model(NEFF), Neuron Runtime checks whether the hardware operators are supported or not. If unsupported,
Neuron Runtime will display the following error messages:

::
    2023-Jul-28 22:23:13.0357 101413:101422 ERROR  TDRV:translate_one_pseudo_instr_v2           Unsupported hardware operator code 214 found in neff.
    2023-Jul-28 22:23:13.0357 101413:101422 ERROR  TDRV:translate_one_pseudo_instr_v2           Please make sure to upgrade to latest aws-neuronx-runtime-lib and aws-neuronx-collective; for detailed installation instructions visit Neuron documentation.    

Solution
''''''''

Upgrade to latest Neuron Runtime and Neuron Collectives.

------------

Insufficient Memory
-------------------

While loading a model(NEFF), Neuron Runtime reserves both device and host memory
for storing weights, ifmap and ofmap of the Model. The memory consumption of
each model is different. If Neuron Runtime is unable to allocate memory then
the model load would fail with the following message in syslog

::

   kernel: [XXXXX] neuron:mc_alloc: device mempool [0:0] total 1073741568 occupied 960539030 needed 1272 available 768


Solution
''''''''

As the error is contextual to what's going on with your instance, the
exact next step is unclear. Try unloading some of the loaded models
which will free up device DRAM space. If this is still a problem, moving
to a larger Inf1 instance size with additional NeuronCores may help.

------------

Insufficient number of NeuronCores
----------------------------------

The NEFF requires more NeuronCores than available on the instance.

Check for error messages in syslog similar to:

::

  NRT:  26638:26638 ERROR  TDRV:db_vtpb_get_mla_and_tpb                 Could not find VNC id n
  NRT:  26638:26638 ERROR  NMGR:dlr_kelf_stage                          Failed to create shared io
  NRT:  26638:26638 ERROR  NMGR:stage_kelf_models                       Failed to stage graph: kelf-a.json to NeuronCore
  NRT:  26638:26638 ERROR  NMGR:kmgr_load_nn_post_metrics               Failed to load NN: xxxxxxx, err: 2

Solution
''''''''

The NeuronCores may be in use by models you are not actively using.
Ensure you've unloaded models you're not using and terminated unused applications.
If this is still a problem, moving to a larger Inf1 instance
size with additional NeuronCores may help.

--------------

Numerical Error
---------------

Neuron Devices will detect any NaN generated during execution and
report it. If Neuron Runtime sees NaNs are generated then it would
fail the execution request with Numerical Error with the following
message:

::

   nrtd[nnnnn]: ....  Error notifications found on NC .... INFER_ERROR_SUBTYPE_NUMERICAL

Solution
''''''''

This usually an indication of either error in the model or error in the
input.

Report issue to Neuron by posting the relevant details on GitHub
`issues <https://github.com/aws/aws-neuron-sdk/issues>`__.


Memory Errors
$$$$$$$$$$$$$


Transient memory errors
-----------------------

::

   Uncorrectable memory error is detected on Neuron device: 5:1 metadata: 0x2. The error might cause incorrect computational results and might affect training convergence. Please 
   terminate and restart from the last checkpoint if the convergence is impacted.

Solution
^^^^^^^^

Neuron detected a single uncorrectable bit flip in the device memory.
The execution can continue but there is a possibility of a numerical
error. If this is a concern, terminate and restart from the last known
good check point.

Persistent memory errors
------------------------

::

   Uncorrectable memory error is detected on Neuron device: 5:1 metadata: 0x2. Failing execution.

.. _solution-1:

Solution
^^^^^^^^

Multiple uncorrectable errors are detected during execution. The
execution cannot continue. This is most likely caused by faulty
hardware. Terminate and move to a different instance.

Failure to initialize Neuron
----------------------------

::

   nd0 nc0 Timestamp program stop timeout (1000 ms)
   nd0 nc0 Error while waiting for timestamp program to end on TPB eng 0
   nd0 nc0 Failed to stop neuron core
   nd0 nc0 Failed to end timestamp sync programs
   TDRV not initialized
   Failed to initialize devices, error:5

.. _solution-2:

Solution
^^^^^^^^

Previously executed application left Neuron devices in running state.
Reset Neuron devices but reloading Neuron Driver. Note, this is a
temporary workaround, future versions of Neuron will reset running
devices automatically.

::

   sudo rmmod neuron; sudo modprobe neuron

An application is trying to use more cores that are available on the instance
-----------------------------------------------------------------------------

::

   Could not open the nd1

.. _solution-3:

Solution
^^^^^^^^

Use properly sized instance. trn1.32xlarge has 32 Neuron Cores,
trn1.2xlarge has 2 Neuron Cores.


EFA and Collective Communication Errors
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Missing aws-neuronx-collectives package
---------------------------------------

**aws-neuronx-collectives** package is required to execute Collective
Communication on a single instance and across multiple instances.

::

   NCCL init error: Error opening libnccom.so, cannot use collective operations! Please set LD_LIBRARY_PATH to library location. Error: libnccom.so: cannot open shared object 
   file: No such file or directory
   Please make sure to install correct version of aws-neuronx-collectives; for detailed installation instructions visit Neuron documentation

.. _solution-4:

Solution
^^^^^^^^

Install aws-neuornx-collectives package. If the installation used
non-default destination set LD_LIBRARY_PATH.

.. _missing-efa-installer-package:

Missing efa installer package.
------------------------------

**efa-installer** package is required to execute Collective
Communication across multiple instances.

::

   Unable to run multi-instance workload.  Ofi plugin is not installed or EFA is not enabled

.. _solution-5:

Solution
^^^^^^^^

Follow the directions to install efa-installer package. Make sure to add
the path to to libfabric library to LD_LIBRARY_PATH

.. _efa-is-not-enabled-in-trn132xlarage:

EFA is not enabled in trn1.32xlarage
------------------------------------

EFA is used as a transport for Collective Communication among multiple
instances. EFA must be enabled on the instances used for multi-node
training.

::

    OFI plugin initNet() failed is EFA enabled?

.. _solution-6:

Solution
^^^^^^^^

Confirm that EFA is enabled by running lspci command and making sure
there are eight EFA devices. For example:

::

   [ec2-user@ip-10-0-13-247 ~]$ lspci -tv
   -+-[0000:a0]-+-00.0  Amazon.com, Inc. Elastic Network Adapter (ENA)
    |           +-01.0  Amazon.com, Inc. Elastic Network Adapter (ENA)
    |           +-19.0  Amazon.com, Inc. Elastic Fabric Adapter (EFA)
    |           +-1a.0  Amazon.com, Inc. Elastic Fabric Adapter (EFA)
    |           +-1b.0  Amazon.com, Inc. NeuronDevice
    |           +-1c.0  Amazon.com, Inc. NeuronDevice
    |           +-1d.0  Amazon.com, Inc. NeuronDevice
    |           +-1e.0  Amazon.com, Inc. NeuronDevice
    |           \-1f.0  Amazon.com, Inc. NVMe SSD Controller
    +-[0000:90]-+-00.0  Amazon.com, Inc. Elastic Network Adapter (ENA)
    |           +-01.0  Amazon.com, Inc. Elastic Network Adapter (ENA)
    |           +-19.0  Amazon.com, Inc. Elastic Fabric Adapter (EFA)
    |           +-1a.0  Amazon.com, Inc. Elastic Fabric Adapter (EFA)
    |           +-1b.0  Amazon.com, Inc. NeuronDevice
    |           +-1c.0  Amazon.com, Inc. NeuronDevice
    |           +-1d.0  Amazon.com, Inc. NeuronDevice
    |           +-1e.0  Amazon.com, Inc. NeuronDevice
    |           \-1f.0  Amazon.com, Inc. NVMe SSD Controller
    +-[0000:20]-+-00.0  Amazon.com, Inc. Elastic Network Adapter (ENA)
    |           +-01.0  Amazon.com, Inc. Elastic Network Adapter (ENA)
    |           +-19.0  Amazon.com, Inc. Elastic Fabric Adapter (EFA)
    |           +-1a.0  Amazon.com, Inc. Elastic Fabric Adapter (EFA)
    |           +-1b.0  Amazon.com, Inc. NeuronDevice
    |           +-1c.0  Amazon.com, Inc. NeuronDevice
    |           +-1d.0  Amazon.com, Inc. NeuronDevice
    |           +-1e.0  Amazon.com, Inc. NeuronDevice
    |           \-1f.0  Amazon.com, Inc. NVMe SSD Controller
    +-[0000:10]-+-00.0  Amazon.com, Inc. Elastic Network Adapter (ENA)
    |           +-01.0  Amazon.com, Inc. Elastic Network Adapter (ENA)
    |           +-19.0  Amazon.com, Inc. Elastic Fabric Adapter (EFA)
    |           +-1a.0  Amazon.com, Inc. Elastic Fabric Adapter (EFA)
    |           +-1b.0  Amazon.com, Inc. NeuronDevice
    |           +-1c.0  Amazon.com, Inc. NeuronDevice
    |           +-1d.0  Amazon.com, Inc. NeuronDevice
    |           +-1e.0  Amazon.com, Inc. NeuronDevice
    |           \-1f.0  Amazon.com, Inc. NVMe SSD Controller
    \-[0000:00]-+-00.0  Intel Corporation 440FX - 82441FX PMC [Natoma]
                +-01.0  Intel Corporation 82371SB PIIX3 ISA [Natoma/Triton II]
                +-01.3  Intel Corporation 82371AB/EB/MB PIIX4 ACPI
                +-03.0  Amazon.com, Inc. Device 1111
                +-04.0  Amazon.com, Inc. NVMe EBS Controller
                \-1f.0  Amazon.com, Inc. NVMe EBS Controller

Launch instances with EFA enabled and try again. If not planning to use
the instances for multi-node training or running on trn1.2xlarge, this
error message can be ignored.

Communication timeout
---------------------

Ranks exchange information during NEFF loading and before the start of
the execution. The loading/execution cannot move forward until all ranks
are ready.

::

   Timeout waiting for RX (waited 120 sec) - retrying

::

   Timeout waiting for incoming connection (waited 120 sec) - retrying

::

   Connect to localhost:33666 failed - retrying

.. _solution-7:

Solution
^^^^^^^^

The communication timeouts are not fatal. The ranks will continue
waiting forever. In most case the timeouts are caused by one of the
ranks getting delayed, usually be recompilation of a graph. The
execution is resumed after the graph is compiled (might take significant
amount of time). It is possible to determine if compilation is in
progress by checking the logs on all nodes.

Communication timeouts might also indicate that one of the nodes or
ranks is hang. If that is the case, terminate the run and restart from
the last known good check point.

.. _communication-errors:

Communication errors.
---------------------

::

   RX, connection closed by remote peer

There could be other similar messages indicating that ranks failed to
communicate.

.. _solution-8:

Solution
^^^^^^^^

One of the ranks or nodes encountered a problem and terminated.
Terminate the run and restart from the last known check point.

.. _efa-kernel-messages-dmesg-after-process-termination:

EFA Kernel messages (dmesg) after process termination.
------------------------------------------------------

::

   [298850.502143] neuron:npid_detach: neuron:npid_detach: pid=90193, slot=0
   [298850.919248] efa 0000:a0:1a.0 rdmap160s26: Failed to process command DEREG_MR (opcode 8) comp_status 7 err -22

.. _solution-9:

Solution
^^^^^^^^

When a process that executed Collective Communication terminates it
deregisters buffers that were registered with the networking stack.
There is a race condition because the Neuron driver deregisters buffers
owned by terminating process as part of the memory cleanup. The error is
benign and will be removed in the future releases.

Failure to find bootstrap interface
-----------------------------------

::

   No interface found in the same subnet as remote address fe80::1461:22ff:fe33:b471<45015>
   No usable listening interface found

.. _solution-10:

Solution
^^^^^^^^

Bootstrap code incorrectly trying to use link-local IPv6 address for
communication. This error will be fixed in the next Neuron release. In
the meantime, as a workaround, disable IPv6 on the instances.

::

   sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
   sudo sysctl -w net.ipv6.conf.default.disable_ipv6=1

Name resolution failure
-----------------------

.. code:: bash
   
     WARN Invalid NCCL_COMM_ID [compute1-st-kaena-training-0-1.pcluster-trn1-24-pdx80-2n.pcluster:41211], please use format: <ipv4>:<port> or [<ipv6>]:<port>

.. _solution-11:

Solution
^^^^^^^^

Verify that the name can be resolved by DNS by using nslookup or dig.  Currently released version fails to resolve FQDN longer than 63 characters.  This error will be fixed in the upcoming Neuron release.  In the mean time use shorter names to ensure that FQDN length does not exceed the maximum of 63 characters.


Usage of Neuron Custom C++ Operators
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Neuron Runtime timeout or GPSIMD exception
------------------------------------------

At this point, reset of Neuron Runtime is required after running a model which
invoked a Neuron Custom C++ operator. Otherwise, a Neuron Runtime timeout or
GPSIMD exception may occur.

Example Neuron Runtime timeout:

::

   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_tpb_status_notifications   Missing infer_status notification: (end:1)
   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_tpb_status_notifications   Missing infer_status notification: (end:2)
   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_tpb_status_notifications   Missing infer_status notification: (end:3)
   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_tpb_status_notifications   Missing infer_status notification: (end:4)
   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_tpb_status_notifications   Missing infer_status notification: (end:0)
   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_infer_status_notifications (FATAL-RT-UNDEFINED-STATE) inference timeout (600000 ms) on Neuron Device 0 NC 0, waiting for execution completion notification
   2023-Jan-09 20:27:41.0600 15042:15042 ERROR  NMGR:dlr_infer                               Inference completed with err: 5

Example GPSIMD exception:

::

   2023-Jan-06 22:28:01.0845 137472:137472 ERROR TDRV:pool_stdio_queue_consume_all_entries  Printing stderr from GPSIMD:
   GPSIMD EXCEPTION OCCURRED: ILLEGAL INSTRUCTION
   Subtype/Type/Cause: 0x201
   Exception PC: 0x840001E8

Solution
''''''''

If either of the above errors are seen, and ``NEURON_RT_RESET_CORES`` is set to
0, either unset it or set it to 1. This will enable the default runtime
behaviour of resetting NeuronCores when initializing applications. See
:ref:`nrt-configuration` for more information.

Also note that the timeout period can be changed by setting
``NEURON_RT_EXEC_TIMEOUT``. See :ref:`nrt-configuration` for more information.


FI_EFA_FORK_SAFE
----------------

Older Linux (<5.15) kernels require environment variable FI_EFA_FORK_SAFE to be set to 1 for the libfabric to operate correctly.  Specifically Amazon Linux 2 uses 5.10 kernel and requires the variable to be set.

When the variable is not set multi-node collective communication will be disabled.  Intra-node collective communication is still possible.  The following error message will be logged the first time a model containing collective communication is loaded:

::

   Linux kernel 5.10 requires setting FI_EFA_FORK_SAFE=1 environment variable.  Multi-node support will be disabled.  
   Please restart with FI_EFA_FORK_SAFE=1 set."
