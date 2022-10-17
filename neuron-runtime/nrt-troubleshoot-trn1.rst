.. _trouble-shoot-trn1:

Neuron Runtime Troubleshooting on Trn1
======================================

.. contents::  Table of contents
   :local:
   :depth: 2


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
