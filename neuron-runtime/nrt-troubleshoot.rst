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
^^^^^^^^

1. Stop all applications using the NeuronCores.

2. Uninstall aws-neuron-dkms ``sudo apt remove aws-neuron-dkms`` or
   ``sudo dnf remove aws-neuron-dkms``

3. Install kernel headers for the current kernel
   ``sudo apt install -y linux-headers-$(uname -r)`` or
   ``sudo dnf install -y "kernel-devel-uname-r = $(uname -r)"``

4. Install aws-neuron-dkms ``sudo apt install aws-neuron-dkms`` or
   ``sudo dnf install aws-neuron-dkms``

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
^^^^^^^^

Please follow the installation steps in :ref:`setup-guide-index` to install ``aws-neuronx-dkms``.

This Neuron Runtime (compatibility id: X) is not compatible with the installed aws-neuron-dkms package
------------------------------------------------------------------------------------------------------

This error is caused by incompatibility between the Neuron Driver (dkms package) and the Runtime Library (runtime-lib package).  The driver remains backwards compatible with older versions of Neuron Runtime, but newer versions of the Runtime might rely on the functionality that is only provided by a newer driver.  In that case, an update to the newer driver is required.

In some cases the compatibility error persists even after the driver has been updated.  That happens when the update process fails to reload the driver at the end of the update.  Note that ``$ modinfo neuron``  will misleadingly show the new version because modinfo reads the version information for neuron.ko file that's been successfully replaced.

Reload failure happens because one of the processes is still using Neuron Devices and thus the driver cannot be reloaded.

Solution
^^^^^^^^

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

.. code:: bash

   $ sudo dmesg
   ...
   [21531.105295] Neuron Driver Started with Version:2.9.4.0-8a6fdf292607dccc3b7059ebbe2fb24c60dfc7c4

A common culprit is a Jupyter process.  If you are using Jupyter on the instance, make sure to terminate Jupyter process before updating the driver.

Neuron Core is in use
---------------------

A Neuron Core cant be shared between two applications. If an application
started using a Neuron Core all other applications trying to use the
NeuronCore would fail during runtime initialization with the following
message in the console and in syslog:

.. code:: bash

   2021-Aug-27 23:22:12.0323 28078:28078 ERROR   NRT:nrt_allocate_neuron_cores               NeuronCore(s) not available - Requested:nc1-nc1 Available:0

Solution
^^^^^^^^

Terminate any other processes that are using NeuronCore and then try launching the application again. If you are using Jupyter, ensure that you only have a single Jupyter kernel attempting to access the NeuronCores by restarting or shutting-down any other kernels, which will release any NeuronCores that might be in use.

Unsupported NEFF Version
------------------------

While loading a model(NEFF), Neuron Runtime checks the version compatibility.
If the version the NEFF is incompatible with Runtime then it would fail the
model load with following error message:

::

   NEFF version mismatch supported: 1.1 received: 2.0

Solution
^^^^^^^^

Use compatible versions of Neuron Compiler and Runtime. Updating to the
latest version of both Neuron Compiler and Neuron Runtime is the
simplest solution. If updating one of the two is not an option, please
refer to the :ref:`runtime_rn`
of the Neuron Runtime to determine NEFF version support.

Unsupported Hardware Operator Code
----------------------------------

While loading a model(NEFF), Neuron Runtime checks whether the hardware operators are supported or not. If unsupported,
Neuron Runtime will display the following error messages:

::

    2023-Jul-28 22:23:13.0357 101413:101422 ERROR  TDRV:translate_one_pseudo_instr_v2           Unsupported hardware operator code 214 found in neff.
    2023-Jul-28 22:23:13.0357 101413:101422 ERROR  TDRV:translate_one_pseudo_instr_v2           Please make sure to upgrade to latest aws-neuronx-runtime-lib and aws-neuronx-collective; for detailed installation instructions visit Neuron documentation.

Solution
^^^^^^^^

Upgrade to latest Neuron Runtime and Neuron Collectives.


Insufficient Memory
-------------------

While loading a model(NEFF), Neuron Runtime reserves both device and host memory
for storing weights, ifmap and ofmap of the Model. The memory consumption of
each model is different. If Neuron Runtime is unable to allocate memory then
the model load would fail with the following message in syslog

::

   kernel: [XXXXX] neuron:mc_alloc: device mempool [0:0] total 1073741568 occupied 960539030 needed 1272 available 768


Solution
^^^^^^^^

As the error is contextual to what's going on with your instance, the
exact next step is unclear. Try unloading some of the loaded models
which will free up device DRAM space. If this is still a problem, moving
to a larger Inf1 instance size with additional NeuronCores may help.

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
^^^^^^^^

The NeuronCores may be in use by models you are not actively using.
Ensure you've unloaded models you're not using and terminated unused applications.
If this is still a problem, moving to a larger Inf1 instance
size with additional NeuronCores may help.

Numerical Error
---------------

Neuron Devices will detect any NaN generated during execution and
report it. If Neuron Runtime sees NaNs are generated then it would
fail the execution request with Numerical Error with the following
message:

::

   nrtd[nnnnn]: ....  Error notifications found on NC .... INFER_ERROR_SUBTYPE_NUMERICAL

Solution
^^^^^^^^

This usually an indication of either error in the model or error in the
input.

Report issue to Neuron by posting the relevant details on GitHub
`issues <https://github.com/aws/aws-neuron-sdk/issues>`__.

RuntimeError: module compiled against API version 0xf but this version of numpy is 0xe
--------------------------------------------------------------------------------------
This usually means that the numpy version used during compilation is different than the one used when executing the model.
As of Neuron SDK release 2.15, numpy versions supported in Neuron SDK are following:  numpy<=1.25.2, >=1.22.2.  Check and confirm the right
numpy version is installed and re-compile/execute the model.


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


Neuron DGE notification queue overflow
----------------------------------------

.. code:: bash

   2025-Oct-01 23:48:34.002205 516278:516289 ERROR  TDRV:exec_consume_topsp_cc_notifications     [ND 1][NC 4] execution on model /home/ubuntu/compiled-models/model.MODULE_7c055c4ac6e2851a63bb+7d89256e.neff, is stuck after all collectives operation have completed
   2025-Oct-01 23:48:34.002207 516278:516288 ERROR   NRT:nrt_infodump                            Failure: NRT_EXEC_SW_NQ_OVERFLOW in nrt_execute()
   2025-Oct-01 23:48:34.002234 516278:516288 ERROR   NRT:nrt_infodump                            LNC: 0
   2025-Oct-01 23:48:34.002260 516278:516285 ERROR  TDRV:exec_request_process_errors             [ND 1][NC 0] execution timeout (30000 ms) on model /home/ubuntu/compiled-models/model.MODULE_7c055c4ac6e2851a63bb+7d89256e.neff, potentially caused by DGE notifications enabled. Please disable it (set NEURON_RT_ENABLE_DGE_NOTIFICATIONS to 0) and try again.

Solution
^^^^^^^^

Set the environment variable ``NEURON_RT_ENABLE_DGE_NOTIFICATIONS`` to ``0`` to disable DMA Generation Engine notifications.


Neuron Runtime execution fails at out-of-bound access
-----------------------------------------------------

When a Neuron Runtime execution encounters an out-of-bound access error, the runtime logs in the stdout console will display one of the following error messages:

::

    2024-08-12 18:34:56,116::ERROR: 2024-Aug-12 18:34:56.067150 159612:159612 ERROR  TDRV:generate_custom_notification_msg        nd0:nc0:h_model.id1107: Received notification generated at runtime: failed to run embedding table update, due to out-of-bound access.
    2024-08-12 18:34:56,116::ERROR: 2024-Aug-12 18:34:56.067151 159602:159602 ERROR  TDRV:generate_custom_notification_msg        nd0:nc1:h_model.id1109: Received notification generated at runtime: failed to run scatter/gather (indirect memory copy), due to out-of-bound access.

**Cause of the Error**

An out-of-bound access error typically indicates that incorrect inputs have been provided to the model.

**How to Debug**

To troubleshoot this issue, you need to examine both the High-Level Operation (HLO) and all inputs.
Neuron Runtime can automatically dump all inputs in binary format, which can be instrumental in debugging.
To enable input dumping for each failed execution, set the following environment variable:

::

    export NEURON_RT_DBG_DUMP_INPUTS_ON_ERR=<an NRT_STATUS value>

A complete set of ``NRT_STATUS`` can be found under :ref:`The LIBNRT API Return Codes <nrt_api>`.

Once this variable is set, Neuron Runtime generates a directory in the current working directory for each failed execution at this `NRT_STATUS` value. The directory name follows this pattern:

::

    input_dump_<runtime_generated_random_number>_h_nn_<runtime_generated_execution_id>


Inside each directory, you'll find all the inputs that led to this failure, stored in binary format.
Additionally, the model name is saved in a separate file called model_name.txt within the same directory.

To disable input dump, you can set the environment variable back to 0

::

    export NEURON_RT_DBG_DUMP_INPUTS_ON_ERR=0

**Example: Debug an out-of-bound access execution**

To debug an out-of-bound (OOB) execution, which returns an NRT_STATUS code of 1006, both HLO and all inputs are required.
By setting the ``NEURON_RT_DBG_DUMP_INPUTS_ON_ERR`` environment variable to 1006, you can capture the inputs leading to an OOB execution.

For example, when an OOB error occurs, Neuron Runtime creates a directory named input_dump_424238335_h_nn_10001.
Here, 424238335 is a randomly generated number by Neuron Runtime, and 10001 is the Neuron Runtime generated execution ID.
All relevant inputs, labeled from input0 to input14, are saved in binary format within this directory.

::

    ubuntu@ip-172-31-53-90:~$ NEURON_RT_DBG_DUMP_INPUTS_ON_ERR=1006 torchrun --nproc_per_node=2 train_torchrun.py
    ......
    2024-Jun-26 00:32:47.943821 30294:32381 ERROR  TDRV:generate_custom_notification_msg        nd0:nc0:h_model.id1001: Received notification generated at runtime: failed to run scatter/gather (indirect memory copy), due to out-of-bound access. isa instruction line number = 11. model name = /home/ubuntu/token-seqlen1280-batch128-FullyUnrolled.736.2.0.62758.0a0+44863561.93f365ce40ab99133659.pb.neff
    ......
    2024-Jun-26 00:32:47.948678 30294:32381 ERROR  NMGR:dlr_infer                               Inference completed with err: 1006. mode->h_nn=1001, start_nc=0, nc_count=1
    2024-Jun-26 00:32:50.801487 30294:32381 ERROR  TDRV:tensor_dump_inputs                      15 input tensors were dumped successfully to directory /home/ubuntu/input_dump_424238335_h_nn_10001. Model name is /home/ubuntu/token-seqlen1280-batch128-FullyUnrolled.736.2.0.62758.0a0+44863561.93f365ce40ab99133659.pb.neff
    ......

    ubuntu@ip-172-31-53-90:~$ ls -lt
    total 3908900
    drwxrwxr-x 2 ubuntu ubuntu 4096 Jun 26 00:32 input_dump_424238335_h_nn_10001
    .....

    ubuntu@ip-172-31-53-90:~$ ls -lt input_dump_424238335_h_nn_10001
    total 1405192
    -rw-r—r-- 1 ubuntu ubuntu 5242880 Jun 26 00:32 input14.bin
    -rw-r—r-- 1 ubuntu ubuntu 5242880 Jun 26 00:32 input13.bin
    -rw-r—r-- 1 ubuntu ubuntu 5242880 Jun 26 00:32 input12.bin
    -rw-r—r-- 1 ubuntu ubuntu 5242880 Jun 26 00:32 input11.bin
    -rw-r—r-- 1 ubuntu ubuntu 13967360 Jun 26 00:32 input10.bin
    -rw-r—r-- 1 ubuntu ubuntu 81920 Jun 26 00:32 input8.bin
    -rw-r—r-- 1 ubuntu ubuntu 4 Jun 26 00:32 input9.bin
    -rw-r—r-- 1 ubuntu ubuntu 4 Jun 26 00:32 input6.bin
    -rw-r—r-- 1 ubuntu ubuntu 81920 Jun 26 00:32 input7.bin
    -rw-r—r-- 1 ubuntu ubuntu 16777216 Jun 26 00:32 input5.bin
    -rw-r—r-- 1 ubuntu ubuntu 131072 Jun 26 00:32 input3.bin
    -rw-r—r-- 1 ubuntu ubuntu 13967360 Jun 26 00:32 input4.bin
    -rw-r—r-- 1 ubuntu ubuntu 16777216 Jun 26 00:32 input2.bin
    -rw-r—r-- 1 ubuntu ubuntu 13967360 Jun 26 00:32 input1.bin
    -rw-r—r-- 1 ubuntu ubuntu 1342177280 Jun 26 00:32 input0.bin
    -rw-r—r-- 1 ubuntu ubuntu 9 Jun 26 00:32 model_name.txt

    ubuntu@ip-172-31-53-96:~$ cat input_dump_424238335_h_nn_10001/model_name.txt
    /home/ubuntu/token-seqlen1280-batch128-FullyUnrolled.736.2.0.62758.0a0+44863561.93f365ce40ab99133659.pb.neff


**Known Limitations**

* **HLO Access**: Neuron Runtime does not have direct access to the HLO; it must be deduced from the model name.

* **Partial Input Dumps**: If a Neuron Runtime execution fails and an exception is raised to the Neuron Framework, other ongoing Neuron Runtime executions may be terminated by the Neuron Framework. This means only one set of inputs may be fully captured, while others may be incomplete if terminated prematurely.

  * An input dump folder is considered complete when the model_name.txt file is fully written, as Neuron Runtime saves all inputs first and then writes the model_name.txt file. So you might find out the folder with the complete set of inputs by searching for the model_name.txt file.


Hardware Errors
----------------


For Trn and Inf instances, the following hardware errors are monitored by Neuron Runtime:


+-------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Error Types                         | Description                                               | Behaviors                                                                                                                     | Recommended Actions                                                                                                                                                  |
+=====================================+===========================================================+===============================================================================================================================+======================================================================================================================================================================+
| SRAM Uncorrectable                  | An on-chip SRAM encountered a parity error and produced   | 1. Instance Retirement Notice:                                                                                                | 1. Replace the EC2 instance by                                                                                                                                       |
|                                     | incorrect results.                                        | You will receive an `EC2 instance retirement notice                                                                           | `terminating <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html>`_                                                                      |
|                                     |                                                           | <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-retirement.html>`_                                              | it or `stopping then starting <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html>`_ it.                                                            |
|                                     |                                                           | within 15 minutes of experiencing this message.                                                                               |                                                                                                                                                                      |
|                                     |                                                           | EKS, EC2 Auto Scaling Groups, and AWS ParallelCluster will react to                                                           | 2. Utilize `Neuron Sysfs <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-sysfs-user-guide.html#description-for-each-metric>`_ |
|                                     |                                                           | these retirement notices according to their configured policies,                                                              | and `Neuron Monitor <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html#system-level-metric-groups>`_     |
|                                     |                                                           | but you can also automate responses to these notices yourself with                                                            | to monitor the ``sram_ecc_uncorrected`` error counts.                                                                                                                |
|                                     |                                                           | `EventBridge rules <https://repost.aws/knowledge-center/eventbridge-notification-scheduled-events>`_.                         |                                                                                                                                                                      |
|                                     |                                                           |                                                                                                                               |                                                                                                                                                                      |
|                                     |                                                           | 2. Neuron Runtime Behavior:                                                                                                   |                                                                                                                                                                      |
|                                     |                                                           | Neuron Runtime will timeout and exit with ``NRT_EXEC_COMPLETED_WITH_ERR (1004)``                                              |                                                                                                                                                                      |
|                                     |                                                           | or ``NRT_EXEC_HW_ERR_NC_UE (1202)`` return code.                                                                              |                                                                                                                                                                      |
|                                     |                                                           | You will see the following error message in runtime logs from stdout console: ``(FATAL-RT-UNDEFINED-STATE)                    |                                                                                                                                                                      |
|                                     |                                                           | [ND 0][NC 0] Uncorrectable memory error is detected, metadata: 0x16. Please terminate or stop/start this instance to prevent  |                                                                                                                                                                      |
|                                     |                                                           | future impact from the hardware error.``                                                                                      |                                                                                                                                                                      |
+-------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| HBM Unrepairable Uncorrectable      | An HBM encountered an unrepairable uncorrectable error    | 1. Instance Retirement Notice:                                                                                                | 1. Replace the EC2 instance by                                                                                                                                       |
|                                     | and produced incorrect results.                           | You will receive an `EC2 instance retirement notice                                                                           | `terminating <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html>`_                                                                      |
|                                     |                                                           | <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-retirement.html>`_                                              | it or `stopping then starting <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html>`_ it.                                                            |
|                                     |                                                           | within 15 minutes of experiencing this message.                                                                               |                                                                                                                                                                      |
|                                     |                                                           | EKS, EC2 Auto Scaling Groups, and AWS ParallelCluster will react to                                                           | 2. Utilize `Neuron Sysfs <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-sysfs-user-guide.html#description-for-each-metric>`_ |
|                                     |                                                           | these retirement notices according to their configured policies,                                                              | and `Neuron Monitor <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html#system-level-metric-groups>`_     |
|                                     |                                                           | but you can also automate responses to these notices yourself with                                                            | to monitor the ``mem_ecc_uncorrected`` error counts.                                                                                                                 |
|                                     |                                                           | `EventBridge rules <https://repost.aws/knowledge-center/eventbridge-notification-scheduled-events>`_.                         |                                                                                                                                                                      |
|                                     |                                                           |                                                                                                                               |                                                                                                                                                                      |
|                                     |                                                           | 2. Neuron Runtime Behavior:                                                                                                   |                                                                                                                                                                      |
|                                     |                                                           | Neuron Runtime will timeout and exit with ``NRT_TIMEOUT (5)``                                                                 |                                                                                                                                                                      |
|                                     |                                                           | or ``NRT_EXEC_HW_ERR_HBM_UE (1201)`` return code.                                                                             |                                                                                                                                                                      |
|                                     |                                                           | You will see the following error message in runtime logs from stdout console: ``(FATAL-RT-UNDEFINED-STATE)                    |                                                                                                                                                                      |
|                                     |                                                           | Uncorrectable HBM memory error is detected. Execution results may be invalid.                                                 |                                                                                                                                                                      |
|                                     |                                                           | Please terminate this instance to prevent future impact from the hardware error.``                                            |                                                                                                                                                                      |
+-------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| HBM Repairable Uncorrectable        | An HBM encountered a repairable uncorrectable error       | Neuron Runtime Behavior:                                                                                                      | 1. Reload the neuron driver or                                                                                                                                       |
|                                     | and produced incorrect results.                           | Neuron Runtime will timeout and exit with ``NRT_TIMEOUT (5)``                                                                 | `reboot <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-reboot.html>`_ the EC2 instance.                                                           |
|                                     |                                                           | or ``NRT_EXEC_HW_ERR_REPAIRABLE_HBM_UE (1205)`` return code.                                                                  |                                                                                                                                                                      |
|                                     |                                                           | You will see the following error message in runtime logs from stdout console: ``(FATAL-RT-UNDEFINED-STATE)                    | 2. Utilize `Neuron Sysfs <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-sysfs-user-guide.html#description-for-each-metric>`_ |
|                                     |                                                           | Uncorrectable HBM memory error is detected. Execution results may be invalid.                                                 | and `Neuron Monitor <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html#system-level-metric-groups>`_     |
|                                     |                                                           | Please reload the neuron driver or reboot your EC2 instance to prevent future impact from the hardware error.``               | to monitor the ``mem_ecc_repairable_uncorrected`` error counts.                                                                                                      |
+-------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| DMA Aborts                          | A DMA engine encountered an unrecoverable error.          | Neuron Runtime Behavior:                                                                                                      | Replace the EC2 instance by                                                                                                                                          |
|                                     |                                                           | Neuron Runtime will timeout and exit with ``NRT_TIMEOUT (5)``                                                                 | `terminating <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html>`_                                                                      |
|                                     |                                                           | or ``NRT_EXEC_HW_ERR_DMA_ABORT (1203)`` return code.                                                                          | it or `stopping then starting <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html>`_ it.                                                            |
|                                     |                                                           | You will see the following error messages in runtime logs from stdout console:                                                |                                                                                                                                                                      |
|                                     |                                                           | ``[MLA 0][NC 0] DMA TX engine 0 is in an abort state`` or                                                                     |                                                                                                                                                                      |
|                                     |                                                           | ``[MLA 0][NC 0] DMA RX engine 0 is in an abort state``                                                                        |                                                                                                                                                                      |
+-------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Hang on Collectives                 | Possibly caused by a hardware error on another worker.    | Neuron Runtime Behavior:                                                                                                      | Search for SRAM Uncorrectable, HBM Uncorrectable, DMA Aborts, and Hang on Compute errors on the other workers, and implement the recommended actions on the          |
|                                     |                                                           | Neuron Runtime will timeout and exit with ``NRT_TIMEOUT (5)``                                                                 | affected worker. Afterward, restart your workload and attempt again.                                                                                                 |
|                                     |                                                           | or ``NRT_EXEC_HW_ERR_COLLECTIVES (1200)`` return code.                                                                        |                                                                                                                                                                      |
|                                     |                                                           | You will see the following error messages in runtime logs from stdout console:                                                |                                                                                                                                                                      |
|                                     |                                                           | ``(FATAL-RT-UNDEFINED-STATE) missing collectives status                                                                       |                                                                                                                                                                      |
|                                     |                                                           | on Neuron Device 0 NC 0, model 0 - suspected hang in collectives operation 0 out of 100``                                     |                                                                                                                                                                      |
+-------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Hang on Compute                     | Unexpected software or hardware issue.                    | Neuron Runtime Behavior:                                                                                                      | Replace the EC2 instance by                                                                                                                                          |
|                                     |                                                           | Neuron Runtime will timeout and exit with ``NRT_TIMEOUT (5)``.                                                                | `terminating <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html>`_                                                                      |
|                                     |                                                           | You will see the following error messages in runtime logs from stdout console:                                                | it or `stopping then starting <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html>`_ it.                                                            |
|                                     |                                                           | ``(FATAL-RT-UNDEFINED-STATE) execution timeout (30000 ms)                                                                     |                                                                                                                                                                      |
|                                     |                                                           | on Neuron Device 0 NC 0, model xxx.neff, waiting for execution completion notification``                                      |                                                                                                                                                                      |
+-------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Upon any hardware errors, you should also expect to see the error message like the following in ``dmesg``:
``NEURON_HW_ERR=SRAM_UNCORRECTABLE_ERROR instance-id=i-0592464924bd45322 hostname=ip-172-31-61-252 nd-id=0 nc-id=0 serial-num=19fcda00f5ff6eb9 action=TERMINATE_INSTANCE``


EFA and Collective Communication Errors
-----------------------------------------

Missing aws-neuronx-collectives package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**aws-neuronx-collectives** package is required to execute Collective
Communication on a single instance and across multiple instances.

::

   NCCL init error: Error opening libnccom.so, cannot use collective operations! Please set LD_LIBRARY_PATH to library location. Error: libnccom.so: cannot open shared object
   file: No such file or directory
   Please make sure to install correct version of aws-neuronx-collectives; for detailed installation instructions visit Neuron documentation

.. _solution-4:

Solution
~~~~~~~~~

Install aws-neuornx-collectives package. If the installation used
non-default destination set LD_LIBRARY_PATH.

.. _missing-efa-installer-package:

Missing efa installer package.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**efa-installer** package is required to execute Collective
Communication across multiple instances.

::

   Unable to run multi-instance workload.  Ofi plugin is not installed or EFA is not enabled

.. _solution-5:

Solution
~~~~~~~~~

Follow the directions to install efa-installer package. Make sure to add
the path to to libfabric library to LD_LIBRARY_PATH

.. _efa-is-not-enabled-in-trn132xlarage:

EFA is not enabled in trn1.32xlarge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EFA is used as a transport for Collective Communication among multiple
instances. EFA must be enabled on the instances used for multi-node
training.

::

    OFI plugin initNet() failed is EFA enabled?

.. _solution-6:

Solution
~~~~~~~~~

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
^^^^^^^^^^^^^^^^^^^^^^

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
~~~~~~~~~

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

Communication errors
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

     WARN Invalid NCCL_COMM_ID [compute1-dy-training-0-1.pcluster-trn1-24-pdx80-2n.pcluster:41211], please use format: <ipv4>:<port> or [<ipv6>]:<port>

.. _solution-11:

Solution
^^^^^^^^

Verify that the name can be resolved by DNS by using nslookup or dig.  Currently released version fails to resolve FQDN longer than 63 characters.  This error will be fixed in the upcoming Neuron release.  In the mean time use shorter names to ensure that FQDN length does not exceed the maximum of 63 characters.

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
^^^^^^^^

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

.. code-block::

   Linux kernel 5.10 requires setting FI_EFA_FORK_SAFE=1 environment variable.  Multi-node support will be disabled.
   Please restart with FI_EFA_FORK_SAFE=1 set."


Neuron driver cannot be uninstalled
------------------------------------

If you attempt to uninstall the Neuron driver on Ubuntu with ``sudo dpkg -r aws-neuronx-dkms``, you may get an error like this:

.. code-block::

   Removing aws-neuronx-dkms (2.x) ...
   Neuron module is currently loaded. Attempting to unload...
   ERROR: Cannot unload neuron module - it is currently in use.
   Please stop all processes using the neuron module before uninstalling.
   dpkg: error processing package aws-neuronx-dkms (--remove):
   installed aws-neuronx-dkms package pre-removal script subprocess returned error exit status 1
   Errors were encountered while processing:
   aws-neuronx-dkms

On Amazon Linux, you get a similar error if you run ``sudo rpm -e aws-neuronx-dkms`` to uninstall the driver:

.. code-block::
   
   Uninstall of aws-neuronx module (version 2.x) beginning:
   Neuron module is currently loaded. Attempting to unload...
   ERROR: Cannot unload neuron module - it is currently in use.
   Please stop all processes using the neuron module before uninstalling.
   error: %preun(aws-neuronx-dkms-2.x-dkms.noarch) scriptlet failed, exit status 1
   error: aws-neuronx-dkms-2.x-dkms.noarch: erase failed

Usually, this just means you still have an active process using the driver. Killing that process will allow the driver to be unloaded/uninstalled. But if for some rare reason the driver is stuck, one remediation is to first force uninstall the driver, and then reboot. 

Solution
^^^^^^^^

Force-uninstall the Neuron driver.

.. warning:: Force-uninstalling the driver runs the risk of causing system instability or causing a kernel panic. Reboot your instance immediately after uninstalling it.

To force-uninstall the driver on Ubuntu instances:

.. code-block::

   sudo dkms remove aws-neuronx/<version> --all
   sudo dpkg -r --force-all aws-neuronx-dkms

To force-uninstall the driver on Amazon Linux instances:

.. code-block::

   sudo dkms remove aws-neuronx/<version> --all
   sudo rpm -e --noscript aws-neuronx-dkms
