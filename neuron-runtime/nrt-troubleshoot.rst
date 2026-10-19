.. meta::
   :description: Diagnose and fix common AWS Neuron Runtime issues on Inf1, Inf2, and Trn1, including driver installation, model load, memory, hardware, and collective communication errors.
   :date-modified: 2026-10-13

.. _nrt-troubleshooting:

Neuron Runtime troubleshooting on Inf1, Inf2, and Trn1
======================================================

This page covers common issues with Neuron Runtime 2.x and later. Each entry
describes the symptom, explains the cause, and gives the steps to correct it.

If your issue is not listed here, contact us through
`AWS Neuron GitHub issues <https://github.com/aws-neuron/aws-neuron-sdk/issues>`__
or AWS support.

.. contents::  Table of contents
   :local:
   :depth: 2

Neuron driver installation fails
--------------------------------

``aws-neuronx-dkms`` is a driver package that is compiled during installation.
Compilation requires the kernel headers for the instance's kernel. Use
``uname -r`` to find the kernel version on the instance. In some cases the
installed kernel headers are newer than the instance's kernel.

Check the ``aws-neuronx-dkms`` installation log for a message like the
following:

.. code-block:: text

   Building for 4.14.193-149.317.amzn2.x86_64
   Module build for kernel 4.14.193-149.317.amzn2.x86_64 was skipped since the
   kernel headers for this kernel does not seem to be installed.

If the installation log is not available, check whether the module is loaded:

.. code-block:: bash

   $ lsmod | grep neuron

If this produces no output, the ``aws-neuronx-dkms`` installation failed.

Solution
^^^^^^^^

1. Stop all applications using the NeuronCores.

2. Uninstall the driver: ``sudo apt remove aws-neuronx-dkms`` or
   ``sudo dnf remove aws-neuronx-dkms``.

3. Install kernel headers for the current kernel:
   ``sudo apt install -y linux-headers-$(uname -r)`` or
   ``sudo dnf install -y "kernel-devel-uname-r = $(uname -r)"``.

4. Install the driver: ``sudo apt install aws-neuronx-dkms`` or
   ``sudo dnf install aws-neuronx-dkms``.

Application fails to start
--------------------------

Neuron Runtime requires the Neuron driver (``aws-neuronx-dkms`` package) to
access Neuron devices. If the driver is not installed, Neuron Runtime cannot
access the devices and fails with an error message in the console and syslog.

If ``aws-neuronx-dkms`` is not installed, the error message looks like the
following:

.. code-block:: text

   2021-Aug-11 18:38:27.0917 13713:13713 ERROR   NRT:nrt_init      Unable to determine Neuron Driver version. Please check aws-neuron-dkms package is installed.

If ``aws-neuronx-dkms`` is installed but does not support the latest runtime,
the error message looks like the following:

.. code-block:: text

   2021-Aug-11 19:18:21.0661 24616:24616 ERROR   NRT:nrt_init      This runtime requires Neuron Driver version 2.0 or greater. Please upgrade aws-neuron-dkms package.

When using a supported framework from Neuron SDK 2.5.0 with Neuron driver
(``aws-neuronx-dkms``) version 2.4 or older, Neuron Runtime returns the
following error:

.. code-block:: text

   2022-Dec-01 09:34:12.0559   138:138   ERROR   HAL:aws_hal_tpb_pooling_write_profile       failed programming the engine

Solution
^^^^^^^^

Follow the installation steps in :ref:`setup-guide-index` to install
``aws-neuronx-dkms``.

This Neuron Runtime is not compatible with the installed aws-neuronx-dkms package
---------------------------------------------------------------------------------

This error (``compatibility id: X``) indicates an incompatibility between the
Neuron driver (``dkms`` package) and the runtime library (``runtime-lib``
package). The driver stays backward compatible with older Neuron Runtime
versions, but a newer runtime can require functionality that only a newer
driver provides. When that happens, update to the newer driver.

Sometimes the compatibility error persists after you update the driver. That
happens when the update process fails to reload the driver at the end of the
update. Note that ``modinfo neuron`` misleadingly shows the new version,
because ``modinfo`` reads the version information from the ``neuron.ko`` file
that was already replaced.

The reload fails when a process is still using Neuron devices, which prevents
the driver from reloading.

Solution
^^^^^^^^

Check for any process still using the Neuron driver by running ``lsmod``:

.. code-block:: bash

   ubuntu@ip-10-1-200-50:~$ lsmod | grep neuron
   neuron                237568  0
   ubuntu@ip-10-1-200-50:~$

The "Used by" counter, the second number, should be 0. If it is not, a running
process is still using Neuron. Terminate that process, then either reload the
driver:

.. code-block:: bash

   $ sudo rmmod neuron
   $ sudo modprobe neuron

or rerun the installation once more. The driver logs its version in ``dmesg``:

.. code-block:: bash

   $ sudo dmesg
   ...
   [21531.105295] Neuron Driver Started with Version:2.9.4.0-8a6fdf292607dccc3b7059ebbe2fb24c60dfc7c4

A common cause is a Jupyter process. If you are running Jupyter on the
instance, terminate it before updating the driver.

NeuronCore is in use
--------------------

A NeuronCore cannot be shared between two applications. After one application
starts using a NeuronCore, any other application that tries to use it fails
during runtime initialization with the following message in the console and
syslog:

.. code-block:: text

   2021-Aug-27 23:22:12.0323 28078:28078 ERROR   NRT:nrt_allocate_neuron_cores               NeuronCore(s) not available - Requested:nc1-nc1 Available:0

Solution
^^^^^^^^

Terminate any other processes using the NeuronCore, then launch the
application again. If you are using Jupyter, ensure that only a single Jupyter
kernel accesses the NeuronCores by restarting or shutting down other kernels,
which releases any NeuronCores in use.

Unsupported NEFF version
------------------------

When loading a model (NEFF), Neuron Runtime checks version compatibility. If
the NEFF version is incompatible with the runtime, the model load fails with
the following error:

.. code-block:: text

   NEFF version mismatch supported: 1.1 received: 2.0

Solution
^^^^^^^^

Use compatible versions of the Neuron compiler and runtime. Updating both to
the latest version is the simplest fix. If you cannot update one of the two,
see the :ref:`runtime_rn` to determine NEFF version support.

Unsupported hardware operator code
----------------------------------

When loading a model (NEFF), Neuron Runtime checks whether the hardware
operators are supported. If they are not, Neuron Runtime displays the following
error messages:

.. code-block:: text

    2023-Jul-28 22:23:13.0357 101413:101422 ERROR  TDRV:translate_one_pseudo_instr_v2           Unsupported hardware operator code 214 found in neff.
    2023-Jul-28 22:23:13.0357 101413:101422 ERROR  TDRV:translate_one_pseudo_instr_v2           Please make sure to upgrade to latest aws-neuronx-runtime-lib and aws-neuronx-collective; for detailed installation instructions visit Neuron documentation.

Solution
^^^^^^^^

Upgrade to the latest Neuron Runtime and Neuron Collectives.

Insufficient memory
-------------------

When loading a model (NEFF), Neuron Runtime reserves both device and host
memory to store the model's weights, ifmap, and ofmap. Memory consumption
differs per model. If Neuron Runtime cannot allocate the memory, the model load
fails with the following message in syslog:

.. code-block:: text

   kernel: [XXXXX] neuron:mc_alloc: device mempool [0:0] total 1073741568 occupied 960539030 needed 1272 available 768

Solution
^^^^^^^^

Because this error depends on what is running on your instance, the exact next
step varies. Try unloading some of the loaded models to free device DRAM. If
the problem persists, moving to a larger instance size with more NeuronCores
can help.

Insufficient number of NeuronCores
----------------------------------

The NEFF requires more NeuronCores than are available on the instance. Check
syslog for error messages similar to:

.. code-block:: text

  NRT:  26638:26638 ERROR  TDRV:db_vtpb_get_mla_and_tpb                 Could not find VNC id n
  NRT:  26638:26638 ERROR  NMGR:dlr_kelf_stage                          Failed to create shared io
  NRT:  26638:26638 ERROR  NMGR:stage_kelf_models                       Failed to stage graph: kelf-a.json to NeuronCore
  NRT:  26638:26638 ERROR  NMGR:kmgr_load_nn_post_metrics               Failed to load NN: xxxxxxx, err: 2

Solution
^^^^^^^^

The NeuronCores may be in use by models you are not actively using. Unload
models you are not using and terminate unused applications. If the problem
persists, moving to a larger instance size with more NeuronCores can help.

Numerical error
---------------

Neuron devices detect any NaN generated during execution and report it. When
Neuron Runtime detects generated NaNs, it fails the execution request with a
numerical error and the following message:

.. code-block:: text

   nrtd[nnnnn]: ....  Error notifications found on NC .... INFER_ERROR_SUBTYPE_NUMERICAL

Solution
^^^^^^^^

This usually indicates an error in either the model or the input. Report the
issue to Neuron by posting the relevant details on GitHub
`issues <https://github.com/aws/aws-neuron-sdk/issues>`__.

RuntimeError: module compiled against API version 0xf but this version of numpy is 0xe
--------------------------------------------------------------------------------------

This usually means the numpy version used during compilation differs from the
one used when executing the model. As of Neuron SDK release 2.15, the supported
numpy versions are ``numpy<=1.25.2, >=1.22.2``. Confirm the correct numpy
version is installed, then recompile and run the model again.

Failure to initialize Neuron
----------------------------

.. code-block:: text

   nd0 nc0 Timestamp program stop timeout (1000 ms)
   nd0 nc0 Error while waiting for timestamp program to end on TPB eng 0
   nd0 nc0 Failed to stop neuron core
   nd0 nc0 Failed to end timestamp sync programs
   TDRV not initialized
   Failed to initialize devices, error:5

Solution
^^^^^^^^

A previously executed application left the Neuron devices in a running state.
Reset the Neuron devices by reloading the Neuron driver. This is a temporary
workaround; future versions of Neuron will reset running devices automatically.

.. code-block:: bash

   sudo rmmod neuron; sudo modprobe neuron

An application is trying to use more cores than are available on the instance
-----------------------------------------------------------------------------

.. code-block:: text

   Could not open the nd1

Solution
^^^^^^^^

Use a properly sized instance. trn1.32xlarge has 32 NeuronCores;
trn1.2xlarge has 2 NeuronCores.

Neuron DGE notification queue overflow
--------------------------------------

.. code-block:: text

   2025-Oct-01 23:48:34.002205 516278:516289 ERROR  TDRV:exec_consume_topsp_cc_notifications     [ND 1][NC 4] execution on model /home/ubuntu/compiled-models/model.MODULE_7c055c4ac6e2851a63bb+7d89256e.neff, is stuck after all collectives operation have completed
   2025-Oct-01 23:48:34.002207 516278:516288 ERROR   NRT:nrt_infodump                            Failure: NRT_EXEC_SW_NQ_OVERFLOW in nrt_execute()
   2025-Oct-01 23:48:34.002234 516278:516288 ERROR   NRT:nrt_infodump                            LNC: 0
   2025-Oct-01 23:48:34.002260 516278:516285 ERROR  TDRV:exec_request_process_errors             [ND 1][NC 0] execution timeout (30000 ms) on model /home/ubuntu/compiled-models/model.MODULE_7c055c4ac6e2851a63bb+7d89256e.neff, potentially caused by DGE notifications enabled. Please disable it (set NEURON_RT_ENABLE_DGE_NOTIFICATIONS to 0) and try again.

Solution
^^^^^^^^

Set the environment variable ``NEURON_RT_ENABLE_DGE_NOTIFICATIONS`` to ``0`` to
disable DMA Generation Engine notifications.

Neuron Runtime execution fails at out-of-bound access
-----------------------------------------------------

When a Neuron Runtime execution encounters an out-of-bound access error, the
runtime logs in the stdout console show one of the following messages:

.. code-block:: text

    2024-08-12 18:34:56,116::ERROR: 2024-Aug-12 18:34:56.067150 159612:159612 ERROR  TDRV:generate_custom_notification_msg        nd0:nc0:h_model.id1107: Received notification generated at runtime: failed to run embedding table update, due to out-of-bound access.
    2024-08-12 18:34:56,116::ERROR: 2024-Aug-12 18:34:56.067151 159602:159602 ERROR  TDRV:generate_custom_notification_msg        nd0:nc1:h_model.id1109: Received notification generated at runtime: failed to run scatter/gather (indirect memory copy), due to out-of-bound access.

Cause
^^^^^

An out-of-bound access error typically indicates that incorrect inputs were
provided to the model.

How to debug
^^^^^^^^^^^^

To troubleshoot this issue, examine both the High-Level Operation (HLO) and all
inputs. Neuron Runtime can dump all inputs in binary format, which helps with
debugging. To enable input dumping for each failed execution, set the following
environment variable:

.. code-block:: bash

    export NEURON_RT_DBG_DUMP_INPUTS_ON_ERR=<an NRT_STATUS value>

For the complete set of ``NRT_STATUS`` values, see
:ref:`The LIBNRT API Return Codes <nrt_api>`.

Once this variable is set, Neuron Runtime creates a directory under
``/tmp/neuron-input-dump`` for each failed execution at this ``NRT_STATUS``
value. To write the dumps somewhere else, set
``NEURON_RT_DBG_INPUT_DUMP_DIRECTORY`` to the directory you want. The
per-execution directory name follows this pattern:

.. code-block:: text

    input_dump_<runtime_generated_random_number>_h_nn_<runtime_generated_execution_id>

Each directory contains all the inputs that led to the failure, stored in
binary format. The model name is saved in a separate file, ``model_name.txt``,
in the same directory.

To disable the input dump, set the environment variable back to 0:

.. code-block:: bash

    export NEURON_RT_DBG_DUMP_INPUTS_ON_ERR=0

Example: debug an out-of-bound access execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To debug an out-of-bound (OOB) execution, which returns an ``NRT_STATUS`` code
of 1006, you need both the HLO and all inputs. Set
``NEURON_RT_DBG_DUMP_INPUTS_ON_ERR`` to 1006 to capture the inputs leading to an
OOB execution.

For example, when an OOB error occurs, Neuron Runtime creates a directory named
``input_dump_424238335_h_nn_10001``. Here, 424238335 is a randomly generated
number, and 10001 is the Neuron Runtime execution ID. All relevant inputs,
labeled input0 to input14, are saved in binary format in this directory.

.. code-block:: text

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
    -rw-r--r-- 1 ubuntu ubuntu 5242880 Jun 26 00:32 input14.bin
    -rw-r--r-- 1 ubuntu ubuntu 5242880 Jun 26 00:32 input13.bin
    -rw-r--r-- 1 ubuntu ubuntu 5242880 Jun 26 00:32 input12.bin
    -rw-r--r-- 1 ubuntu ubuntu 5242880 Jun 26 00:32 input11.bin
    -rw-r--r-- 1 ubuntu ubuntu 13967360 Jun 26 00:32 input10.bin
    -rw-r--r-- 1 ubuntu ubuntu 81920 Jun 26 00:32 input8.bin
    -rw-r--r-- 1 ubuntu ubuntu 4 Jun 26 00:32 input9.bin
    -rw-r--r-- 1 ubuntu ubuntu 4 Jun 26 00:32 input6.bin
    -rw-r--r-- 1 ubuntu ubuntu 81920 Jun 26 00:32 input7.bin
    -rw-r--r-- 1 ubuntu ubuntu 16777216 Jun 26 00:32 input5.bin
    -rw-r--r-- 1 ubuntu ubuntu 131072 Jun 26 00:32 input3.bin
    -rw-r--r-- 1 ubuntu ubuntu 13967360 Jun 26 00:32 input4.bin
    -rw-r--r-- 1 ubuntu ubuntu 16777216 Jun 26 00:32 input2.bin
    -rw-r--r-- 1 ubuntu ubuntu 13967360 Jun 26 00:32 input1.bin
    -rw-r--r-- 1 ubuntu ubuntu 1342177280 Jun 26 00:32 input0.bin
    -rw-r--r-- 1 ubuntu ubuntu 9 Jun 26 00:32 model_name.txt

    ubuntu@ip-172-31-53-96:~$ cat input_dump_424238335_h_nn_10001/model_name.txt
    /home/ubuntu/token-seqlen1280-batch128-FullyUnrolled.736.2.0.62758.0a0+44863561.93f365ce40ab99133659.pb.neff

Known limitations
^^^^^^^^^^^^^^^^^

* **HLO access:** Neuron Runtime does not have direct access to the HLO; it must
  be deduced from the model name.

* **Partial input dumps:** If a Neuron Runtime execution fails and raises an
  exception to the Neuron framework, other ongoing Neuron Runtime executions may
  be terminated by the framework. This means only one set of inputs may be fully
  captured, while others may be incomplete if terminated prematurely.

  * An input dump folder is complete when the ``model_name.txt`` file is fully
    written, because Neuron Runtime saves all inputs first and then writes
    ``model_name.txt``. To find the folder with the complete set of inputs,
    search for the ``model_name.txt`` file.

Hardware errors
---------------

For Trn and Inf instances, Neuron Runtime monitors the following hardware
errors. Each error below lists its cause, the runtime behavior and return
codes, and the recommended actions.

SRAM uncorrectable
^^^^^^^^^^^^^^^^^^

An on-chip SRAM encountered a parity error and produced incorrect results.

Instance retirement notice: you receive an
`EC2 instance retirement notice <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-retirement.html>`__
within 15 minutes of this message. EKS, EC2 Auto Scaling groups, and AWS
ParallelCluster react to these retirement notices according to their configured
policies, and you can also automate responses with
`EventBridge rules <https://repost.aws/knowledge-center/eventbridge-notification-scheduled-events>`__.

Neuron Runtime behavior: Neuron Runtime times out and exits with return code
``NRT_EXEC_COMPLETED_WITH_ERR (1004)`` or ``NRT_EXEC_HW_ERR_NC_UE (1202)``. The
stdout console shows:

.. code-block:: text

   (FATAL-RT-UNDEFINED-STATE) [ND 0][NC 0] Uncorrectable memory error is detected, metadata: 0x16. Please terminate or stop/start this instance to prevent future impact from the hardware error.

Recommended actions:

1. Replace the EC2 instance by
   `terminating <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html>`__
   it or `stopping then starting <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html>`__ it.
2. Use `Neuron Sysfs <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-sysfs-user-guide.html#description-for-each-metric>`__
   and `Neuron Monitor <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html#system-level-metric-groups>`__
   to monitor the ``sram_ecc_uncorrected`` error counts.

HBM unrepairable uncorrectable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An HBM encountered an unrepairable uncorrectable error and produced incorrect
results.

Instance retirement notice: you receive an
`EC2 instance retirement notice <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-retirement.html>`__
within 15 minutes of this message. EKS, EC2 Auto Scaling groups, and AWS
ParallelCluster react to these retirement notices according to their configured
policies, and you can also automate responses with
`EventBridge rules <https://repost.aws/knowledge-center/eventbridge-notification-scheduled-events>`__.

Neuron Runtime behavior: Neuron Runtime times out and exits with return code
``NRT_TIMEOUT (5)`` or ``NRT_EXEC_HW_ERR_HBM_UE (1201)``. The stdout console
shows:

.. code-block:: text

   (FATAL-RT-UNDEFINED-STATE) Uncorrectable HBM memory error is detected. Execution results may be invalid. Please terminate this instance to prevent future impact from the hardware error.

Recommended actions:

1. Replace the EC2 instance by
   `terminating <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html>`__
   it or `stopping then starting <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html>`__ it.
2. Use `Neuron Sysfs <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-sysfs-user-guide.html#description-for-each-metric>`__
   and `Neuron Monitor <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html#system-level-metric-groups>`__
   to monitor the ``mem_ecc_uncorrected`` error counts.

HBM repairable uncorrectable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An HBM encountered a repairable uncorrectable error and produced incorrect
results.

Neuron Runtime behavior: Neuron Runtime times out and exits with return code
``NRT_TIMEOUT (5)`` or ``NRT_EXEC_HW_ERR_REPAIRABLE_HBM_UE (1205)``. The stdout
console shows:

.. code-block:: text

   (FATAL-RT-UNDEFINED-STATE) Uncorrectable HBM memory error is detected. Execution results may be invalid. Please reload the neuron driver or reboot your EC2 instance to prevent future impact from the hardware error.

Recommended actions:

1. Reload the neuron driver or
   `reboot <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-reboot.html>`__
   the EC2 instance.
2. Use `Neuron Sysfs <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-sysfs-user-guide.html#description-for-each-metric>`__
   and `Neuron Monitor <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html#system-level-metric-groups>`__
   to monitor the ``mem_ecc_repairable_uncorrected`` error counts.

DMA aborts
^^^^^^^^^^

A DMA engine encountered an unrecoverable error.

Neuron Runtime behavior: Neuron Runtime times out and exits with return code
``NRT_TIMEOUT (5)`` or ``NRT_EXEC_HW_ERR_DMA_ABORT (1203)``. The stdout console
shows one of:

.. code-block:: text

   [MLA 0][NC 0] DMA TX engine 0 is in an abort state
   [MLA 0][NC 0] DMA RX engine 0 is in an abort state

Recommended actions: replace the EC2 instance by
`terminating <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html>`__
it or `stopping then starting <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html>`__ it.

Hang on collectives
^^^^^^^^^^^^^^^^^^^

Possibly caused by a hardware error on another worker.

Neuron Runtime behavior: Neuron Runtime times out and exits with return code
``NRT_TIMEOUT (5)`` or ``NRT_EXEC_HW_ERR_COLLECTIVES (1200)``. The stdout
console shows:

.. code-block:: text

   (FATAL-RT-UNDEFINED-STATE) missing collectives status on Neuron Device 0 NC 0, model 0 - suspected hang in collectives operation 0 out of 100

Recommended actions: search for SRAM uncorrectable, HBM uncorrectable, DMA
aborts, and hang-on-compute errors on the other workers, and implement the
recommended actions on the affected worker. Afterward, restart your workload
and try again.

Hang on compute
^^^^^^^^^^^^^^^

Unexpected software or hardware issue.

Neuron Runtime behavior: Neuron Runtime times out and exits with return code
``NRT_TIMEOUT (5)``. The stdout console shows:

.. code-block:: text

   (FATAL-RT-UNDEFINED-STATE) execution timeout (30000 ms) on Neuron Device 0 NC 0, model xxx.neff, waiting for execution completion notification

Recommended actions: replace the EC2 instance by
`terminating <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html>`__
it or `stopping then starting <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html>`__ it.

For any hardware error, you also see a message like the following in ``dmesg``:

.. code-block:: text

   NEURON_HW_ERR=SRAM_UNCORRECTABLE_ERROR instance-id=i-0592464924bd45322 hostname=ip-172-31-61-252 nd-id=0 nc-id=0 serial-num=19fcda00f5ff6eb9 action=TERMINATE_INSTANCE

EFA and collective communication errors
----------------------------------------

Missing aws-neuronx-collectives package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``aws-neuronx-collectives`` package is required to run collective
communication on a single instance and across multiple instances.

.. code-block:: text

   NCCL init error: Error opening libnccom.so, cannot use collective operations! Please set LD_LIBRARY_PATH to library location. Error: libnccom.so: cannot open shared object
   file: No such file or directory
   Please make sure to install correct version of aws-neuronx-collectives; for detailed installation instructions visit Neuron documentation

Solution
~~~~~~~~

Install the ``aws-neuronx-collectives`` package. If you installed it to a
non-default location, set ``LD_LIBRARY_PATH``.

Missing efa-installer package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``efa-installer`` package is required to run collective communication
across multiple instances.

.. code-block:: text

   Unable to run multi-instance workload.  Ofi plugin is not installed or EFA is not enabled

Solution
~~~~~~~~

Install the ``efa-installer`` package, then add the path to the libfabric
library to ``LD_LIBRARY_PATH``.

EFA is not enabled in trn1.32xlarge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EFA is the transport for collective communication across multiple instances.
EFA must be enabled on the instances used for multi-node training.

.. code-block:: text

    OFI plugin initNet() failed is EFA enabled?

Solution
~~~~~~~~

Confirm that EFA is enabled by running ``lspci`` and verifying that there are
eight EFA devices. For example:

.. code-block:: text

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

Launch instances with EFA enabled and try again. If you do not plan to use the
instances for multi-node training, or you are running on trn1.2xlarge, you can
ignore this error.

Communication timeout
^^^^^^^^^^^^^^^^^^^^^^

Ranks exchange information during NEFF loading and before execution starts.
Loading and execution cannot proceed until all ranks are ready.

.. code-block:: text

   Timeout waiting for RX (waited 120 sec) - retrying

.. code-block:: text

   Timeout waiting for incoming connection (waited 120 sec) - retrying

.. code-block:: text

   Connect to localhost:33666 failed - retrying

Solution
~~~~~~~~

Communication timeouts are not fatal; the ranks wait indefinitely. In most
cases a timeout is caused by one rank getting delayed, usually by recompilation
of a graph. Execution resumes after the graph compiles, which can take a
significant amount of time. To determine whether compilation is in progress,
check the logs on all nodes.

A communication timeout can also indicate that a node or rank is hung. If so,
terminate the run and restart from the last known good checkpoint.

Communication errors
--------------------

.. code-block:: text

   RX, connection closed by remote peer

Other similar messages can indicate that ranks failed to communicate.

Solution
^^^^^^^^

One of the ranks or nodes encountered a problem and terminated. Terminate the
run and restart from the last known checkpoint.

EFA kernel messages (dmesg) after process termination
------------------------------------------------------

.. code-block:: text

   [298850.502143] neuron:npid_detach: neuron:npid_detach: pid=90193, slot=0
   [298850.919248] efa 0000:a0:1a.0 rdmap160s26: Failed to process command DEREG_MR (opcode 8) comp_status 7 err -22

Solution
^^^^^^^^

When a process that ran collective communication terminates, it deregisters
buffers that were registered with the networking stack. A race condition occurs
because the Neuron driver deregisters buffers owned by the terminating process
as part of memory cleanup. The error is benign and will be removed in a future
release.

Failure to find bootstrap interface
-----------------------------------

.. code-block:: text

   No interface found in the same subnet as remote address fe80::1461:22ff:fe33:b471<45015>
   No usable listening interface found

Solution
^^^^^^^^

The bootstrap code incorrectly tries to use a link-local IPv6 address for
communication. A fix is planned for an upcoming Neuron release. As a
workaround, disable IPv6 on the instances:

.. code-block:: bash

   sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
   sudo sysctl -w net.ipv6.conf.default.disable_ipv6=1

Name resolution failure
-----------------------

.. code-block:: text

     WARN Invalid NCCL_COMM_ID [compute1-dy-training-0-1.pcluster-trn1-24-pdx80-2n.pcluster:41211], please use format: <ipv4>:<port> or [<ipv6>]:<port>

Solution
^^^^^^^^

Verify that the name resolves through DNS using ``nslookup`` or ``dig``. The
currently released version fails to resolve an FQDN longer than 63 characters.
A fix is planned for an upcoming Neuron release. In the meantime, use shorter
names so the FQDN does not exceed the maximum of 63 characters.

Neuron Runtime timeout or GPSIMD exception
------------------------------------------

Currently, you must reset Neuron Runtime after running a model that invoked a
Neuron Custom C++ operator. Otherwise, a Neuron Runtime timeout or GPSIMD
exception can occur.

Example Neuron Runtime timeout:

.. code-block:: text

   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_tpb_status_notifications   Missing infer_status notification: (end:1)
   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_tpb_status_notifications   Missing infer_status notification: (end:2)
   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_tpb_status_notifications   Missing infer_status notification: (end:3)
   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_tpb_status_notifications   Missing infer_status notification: (end:4)
   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_tpb_status_notifications   Missing infer_status notification: (end:0)
   2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_infer_status_notifications (FATAL-RT-UNDEFINED-STATE) inference timeout (600000 ms) on Neuron Device 0 NC 0, waiting for execution completion notification
   2023-Jan-09 20:27:41.0600 15042:15042 ERROR  NMGR:dlr_infer                               Inference completed with err: 5

Example GPSIMD exception:

.. code-block:: text

   2023-Jan-06 22:28:01.0845 137472:137472 ERROR TDRV:pool_stdio_queue_consume_all_entries  Printing stderr from GPSIMD:
   GPSIMD EXCEPTION OCCURRED: ILLEGAL INSTRUCTION
   Subtype/Type/Cause: 0x201
   Exception PC: 0x840001E8

Solution
^^^^^^^^

If you see either error and ``NEURON_RT_RESET_CORES`` is set to 0, either unset
it or set it to 1. This enables the default runtime behavior of resetting
NeuronCores when initializing applications. See :ref:`nrt-configuration` and
:ref:`nrt-env-var-reference` for more information.

You can also change the timeout period by setting ``NEURON_RT_EXEC_TIMEOUT``.
See :ref:`nrt-configuration` and :ref:`nrt-env-var-reference` for more
information.

FI_EFA_FORK_SAFE
----------------

Linux kernels older than 5.15 require the environment variable
``FI_EFA_FORK_SAFE`` to be set to 1 for libfabric to operate correctly. Amazon
Linux 2 uses the 5.10 kernel and requires this variable to be set.

When the variable is not set, multi-node collective communication is disabled.
Intra-node collective communication still works. The following error is logged
the first time a model containing collective communication is loaded:

.. code-block:: text

   Linux kernel 5.10 requires setting FI_EFA_FORK_SAFE=1 environment variable.  Multi-node support will be disabled.
   Please restart with FI_EFA_FORK_SAFE=1 set."

Neuron driver cannot be uninstalled
-----------------------------------

If you try to uninstall the Neuron driver on Ubuntu with
``sudo dpkg -r aws-neuronx-dkms``, you might get an error like this:

.. code-block:: text

   Removing aws-neuronx-dkms (2.x) ...
   Neuron module is currently loaded. Attempting to unload...
   ERROR: Cannot unload neuron module - it is currently in use.
   Please stop all processes using the neuron module before uninstalling.
   dpkg: error processing package aws-neuronx-dkms (--remove):
   installed aws-neuronx-dkms package pre-removal script subprocess returned error exit status 1
   Errors were encountered while processing:
   aws-neuronx-dkms

On Amazon Linux, you get a similar error if you run
``sudo rpm -e aws-neuronx-dkms`` to uninstall the driver:

.. code-block:: text

   Uninstall of aws-neuronx module (version 2.x) beginning:
   Neuron module is currently loaded. Attempting to unload...
   ERROR: Cannot unload neuron module - it is currently in use.
   Please stop all processes using the neuron module before uninstalling.
   error: %preun(aws-neuronx-dkms-2.x-dkms.noarch) scriptlet failed, exit status 1
   error: aws-neuronx-dkms-2.x-dkms.noarch: erase failed

Usually this means a process is still using the driver. Terminating that
process lets the driver be unloaded and uninstalled. In rare cases the driver
is stuck; the remediation is to force-uninstall the driver and then reboot.

Solution
^^^^^^^^

Force-uninstall the Neuron driver.

.. warning:: Force-uninstalling the driver risks system instability or a kernel
   panic. Reboot your instance immediately after uninstalling it.

To force-uninstall the driver on Ubuntu instances:

.. code-block:: bash

   sudo dkms remove aws-neuronx/<version> --all
   sudo dpkg -r --force-all aws-neuronx-dkms

To force-uninstall the driver on Amazon Linux instances:

.. code-block:: bash

   sudo dkms remove aws-neuronx/<version> --all
   sudo rpm -e --noscript aws-neuronx-dkms
