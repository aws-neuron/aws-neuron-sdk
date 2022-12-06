.. _nrt-troubleshooting:

Neuron Runtime Troubleshooting on Inf1
======================================

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

Terminate the the process using NeuronCore and then try launching the application again.

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

