.. _rtd-troubleshooting:

Troubleshooting Neuron Runtime
==============================

This document aims to provide more information on how to fix issues you
might encounter while using the Neuron Runtime 1.1 or above. For each
issue we will provide an explanation of what happened and what can
potentially correct the issue.


If you haven't read it already, please familiarize yourself with our
:ref:`rtd-getting-started` documentation and usage examples. If your
issue is still not resolved or you have a more nuanced problem, contact
us via `issues <https://github.com/aws/aws-neuron-sdk/issues>`__ posted
to this repo, the `AWS Neuron developer
forum <https://forums.aws.amazon.com/forum.jspa?forumID=355>`__, or
through AWS support.

--------------

Topics
~~~~~~

**What is going wrong?**

`Runtime installation failed <#installation-failed>`__

`Neuron Runtime services fail to
start <#neuron-services-fail-to-start>`__

`Load model failure <#load-model-failure>`__

`Inferences are failing <#inferences-are-failing>`__

.. raw:: html

   <br/>

**Additional helpers:**

:ref:`rtd-getting-started`

:ref:`rtd-return-codes`

--------------

Runtime installation failed
---------------------------

Refer to the :ref:`neuron-install-guide`
for details.

--------------

Neuron Runtime service fails to start
-------------------------------------

If the neuron-rtd service is failing to start, you may be experiencing
failure due to (1) a conflict with another instance of neuron-rtd, (2)
neuron driver(aws-neuron-dkms) package is not installed.

Neuron Driver is not installed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What Went Wrong?
''''''''''''''''

Neuron Runtime requires Neuron Driver(aws-neuron-dkms) to access inf1
devices. If the driver is not installed then Neuron Runtime service wont
start.

How To Find Out?
''''''''''''''''

``systemctl status`` command can be used to check whether neuron-rtd is
active or not. If Neuron Driver is not installed then output would look
similar to the following

::

   $ sudo systemctl status neuron-rtd
   ● neuron-rtd.service - Neuron Runtime Daemon
      Loaded: loaded (/lib/systemd/system/neuron-rtd.service; enabled; vendor preset: enabled)
      Active: inactive (dead) since Wed 2020-10-14 16:28:18 UTC; 1 day 8h ago
   Condition: start condition failed at Fri 2020-10-16 01:08:31 UTC; 4s ago
              └─ ConditionPathExistsGlob=/dev/neuron* was not met
    Main PID: 27911 (code=killed, signal=TERM)

How To Fix?
'''''''''''

Please follow the installation steps in :ref:`neuron-install-guide` to install ``aws-neuron-dkms``
package and then restart runtime using
``sudo systemctl restart neuron-rtd`` command.

.. _neuron-driver-installation-fails:

Neuron Driver installation fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _what-went-wrong-1:

What Went Wrong?
''''''''''''''''

aws-neuron-dkms is a driver pacakge which needs to be compiled during
installation. The compilation requires kernel headers for the instance's
kernel. ``uname -r`` can be used to find kernel version in the instance.
In some cases, the installed kernel headers might be newer than the
instance's kernel itself.

.. _how-to-find-out-1:

How To Find Out?
''''''''''''''''

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

.. _how-to-fix-1:

How To Fix?
'''''''''''

1. Uninstall aws-neuron-dkms ``sudo apt remove aws-neuron-dkms`` or
   ``sudo yum remove aws-neuron-dkms``

2. Install kernel headers for the current kernel
   ``sudo apt install -y linux-headers-$(uname -r)`` or
   ``sudo yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)``

3. Install aws-neuron-dkms ``sudo apt install aws-neuron-dkms`` or
   ``sudo yum install aws-neuron-dkms``

4. Restart runtime using ``sudo systemctl restart neuron-rtd`` command.

Another Instance of Runtime is Running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _what-went-wrong-2:

What Went Wrong?
''''''''''''''''

A new instance of neuron-rtd cannot start if another neuron-rtd is
already running and bound to the same Neuron devices. Please read on for
how to detect this scenario, but if you're having trouble configuring
two or more runtimes on the same Inf1 instance, see detailed config
instructions at :ref:`multiple-neuron-rtd`.

Check for error messages in syslog similar to:

.. code:: bash

   Oct 16 01:07:00 xxxxxxxx kernel: [ 7638.723761] neuron:ncdev_device_init: device inuse by pid:9428

.. _how-to-fix-2:

How To Fix?
'''''''''''

Terminate the current neuron-rtd that is already running before starting
the new instance.

.. code:: bash

   sudo systemctl stop neuron-rtd

--------------

Load Model Failure
------------------

There are a variety of reasons for a model load to fail. The most common
ones are listed below. If the solutions below are insufficient, please
reach out to the Neuron team by posting the relevant details on GitHub
`issues <https://github.com/aws/aws-neuron-sdk/issues>`__.

Neff couldn't be extracted
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _what-went-wrong-3:

What went wrong?
''''''''''''''''

Host ran out of disk space while trying to extract the NEFF object

.. _how-to-find-out-2:

How to find out?
''''''''''''''''

Syslog will show an error similar to the following:

::

   nrtd[nnnnn]: ....  Failed to untar (tar -xsvf /tmp/neff.XXXXX -C /tmp/neff.YYYYY > /dev/null)

.. _how-to-fix-3:

How to fix?
'''''''''''

Increase /tmp space by removing unused files or taking other measures to
increase the available disk size under /tmp.

Unsupported NEFF Version
~~~~~~~~~~~~~~~~~~~~~~~~

.. _what-went-wrong-4:

What Went Wrong?
''''''''''''''''

The version of the NEFF file is incompatible with the version of Neuron
Runtime that has received it. The NEFF is generated by the compiler and
the Neuron Runtime is intended to support multiple NEFF versions;
however, this may require updating the runtime to gain support for newer
NEFF formats.

.. _how-to-find-out-3:

How To Find Out?
''''''''''''''''

Check for error messages in syslog similar to:

::

   nrtd[nnnnn]: ....  NEFF version mismatch supported: 1.1 received: 2.0

Error Code: 10

.. _how-to-fix-4:

How To Fix?
'''''''''''

Use compatible versions of Neuron Compiler and Runtime. Updating to the
latest version of both Neuron Compiler and Neuron Runtime is the
simplest solution. If updating one of the two is not an option, please
refer to the :ref:`neuron-runtime-release-notes`
of the Neuron Runtime to determine NEFF version support.

Invalid NEFF
~~~~~~~~~~~~

.. _what-went-wrong-5:

What Went Wrong?
''''''''''''''''

Validation is performed on the NEFF file before attempting to load it.
When that validation fails, it usually indicates that the compiler
produced an invalid NEFF (possible bug in the Neuron Compiler).

.. _how-to-find-out-4:

How To Find Out?
''''''''''''''''

Check for error messages in syslog similar to:

::

   nrtd[nnnn]: .... Failed .... neff.json
   nrtd[nnnn]: .... Failed/Unsupported/Invalid .... NEFF
   nrtd[nnnn]: .... Wrong NEFF file size
   nrtd[nnnn]: .... NEFF upload failed

Error Code: 2

.. _how-to-fix-5:

How To Fix?
'''''''''''

Try recompiling with the latest version of Neuron Compiler. If that does
not work, report issue to Neuron by posting the relevant details on
GitHub `issues <https://github.com/aws/aws-neuron-sdk/issues>`__.

Bad Memory Access by NEFF
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _what-went-wrong-6:

What Went Wrong?
''''''''''''''''

To ensure the execution of the NEFF, neuron-rtd is monitoring for
illegal and unaligned access to Inferentia memory. When this occurs, the
NEFF will fail to load.

.. _how-to-find-out-5:

How To Find Out?
''''''''''''''''

Check for error messages in syslog similar to:

::

   nrtd[nnnn]:.... address ... must be X byte aligned

Error Code: 2

.. _how-to-fix-6:

How To Fix?
'''''''''''

Report issue to Neuron by posting the relevant details on GitHub
`issues <https://github.com/aws/aws-neuron-sdk/issues>`__.

Insufficient resources
~~~~~~~~~~~~~~~~~~~~~~

.. _what-went-wrong-7:

What Went Wrong?
''''''''''''''''

Loading the NEFF requires more host or Inferentia resources (usually
memory on the host or Inferentia) then available on the instance. This
issue may be contextual in that other applications or models consumed
the needed resources before the current NEFF could be loaded.

.. _how-to-find-out-6:

How To Find Out?
''''''''''''''''

Check for error messages in syslog similar to:

::

   kernel: [XXXXX] neuron:mc_alloc: device mempool [0:0] total 1073741568 occupied 960539030 needed 1272 available 768

Error Code: 4

.. _how-to-fix-7:

How To Fix?
'''''''''''

As the error is contextual to what's going on with your instance, the
exact next step is unclear. Try unloading some of the loaded models
which will free up device DRAM space. If this is still a problem, moving
to a larger Inf1 instance size with additional NeuronCores may help.

Insufficient number of NeuronCores available to load a NEFF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _what-went-wrong-8:

What Went Wrong?
''''''''''''''''

The NEFF requires more NeuronCores than available on the instance.

.. _how-to-find-out-7:

How To Find Out?
''''''''''''''''

Check for error messages in syslog similar to:

::

   nrtd[nnnn]:.... Requested number of NCs: X exceeds the total available number: Y
   nrtd[nnnn]:.... Insufficient number of VNCs: X, required: Y

Error Code: 9

.. _how-to-fix-8:

How To Fix?
'''''''''''

The NeuronCores may be in use by models you are not actively using.
Ensure you've unloaded models you're not using and deleted unused
NCGroups. If this is still a problem, moving to a larger Inf1 instance
size with additional NeuronCores may help.

--------------

Inferences are failing
----------------------

Wrong Model Id
~~~~~~~~~~~~~~

.. _what-went-wrong-9:

What Went Wrong?
''''''''''''''''

An inference request had a model id that is invalid or not in a running
state.

.. _how-to-find-out-8:

How To Find Out?
''''''''''''''''

Check for error messages in syslog similar to:

::

   nrtd[nnnnn]:....Failed to find model: 10001

.. _how-to-fix-9:

How To Fix?
'''''''''''

Ensure your application is only inferring against models that are
running on the Inferentia.

Bad or incorrect inputs
~~~~~~~~~~~~~~~~~~~~~~~

NEFF contains information of the number of input feature maps required
by the model. If inputs to the model don't match the expected
number/size of the input, inference will fail.

.. _what-went-wrong-10:

What Went Wrong?
''''''''''''''''

Mismatch in either the number of expected inputs or the size of the
inputs.

.. _how-to-find-out-9:

How To Find Out?
''''''''''''''''

Check for error messages in syslog similar to:

::

   nrtd[nnnnn]: ....  Wrong number of ifmaps, expected: X, received: Y
   nrtd[nnnnn]: ....  Invalid data length for [input:0], received X, expected Y

.. _how-to-fix-10:

How To Fix?
'''''''''''

Ensure the correct number of inputs and correct sizes are used.

Numerical errors on the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _what-went-wrong-11:

What Went Wrong?
''''''''''''''''

The inference generated NaNs during execution, which is usually an
indication of model or input errors.

.. _how-to-find-out-10:

How To Find Out?
''''''''''''''''

Check for error messages in syslog similar to:

::

   nrtd[nnnnn]: ....  Error notifications found on NC .... INFER_ERROR_SUBTYPE_NUMERICAL

.. _how-to-fix-11:

How To Fix?
'''''''''''

Report issue to Neuron by posting the relevant details on GitHub
`issues <https://github.com/aws/aws-neuron-sdk/issues>`__.

Inference Timeout
~~~~~~~~~~~~~~~~~

.. _what-went-wrong-12:

What Went Wrong?
''''''''''''''''

It's possible that the Neuron Compiler built a NEFF with errors, e.g.
the NEFF might describe incorrect internal data flows or contain
incorrect instruction streams. When this happens, it will potentially
result in a timeout during inference.

.. _how-to-find-out-11:

How To Find Out?
''''''''''''''''

Check for error messages in syslog similar to:

::

   nrtd[nnnnn]: ....  Error: DMA completion timeout in ....

Error Code: 5

.. _how-to-fix-12:

How To Fix?
'''''''''''

Report issue to Neuron by posting the relevant details on GitHub
`issues <https://github.com/aws/aws-neuron-sdk/issues>`__.

--------------
