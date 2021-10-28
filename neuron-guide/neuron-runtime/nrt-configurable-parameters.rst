.. _nrt-configuration:

Neuron Runtime Configuration
============================

Runtime is responsible of executing ML models on Neuron devices and it determines which NeuronCore will execute which model and how to execute it.
User application should configure the Runtime to change the default behavior, Runtime can be configured through environmental variables,
in most cases Neuron framework extensions will take care of the proper configuration in other cases the user may need to explicitly configure the runtime
 to achieve the desired behavior.

This guide provides an overview of the different environment variables available to
configure Neuron runtime behavior.

.. list-table:: Environment Variables
   :widths: 25 60 20 50 20
   :header-rows: 1

   * - Name
     - Description
     - Type
     - Expected Values
     - Default Value
   * - ``NEURON_RT_VISIBLE_CORES``
     - Range of specific NeuronCores needed by the process
     - Integer range (like 1-3)
     - Any value or range between 0 to Max NeuronCore in the system.
     - None
   * - ``NEURON_RT_LOG_LOCATION``
     - Runtime log location
     - string
     - console or syslog
     - console
   * - ``NEURON_RT_LOG_LEVEL``
     - Runtime log verbose level
     - string
     - ERROR, WARNING, INFO, DEBUG, TRACE
     - ERROR
   * - ``NEURON_RT_EXEC_TIMEOUT``
     - Timeout for execution in seconds
     - Integer
     - 0 to INT_MAX
     - 2
   * - ``NEURON_RT_VALIDATE_HASH``
     - Validate NEFF contents before loading into accelerator
     - Boolean
     - TRUE or FALSE
     - FALSE


NeuronCore Allocation with NEURON_RT_VISIBLE_CORES
--------------------------------------------------

.. important ::

  ``NEURONCORE_GROUP_SIZES`` is being deprecated, if your application is using ``NEURONCORE_GROUP_SIZES`` please 
  see :ref:`neuron-migrating-apps-neuron-to-libnrt` for more details.


By default, Neuron Runtime initializes all the cores present in the system and reserves them for the current process.

.. note ::

  Once a NeuronCore is reserved for a process, it cant be used by another process at all, until the process reserving that NeuronCore dies.

For parallel processing, it is necessary multiple processes need to use different NeuronCores.
For this purpose **``NEURON_RT_VISIBLE_CORES``** can be used which controls what NeuronCores the process would reserve.
This variable takes a NeuronCore index or an inclusive range.

For example, if an application(myapp.py) requires one NeuronCore, then it can be started with
``NEURON_RT_VISIBLE_CORES=0`` to use only NeuronCore 0. To do parallel processing, multiple process can be
started(without any change to application) with different ``NEURON_RT_VISIBLE_CORES`` values.
Here is an example which runs myapp.py on inf1.xl parallely by using different NeuronCores available.

::

 NEURON_RT_VISIBLE_CORES=0 myapp.py
 NEURON_RT_VISIBLE_CORES=1 myapp.py
 NEURON_RT_VISIBLE_CORES=2 myapp.py
 NEURON_RT_VISIBLE_CORES=3 myapp.py


Another example, where myapp2.py requires 3 NeuronCores and being run on inf1.6xl.
In the following example, the first instance of myapp2 would use NeuronCores 0, 1 and 2, then next instance would use 3, 4, and 4 and so on.

::

 NEURON_RT_VISIBLE_CORES=0-2 myapp2.py
 NEURON_RT_VISIBLE_CORES=3-5 myapp2.py
 NEURON_RT_VISIBLE_CORES=6-8 myapp2.py
 NEURON_RT_VISIBLE_CORES=9-11 myapp2.py
 NEURON_RT_VISIBLE_CORES=12-14 myapp2.py


Notes
~~~~~

1. Number of NeuronCores in a inferentia device is 4
2. Number of inferentia is depends on the instance size.
3. The NeuronCore index in NEURON_RT_VISIBLE_CORES starts from 0 and ends at (number of NeuronDevices * number of NeuronCores) - 1.


Logging and debug-ability
-------------------------
By default, Neuron Runtime logs to syslog with verbose level of *INFO* and only *ERROR* s are logged in console.
The following code snippet shows ways to increase/decrease the log level.

::

 NEURON_RT_LOG_LEVEL=INFO myapp.py         # Sets the log level for syslog and console to INFO
 NEURON_RT_LOG_LOCATION=console NEURON_RT_LOG_LEVEL=QUIET myapp.py    # Completely disables console logging.

By default, Neuron Runtime expects the NeuronCore to complete execution of any model with in 2 seconds.
If NeuronCore didnt complete the execution within 2 seconds then runtime would fail the execution with timeout error.
Most of the models takes few milliseconds to complete so 2 seconds(2000 milliseconds) is more than adequate.
However if your model is expected to run more than 2 seconds then you can increase the timeout with NEURON_RT_EXEC_TIMEOUT.

::

 NEURON_RT_EXEC_TIMEOUT=5 myapp.py       # increases the timeout to 5 seconds

Checksum
--------
To execute a model(NEFF), Neuron Runtime needs to load the NEFF file into NeuronCore and run.
Neuron Runtime provides a way to do checksum validation on each NEFF file while loading to validate the file is not corrupted.
This option is off by default to avoid performance penalty during model load time(~50%).

::

 NEURON_RT_VALIDATE_HASH=true myapp1.py     # enables model checksum validation while loading
 NEURON_RT_VALIDATE_HASH=false myapp2.py    # disables(default) model checksum validation while loading