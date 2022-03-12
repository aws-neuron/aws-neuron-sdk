.. _nrt-configuration:

Neuron Runtime Configuration
============================

Neuron Runtime is responsible for executing ML models on Neuron Devices.  Neuron Runtime determines which NeuronCore will execute which model and how to execute it.
Configuration of the Neuron Runtime is controlled through the use of environmental variables at the process level.  By default, Neuron framework extensions will take care of Neuron Runtime configuration on the user's behalf.  Explicit configurations are also possible when attempting to achieve a desired behavior.

This guide provides an overview of the different environment variables available to
configure Neuron Runtime behavior.

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
   * - ``NEURON_RT_NUM_CORES``
     - Number of NeuronCores required by the process.
     - Integer
     - A value from 1 to Max NeuronCore in the system.
     - 0, which is interpretted as "all"
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


NeuronCore Allocation
---------------------

.. important ::

  ``NEURONCORE_GROUP_SIZES`` is being deprecated, if your application is using ``NEURONCORE_GROUP_SIZES`` please 
  see :ref:`neuron-migrating-apps-neuron-to-libnrt` for more details.


By default, Neuron Runtime initializes all the cores present in the system and reserves them for the current process.

.. note ::

  Once a NeuronCore is reserved for a process, it cannot be used by another process at all, until the process reserving that NeuronCore is terminated.
  
Using NEURON_RT_VISIBLE_CORES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For parallel processing, ``NEURON_RT_VISIBLE_CORES`` can be used to control which NeuronCores each process would reserve.  This variable is specified with a single NeuronCore index or an inclusive range value.

For example, if a process (myapp.py) requires one NeuronCore, then it can be started with
``NEURON_RT_VISIBLE_CORES=0`` to limit the process to NeuronCore 0. For parallel processing, multiple process can be
started (without any change to myapp.py code) with different ``NEURON_RT_VISIBLE_CORES`` values.
Here is an example that runs myapp.py on inf1.xlarge in parallel across the four different NeuronCores available in the inf1.xlarge.

::

 NEURON_RT_VISIBLE_CORES=0 myapp.py &
 NEURON_RT_VISIBLE_CORES=1 myapp.py &
 NEURON_RT_VISIBLE_CORES=2 myapp.py &
 NEURON_RT_VISIBLE_CORES=3 myapp.py &


If myapp.py required 3 NeuronCores and was running on a inf1.6xlarge (16 NeuronCores maximum), the first instance of myapp.py could use NeuronCores 0-2, the next instance could use 3-5 and so on:

::

 NEURON_RT_VISIBLE_CORES=0-2 myapp.py &
 NEURON_RT_VISIBLE_CORES=3-5 myapp.py &
 NEURON_RT_VISIBLE_CORES=6-8 myapp.py &
 NEURON_RT_VISIBLE_CORES=9-11 myapp.py &
 NEURON_RT_VISIBLE_CORES=12-14 myapp.py &


Using NEURON_RT_NUM_CORES
~~~~~~~~~~~~~~~~~~~~~~~~~

If ``NEURON_RT_NUM_CORES`` is set to a value between 1 and the maximum number of NeuronCores in the instance, Neuron Runtime will attempt to automatically reserve the number of free NeuronCores specified for the process. The difference between ``NEURON_RT_VISIBLE_CORES`` and ``NEURON_RT_NUM_CORES`` is that, ``NEURON_RT_VISIBLE_CORES`` specifies exact NeuronCores to allocate where as ``NEURON_RT_NUM_CORES`` specifies the number of NeuronCores needed and Neuron Runtime selects free NeuronCores.

Using the same example earlier where myapp.py needed 3 cores, but _which_ 3 cores was of no concern, the same application could be executed in parallel up to 5 times on an inf1.6xlarge (16 NeuronCore max):

::

 NEURON_RT_NUM_CORES=3 myapp.py &
 NEURON_RT_NUM_CORES=3 myapp.py &
 NEURON_RT_NUM_CORES=3 myapp.py &
 NEURON_RT_NUM_CORES=3 myapp.py &
 NEURON_RT_NUM_CORES=3 myapp.py &

Executing a 6th ``NEURON_RT_NUM_CORES=3 myapp.py &`` in the above example would fail as there is only a single NeuronCore still free.


Notes
~~~~~

1. Number of NeuronCores in a inferentia device is 4
2. Number of inferentia is depends on the instance size.
3. The NeuronCore index in NEURON_RT_VISIBLE_CORES starts from 0 and ends at (number of NeuronDevices * number of NeuronCores) - 1.
4. By default, ``NEURON_RT_NUM_CORES`` is set to ``0``, which indicates to RT that all cores are to be used.  
5. NEURON_RT_VISIBLE_CORES takes precedence over NEURON_RT_NUM_CORES.  If specified, all cores within the range will be assigned to the owning process.


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
