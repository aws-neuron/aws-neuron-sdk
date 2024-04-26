.. _nrt-configuration:

NeuronX Runtime Configuration
============================

NeuronX Runtime is responsible for executing ML models on Neuron Devices. NeuronX Runtime determines which NeuronCore will execute which model and how to execute it.
Configuration of the NeuronX Runtime is controlled through the use of Environment variables at the process level.  By default, Neuron framework extensions will take care of NeuronX Runtime configuration on the user's behalf.  Explicit configurations are also possible when attempting to achieve a desired behavior.

This guide provides an overview of the different environment variables available to
configure NeuronX Runtime behavior.

.. list-table:: Environment Variables
   :widths: 25 60 20 50 20 50
   :header-rows: 1
   

   
   * - Name
     - Description
     - Type
     - Expected Values
     - Default Value
     - RT Version
   * - ``NEURON_RT_VISIBLE_CORES``
     - Range of specific NeuronCores needed by the process
     - Integer range (like 1-3)
     - Any value or range between 0 to Max NeuronCore in the system.
     - None
     - 2.0+
   * - ``NEURON_RT_NUM_CORES``
     - Number of NeuronCores required by the process.
     - Integer
     - A value from 1 to Max NeuronCore in the system.
     - 0, which is interpreted as "all"
     - 2.0+
   * - ``NEURON_RT_LOG_LOCATION``
     - Runtime log location
     - string
     - console or syslog
     - console
     - 2.0+
   * - ``NEURON_RT_LOG_LEVEL``
     - Runtime log verbose level
     - string
     - ERROR, WARNING, INFO, DEBUG, TRACE
     - ERROR
     - 2.0+
   * - ``NEURON_RT_EXEC_TIMEOUT``
     - Timeout for execution in seconds
     - Integer
     - 0 to INT_MAX
     - 30
     - 2.0+
   * - ``NEURON_RT_VALIDATE_HASH``
     - Validate NEFF contents before loading into accelerator
     - Boolean
     - TRUE or FALSE
     - FALSE
     - 2.0+
   * - ``NEURON_RT_MULTI_INSTANCE_SHARED_WEIGHTS``
     - Share weights when loading multiple instance versions of the same model on different NeuronCores
     - Boolean
     - TRUE or FALSE
     - FALSE
     - 2.11+
   * - ``NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS``
     - Controls number of asynchronous execution requests to be supported.
     - Integer
     - 0 to INT_MAX; 0 is disabled.
     - 0
     - 2.15+


NeuronCore Allocation
---------------------

.. important ::

  ``NEURONCORE_GROUP_SIZES`` is being deprecated, if your application is using ``NEURONCORE_GROUP_SIZES`` please 
  see :ref:`neuron-migrating-apps-neuron-to-libnrt` for more details.


By default, NeuronX Runtime initializes all the cores present in the system and reserves them for the current process.

.. note::

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
By default, NeuronX Runtime logs to syslog with verbose level of *INFO* and only *ERROR* s are logged in console.
The following code snippet shows ways to increase/decrease the log level.

::

 NEURON_RT_LOG_LEVEL=INFO myapp.py         # Sets the log level for syslog and console to INFO
 NEURON_RT_LOG_LOCATION=console NEURON_RT_LOG_LEVEL=QUIET myapp.py    # Completely disables console logging.

By default, NeuronX Runtime expects the NeuronCore to complete execution of any model with in 2 seconds.
If NeuronCore didn't complete the execution within 2 seconds then runtime would fail the execution with timeout error.
Most of the models takes few milliseconds to complete so 2 seconds(2000 milliseconds) is more than adequate.
However if your model is expected to run more than 2 seconds then you can increase the timeout with NEURON_RT_EXEC_TIMEOUT.

::

 NEURON_RT_EXEC_TIMEOUT=5 myapp.py       # increases the timeout to 5 seconds


Additional Logging Controls
-------------------------
NeuronX Runtime enables detailed control over logging behaviors, including the ability to set separate log levels and log locations for individual components. 
When ``NEURON_RT_LOG_LEVEL`` is set globally, NeuronX Runtime combines the logs from all modules into a single stream. 
For instance, the logs from the modules ``TDRV`` and ``NMGR`` would appear in the same stream as shown in the example below

::
  2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_infer_status_notifications (FATAL-RT-UNDEFINED-STATE) inference timeout (600000 ms) on Neuron Device 0 NC 0, waiting for execution completion notification
  2023-Jan-09 20:27:41.0600 15042:15042 ERROR  NMGR:dlr_infer 

However, it is possible to adjust the log level for individual components to capture more or less detail as required for specific debugging contexts.
These individual components are
- ``TDRV``: the low level driver library
- ``KMGR``: the higher level manager library bridging the driver and runtime
- ``NRT``: the Neuron Runtime library responsible for loading and executing models that is exposed to end users and frameworks

To adjust the log level for individual components, use the environment variable ``NEURON_RT_LOG_LEVEL_<component>``, where ``<component>`` is the identifier of the component 
(either ``TDRV``, ``NMGR``, or ``NRT``). 
This allows for precise control over the verbosity of logs generated by each component, facilitating more targeted debugging.
For example, the following sets different log levels for the ``TDRV`` and ``NMGR`` components.

::
  export NEURON_RT_LOG_LEVEL_TDRV=DEBUG
  export NEURON_RT_LOG_LEVEL_NMGR=ERROR


Similarly, to specify separate log locations for individual components, use the environment variable ``NEURON_RT_LOG_LOCATION_<component>``, following the same naming convention as for log levels. 
This feature enables logs from different components to be directed to separate files or destinations, making it easier to organize and analyze the log output.
For example, the following sets different log locations for the ``TDRV`` and ``NMGR`` components.

::
  export NEURON_RT_LOG_LOCATION_TDRV=tdrv.log
  export NEURON_RT_LOG_LOCATION_NMGR=nmgr.log



Checksum
--------
To execute a model(NEFF), NeuronX Runtime needs to load the NEFF file into NeuronCore and run.
Neuron Runtime provides a way to do checksum validation on each NEFF file while loading to validate the file is not corrupted.
This option is off by default to avoid performance penalty during model load time(~50%).

::

 NEURON_RT_VALIDATE_HASH=true myapp1.py     # enables model checksum validation while loading
 NEURON_RT_VALIDATE_HASH=false myapp2.py    # disables(default) model checksum validation while loading
 
 
Shared Weights (NEURON_RT_MULTI_INSTANCE_SHARED_WEIGHTS)
--------------------------------------------------------
By default, NeuronX Runtime will make copies of model weights when loading the same instance of a model to multiple NeuronCores. Changing this default to a weight sharing mechanism is possible with NeuronX Runtime 2.11 or higher by setting ``NEURON_RT_MULTI_INSTANCE_SHARED_WEIGHTS=TRUE``. Use of this flag will allow for more models to be loaded by reducing the memory requirements, but will potentially come at a cost of throughput by forcing the execution across cores to compete for memory bandwidth.

Note: the use of this flag requires the model to be loaded with the multi-instance feature (see :ref:`torch_core_placement_api`).

See the :pytorch-neuron-src:`[BERT tutorial with shared weights notebook] <bert_tutorial/tutorial_pretrained_bert_shared_weights.ipynb>` for an example of how this is used in ``Torch-Neuron``.

::

 NEURON_RT_MULTI_INSTANCE_SHARED_WEIGHTS=TRUE myapp1.py     # enables model weight sharing
 NEURON_RT_MULTI_INSTANCE_SHARED_WEIGHTS=FALSE myapp2.py    # disables(default) model weight sharing


Aynchronous Execution (NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS)
--------------------------------------------------------
A beta asynchronous execution feature which can reduce latency by roughly 12% for training workloads. Starting in Neuron Runtime version 2.15, the feature is available, but disabled.  To enable the feature for possible improvement, recommendation is to set NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS to 3.  Setting the number of inflight requests above 3 may lead to Out-Of-Memory (OOM) errors during execution.  For developers using libnrt.so directly, please use nrt_register_async_exec_callback to register a callback for the nrt execution thread to post the execution status to. A default callback will be registered if one is not set by the developer.

::

 NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3 myapp.py     # Up to 3 async exec requests at once.
 NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0 myapp.py     # disables async execution (default behavior)
