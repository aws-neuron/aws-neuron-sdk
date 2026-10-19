.. meta::
   :description: Configure the Neuron Runtime on AWS Trainium and Inferentia through environment variables that control NeuronCore allocation, execution timeouts, logging, checksum validation, shared weights, and async execution.
   :date-modified: 2026-09-10
   :keywords: Neuron, Neuron Runtime, NRT, configuration, environment variables, NEURON_RT_VISIBLE_CORES, NEURON_RT_NUM_CORES, NEURON_RT_LOG_LEVEL

.. _nrt-configuration:

=============================
Neuron Runtime configuration
=============================

Neuron Runtime is responsible for executing ML models on Neuron devices. Neuron Runtime determines which NeuronCore will execute which model and how to execute it. Configuration of Neuron Runtime is controlled through environment variables at the process level. By default, Neuron framework extensions take care of Neuron Runtime configuration on the user's behalf. Explicit configuration is also possible when attempting to achieve a desired behavior.

This guide walks through the most common configuration tasks with worked examples. For the complete, categorized list of runtime environment variables — including their types, defaults, and use classes — see :ref:`nrt-env-var-reference`.

NeuronCore allocation
---------------------

.. important::

  ``NEURONCORE_GROUP_SIZES`` is being deprecated. If your application is using ``NEURONCORE_GROUP_SIZES``, see :ref:`neuron-migrating-apps-neuron-to-libnrt` for more details.


By default, Neuron Runtime initializes all the cores present in the system and reserves them for the current process.

.. note::

  Once a NeuronCore is reserved for a process, it cannot be used by another process at all, until the process reserving that NeuronCore is terminated.

Using NEURON_RT_VISIBLE_CORES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For parallel processing, ``NEURON_RT_VISIBLE_CORES`` can be used to control which NeuronCores each process would reserve. This variable is specified with a single NeuronCore index or an inclusive range value.

For example, if a process (``myapp.py``) requires one NeuronCore, it can be started with ``NEURON_RT_VISIBLE_CORES=0`` to limit the process to NeuronCore 0. For parallel processing, multiple processes can be started (without any change to ``myapp.py`` code) with different ``NEURON_RT_VISIBLE_CORES`` values. Here is an example that runs ``myapp.py`` on inf1.xlarge in parallel across the four different NeuronCores available in the inf1.xlarge.

::

 NEURON_RT_VISIBLE_CORES=0 myapp.py &
 NEURON_RT_VISIBLE_CORES=1 myapp.py &
 NEURON_RT_VISIBLE_CORES=2 myapp.py &
 NEURON_RT_VISIBLE_CORES=3 myapp.py &


If ``myapp.py`` required 3 NeuronCores and was running on an inf1.6xlarge (16 NeuronCores maximum), the first instance of ``myapp.py`` could use NeuronCores 0-2, the next instance could use 3-5, and so on:

::

 NEURON_RT_VISIBLE_CORES=0-2 myapp.py &
 NEURON_RT_VISIBLE_CORES=3-5 myapp.py &
 NEURON_RT_VISIBLE_CORES=6-8 myapp.py &
 NEURON_RT_VISIBLE_CORES=9-11 myapp.py &
 NEURON_RT_VISIBLE_CORES=12-14 myapp.py &


Using NEURON_RT_NUM_CORES
~~~~~~~~~~~~~~~~~~~~~~~~~

If ``NEURON_RT_NUM_CORES`` is set to a value between 1 and the maximum number of NeuronCores in the instance, Neuron Runtime will attempt to automatically reserve the number of free NeuronCores specified for the process. The difference between ``NEURON_RT_VISIBLE_CORES`` and ``NEURON_RT_NUM_CORES`` is that ``NEURON_RT_VISIBLE_CORES`` specifies exact NeuronCores to allocate, whereas ``NEURON_RT_NUM_CORES`` specifies the number of NeuronCores needed and Neuron Runtime selects free NeuronCores.

Using the same example where ``myapp.py`` needed 3 cores but *which* 3 cores was of no concern, the same application could be executed in parallel up to 5 times on an inf1.6xlarge (16 NeuronCore max):

::

 NEURON_RT_NUM_CORES=3 myapp.py &
 NEURON_RT_NUM_CORES=3 myapp.py &
 NEURON_RT_NUM_CORES=3 myapp.py &
 NEURON_RT_NUM_CORES=3 myapp.py &
 NEURON_RT_NUM_CORES=3 myapp.py &

Executing a 6th ``NEURON_RT_NUM_CORES=3 myapp.py &`` in the above example would fail because only a single NeuronCore is still free.


Notes
~~~~~

1. The number of NeuronCores in an Inferentia device is 4.
2. The number of Inferentia devices depends on the instance size.
3. The NeuronCore index in ``NEURON_RT_VISIBLE_CORES`` starts from 0 and ends at (number of Neuron devices * number of NeuronCores) - 1.
4. By default, ``NEURON_RT_NUM_CORES`` is set to ``0``, which indicates to Neuron Runtime that all cores are to be used.
5. ``NEURON_RT_VISIBLE_CORES`` takes precedence over ``NEURON_RT_NUM_CORES``. If specified, all cores within the range will be assigned to the owning process.


Logging and debugging
---------------------

By default, Neuron Runtime logs to both console and syslog at level ``WARN`` for every submodule except ``HAL``, which logs at ``ERROR``. Use ``NEURON_RT_LOG_LEVEL`` to raise or lower the global verbosity, and ``NEURON_RT_LOG_LOCATION`` to choose where logs go.

Setting only ``NEURON_RT_LOG_LEVEL`` sends logs to the console alone and stops syslog output, and setting only ``NEURON_RT_LOG_LOCATION`` lowers the level to ``ERROR``, so warnings are no longer logged.

::

 NEURON_RT_LOG_LEVEL=INFO myapp.py       # raises console verbosity to INFO

By default, Neuron Runtime allows each model execution 30 seconds to complete. If the NeuronCore doesn't complete execution within that window, the runtime fails the execution with a timeout error. Most models take a few milliseconds, so 30 seconds is more than adequate. However, distributed workloads can legitimately need longer — network interruptions are the usual cause of spurious timeouts — so you can increase the timeout with ``NEURON_RT_EXEC_TIMEOUT``.

::

 NEURON_RT_EXEC_TIMEOUT=120 myapp.py       # increases the timeout to 120 seconds


Additional logging controls
---------------------------

Neuron Runtime enables detailed control over logging behaviors, including the ability to set separate log levels and log locations for individual components. When ``NEURON_RT_LOG_LEVEL`` is set globally, Neuron Runtime combines the logs from all modules into a single stream. For instance, the logs from the modules ``TDRV`` and ``NMGR`` would appear in the same stream as shown in the example below:

::

  2023-Jan-09 20:27:41.0593 15042:15042 ERROR  TDRV:exec_consume_infer_status_notifications (FATAL-RT-UNDEFINED-STATE) inference timeout (600000 ms) on Neuron Device 0 NC 0, waiting for execution completion notification
  2023-Jan-09 20:27:41.0600 15042:15042 ERROR  NMGR:dlr_infer

However, it is possible to adjust the log level for individual submodules to capture more or less detail as required for specific debugging contexts. Commonly used submodules include:

- ``TDRV``: the low-level driver library
- ``NMGR``: the higher-level manager library bridging the driver and runtime
- ``NRT``: the Neuron Runtime library responsible for loading and executing models that is exposed to end users and frameworks

For the complete list of loggable submodules, see :ref:`nrt-env-var-reference`.

To adjust the log level for an individual submodule, use the environment variable ``NEURON_RT_LOG_LEVEL_<submodule>``, where ``<submodule>`` is the submodule identifier (for example ``TDRV``, ``NMGR``, or ``NRT``). This allows for precise control over the verbosity of logs generated by each submodule, facilitating more targeted debugging. For example, the following sets different log levels for the ``TDRV`` and ``NMGR`` submodules:

::

  export NEURON_RT_LOG_LEVEL_TDRV=DEBUG
  export NEURON_RT_LOG_LEVEL_NMGR=ERROR


Similarly, to specify separate log locations for individual submodules, use the environment variable ``NEURON_RT_LOG_LOCATION_<submodule>``, following the same naming convention as for log levels. As with the global controls, selecting one sink for a submodule disables the other for that submodule. This feature enables logs from different submodules to be directed to separate destinations, making it easier to organize and analyze the log output. For example, the following sets different log locations for the ``TDRV`` and ``NMGR`` submodules:

::

  export NEURON_RT_LOG_LOCATION_TDRV=tdrv.log
  export NEURON_RT_LOG_LOCATION_NMGR=nmgr.log



Checksum
--------

To execute a model (NEFF), Neuron Runtime needs to load the NEFF file into the NeuronCore and run. Neuron Runtime provides a way to do checksum validation on each NEFF file while loading to validate the file is not corrupted. This option is off by default to avoid a performance penalty during model load time (~50%).

::

 NEURON_RT_VALIDATE_HASH=true myapp1.py     # enables model checksum validation while loading
 NEURON_RT_VALIDATE_HASH=false myapp2.py    # disables (default) model checksum validation while loading


Related topics
--------------

* :ref:`nrt-env-var-reference` — complete reference for all runtime environment variables, grouped by use class
* :ref:`neuron-runtime-guides` — all Neuron Runtime how-to guides
* :ref:`nrt-api-guide` — Neuron Runtime developer guide for C/C++ applications
* :ref:`nrt-migrate-to-explicit-async` — migrate to the explicit async APIs
* :ref:`nrt-troubleshooting` — troubleshooting runtime issues on Inf1 and Trn1
