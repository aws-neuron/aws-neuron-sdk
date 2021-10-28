.. _neuron-monitor-upg:

Migrating to Neuron Monitor 2.x
===============================

The new version of neuron-monitor is very similar to its predecessor,
with a few differences.

.. contents::
   :local:
   :depth: 2

Configuration file changes
--------------------------
* The ``"hw_counters"`` metric group has been renamed to ``"neuron_hw_counters"`` and is
  now part of the ``"system_metrics"`` category.
  ::

    ...
    "system_metrics": [
       ...
       {
          "period": "2s",
          "type": "neuron_hw_counters"
       }
     ]
* Each entry in the ``"neuron_runtimes"`` array can either have a ``"tag_filter"`` which
  is a regex that will be used to filter Neuron applications based on their tag; or an
  ``"address"`` field which is used to specify a ``neuron-rtd`` daemon GRPC address that should
  be monitored. For more details on backwards compatibility, read
  ::

    {
     "period": "1s",
     "neuron_runtimes": [
       {
         "address": "unix:/run/neuron.sock",
         "metrics": [
           ...
         ]
       },
       {
         "tag_filter": ".*",
         "metrics": [
           ...
         ]
       }
       ...

Output JSON structure changes
-----------------------------
* The ``"hw_counters"`` metric group has been renamed to ``"neuron_hw_counters"`` and is
  now part of the ``"system_metrics"`` category.
  ::

     "system_metrics": [
       {
         "type": "vcpu_usage"
       },
       {
         "type": "memory_info"
       },
       {
          "period": "2s",
          "type": "neuron_hw_counters"
       }
     ]
* Each ``"neuron_runtime_data"`` object now contains 3 new properties:
  ``"pid"``, ``"address"`` and ``"neuron_runtime_tag"``. The
  ``"neuron_runtime_index"`` property has been removed:
  ::

   {
     "neuron_runtime_data": [
       {
         "pid": 0,
         "address": "",
         "neuron_runtime_tag", "my_app_1",
         ...
* The ``"loaded_models"`` array has been removed from the objects representing
  ``"neuroncore_counters"`` entries since the same information can be found in the ``"memory_used"``
  group.
* A new ``"model"`` error category has been added:
  ::

     "error_summary": {
       "generic": 0,
       "numerical": 0,
       "transient": 0,
       "model": 0,
       "runtime": 0,
       "hardware": 0
     },
