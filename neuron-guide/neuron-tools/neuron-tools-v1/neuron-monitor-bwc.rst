.. _neuron-monitor-bwc:

Neuron Monitor 2.x Backwards Compatibility with Neuron Runtime 1.x
==================================================================

neuron-Monitor 2.x can also monitor ``neuron-rtd`` daemons by adding an entry in the configuration file
for each of them and specifying their GRPC address instead of a ``"tag_filter"``. These entries can coexist
with entries for Neuron applications (which use the ``"tag_filter"`` field):

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
     ],
     "system_metrics": [
         ...
     ]
   }

``neuron-rtd`` entries in the output JSON will have a non-empty ``"address"`` field and the tag
will contain its GRPC address:

::

   {
     "neuron_runtime_data": [
       {
         "pid": 0,
         "address": "unix:/run/neuron.sock",
         "neuron_runtime_tag": "unix:/run/neuron.sock",
         "error": "",
         "report": {
           "neuroncore_counters": {
               [...]
           },
