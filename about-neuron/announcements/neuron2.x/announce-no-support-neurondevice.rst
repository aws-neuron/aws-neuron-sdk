.. post:: September 16, 2024
    :language: en
    :tags: eos-neuron-device-version, neuron-device-version

.. _eos-neuron-device-version:

'neuron-device-version' field in neuron-monitor no longer supported
--------------------------------------------------------------------

Starting with :ref:`Neuron release 2.20 <neuron-2-20-2-whatsnew>`, Neuron no longer supports the field 'neuron-device-version' in neuron-monitor.

Customers who are using the field 'neuron-device-version' will instead need to use 'instance_type' field in the 'instance_info' section and the 'neuroncore_version' field to obtain neuron device information.

Please see :ref:`neuron-monitor-ug` for more details.
