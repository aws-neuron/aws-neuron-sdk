.. post:: June 28, 2024
    :language: en
    :tags: announce-eos-neuron-device-version, neuron-device-version

.. _announce-eos-neuron-device-version:

Announcing end of support for 'neuron-device-version' field in neuron-monitor
-------------------------------------------------------------------------------

:ref:`Neuron release 2.19 <neuron-2.19.0-whatsnew>` will be the last release to include the field 'neuron-device-version' in neuron-monitor.

In future releases, customers who are using the field 'neuron-device-version' will instead need to use 'instance_type' field in the 'instance_info' section and the 'neuroncore_version' field to obtain neuron device information.

Please see :ref:`neuron-monitor-ug` for more details.
