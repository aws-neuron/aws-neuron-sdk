.. post:: Mar 25, 2022
    :language: en
    :tags: announce-eol


Announcing end of support for ``NEURONCORE_GROUP_SIZES`` starting with Neuron 1.20.0 release
--------------------------------------------------------------------------------------------

Starting with Neuron SDK 1.20.0, ``NEURONCORE_GROUP_SIZES`` environment variable will no longer be supported. Setting 
``NEURONCORE_GROUP_SIZES`` environment variable will no longer affect applications behavior.
Current customers using ``NEURONCORE_GROUP_SIZES`` environment variable are advised to use ``NEURON_RT_VISIBLE_CORES`` environment variable  or ``NEURON_RT_NUM_CORES`` environment variable instead.

See :ref:`eol-ncg`, :ref:`nrt-configuration` and :ref:`neuron-migrating-apps-neuron-to-libnrt` for more information.
