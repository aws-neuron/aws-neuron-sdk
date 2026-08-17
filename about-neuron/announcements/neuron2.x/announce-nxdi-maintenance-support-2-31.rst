.. post:: August 12, 2026
    :language: en
    :tags: announce-nxdi-maintenance-update

.. _announce-nxdi-maintenance-support-2-31:

NxD Inference and NxD Core maintenance support with Neuron SDK 2.31
--------------------------------------------------------------------

As announced in :ref:`Neuron 2.30 <announce-maintenance-nxdi-nxd-core-inference>`, NxD Inference and NxD Core Inference APIs are in maintenance mode. NxD Inference 0.10.18399 and NxD Core 0.19.28492 (in Neuron SDK 2.31.0) will continue to be supported with critical bug fixes and security patches. No new releases are planned.

The `vLLM Neuron plugin <https://github.com/vllm-project/vllm-neuron>`_ for Trn2 and Trn3 (which does not have a dependency on the NxD Inference libraries) is now in Beta. Consider migrating to the vLLM Neuron plugin for access to new features and performance improvements. For migration guidance, see  `Migrating from NxD Inference to vLLM Neuron <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/getting-started/migration-nxdi-to-vllm-neuron.html>`_
