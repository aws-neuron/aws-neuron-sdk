.. post:: May 20, 2026
    :language: en
    :tags: announce-maintenance-nxdt

.. _announce-maintenance-nxdt-nxd-core-training:

NxDT and NxD Core Training APIs are now in maintenance mode starting with Neuron 2.30
--------------------------------------------------------------------------------------

Starting with :ref:`Neuron Release 2.30.0 <neuron-2.30.0-whatsnew>`, NxDT and NxD Core Training APIs are in maintenance mode. Future releases will address critical security issues only and we will gradually end support.

Existing NxDT/NxD Core users should stay on Neuron 2.28 and PyTorch 2.9 until ready to migrate to native PyTorch on Neuron (starting Neuron 2.31 and PyTorch 2.12). Customers are recommended to use native PyTorch with standard distributed primitives (DTensor, FSDP, DDP) and TorchTitan starting with Neuron 2.31 and PyTorch 2.12.

Native PyTorch on Neuron is currently available as a Private Beta. To request access, contact your AWS account manager. A migration guide will be published in a coming release.

