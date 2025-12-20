.. post:: December 16, 2025
    :language: en
    :tags: announcement-end-of-support-nxdt-nxd-core

.. _announcement-end-of-support-nxdt-nxd-core:

Announcing End of Support for NxDT and NxD Core Training APIs Starting with PyTorch 2.10
-----------------------------------------------------------------------------------------

Neuron support for PyTorch 2.9 will be the last to include the NeuronX Distributed Training (NxDT) libraries, NxD Core training APIs, and PyTorch/XLA for training. Starting with Neuron support for PyTorch 2.10, these components will no longer be supported.

How does this impact you
^^^^^^^^^^^^^^^^^^^^^^^^^

Existing NxDT/NxD Core users should stay on PyTorch 2.9 until ready to migrate to native PyTorch on Neuron (starting PyTorch 2.10). Customers are recommended to use native PyTorch with standard distributed primitives (DTensor, FSDP, DDP) and TorchTitan starting with Neuron 2.28 and PyTorch 2.10. A migration guide will be published in a coming release.

See :doc:`Native PyTorch on Neuron Overview </frameworks/torch/pytorch-native-overview>` for more information.
