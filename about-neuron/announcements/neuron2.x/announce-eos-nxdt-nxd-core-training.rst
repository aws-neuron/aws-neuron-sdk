.. post:: February 26, 2026
    :language: en
    :tags: announce-eos-nxdt

.. _announce-eos-nxdt-nxd-core-training:

Announcing end of support for NxDT and NxD Core Training APIs starting with Neuron SDK release 2.29 (PyTorch 2.10)
-------------------------------------------------------------------------------------------------------------------

Neuron SDK release 2.28 (PyTorch 2.9) will be the last release to include the NeuronX Distributed Training (NxDT) library. Starting with Neuron SDK release 2.29 (PyTorch 2.10), the use of NxD Core training APIs and the PyTorch/XLA package for training will no longer be supported.

How does this impact you?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Existing NxDT/NxD Core users should stay on Neuron SDK 2.28 (PyTorch 2.9) until ready to migrate to native PyTorch on Neuron. Native PyTorch on Neuron uses standard distributed primitives (DTensor, FSDP, DDP). A migration guide will be published in a coming release.

See :doc:`Native PyTorch on Neuron Overview </frameworks/torch/pytorch-native-overview>` for more information.
