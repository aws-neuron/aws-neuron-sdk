.. post:: June 28, 2026
    :language: en
    :tags: announce-no-longer-support, neuron-dlami, nxdt

.. _announce-no-longer-include-nxdt-dlami:

Neuron DLAMIs no longer include NxD Training starting with Neuron 2.31.0
-------------------------------------------------------------------------

Starting with Neuron SDK 2.31.0, NeuronX Distributed Training (NxDT) is no longer included in Neuron Deep Learning AMIs (DLAMIs).

To use a DLAMI with NxDT, use a Neuron DLAMI from previous Neuron releases (Neuron 2.30.0 or earlier). The Neuron SDK version is included in each DLAMI's name and description.

NxDT and NxD Core training APIs are in maintenance mode and receive critical security fixes only. We recommend migrating to native PyTorch on Neuron with standard distributed primitives (DTensor, FSDP, DDP). See :doc:`Native PyTorch on Neuron Overview </frameworks/torch/pytorch-native-overview>` for more information.
