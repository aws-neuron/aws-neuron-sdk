.. post:: May 20, 2026
    :language: en
    :tags: announce-no-support-pytorch-xla-training

.. _announce-no-support-pytorch-xla-training:

Neuron no longer supports PyTorch/XLA for training starting with Neuron 2.30
-----------------------------------------------------------------------------

Starting with :ref:`Neuron Release 2.30.0 <neuron-2.30.0-whatsnew>`, PyTorch/XLA is no longer supported for training workloads on Trainium. PyTorch 2.9 was the last version based on the PyTorch/XLA backend for training.

Customers using PyTorch/XLA for training should migrate to native PyTorch on Neuron (TorchNeuron), which uses standard distributed primitives (DTensor, FSDP, DDP) and TorchTitan starting with Neuron 2.31 and PyTorch 2.12. PyTorch/XLA continues to be supported for inference workloads.

Native PyTorch on Neuron is currently available as a Private Beta. To request access, contact your AWS account manager.

For migration guidance, see :doc:`/frameworks/torch/index`.

