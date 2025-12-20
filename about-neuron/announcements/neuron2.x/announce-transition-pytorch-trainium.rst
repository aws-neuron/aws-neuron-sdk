.. post:: December 16, 2025
    :language: en
    :tags: announce-transition-pytorch-trainium

.. _announce-transition-pytorch-trainium:

Announcing Transition to PyTorch Native Support for AWS Trainium in the Next Neuron Release Supporting PyTorch 2.10
------------------------------------------------------------------------------------------------------------------------

Starting with the introduction of Neuron support for PyTorch 2.10, AWS Neuron will begin a transition from PyTorch/XLA to native PyTorch support via TorchNeuron. PyTorch 2.9 will be the last version based on PyTorch/XLA.

What's changing
^^^^^^^^^^^^^^^^

* If you are using PyTorch 2.9, it will be the last version of it that uses the PyTorch/XLA backend in Neuron.
* For PyTorch 2.10 and later users, Neuron will provide Native PyTorch support via TorchNeuron.

Customers using PyTorch/XLA-based training should migrate to native PyTorch with TorchNeuron, which provides:

* Native PyTorch eager execution mode
* Standard distributed primitives (DTensor, FSDP, DDP)
* ``torch.compile`` support
* Compatibility with frameworks like TorchTitan (PyTorch Training Library)

For more information about native PyTorch on Neuron and migration guidance, see :doc:`Native PyTorch for AWS Trainium </frameworks/torch/pytorch-native-overview>`.

