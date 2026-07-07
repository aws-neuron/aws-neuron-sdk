.. post:: June 28, 2026
    :language: en
    :tags: announce-no-longer-support, neuron-containers, dlc

.. _announce-no-longer-publish-pytorch-training-dlc:

Neuron no longer publishes the PyTorch/XLA training DLC starting with Neuron 2.31.0
------------------------------------------------------------------------------------

Starting with Neuron SDK 2.31.0, the PyTorch/XLA training Deep Learning Container (DLC) (``pytorch-training-neuronx``) is no longer published.

If you require this DLC, use an image associated with a previous Neuron SDK release (2.30.0 or earlier). The SDK version is included in each image tag.

For PyTorch-based training on Neuron, we recommend using the Native PyTorch on Neuron DLC. For the available container images, see :ref:`containers_rn`.

