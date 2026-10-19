.. post:: October 16, 2026
    :language: en
    :tags: announce-eos-torch-xla-inference-trn

.. _announce-eos-torch-xla-inference-trn:

Announcing end of support for the Torch/XLA inference path on Trn2 and Trn3 starting in Neuron 2.34
----------------------------------------------------------------------------------------------------

As part of the :ref:`transition to native PyTorch support on Trainium <announce-transition-pytorch-trainium>`, native PyTorch support through TorchNeuron is now available starting with :ref:`Neuron Release 2.33 <whats-new-2026-10-19-v2_33>`. With native PyTorch support in place, the Torch/XLA inference path is no longer required on Trn2 and Trn3.

Starting with the next full Neuron release, Neuron will end support for the Torch/XLA inference path on Trn2 and Trn3 instances. The native PyTorch path remains the default and recommended path.

On Inf2 and Trn1 instances, Torch/XLA inference support remains in maintenance mode, as described in :ref:`announce-maintenance-pytorch-xla-inference`.
