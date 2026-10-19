.. post:: October 16, 2026
    :language: en
    :tags: announce-eos-vllm-neuron-torch-xla

.. _announce-eos-vllm-neuron-torch-xla:

Announcing end of support for the Torch/XLA path in the vLLM Neuron plugin on Trn2 and Trn3 starting in Neuron 2.34
-------------------------------------------------------------------------------------------------------------------

As part of the :ref:`transition to native PyTorch support on Trainium <announce-transition-pytorch-trainium>`, native PyTorch support through TorchNeuron is the default execution path in the `vLLM Neuron plugin <https://github.com/vllm-project/vllm-neuron>`_ starting with :ref:`Neuron Release 2.33 <whats-new-2026-10-19-v2_33>`.

Starting with Neuron Release 2.34, Neuron will end support for the Torch/XLA path in the vLLM Neuron plugin on Trn2 and Trn3 instances.

Current users should migrate to the native PyTorch API supported through the vLLM Neuron plugin, which remains the default and recommended path. For details and migration guidance, see the `vLLM Neuron plugin documentation <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/index.html>`_.
