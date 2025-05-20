.. post:: May 15, 2025
    :language: en
    :tags: announce-eos-mllama-checkpoint

.. _announce-eos-mllama-checkpoint:

Announcing end of support for mllama 3.2 Meta Checkpoint API starting next release
--------------------------------------------------------------------------------------

:ref:`Neuron Release 2.23 <neuron-2.23.0-whatsnew>` will be the last release to include support for the mllama 3.2 Meta checkpoint API. In the next release (Neuron 2.24), Neuron will end support.

 All previously converted checkpoints will continue to function without disruption. Customers' existing workflows and converted models remain fully operational. For new checkpoint conversions, the HuggingFace solution provides equivalent functionality. Customers are recommended to use HuggingFace's official conversion script, available here:
:ref:`HuggingFace Conversion Script <https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/convert_mllama_weights_to_hf.py>`_
