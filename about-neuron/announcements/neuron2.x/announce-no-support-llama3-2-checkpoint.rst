.. post:: June 24, 2025
    :language: en
    :tags: announce-no-longer-support-llama-checkpoint

.. _announce-no-longer-support-llama-32-meta-checkpoint:

Announcing end of support for Llama 3.2 Meta checkpoint
---------------------------------------------------------

Starting with :ref:`Neuron Release 2.24 <neuron-2-24-0-whatsnew>`, the mllama 3.2 Meta checkpoint API is no longer be supported.

**I currently use the mllama 3.2 Meta checkpoint in my applications. What do I do?**

All previously converted checkpoints will continue to function without disruption. Customers' existing workflows and converted models remain fully operational. For new checkpoint conversions, customers are advised to use the Hugging Face solution which provides equivalent functionality. Hugging Face's official conversion script is available here:
`HuggingFace Conversion Script <https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/convert_mllama_weights_to_hf.py>`_
