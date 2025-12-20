.. meta::
  :description: Reference guide for running inference with NeuronX Distributed Inference (NxDI) on AWS Neuron for Trainium and Inferentia ML chips.

.. _nxdi-models-index:

Neuron Inference Model Support
=================================

This section provides information on model support in **NeuronX Distributed Inference (NxDI)** and how to determine appropriate configurations for both online and offline use cases.

.. _nxdi-models-llama3-index:

Llama 3
---------------------------

Meta's Llama 3 family includes large language models available in multiple sizes and versions. Select the model variant that matches your application requirements:

.. grid:: 1
  :gutter: 1

  .. grid-item-card:: Llama 3.3 70B

    Meta's multilingual LLM, featuring 70B parameters and Grouped Query Attention.

    :bdg-ref-primary:`Quickstart <nxdi-models-llama-3-3-70b-instruct-quickstart>`

.. _nxdi-models-qwen3-index:

Qwen 3
---------------------------

Qwen 3 family includes large language models available in multiple sizes and versions. Select the model variant that matches your application requirements:

.. grid:: 1
  :gutter: 1

  .. grid-item-card:: Qwen3 MoE 235B

    Qwen family multilingual LLM, featuring sparse Mixture-of-Experts and Grouped Query Attention

    :bdg-ref-primary:`Quickstart <nxdi-models-qwen3-235b-a22b-quickstart>`

.. note::
   Instructions for additional models will be available soon. For a complete list of supported model architectures, refer to this :ref:`developer guide <nxdi-model-reference>`.

.. toctree::
   :maxdepth: 1
   :hidden:

   llama3/llama_33_70b
   qwen3/qwen3_moe_235b

