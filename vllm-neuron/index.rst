:orphan:

.. meta::
   :description: vLLM Neuron plugin documentation home — serve LLMs on AWS Trainium (Trn2 and Trn3) with quickstarts, feature guides, model recipes, and tutorials.
   :keywords: vLLM, vLLM Neuron, AWS Neuron, Trainium, LLM serving, continuous batching, speculative decoding
   :date-modified: 2026-09-01

.. _vllm-neuron-home:

==================================
vLLM Neuron: LLM serving on Neuron
==================================

.. note::
   vLLM Neuron is currently in Beta.

The vLLM Neuron plugin brings the full vLLM serving stack to AWS Trainium (Trn2 and Trn3). It
is the recommended path for serving large language models (LLMs) on Neuron, exposing the
standard ``vllm serve`` command and an OpenAI-compatible API. This page is the section home
for vLLM Neuron: it points you to the setup guide, quickstarts, feature and configuration
references, model recipes, and tutorials.

The plugin supports continuous batching, prefix caching, speculative decoding (EAGLE3),
disaggregated inference, structured outputs, quantization, and multimodal models. Its
documentation is sourced from the vLLM Neuron plugin repository and maintained alongside the
plugin itself.

Applies to
----------

This section is applicable to:

- Online, OpenAI-compatible LLM serving on AWS Trainium (Trn2/Trn3).
- High-throughput offline batch inference with the ``vllm.LLM`` Python API.
- Migrating existing NxD Inference deployments to the vLLM Neuron plugin.

Get started
-----------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Setup guide
      :link: /vllm-neuron/docs/getting-started/setup-guide
      :link-type: doc
      :class-card: sd-border-2

      Install and configure vLLM Neuron on a Trainium instance.

   .. grid-item-card:: Online serving quickstart
      :link: /vllm-neuron/docs/getting-started/quickstart-online-serving
      :link-type: doc
      :class-card: sd-border-2

      Launch an OpenAI-compatible API server and send your first chat request.

   .. grid-item-card:: Offline serving quickstart
      :link: /vllm-neuron/docs/getting-started/quickstart-offline-serving
      :link-type: doc
      :class-card: sd-border-2

      Run high-throughput batch inference with the ``vllm.LLM`` Python API.

   .. grid-item-card:: Migrate from NxD Inference
      :link: /vllm-neuron/docs/getting-started/migration-nxdi-to-vllm-neuron
      :link-type: doc
      :class-card: sd-border-2

      Move an existing NxDI deployment to the vLLM Neuron plugin.

Deploy and serve
----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Features guide
      :link: /vllm-neuron/docs/guides/features-guide
      :link-type: doc
      :class-card: sd-border-2

      Configure and tune serving features — bucketing, quantization, disaggregated
      inference, and speculation.

   .. grid-item-card:: Configuration reference
      :link: /vllm-neuron/docs/guides/reference-configuration
      :link-type: doc
      :class-card: sd-border-2

      All Neuron-specific options in ``additional_config`` and the supported environment
      variables.

   .. grid-item-card:: Model recipes
      :link: /vllm-neuron/docs/model-recipes/index
      :link-type: doc
      :class-card: sd-border-2

      Supported models and their feature tables — Llama 3, GPT-OSS, Qwen3-VL, and more.

   .. grid-item-card:: Tutorials
      :link: /vllm-neuron/docs/tutorials/index
      :link-type: doc
      :class-card: sd-border-2

      End-to-end walk-throughs for speculative decoding, disaggregated inference, and
      multimodal serving.

Develop and understand
----------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Model development
      :link: /vllm-neuron/docs/model-dev/index
      :link-type: doc
      :class-card: sd-border-2

      Onboard new architectures, add vision-language models, and debug accuracy on Neuron.

   .. grid-item-card:: Concepts and architecture
      :link: /vllm-neuron/docs/design/index
      :link-type: doc
      :class-card: sd-border-2

      Design deep dives — parallelism, speculation, compilation, and vLLM integration
      internals.

The source code for the vLLM Neuron plugin is hosted in the public
`vLLM Neuron GitHub repository <https://github.com/vllm-project/vllm-neuron>`_, where external
contributors can clone the plugin and open documentation pull requests.

Related
-------

- :ref:`Inference on AWS Neuron <inference-home>` — the task-oriented entry point that places
  vLLM Neuron alongside NxD Inference and Hugging Face serving paths.

.. toctree::
   :hidden:
   :maxdepth: 2

   docs/index
