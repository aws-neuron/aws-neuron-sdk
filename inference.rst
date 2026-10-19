.. meta::
   :description: Task-oriented entry point for inference on AWS Neuron — serve models with the vLLM Neuron and vLLM Omni Neuron plugins or native PyTorch. NxD Inference and NxD Core are legacy, end of support announced starting Neuron 2.34.
   :keywords: AWS Neuron, Trainium, Inferentia, inference, vLLM, vLLM Neuron, native PyTorch, NxD Inference, legacy, LLM serving, NKI
   :date-modified: 2026-08-28

.. _inference-home:
.. _neuron-inference-overview:

=======================
Inference on AWS Neuron
=======================

AWS Neuron runs inference on AWS Trainium and Inferentia for large language models (LLMs),
image and video generation, and custom model architectures. This page helps you pick a
serving path — the vLLM Neuron plugin, vLLM Omni Neuron, or native PyTorch. Neuron supports
both low-latency online serving and high-throughput batch inference. NxD Inference and NxD Core
are legacy frameworks, announced end of support starting the Neuron 2.34 release; see
:ref:`Legacy inference frameworks <inference-legacy-home>` below.

.. admonition:: Does this page fit your task?
   :class: tip

   **Use this page if** you are choosing how to serve a model on AWS Trainium or Inferentia and
   want the right path without knowing which library implements it.
   **Scenario:** you have a trained model (an open-weight LLM, a fine-tuned checkpoint, or a
   diffusion model) and want to deploy it for online or offline inference on Neuron.
   **Not this page if** you are training a model — see :ref:`training-home` instead.

Choose your inference path
==========================

The recommended serving path for new inference work on Neuron is the **vLLM Neuron plugin**. For
diffusion and multimodal generation, use **vLLM Omni Neuron**. You can also build on **native
PyTorch** directly. Pick the path that matches your model and hardware.

vLLM Neuron (recommended)
-------------------------

The :ref:`vLLM Neuron plugin <vllm-neuron-home>` is the recommended path for serving
LLMs such as Llama 3.3 70B and gpt-oss on Neuron. It offers continuous
batching, prefix caching, speculative decoding, and quantization, and exposes both offline
batch inference and OpenAI-compatible online serving. Start from the
:doc:`online </vllm-neuron/docs/getting-started/quickstart-online-serving>` or
:doc:`offline </vllm-neuron/docs/getting-started/quickstart-offline-serving>` serving
quickstart, then follow a :doc:`model recipe </vllm-neuron/docs/model-recipes/index>`.
Setup, CLI flags, and feature configuration live on the vLLM Neuron landing page rather
than here.

vLLM Omni Neuron (Beta — diffusion and multimodal)
--------------------------------------------------

For diffusion and multimodal generation models such as video generation, use the
:doc:`vLLM Omni Neuron plugin </vllm-omni-neuron/docs/index>`. It is an out-of-tree Neuron
backend for `vLLM Omni <https://github.com/vllm-project/vllm-omni>`_ that reuses vLLM Omni's
upstream serving components and adds Neuron-specific execution, caching, and sharding. The
initial Beta release serves the Wan2.2-T2V-A14B (text-to-video) and Wan2.2-I2V-A14B
(image-to-video) models on Trn2 and Trn3, with offline generation through the ``Omni.generate``
API and OpenAI-compatible online serving at the ``/v1/videos`` endpoint. Start from the
:doc:`online </vllm-omni-neuron/docs/getting-started/quickstart-online-serving-wan22>` or
:doc:`offline </vllm-omni-neuron/docs/getting-started/quickstart-offline-serving-wan22>`
serving quickstart. vLLM Omni Neuron is also published as a Neuron Deep Learning Container; see
the :doc:`vLLM Omni Neuron setup guide </vllm-omni-neuron/docs/getting-started/setup-guide>`.

Native PyTorch
--------------

All inference on Trainium runs on native PyTorch (TorchNeuron) underneath, and you can build on
it directly — running native PyTorch models with ``torch.compile``, developing a custom
inference framework, or adapting another model-serving framework for Trainium. A serving plugin
is the recommended path when it fits your model, but it is not required. See
:ref:`Native PyTorch inference on Neuron <inference-native-pytorch>` for how inference works on
Trainium and whether to use the ``torch-neuronx`` or ``libtorch-neuronx-lite`` package. If your
workflow is built around the Hugging Face ecosystem, you can also serve Neuron-compiled models
with `Optimum Neuron <https://huggingface.co/docs/optimum-neuron/index>`_, maintained by Hugging
Face.

Custom models and performance optimization
-------------------------------------------

To bring up a custom diffusion or multimodal generation model on Trn2/Trn3, the vLLM Omni
Neuron plugin (Beta) ships Neuron-optimized NKI kernels and reference implementations you can
adapt. See its :doc:`model onboarding guide </vllm-omni-neuron/docs/model-dev/onboarding-models>`.

To implement a custom model directly on native PyTorch, or to build a custom inference stack,
see :ref:`Native PyTorch inference on Neuron <inference-native-pytorch>`. The legacy
:ref:`NxD Inference and NxD Core frameworks <inference-legacy-home>` are also available for
custom NxDI-API serving and for building custom models on the NxD Core primitives, but they are
announced end of support starting Neuron 2.34. See the NKI note below for where custom kernels
fit.

Learn the concepts
==================

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: vLLM Neuron plugin
      :link: /vllm-neuron/index
      :link-type: doc
      :class-card: sd-border-2

      :bdg-info:`overview`
      ^^^
      The recommended serving path for new inference work — continuous batching, speculative decoding,
      quantization, and OpenAI-compatible online serving.

   .. grid-item-card:: vLLM Neuron features guide
      :link: /vllm-neuron/docs/guides/features-guide
      :link-type: doc
      :class-card: sd-border-2

      :bdg-info:`deep dive`
      ^^^
      The serving features vLLM Neuron supports and how to configure them for your workload.

   .. grid-item-card:: Native PyTorch inference
      :link: inference-native-pytorch
      :link-type: ref
      :class-card: sd-border-2

      :bdg-info:`overview`
      ^^^
      Run inference with native PyTorch on Trainium, and build a custom inference stack with
      the ``torch-neuronx`` or ``libtorch-neuronx-lite`` package.

Start serving
=============

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Online serving quickstart
      :link: /vllm-neuron/docs/getting-started/quickstart-online-serving
      :link-type: doc
      :class-card: sd-border-1

      :bdg-primary:`quickstart`
      ^^^
      Stand up an OpenAI-compatible vLLM Neuron server and send your first request.

   .. grid-item-card:: Offline serving quickstart
      :link: /vllm-neuron/docs/getting-started/quickstart-offline-serving
      :link-type: doc
      :class-card: sd-border-1

      :bdg-primary:`quickstart`
      ^^^
      Run offline batch inference with vLLM Neuron for high-throughput workloads.

   .. grid-item-card:: Llama 3 recipe
      :link: /vllm-neuron/docs/model-recipes/llama-3
      :link-type: doc
      :class-card: sd-border-1

      :bdg-primary:`tutorial`
      ^^^
      End-to-end recipe for serving Llama 3 on Neuron with vLLM Neuron.

   .. grid-item-card:: EAGLE3 speculative decoding
      :link: /vllm-neuron/docs/tutorials/tutorial-eagle3-speculative-decoding-llama-3-1
      :link-type: doc
      :class-card: sd-border-1

      :bdg-primary:`tutorial`
      ^^^
      Accelerate Llama 3.1 serving with EAGLE3 speculative decoding on Trn2/Trn3.

.. note:: Where NKI fits in inference

   The serving backends use :doc:`NKI </nki/index>` kernels internally — the
   :doc:`NKI Library </nki/library/index>` ships tuned attention, MoE, and quantization
   kernels that the vLLM Neuron plugin and NxD Inference call for supported architectures.
   Write a custom NKI kernel when you need to close a remaining performance gap.

Legacy inference frameworks
===========================

NxD Inference (NxDI, the ``neuronx-distributed-inference`` library) and NxD Core are legacy and
are **announced end of support on Trn2 and Trn3 starting the Neuron 2.34 release**. For new
inference work, use the recommended paths above.

The documentation for both lives under :ref:`Legacy inference frameworks
<inference-legacy-home>`.

Related
=======

- :ref:`vLLM Neuron <vllm-neuron-home>` — the recommended inference section for new work, with
  setup, feature guides, model recipes, and tutorials.
- :ref:`Legacy inference frameworks <inference-legacy-home>` — maintained
  documentation for Trn1/Inf2 inference and custom NxDI-API serving.
- :ref:`Training on Neuron <training-home>` — the task-oriented entry point for training
  models on Neuron.
