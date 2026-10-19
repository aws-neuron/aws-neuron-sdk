.. meta::
   :description: AWS Neuron SDK enables high-performance deep learning and generative AI on AWS Inferentia and Trainium instances. Get started with native PyTorch, vLLM, NKI kernels, and AI-driven optimization with Neuron Explorer.
   :date-modified: 2026-10-13

.. _neuron_home:

AWS Neuron Documentation
========================

:ref:`AWS Neuron <what-is-neuron>` is the software development kit for deep learning and generative AI on `AWS Inferentia <https://aws.amazon.com/ai/machine-learning/inferentia/>`_ and `AWS Trainium <https://aws.amazon.com/ai/machine-learning/trainium/>`_ instances. Neuron supports multiple development paths: serving large language models with vLLM, training and inference with PyTorch (native) and JAX, authoring custom kernels with NKI, and direct use of the Neuron Graph Compiler and Runtime.

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: :octicon:`tag;1em;sd-text-primary` Current release: Neuron 2.33.0
      :link: /release-notes/2.33.0
      :link-type: doc
      :class-card: sd-border-2
      :class-title: sd-fs-6

      **Released October 19, 2026**. Select this card for the details!

----

Train and deploy models on Neuron
---------------------------------

Neuron now supports native PyTorch on Trn2 and Trn3 platforms with TorchNeuron. Run your existing PyTorch code in eager mode or with ``torch.compile``, and change the device from ``cuda`` to ``neuron``. TorchNeuron is currently in Beta.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: :octicon:`flame;1em;sd-text-primary` TorchNeuron: Native PyTorch on Trainium (Beta)
      :link: torch-neuron-home
      :link-type: ref
      :class-card: sd-border-2
      :class-title: sd-fs-5

      Use standard PyTorch APIs on Trainium: eager mode, ``torch.compile`` with ``backend="neuron"``, and the FSDP, DTensor, DDP, and Tensor Parallelism distributed APIs. Most PyTorch workflows run with few or no changes.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`graph;1em;sd-text-primary` Train models on Neuron
      :link: training-home
      :link-type: ref
      :class-card: sd-border-1

      What Neuron provides for model training: native PyTorch (TorchNeuron), TorchTitan Neuron, and Hugging Face.

   .. grid-item-card:: :octicon:`server;1em;sd-text-primary` Deploy models on Neuron
      :link: inference-home
      :link-type: ref
      :class-card: sd-border-1

      What Neuron provides for inference: vLLM Neuron and vLLM Omni Neuron for model serving.

----

Monitor and optimize your model
-------------------------------

Use Neuron Observer to monitor the health of your Trainium cluster, and use Neuron Explorer to profile your model and find ways to make it faster.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`pulse;1em;sd-text-primary` Neuron Observer (Beta)
      :link: neuron-observer-index
      :link-type: ref
      :class-card: sd-border-1

      Monitor the health of Trainium clusters on Amazon EKS. Neuron Observer detects hardware faults and finds the affected node and device. Supported on Trn2 and Trn3.

   .. grid-item-card:: :octicon:`telescope;1em;sd-text-primary` Neuron Explorer
      :link: neuron-explorer-home
      :link-type: ref
      :class-card: sd-border-1

      Profile and debug your model and kernels on Trainium and Inferentia. Use the CLI, the UI, or the VS Code integration, and get AI-driven optimization recommendations.

----

Develop with agent support
--------------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: :octicon:`dependabot;1em;sd-text-primary` Neuron Agentic Development
      :link: neuron-agentic-development
      :link-type: ref
      :class-card: sd-border-1

      Use open-source AI agents and skills in coding environments such as Kiro and Claude Code to develop on Trainium. Write, debug, and profile NKI kernels, or port Hugging Face transformer models to run on Trainium, with natural-language prompts.

----

Who Neuron is for
-----------------

* **ML engineers deploying production models** — Deploy prepared :doc:`Neuron Deep Learning AMIs (DLAMIs) </deploy/environments/dlami>` and :doc:`Deep Learning Containers (DLCs) </deploy/environments/index>` on Amazon EC2 Trainium and Inferentia instances. Start with the :doc:`DLAMI setup guide </deploy/environments/dlami>` or the :doc:`DLC quickstart </deploy/environments/quickstart-deploy-dlc>`.

  * **Serving LLMs** — Use vLLM Neuron to serve open-source LLMs with minimal code changes. Start with the :doc:`online serving quickstart </vllm-neuron/docs/getting-started/quickstart-online-serving>` or :doc:`offline serving quickstart </vllm-neuron/docs/getting-started/quickstart-offline-serving>`.
* **ML researchers and model developers** — Use native PyTorch on Trainium with eager mode, ``torch.compile``, and standard distributed APIs. Start with :doc:`TorchNeuron </torch/index>`.
* **Performance engineers optimizing kernels** — Use NKI to write custom kernels with direct NeuronCore access, or pick from the NKI Library's pre-optimized kernels. Start with the :doc:`NKI quickstart </nki/get-started/quickstart-implement-run-kernel>` and :doc:`NKI Library </nki/library/index>`.

----

Start here
----------

Pick the task that matches what you want to do.

.. grid:: 1 1 1 1
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1em;sd-text-primary` Get started with a Neuron DLAMI and PyTorch
      :link: /deploy/environments/dlami
      :link-type: doc
      :class-card: sd-border-1

      Launch a Trainium or Inferentia EC2 instance with a pre-configured **Neuron Deep Learning AMI (DLAMI)** and PyTorch. The DLAMI bundles the Neuron SDK, framework virtual environments (PyTorch, JAX, vLLM), and the system tools — no manual install required. See :doc:`Install PyTorch via Deep Learning AMI </setup/pytorch/dlami>` and get started!

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`server;1em;sd-text-primary` Serve a large language model
      :link: /vllm-neuron/docs/index
      :link-type: doc
      :class-card: sd-border-1

      Run LLM inference on Trainium and Inferentia with **vLLM Neuron**. Supports OpenAI-compatible APIs, continuous batching, and speculative decoding. See the :doc:`offline </vllm-neuron/docs/getting-started/quickstart-offline-serving>` or :doc:`online </vllm-neuron/docs/getting-started/quickstart-online-serving>` serving quickstart.

   .. grid-item-card:: :octicon:`device-camera-video;1em;sd-text-primary` Serve a diffusion or video model
      :link: /vllm-omni-neuron/docs/index
      :link-type: doc
      :class-card: sd-border-1

      Run diffusion and multimodal generation models (such as WAN 2.2 video generation) on Trainium 2 and Trainium 3 with **vLLM Omni Neuron** (Beta). See the :doc:`offline </vllm-omni-neuron/docs/getting-started/quickstart-offline-serving-wan22>` or :doc:`online </vllm-omni-neuron/docs/getting-started/quickstart-online-serving-wan22>` serving quickstart.

   .. grid-item-card:: :octicon:`graph;1em;sd-text-primary` Train a model with PyTorch
      :link: /torch/index
      :link-type: doc
      :class-card: sd-border-1

      Use **native PyTorch on Trainium** (TorchNeuron) with eager mode, ``torch.compile``, and the standard distributed APIs (FSDP, DTensor, DDP). Existing PyTorch code runs with minimal changes; primarily swap ``cuda`` for ``neuron`` on your tensors.

   .. grid-item-card:: :octicon:`code;1em;sd-text-primary` Write custom NKI kernels
      :link: /nki/get-started/quickstart-implement-run-kernel
      :link-type: doc
      :class-card: sd-border-1

      Program NeuronCores directly with **NKI** when you need finer control than framework-level compilation provides. NKI offers tile-level programming with Python and NumPy-like syntax, and ships with a library of pre-optimized kernels (attention, MoE, and others).

----

Neuron SDK organization
-----------------------

The Neuron SDK includes these major components.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - What it does and when to use it
   * - :octicon:`flame;1em;sd-text-primary` :doc:`TorchNeuron </torch/index>`
     - Native PyTorch backend for Trainium, with eager mode, ``torch.compile``, and standard distributed APIs. Use it to train or run PyTorch models with few code changes.
   * - :octicon:`graph;1em;sd-text-primary` :doc:`TorchTitan Neuron </training/torchtitan/index>`
     - PyTorch-native framework for large-scale model pretraining. Use it to pretrain LLMs on Trainium clusters.
   * - :octicon:`server;1em;sd-text-primary` :doc:`vLLM Neuron </vllm-neuron/docs/index>`
     - vLLM plugin for LLM serving with OpenAI-compatible APIs and continuous batching. Use it to serve open-source LLMs in production.
   * - :octicon:`device-camera-video;1em;sd-text-primary` :doc:`vLLM Omni Neuron </vllm-omni-neuron/docs/index>`
     - vLLM plugin for diffusion and multimodal generation models. Use it to serve image and video generation models on Trn2 and Trn3.
   * - :octicon:`beaker;1em;sd-text-primary` :doc:`JAX NeuronX </jax/index>`
     - JAX integration for Neuron (Beta). Use it to run JAX training and inference code on Trainium.
   * - :octicon:`code;1em;sd-text-primary` :doc:`Neuron Kernel Interface (NKI) </nki/index>`
     - Python programming interface for custom kernels with direct NeuronCore access. Use it when framework-level compilation does not give you the performance you need.
   * - :octicon:`package;1em;sd-text-primary` :doc:`NKI Library </nki/library/index>`
     - Open-source, pre-optimized kernels for attention, MLP, MoE, normalization, and more. Use these kernels in your models, or read them as reference implementations.
   * - :octicon:`gear;1em;sd-text-primary` :doc:`Neuron Graph Compiler </compiler/index>`
     - Compiles model graphs and NKI kernels (``neuronx-cc``) into Neuron Executable File Format (NEFF) files. Use the compiler options to tune precision and performance.
   * - :octicon:`cpu;1em;sd-text-primary` :doc:`Neuron Runtime </neuron-runtime/index>`
     - Loads NEFF files and runs them on NeuronCores. Manages devices, memory, and :doc:`collective communications </neuron-runtime/about/collectives>`. Use its settings to configure and debug execution.
   * - :octicon:`telescope;1em;sd-text-primary` :doc:`Neuron Explorer </tools/neuron-explorer/index>`
     - Profiling and debugging suite with AI-driven recommendations, available from the CLI, the UI, or VS Code. Use it to find and fix performance problems.
   * - :octicon:`pulse;1em;sd-text-primary` :ref:`Neuron Observer <neuron-observer-index>`
     - Health monitoring for Trainium clusters on Amazon EKS. Use it to detect hardware faults and find the affected node and device.
   * - :octicon:`tools;1em;sd-text-primary` :doc:`Neuron system tools </tools/neuron-sys-tools/index>`
     - Command-line tools to monitor and inspect Neuron devices on an instance. Use them to check device usage and status.
   * - :octicon:`dependabot;1em;sd-text-primary` :doc:`Neuron Agentic Development </tools/neuron-agentic-development/index>`
     - Skills and context for AI coding agents that work with Neuron. Use them to speed up model and kernel development.
   * - :octicon:`container;1em;sd-text-primary` :doc:`Deploy on AWS </deploy/index>`
     - Pre-configured Deep Learning AMIs (DLAMIs) and Deep Learning Containers (DLCs) for Amazon EC2, EKS, ECS, SageMaker, and ParallelCluster. Use them to set up an environment without a manual install.
   * - :octicon:`archive;1em;sd-text-primary` :doc:`Legacy inference frameworks </inference-legacy/index>` · :doc:`Legacy training frameworks </training/legacy-frameworks>`
     - Earlier components, such as PyTorch NeuronX (``torch-neuronx``) and the NeuronX Distributed (NxD) libraries. Use them only to maintain existing workloads.

----

Learn more
----------

* :doc:`What is AWS Neuron? </about-neuron/what-is-neuron>` — Background on Inferentia, Trainium, and the Neuron SDK.
* :doc:`Release notes </release-notes/index>` — Component-by-component release notes for every Neuron SDK version.
* :doc:`Open source and contribute </about-neuron/oss/index>` — Public GitHub repositories, contribution guidelines, and source for TorchNeuron, NKI Library, NKI Samples, vLLM Neuron, and Neuron Agentic Development.
* :doc:`News and blogs </about-neuron/news-and-blogs/index>` — Feature announcements, technical deep dives, and customer stories.
* :doc:`FAQ and troubleshooting </about-neuron/faq>` — Common questions and solutions for Neuron SDK issues.
* :doc:`Archived documentation </archive/index>` — Reference material for MXNet Neuron, TensorFlow Neuron, torch-neuron (Inf1), and other legacy components.

.. toctree::
   :maxdepth: 1
   :hidden:

   About Neuron </about-neuron/index>
   Neuron architecture </about-neuron/arch/index>
   What's new </about-neuron/whats-new>
   Announcements </about-neuron/announcements/index>
   News & blogs </about-neuron/news-and-blogs/index>
   Contribute </about-neuron/oss/index>

.. toctree::
    :maxdepth: 1
    :caption: Get started
    :hidden:

    Quickstarts </quickstarts>
    Set up environments </setup/index>
    Deploy model workloads </deploy/index>

.. toctree::
   :maxdepth: 1
   :caption: PyTorch support
   :hidden:

   TorchNeuron </torch/index>

.. toctree::
   :maxdepth: 1
   :caption: Training
   :hidden:

   Training on Neuron </training>
   TorchTitan Neuron </training/torchtitan/index>

.. toctree::
   :maxdepth: 1
   :caption: Model inference
   :hidden:

   Inference on Neuron </inference>
   Native PyTorch inference </inference-native-pytorch>
   vLLM Neuron </vllm-neuron/docs/index>
   vLLM Omni Neuron </vllm-omni-neuron/docs/index>

.. toctree::
   :maxdepth: 1
   :caption: Other frameworks (reference)
   :hidden:

   JAX </jax/index>
   Legacy inference frameworks </inference-legacy/index>
   Legacy training frameworks </training/legacy-frameworks>

.. toctree::
   :maxdepth: 1
   :caption: Use custom kernels
   :hidden:

   Neuron Kernel Interface (NKI) </nki/index>
   NKI API reference </nki/api/index>
   NKI Library </nki/library/index>

.. toctree::
   :maxdepth: 1
   :caption: Developer tools
   :hidden:

   Neuron tools </tools/index>
   Neuron Agentic Development </tools/neuron-agentic-development/index>
   Neuron Explorer </tools/neuron-explorer/index>
   Neuron Observer </tools/neuron-observer/index>

.. toctree::
   :maxdepth: 1
   :caption: Compiler, runtime, & collectives
   :hidden:

   Graph compiler </compiler/index>
   Neuron Runtime </neuron-runtime/index>
   Collectives </neuron-runtime/about/collectives>

.. toctree::
   :maxdepth: 1
   :caption: Archive
   :hidden:

   Archived content </archive/index>

*AWS and the AWS logo are trademarks of Amazon Web Services, Inc. or its affiliates. All rights reserved.*
