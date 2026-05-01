.. meta::
   :description: AWS Neuron SDK enables high-performance deep learning and generative AI on AWS Inferentia and Trainium instances. Get started with native PyTorch, vLLM, NKI kernels, and AI-driven optimization with Neuron Explorer.
   :date-modified: 2026-05-01

.. _neuron_home:

AWS Neuron Documentation
========================

:ref:`AWS Neuron <what-is-neuron>` is the software development kit for deep learning and generative AI on `AWS Inferentia <https://aws.amazon.com/ai/machine-learning/inferentia/>`_ and `AWS Trainium <https://aws.amazon.com/ai/machine-learning/trainium/>`_ instances. Neuron supports multiple development paths: serving large language models with vLLM, training and inference with PyTorch and JAX, authoring custom kernels with NKI, and direct use of the Neuron Graph Compiler and Runtime.

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: :octicon:`tag;1em;sd-text-primary` Current release: Neuron 2.29.1
      :link: /release-notes/2.29.1
      :link-type: doc
      :class-card: sd-border-2
      :class-title: sd-fs-6

      **Released May 1, 2026**. Select this card for the details!
      ^^^
      Patch to Neuron 2.29.0 to address two Neuron Explorer issues.

----

Who Neuron is for
-----------------

* **ML engineers deploying production models** — Deploy prepared :doc:`Neuron Deep Learning AMIs (DLAMIs) </dlami/index>` and :doc:`Deep Learning Containers (DLCs) </containers/index>` on Amazon EC2 Trainium and Inferentia instances. Start with the :doc:`DLAMI setup guide </dlami/index>` or the :doc:`DLC quickstart </containers/get-started/quickstart-configure-deploy-dlc>`.

  * **Serving LLMs** — Use vLLM on Neuron to serve open-source LLMs with minimal code changes. Start with the :doc:`online serving quickstart </libraries/nxd-inference/vllm/quickstart-vllm-online-serving>` or :doc:`offline serving quickstart </libraries/nxd-inference/vllm/quickstart-vllm-offline-serving>`.
* **ML researchers and model developers** — Use native PyTorch on Trainium with eager mode, ``torch.compile``, and standard distributed APIs. Start with :doc:`Native PyTorch on Neuron </frameworks/torch/pytorch-native-overview>`.
* **Performance engineers optimizing kernels** — Use NKI to write custom kernels with direct NeuronCore access, or pick from the NKI Library's pre-optimized kernels. Start with the :doc:`NKI quickstart </nki/get-started/quickstart-implement-run-kernel>` and :doc:`NKI Library </nki/library/index>`.

----

Start here
----------

Pick the task that matches what you want to do.

.. grid:: 1 1 1 1
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1em;sd-text-primary` Get started with a Neuron DLAMI and PyTorch
      :link: /dlami/index
      :link-type: doc
      :class-card: sd-border-1

      Launch a Trainium or Inferentia EC2 instance with a pre-configured **Neuron Deep Learning AMI (DLAMI)** and PyTorch. The DLAMI bundles the Neuron SDK, framework virtual environments (PyTorch, JAX, vLLM), and the system tools — no manual install required. See :doc:`Install PyTorch via Deep Learning AMI </setup/pytorch/dlami>` and get started!

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`server;1em;sd-text-primary` Serve a large language model
      :link: /libraries/nxd-inference/vllm/index
      :link-type: doc
      :class-card: sd-border-1

      Run LLM inference on Trainium and Inferentia with **vLLM on Neuron**. Supports OpenAI-compatible APIs, continuous batching, and speculative decoding. See the :doc:`offline </libraries/nxd-inference/vllm/quickstart-vllm-offline-serving>` or :doc:`online </libraries/nxd-inference/vllm/quickstart-vllm-online-serving>` serving quickstart.

   .. grid-item-card:: :octicon:`graph;1em;sd-text-primary` Train a model with PyTorch
      :link: /frameworks/torch/pytorch-native-overview
      :link-type: doc
      :class-card: sd-border-1

      Use **native PyTorch on Trainium** (TorchNeuron) with eager mode, ``torch.compile``, and the standard distributed APIs (FSDP, DTensor, DDP). Existing PyTorch code runs with minimal changes; primarily swap ``cuda`` for ``neuron`` on your tensors.

   .. grid-item-card:: :octicon:`code;1em;sd-text-primary` Write custom NKI kernels
      :link: /nki/get-started/quickstart-implement-run-kernel
      :link-type: doc
      :class-card: sd-border-1

      Program NeuronCores directly with **NKI** when you need finer control than framework-level compilation provides. NKI offers tile-level programming with Python and NumPy-like syntax, and ships with a library of pre-optimized kernels (attention, MoE, and others).

----

Neuron SDK Organization
-----------------------

The Neuron SDK includes:

* **Frameworks** — Native PyTorch on Trainium (TorchNeuron), PyTorch NeuronX (``torch-neuronx``), and JAX NeuronX.
* **Serving integrations** — vLLM on Neuron V1 (via the ``vllm-neuron`` plugin) and the earlier vLLM integration through NxD Inference, both for OpenAI-compatible LLM serving.
* **NeuronX Distributed (NxD) libraries** — PyTorch libraries for distributed training and inference, including NxD Training, NxD Inference, and NxD Core.
* **Neuron Kernel Interface (NKI)** — Python programming interface for custom kernels on NeuronCores, plus the NKI Library of pre-optimized kernels.
* **Neuron Graph Compiler** (``neuronx-cc``) — Compiles model graphs and NKI kernels into Neuron Executable File Format (NEFF) files.
* **Neuron Runtime** — Loads NEFFs and executes them on NeuronCores, handling device allocation, memory management, and collective communications.
* **Developer tools** — Neuron Explorer and the Neuron system tools for profiling and debugging across every component.

.. grid:: 1
   :gutter: 2

   .. grid-item-card::
      :class-card: sd-border-1

      **Frameworks and serving**
      ^^^
      Write training and inference code with PyTorch or JAX. Serve LLMs with vLLM on Neuron.

      * :doc:`Native PyTorch on Neuron </frameworks/torch/pytorch-native-overview>`
      * :doc:`PyTorch NeuronX (torch-neuronx) </frameworks/torch/index>` · :doc:`JAX NeuronX </frameworks/jax/index>`
      * :doc:`vLLM on Neuron </libraries/nxd-inference/vllm/index>`

   .. grid-item-card::
      :class-card: sd-border-1

      **NKI — Neuron Kernel Interface**
      ^^^
      Programming interface for custom kernels on NeuronCores. Used by the modern framework and serving integrations. Ships with a library of pre-optimized kernels.

      * :doc:`Get started with NKI </nki/get-started/index>` · :doc:`Language guide </nki/get-started/nki-language-guide>`
      * :doc:`NKI Library </nki/library/index>` · :doc:`NKI API reference </nki/api/index>`
      * :doc:`NKI tutorials </nki/guides/tutorials/index>` · :doc:`NKI deep dives </nki/deep-dives/index>`

   .. grid-item-card::
      :class-card: sd-border-1

      **NeuronX Distributed (NxD) libraries**
      ^^^
      PyTorch libraries for distributed training and inference on Neuron. Provide reference model implementations, sharding strategies (tensor, expert, context, pipeline parallelism), and distributed checkpointing. NxD Inference integrates selected NKI kernels for performance-critical operations.

      * :doc:`NxD Training </libraries/nxd-training/index>` · :doc:`NxD Inference </libraries/nxd-inference/index>`
      * :doc:`NxD Core (Training) </libraries/neuronx-distributed/index-training>` · :doc:`NxD Core (Inference) </libraries/neuronx-distributed/index-inference>`

   .. grid-item-card::
      :class-card: sd-border-1

      **Neuron Graph Compiler and Runtime**
      ^^^
      The compiler (``neuronx-cc``) transforms model graphs into NEFF files. The runtime loads NEFFs and executes them on NeuronCores, handling device allocation, memory management, and collective communications. Both framework graphs and NKI kernels compile to NEFF.

      * :doc:`Neuron Graph Compiler </compiler/index>` · :doc:`Compiler error codes </compiler/error-codes/index>`
      * :doc:`Neuron Runtime </neuron-runtime/index>` · :doc:`Collectives </neuron-runtime/about/collectives>`
      * :doc:`Neuron C++ Custom Operators </neuron-customops/index>`

----

Deployment and Tools Support
----------------------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Neuron Explorer
      :link: /tools/neuron-explorer/index
      :link-type: doc
      :class-card: sd-border-1

      Profiling and optimization tool with support for framework, NKI, compiler, and runtime workloads. Covers every Neuron SDK component area.

   .. grid-item-card:: Neuron Agentic Development
      :link: /about-neuron/agentic-development-overview
      :link-type: doc
      :class-card: sd-border-1

      Open-source AI agents and skills for NKI kernel authoring, debugging, profiling, and analysis. Runs inside Claude Code, Kiro, and other agentic IDEs.

   .. grid-item-card:: Deploy on AWS
      :link: /devflows/index
      :link-type: doc
      :class-card: sd-border-1

      Pre-configured :doc:`DLAMIs </dlami/index>` and :doc:`DLCs </containers/index>` for EC2, EKS, ECS, SageMaker, and ParallelCluster.

----

Learn more
----------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: What is AWS Neuron?
      :link: /about-neuron/what-is-neuron
      :link-type: doc
      :class-card: sd-border-1

      Background on Inferentia, Trainium, and the Neuron SDK.

   .. grid-item-card:: Release notes
      :link: /release-notes/index
      :link-type: doc
      :class-card: sd-border-1

      Component-by-component release notes for every Neuron SDK version.

   .. grid-item-card:: Open source and contribute
      :link: /about-neuron/oss/index
      :link-type: doc
      :class-card: sd-border-1

      Public GitHub repositories, contribution guidelines, and source for TorchNeuron, NKI Library, NKI Samples, vLLM Neuron, and Neuron Agentic Development.

   .. grid-item-card:: News and blogs
      :link: /about-neuron/news-and-blogs/index
      :link-type: doc
      :class-card: sd-border-1

      Feature announcements, technical deep dives, and customer stories.

   .. grid-item-card:: FAQ and troubleshooting
      :link: /about-neuron/faq/index
      :link-type: doc
      :class-card: sd-border-1

      Common questions and solutions for Neuron SDK issues.

   .. grid-item-card:: Archived documentation
      :link: /archive/index
      :link-type: doc
      :class-card: sd-border-1

      Reference material for MXNet Neuron, TensorFlow Neuron, torch-neuron (Inf1), and other legacy components.

.. toctree::
   :maxdepth: 1
   :hidden:

   About Neuron </about-neuron/index>
   Neuron Architecture </about-neuron/arch/index>
   What's New </about-neuron/whats-new>
   Announcements </about-neuron/announcements/index>
   News & Blogs </about-neuron/news-and-blogs/index>
   Contribute </about-neuron/oss/index>

.. toctree::
    :maxdepth: 1
    :caption: Get Started
    :hidden:

    Quickstarts </about-neuron/quick-start/index>
    Setup Guides </setup/index>
    Developer Flows </devflows/index>

.. toctree::
   :maxdepth: 1
   :caption: ML Frameworks
   :hidden:

   Home </frameworks/index>
   PyTorch </frameworks/torch/index>
   JAX </frameworks/jax/index>

.. toctree::
   :maxdepth: 1
   :caption: Training
   :hidden:

   NxD Training </libraries/nxd-training/index>
   NxD Core (Training) </libraries/neuronx-distributed/index-training>

.. toctree::
   :maxdepth: 1
   :caption: Inference
   :hidden:

   Overview </libraries/nxd-inference/neuron-inference-overview>
   vLLM </libraries/nxd-inference/vllm/index>
   NxD Inference </libraries/nxd-inference/index>
   NxD Core (Inference) </libraries/neuronx-distributed/index-inference>

.. toctree::
   :maxdepth: 1
   :caption: Developer Tools
   :hidden:

   Home </tools/index>
   Neuron Explorer </tools/neuron-explorer/index>
   Neuron Agentic Development </about-neuron/agentic-development-overview>

.. toctree::
   :maxdepth: 1
   :caption: Orchestrate and Deploy
   :hidden:

   AWS Workload Orchestration </devflows/index>
   Neuron DLAMI </dlami/index>
   Neuron Containers </containers/index>

.. toctree::
   :maxdepth: 1
   :caption: Runtime & Collectives
   :hidden:

   Neuron Runtime </neuron-runtime/index>
   Collectives </neuron-runtime/about/collectives>
   Neuron C++ Custom Operators </neuron-customops/index>

.. toctree::
   :maxdepth: 1
   :caption: Compilers
   :hidden:

   Graph Compiler </compiler/index>
   Compiler Error Codes </compiler/error-codes/index>

.. toctree::
   :maxdepth: 1
   :caption: Neuron Kernel Interface (NKI)
   :hidden:

   Home </nki/index>
   Get Started </nki/get-started/index>
   Guides </nki/guides/index>
   Deep Dives </nki/deep-dives/index>
   Migration Guides </nki/migration/index>
   NKI API Reference </nki/api/index>
   NKI Library </nki/library/index>

.. toctree::
   :maxdepth: 1
   :caption: Archive
   :hidden:

   Archived content </archive/index>

*AWS and the AWS logo are trademarks of Amazon Web Services, Inc. or its affiliates. All rights reserved.*
