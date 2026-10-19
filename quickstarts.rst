.. meta::
   :description: AWS Neuron SDK quickstarts grouped by task — train with TorchNeuron and TorchTitan, serve LLMs with vLLM Neuron, generate video with vLLM Omni Neuron, write NKI kernels, profile with Neuron Explorer, and monitor clusters with Neuron Observer on Trainium.
   :keywords: neuron, quickstart, getting started, TorchNeuron, torch-neuronx, TorchTitan, vLLM Neuron, vLLM Omni Neuron, NKI, Neuron Explorer, Neuron Observer, trainium, trn2, trn3
   :instance-types: trn2, trn3
   :content-type: navigation-hub
   :date-modified: 2026-10-13

.. _neuron-quickstart:

Get started with AWS Neuron
===========================

Use these quickstarts to run your first workload with the AWS Neuron SDK. Each group matches a task. Pick the task you want to do, then start with the first quickstart in that group.

.. note::

   **First time using AWS Neuron?** These quickstarts assume you have:

   - An active AWS account with EC2 access
   - Basic familiarity with PyTorch
   - SSH access to launch and connect to EC2 instances

   Most of these quickstarts run on Trn2 or Trn3 instances. For installation instructions, see the :doc:`Setup Guide </setup/index>`.

Train a model
-------------

Use TorchNeuron, the native PyTorch backend for Trainium, to train with eager mode and ``torch.compile``. Use TorchTitan on Neuron for large-scale pre-training.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Install and verify TorchNeuron
      :link: torchneuron-quickstart-install-verify
      :link-type: ref
      :class-card: sd-border-2

      Install ``torch-neuronx`` and confirm that PyTorch can use the Neuron device.

      **Time**: ~10–15 minutes

      :bdg-primary:`Trn2` :bdg-primary:`Trn3`

   .. grid-item-card:: Build your first model with TorchNeuron
      :link: torchneuron-quickstart-build-first-model
      :link-type: ref
      :class-card: sd-border-2

      Run a PyTorch model in eager mode, with ``torch.compile``, and in a short training loop.

      **Time**: ~10 minutes

      :bdg-primary:`Trn2` :bdg-primary:`Trn3`

   .. grid-item-card:: Set up TorchTitan Neuron
      :link: torchtitan-neuron-setup
      :link-type: ref
      :class-card: sd-border-1

      Install and configure TorchTitan for pre-training on Trainium with TorchNeuron.

      :bdg-primary:`Trn2` :bdg-primary:`Trn3`

   .. grid-item-card:: Run a basic TorchTitan training
      :link: torchtitan-neuron-tutorial-basic
      :link-type: ref
      :class-card: sd-border-1

      Complete a first pre-training run with TorchTitan on Trainium.

      :bdg-primary:`Trn2` :bdg-primary:`Trn3`

Serve LLMs
----------

Use the vLLM Neuron plugin to serve large language models with the standard ``vllm serve`` command and an OpenAI-compatible API.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Online serving with vLLM Neuron
      :link: /vllm-neuron/docs/getting-started/quickstart-online-serving
      :link-type: doc
      :class-card: sd-border-2

      Launch an OpenAI-compatible API server and send your first request.

      :bdg-info:`Trn2` :bdg-info:`Trn3`

   .. grid-item-card:: Offline batch inference with vLLM Neuron
      :link: /vllm-neuron/docs/getting-started/quickstart-offline-serving
      :link-type: doc
      :class-card: sd-border-2

      Run high-throughput batch inference without a server.

      :bdg-info:`Trn2` :bdg-info:`Trn3`

   .. grid-item-card:: Deploy a vLLM server with a Neuron DLC
      :link: quickstart_vllm_dlc_deploy
      :link-type: ref
      :class-card: sd-border-1

      Start a vLLM server from a prebuilt Neuron Deep Learning Container.

Generate images and video
-------------------------

Use the vLLM Omni Neuron plugin to run diffusion models, such as Wan2.2 text-to-video, on Trainium.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Offline video generation with Wan2.2
      :link: /vllm-omni-neuron/docs/getting-started/quickstart-offline-serving-wan22
      :link-type: doc
      :class-card: sd-border-2

      Generate a video clip from a text prompt with Wan2.2-T2V-A14B.

      :bdg-info:`Trn2` :bdg-info:`Trn3`

   .. grid-item-card:: Online video generation with Wan2.2
      :link: /vllm-omni-neuron/docs/getting-started/quickstart-online-serving-wan22
      :link-type: doc
      :class-card: sd-border-2

      Serve Wan2.2-T2V-A14B with the vLLM Omni OpenAI-compatible video API.

      :bdg-info:`Trn2` :bdg-info:`Trn3`

Write custom kernels
--------------------

Use the Neuron Kernel Interface (NKI) to write kernels with direct access to the NeuronCore, or use pre-optimized kernels from the NKI Library.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Implement and run your first NKI kernel
      :link: quickstart-run-nki-kernel
      :link-type: ref
      :class-card: sd-border-2

      Write a simple NKI kernel and run it on a NeuronCore.

      **Time**: ~10 minutes

   .. grid-item-card:: Run an NKI kernel from TorchNeuron
      :link: torchneuron-quickstart-nki-kernel
      :link-type: ref
      :class-card: sd-border-2

      Call an NKI kernel from a PyTorch model with ``nki_op``, in eager mode and with ``torch.compile``.

      **Time**: ~15 minutes

      :bdg-primary:`Trn2` :bdg-primary:`Trn3`

   .. grid-item-card:: Use the NKI Library
      :link: nki-library-quickstart
      :link-type: ref
      :class-card: sd-border-1

      Add pre-optimized NKI Library kernels to your model code.

      **Time**: ~10 minutes

Profile and optimize
--------------------

Use Neuron Explorer to capture profiles and find performance bottlenecks.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Get started with Neuron Explorer
      :link: new-neuron-profiler-setup
      :link-type: ref
      :class-card: sd-border-2

      Set up Neuron Explorer, capture a profile, and view it in the web UI.

Monitor cluster health
----------------------

Use Neuron Observer to detect hardware faults and find the node and device that cause them.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Get started with Neuron Observer
      :link: neuron-observer-get-started
      :link-type: ref
      :class-card: sd-border-2

      Install Neuron Observer on Amazon EKS, verify it with the ``nobs`` CLI, and view metrics.

      **Time**: ~25 minutes

      :bdg-info:`Trn2` :bdg-info:`Trn3`

More quickstarts
----------------

- :ref:`Get started with Neuron agentic development <neuron-agentic-development-getting-started>` - Use AI agents and skills for NKI kernel and model porting work.
- :ref:`Get started with nrtpy <nrtpy-getting-started>` - Load and run compiled NEFFs directly with the Python runtime layer.
- :ref:`Get started with the multi-framework DLAMI <setup-multiframework-dlami>` - Use the preconfigured AMI with PyTorch, JAX, and vLLM virtual environments.
- :ref:`Get started with Neuron DLCs using Docker <containers-getting-started>` - Build and run Neuron containers.

Framework setup guides
----------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: PyTorch
      :link: /setup/pytorch/index
      :link-type: doc
      :class-card: sd-border-1
      :class-body: sphinx-design-class-title-small

      Set up TorchNeuron with a DLAMI, a DLC, or a manual installation.

   .. grid-item-card:: JAX
      :link: /setup/jax/index
      :link-type: doc
      :class-card: sd-border-1
      :class-body: sphinx-design-class-title-small

      Set up JAX 0.7+ with a DLAMI, a DLC, or a manual installation.

Related information
-------------------

- :doc:`/training` - Training on Neuron with TorchNeuron and TorchTitan
- :doc:`/inference` - Inference on Neuron with vLLM Neuron, vLLM Omni Neuron, and native PyTorch
- :doc:`/vllm-neuron/docs/model-recipes/index` - Supported models and features for LLM serving with vLLM Neuron
- :doc:`/deploy/index` - Deployment on EC2, EKS, and other AWS services

.. note::

   Quickstarts for legacy stacks, such as NxD Inference, PyTorch/XLA, and the Inf1 frameworks, are in the :doc:`legacy inference </inference-legacy/index>` and :doc:`archive </archive/index>` sections.
