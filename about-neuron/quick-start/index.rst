.. meta::
   :description: Get started quickly with AWS Neuron SDK on Inferentia and Trainium — training, inference, vLLM serving, NKI, deployment, and tools quickstarts
   :keywords: neuron, quickstart, getting started, pytorch, jax, vllm, nki, inferentia, trainium, training, inference
   :instance-types: inf2, trn1, trn2, trn3
   :content-type: navigation-hub
   :date-modified: 2026-07-21

.. _neuron-quickstart:

Get Started with AWS Neuron
============================

Get up and running with the AWS Neuron SDK on Inferentia and Trainium. Pick a quickstart by task or by component below.

.. note::

   **First time using AWS Neuron?** These quickstarts assume you have:

   - An active AWS account with EC2 access
   - Basic familiarity with your chosen ML framework (PyTorch or JAX)
   - SSH access to launch and connect to EC2 instances

   For detailed installation instructions, see the :doc:`Setup Guide </setup/index>`.

Choose your path
----------------

Start here if you want an end-to-end first workload.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🚀 Training quickstart
      :link: training-quickstart
      :link-type: ref
      :class-card: sd-border-2

      Train your first model on Trainium.

      **Time**: ~15 minutes

      :bdg-primary:`Trn1` :bdg-primary:`Trn2` :bdg-primary:`Trn3`

   .. grid-item-card:: 🎯 Inference quickstart
      :link: inference-quickstart
      :link-type: ref
      :class-card: sd-border-2

      Run your first inference on Inferentia.

      **Time**: ~10 minutes

      :bdg-success:`Inf2` :bdg-success:`Trn1`

LLM serving and agentic tooling
-------------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 💬 LLM serving with vLLM
      :class-card: sd-border-1

      Deploy large language models for production inference:

      - :doc:`Online serving (OpenAI-compatible API) </vllm-neuron/docs/getting-started/quickstart-online-serving>`
      - :doc:`Offline batch inference </vllm-neuron/docs/getting-started/quickstart-offline-serving>`

      :bdg-info:`Trn2` :bdg-info:`Trn3`

   .. grid-item-card:: 🤖 Amazon AI helper tools
      :link: amazon-q-dev
      :link-type: ref
      :class-card: sd-border-1

      Use AI-powered code assistance for Neuron development — code suggestions, debugging, and optimization.

Quickstarts by component
------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Neuron Kernel Interface (NKI)
      :class-card: sd-border-1

      - :ref:`Get started with NKI <nki-get-started>`
      - :ref:`Implement and run your first kernel <quickstart-run-nki-kernel>`
      - :ref:`Get started with the NKI Library <nki-library-quickstart>`

   .. grid-item-card:: Deployment (DLC and DLAMI)
      :class-card: sd-border-1

      - :ref:`Deploy a vLLM server with a Neuron DLC <quickstart_vllm_dlc_deploy>`
      - :ref:`Get started with Neuron DLC using Docker <containers-getting-started>`
      - :ref:`Get started with the multi-framework DLAMI <setup-multiframework-dlami>`

   .. grid-item-card:: Developer tools
      :class-card: sd-border-1

      - :ref:`Get started with Neuron Explorer <new-neuron-profiler-setup>`
      - :ref:`Get started with Neuron agentic development <neuron-agentic-development-getting-started>`

   .. grid-item-card:: Neuron Runtime
      :class-card: sd-border-1

      - :ref:`Generate a Neuron runtime core dump <runtime-core-dump-quickstart>`
      - :ref:`Getting started with nrtpy <nrtpy-getting-started>`

Framework setup guides
----------------------

Need framework-specific setup instructions?

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: PyTorch
      :link: /setup/pytorch/index
      :link-type: doc
      :class-card: sd-border-1
      :class-body: sphinx-design-class-title-small

      PyTorch 2.9+ setup

   .. grid-item-card:: JAX
      :link: /setup/jax/index
      :link-type: doc
      :class-card: sd-border-1
      :class-body: sphinx-design-class-title-small

      JAX 0.7+ setup

Additional resources
--------------------

- :doc:`/about-neuron/models/index` - Pre-tested model samples and tutorials
- :doc:`/deploy/ec2/index` - Detailed EC2 deployment workflows
- :doc:`/deploy/index` - Use Deep Learning Containers
- :doc:`github-samples` - GitHub sample repositories

.. note::

   Legacy Inf1 quickstarts (PyTorch ``torch-neuron``, MXNet, and TensorFlow) have moved to the :doc:`archive </archive/index>` as those instances and frameworks reach end of support. Use the quickstarts above for Inf2, Trn1, Trn2, and Trn3.

.. toctree::
   :hidden:
   :maxdepth: 1
   
   training-quickstart
   inference-quickstart
   /libraries/nxd-inference/vllm/quickstart-vllm-online-serving
   /libraries/nxd-inference/vllm/quickstart-vllm-offline-serving
   /about-neuron/amazonq-getstarted
   github-samples

   