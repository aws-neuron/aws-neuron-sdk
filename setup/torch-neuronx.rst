.. _setup-torch-neuronx:

.. meta::
   :description: Install PyTorch NeuronX (torch-neuronx) on AWS Trainium and Inferentia instances using DLAMI, DLC, or manual pip installation
   :keywords: pytorch, neuron, torch-neuronx, installation, setup, trainium, inferentia, trn1, trn2, trn3, inf2, DLAMI, pip
   :date-modified: 2026-03-30

PyTorch Neuron (``torch-neuronx``) Setup 
========================================

Install PyTorch with Neuron support for training and inference on Inf2, Trn1, Trn2, and Trn3 instances. Choose from a pre-configured DLAMI, a Docker container, or a manual pip installation.

For the full setup guide with all options, see :doc:`Install PyTorch for Neuron </setup/pytorch/index>`.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 🚀 DLAMI Installation
      :link: /setup/pytorch/dlami
      :link-type: doc
      :class-card: sd-border-2

      Pre-configured environment with all dependencies. Recommended for most users.

   .. grid-item-card:: 🚀 Multi-Framework DLAMI
      :link: /setup/multiframework-dlami
      :link-type: doc
      :class-card: sd-border-2

      Pre-configured AMI with PyTorch, JAX, and vLLM virtual environments ready to use.

   .. grid-item-card:: � Deep Learning Container
      :link: /setup/pytorch/dlc
      :link-type: doc
      :class-card: sd-border-2

      Pre-configured Docker images from AWS ECR for containerized deployments.

   .. grid-item-card:: 🔧 Manual Installation
      :link: /setup/pytorch/manual
      :link-type: doc
      :class-card: sd-border-2

      Install on bare OS AMIs or existing systems with full control over dependencies.
