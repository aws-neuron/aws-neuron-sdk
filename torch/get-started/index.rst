.. meta::
   :description: Get started with TorchNeuron, the native PyTorch backend for AWS Trainium. Quickstart guides for eager mode and torch.compile.
   :keywords: TorchNeuron, quickstart, get started, PyTorch, eager mode, torch.compile, AWS Neuron, Trainium
   :date-modified: 2026-08-27

.. _torch-get-started:

Get started with TorchNeuron
=============================

Use these quickstart guides to get your first PyTorch model running on AWS Trainium with TorchNeuron.
Install and verify your environment, then build and train your first model.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: /torch/get-started/quickstart-torch-neuron
      :link-type: doc
      :class-card: sd-border-2

      **Install and verify torch-neuronx**
      :bdg-primary:`quickstart`
      ^^^
      Set up torch-neuronx on a Trainium instance and confirm your environment is ready
      to run models.

   .. grid-item-card::
      :link: /torch/get-started/quickstart-build-first-model
      :link-type: doc
      :class-card: sd-border-2

      **Build your first model with torch-neuronx**
      :bdg-primary:`quickstart`
      ^^^
      Choose a model, build it, and train it end-to-end on Trainium using TorchNeuron's native
      PyTorch backend.

   .. grid-item-card::
      :link: /torch/get-started/quickstart-nki-kernel
      :link-type: doc
      :class-card: sd-border-2

      **Run an NKI kernel from torch-neuronx**
      :bdg-primary:`quickstart`
      ^^^
      Call a custom NKI kernel from a PyTorch model on Trainium using ``@nki_op``.

.. toctree::
   :maxdepth: 1
   :hidden:

   Install and verify torch-neuronx </torch/get-started/quickstart-torch-neuron>
   Build your first model </torch/get-started/quickstart-build-first-model>
   Run an NKI kernel from torch-neuronx </torch/get-started/quickstart-nki-kernel>
