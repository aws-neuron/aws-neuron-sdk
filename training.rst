.. meta::
   :description: Train models on AWS Neuron with native PyTorch (TorchNeuron), TorchTitan Neuron, and Hugging Face training libraries on Trainium.
   :keywords: AWS Neuron, Trainium, training, TorchNeuron, TorchTitan, Hugging Face, FSDP, distributed training, NKI
   :date-modified: 2026-09-01

.. _training-home:

======================
Train models on Neuron
======================

Train and fine-tune models on AWS Trainium. This page organizes training by path —
native PyTorch (TorchNeuron), TorchTitan Neuron, and Hugging Face — so you can start
from your task instead of choosing a library first.

On Neuron, you train and fine-tune models in PyTorch. You can serve supported LLM checkpoints
with vLLM Neuron. To serve a trained model, see :ref:`inference-home`.

Choose your training path
=========================

Neuron supports three training paths on Trainium:

- **Native PyTorch (TorchNeuron)** — the recommended path for new work. TorchNeuron is
  the Neuron native PyTorch backend; it supports eager mode, ``torch.compile`` with the
  ``neuron`` backend, and standard distributed APIs (FSDP, DTensor, DDP, functional
  collectives). Most PyTorch code runs with minimal changes, primarily moving device
  placement from ``cuda`` to ``neuron``.
- **TorchTitan Neuron** — for large-scale pre-training built on TorchNeuron. Use
  TorchTitan when you want a distributed training stack with recipes
  for models such as Llama, DeepSeek (FSDP), and GPT-OSS.
- **Hugging Face libraries** — for fine-tuning and training with Hugging Face
  ``transformers``, ``diffusers``, PEFT, and ``accelerate``, which run natively on
  TorchNeuron with no Neuron-specific path.

Learn the concepts
==================

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: About TorchNeuron
      :link: /torch/about-pytorch-neuron
      :link-type: doc
      :class-card: sd-border-2

      :bdg-info:`overview`
      ^^^
      Understand TorchNeuron's execution modes and how it maps to the PyTorch
      ecosystem.

   .. grid-item-card:: TorchNeuron landing
      :link: torch-neuron-home
      :link-type: ref
      :class-card: sd-border-2

      :bdg-info:`overview`
      ^^^
      The native PyTorch section home: quickstarts, tutorials, guides, and reference.

   .. grid-item-card:: torch.compile on Neuron
      :link: /torch/compile/index
      :link-type: doc
      :class-card: sd-border-2

      :bdg-info:`deep dive`
      ^^^
      How JIT compilation with ``backend="neuron"`` works, including the compilation
      pipeline and caching.

   .. grid-item-card:: TorchTitan Neuron
      :link: /training/torchtitan/index
      :link-type: doc
      :class-card: sd-border-2

      :bdg-info:`overview`
      ^^^
      Large-scale pre-training on Trainium with TorchNeuron and TorchTitan.

   .. grid-item-card:: Training paths (native and legacy XLA)
      :link: training-section-index
      :link-type: ref
      :class-card: sd-border-2

      :bdg-info:`overview`
      ^^^
      Compare the native PyTorch path with the legacy PyTorch/XLA (``torch-neuronx``
      XLA) path.

Start training
==============

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Get started with TorchNeuron
      :link: /torch/get-started/index
      :link-type: doc
      :class-card: sd-border-1

      :bdg-primary:`quickstart`
      ^^^
      Launch a Trainium instance and run your first TorchNeuron training check.

   .. grid-item-card:: TorchNeuron tutorials
      :link: /torch/tutorials/index
      :link-type: doc
      :class-card: sd-border-1

      :bdg-primary:`tutorial`
      ^^^
      End-to-end walk-throughs: a simple training loop and distributed training with
      ``torch.distributed``.

   .. grid-item-card:: Distributed training how-to
      :link: /torch/guides/torch-distributed
      :link-type: doc
      :class-card: sd-border-1

      :bdg-primary:`how-to`
      ^^^
      Use FSDP, DTensor, DDP, and functional collectives on Neuron.

   .. grid-item-card:: TorchTitan tutorials
      :link: /training/torchtitan/tutorials/index
      :link-type: doc
      :class-card: sd-border-1

      :bdg-primary:`tutorial`
      ^^^
      Pre-train Llama, DeepSeek (FSDP), and GPT-OSS on Trainium with TorchTitan.

   .. grid-item-card:: Training best practices
      :link: /training/best-practices
      :link-type: doc
      :class-card: sd-border-1

      :bdg-primary:`how-to`
      ^^^
      Guidance for training performance on Trainium.

.. admonition:: Where NKI fits in training
   :class: note

   Custom kernels written with the :doc:`Neuron Kernel Interface (NKI) </nki/index>`
   plug into the training path as PyTorch custom operators — called like any other
   operator in eager mode and under ``torch.compile``. Reach for NKI when a performance-critical path
   needs finer control than the compiler provides; start from the
   :doc:`NKI Library </nki/library/index>` of pre-optimized kernels.

Related
=======

- :ref:`TorchNeuron <torch-neuron-home>` — the top-level
  TorchNeuron section, with quickstarts, tutorials, guides, and reference.
- :ref:`Deploy models on Neuron <inference-home>` — serving trained models.

.. toctree::
   :maxdepth: 1
   :hidden:

   Training paths (native and legacy XLA) <training/index>
   Best practices for training <training/best-practices>
