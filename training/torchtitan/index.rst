.. meta::
   :description: TorchTitan Neuron. Landing page for running TorchTitan pre-training on Trainium with TorchNeuron.
   :keywords: TorchTitan, TorchNeuron, AWS Neuron, Trainium, pre-training, FSDP, Llama, DeepSeek, GPT-OSS
   :date-modified: 2026-04-28

.. _torchtitan-neuron-landing:

==========================
TorchTitan Neuron
==========================

.. admonition:: Beta Release
   :class: note

   TorchTitan Neuron is in beta for the Neuron 2.33 release. It is supported on Trainium 2 (trn2)
   and Trainium 3 (trn3) instances. We welcome your feedback.

TorchTitan is a PyTorch-native pre-training library. On AWS Neuron, TorchTitan
runs on top of **TorchNeuron** (the native PyTorch backend) — not on the
legacy PyTorch/XLA path.

This section covers how to set up TorchTitan on Trainium, run common
pre-training workloads, and tune performance.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: /training/torchtitan/about-torchtitan-neuron
      :link-type: doc
      :class-card: sd-border-2

      **About TorchTitan Neuron**
      :bdg-info:`overview`
      ^^^
      What TorchTitan adds on Trainium, supported models, and how it fits with
      TorchNeuron.

   .. grid-item-card::
      :link: /training/torchtitan/get-started/index
      :link-type: doc
      :class-card: sd-border-2

      **Get started**
      :bdg-primary:`quickstart`
      ^^^
      Set up TorchTitan on a Neuron instance.

   .. grid-item-card::
      :link: /training/torchtitan/tutorials/index
      :link-type: doc
      :class-card: sd-border-2

      **Tutorials**
      :bdg-primary:`tutorial`
      ^^^
      Basic training, FSDP with Llama and DeepSeek, GPT-OSS.

   .. grid-item-card::
      :link: /training/torchtitan/best-practices
      :link-type: doc
      :class-card: sd-border-2

      **Best practices**
      :bdg-info:`best practice`
      ^^^
      Enablement checklist and performance-tuning guidance.

   .. grid-item-card::
      :link: /training/torchtitan/troubleshooting
      :link-type: doc
      :class-card: sd-border-2

      **Troubleshooting**
      :bdg-info:`troubleshooting`
      ^^^
      Common errors and symptoms on TorchTitan Neuron runs.

.. toctree::
   :maxdepth: 1
   :hidden:

   About TorchTitan Neuron <about-torchtitan-neuron>
   Get started <get-started/index>
   Tutorials <tutorials/index>
   Best practices <best-practices>
   Troubleshooting <troubleshooting>
