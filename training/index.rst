.. meta::
   :description: Training on AWS Neuron with PyTorch — the native PyTorch (TorchNeuron) path for new work and the legacy PyTorch/XLA (torch-neuronx) path for existing workloads.
   :keywords: AWS Neuron, Trainium, training, TorchNeuron, torch-neuronx, PyTorch, PyTorch native, PyTorch XLA, FSDP, DTensor, DDP, NxD, NeuronX Distributed, TorchTitan
   :date-modified: 2026-10-08

.. _training-section-index:

=======================================
Training paths (native and legacy XLA)
=======================================

Train models on AWS Trainium with PyTorch. This section splits by the PyTorch
path you build on: native PyTorch (TorchNeuron), or the legacy PyTorch/XLA path
on the older ``torch-neuronx`` XLA library. Use TorchNeuron for new work. The
XLA path stays documented for workloads already running on it.

.. warning::

   TorchNeuron and the legacy ``torch-neuronx`` (XLA) library share the
   ``torch-neuronx`` package name but are different products with different APIs
   and version lineages. Pick one path and stay on it; do not mix their APIs in
   a single workload.

----

Native PyTorch training (TorchNeuron)
=====================================

TorchNeuron is the Neuron native PyTorch backend. It runs eager mode,
``torch.compile`` with the ``neuron`` backend, and the standard distributed
APIs: FSDP, DTensor, DDP, and functional collectives. Most PyTorch training
code runs on Trainium after you change device placement from ``cuda`` to
``neuron``.

See :doc:`Training best practices </training/best-practices>` for choosing
FSDP, tensor parallelism, pipeline parallelism, and meshes on Trainium.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: /torch/index
      :link-type: doc
      :class-card: sd-border-2

      **TorchNeuron (native PyTorch on Neuron)**
      :bdg-info:`overview`
      ^^^
      Section landing: overview, quickstarts, tutorials, guides, and reference.

   .. grid-item-card::
      :link: /torch/tutorials/index
      :link-type: doc
      :class-card: sd-border-2

      **TorchNeuron training tutorials**
      :bdg-primary:`tutorial`
      ^^^
      Worked examples, from a single training loop to distributed training with
      ``torch.distributed``.

   .. grid-item-card::
      :link: /torch/guides/torch-distributed
      :link-type: doc
      :class-card: sd-border-2

      **Distributed training how-to**
      :bdg-primary:`how-to`
      ^^^
      Use FSDP, DTensor, DDP, and functional collectives on Neuron.


Supported Training Libraries
============================

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: /training/torchtitan/index
      :link-type: doc
      :class-card: sd-border-2

      **TorchTitan Neuron**
      :bdg-info:`overview`
      ^^^
      Pre-train on Trainium with TorchTitan, including Llama, DeepSeek (FSDP),
      and GPT-OSS.

----

Legacy PyTorch/XLA training (``torch-neuronx`` XLA)
===================================================

.. note::

   The ``torch-neuronx`` library referenced here is the older XLA-based
   product, not TorchNeuron. It is in maintenance status for customers with
   existing XLA workloads. For new training, use the native PyTorch path above.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: /archive/nxd-training/index
      :link-type: doc
      :class-card: sd-border-2

      **Training (torch-neuronx XLA)**
      :bdg-info:`overview`
      ^^^
      Entry point for the legacy XLA-based training docs on Trainium.

   .. grid-item-card::
      :link: /archive/nxd-training/index
      :link-type: doc
      :class-card: sd-border-2

      **NeuronX Distributed Training (NxDT)**
      :bdg-info:`overview`
      ^^^
      The XLA-era distributed training library, now in maintenance status and
      archived.

   .. grid-item-card::
      :link: /archive/libraries/neuronx-distributed/index-training
      :link-type: doc
      :class-card: sd-border-2

      **NeuronX Distributed Core (NxD Core)**
      :bdg-info:`overview`
      ^^^
      Distributed-computing primitives the XLA training path builds on.
