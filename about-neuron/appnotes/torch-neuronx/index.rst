.. _torch-neuronx-appnotes:

PyTorch NeuronX Application Notes
==================================

.. toctree::
   :maxdepth: 1
   :hidden:

   introducing-pytorch-2-6
   introducing-pytorch-2-7
   introducing-pytorch-2-8
   introducing-pytorch-2-9
   introducing-pytorch-2-x
   migration-from-xla-downcast-bf16
   torch-neuronx-dataparallel-app-note
   torch-neuronx-graph-partitioner-app-note

This section contains application notes specific to PyTorch NeuronX (``torch-neuronx``) for ``Trn1`` and ``Inf2`` instances. These guides cover PyTorch version migrations, advanced features, optimization techniques, and best practices for training and inference on AWS Trainium and Inferentia2.

PyTorch Version Support
-----------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: introducing-pytorch-2-9
      :link-type: doc

      **PyTorch 2.9 Support**
      ^^^
      New features and migration guide for PyTorch 2.9 on Neuron

   .. grid-item-card::
      :link: introducing-pytorch-2-8
      :link-type: doc

      **PyTorch 2.8 Support**
      ^^^
      New features and migration guide for PyTorch 2.8 on Neuron

   .. grid-item-card::
      :link: introducing-pytorch-2-7
      :link-type: doc

      **PyTorch 2.7 Support**
      ^^^
      Features and improvements introduced with PyTorch 2.7 support

   .. grid-item-card::
      :link: introducing-pytorch-2-x
      :link-type: doc

      **PyTorch 2.x Overview**
      ^^^
      General guide to PyTorch 2.x series support and features

Advanced Features
-----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: torch-neuronx-graph-partitioner-app-note
      :link-type: doc

      **Graph Partitioner**
      ^^^
      Advanced graph partitioning strategies for distributed training and inference

   .. grid-item-card::
      :link: torch-neuronx-dataparallel-app-note
      :link-type: doc

      **Data Parallel Inference**
      ^^^
      Scale inference workloads using ``torch.neuronx.DataParallel`` for multi-core execution

   .. grid-item-card::
      :link: migration-from-xla-downcast-bf16
      :link-type: doc

      **XLA Migration Guide**
      ^^^
      Migrate from deprecated XLA environment variables to PyTorch mixed-precision options
