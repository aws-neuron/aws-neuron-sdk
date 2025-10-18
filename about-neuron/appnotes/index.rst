.. _neuron-appnotes-index:
.. _neuron-appnotes:

.. meta::
   :description: AWS Neuron SDK application notes for support announcements, performance optimization, migration guides, and framework-specific implementations.
   :date-modified: 2025-10-03

Neuron application notes
========================

.. toctree:: 
   :maxdepth: 2
   :hidden:

   Neuron Runtime Library <neuron1x/introducing-libnrt>
   Performance <perf/neuron-cc/performance-tuning>
   Parallel execution <perf/neuron-cc/parallel-ncgs>
   PyTorch for Neuron <torch-neuron/index>
   PyTorch for NeuronX <torch-neuronx/index>

Application notes provide specific documentation for support announcements, migration guides, performance optimization techniques, and framework-specific implementations for AWS Neuron SDK components.


Framework integration
---------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: torch-neuron-r-cnn-app-note
      :link-type: ref

      **PyTorch Neuron (Inf1)**
      ^^^
      R-CNN implementation and optimization techniques for PyTorch on ``Inf1``

   .. grid-item-card::
      :link: torch-neuronx-graph-partitioner-app-note
      :link-type: ref

      **PyTorch NeuronX Graph Partitioner**
      ^^^
      Advanced graph partitioning strategies for distributed training and inference

   .. grid-item-card::
      :link: torch-neuronx-dataparallel-app-note
      :link-type: ref

      **Data Parallel Inference on Torch NeuronX**
      ^^^
      Guide to using ``torch.neuronx.DataParallel`` for scalable inference on ``Inf1``

   .. grid-item-card::
      :link: torch-neuron-dataparallel-app-note
      :link-type: ref

      **Data Parallel Inference on Torch Neuron**
      ^^^
      Guide to using ``torch.neuron.DataParallel`` for scalable inference on ``Inf1``

   .. grid-item-card::
      :link: migration_from_xla_downcast_bf16
      :link-type: ref

      **Migrate from XLA_USE_BF16/XLA_DOWNCAST_BF16**
      ^^^
      Guide to migrating from deprecated XLA environment variables to recommended PyTorch mixed-precision options on NeuronX

   .. grid-item-card::
      :link: introduce-pytorch-2-8
      :link-type: ref

      **PyTorch 2.8 Support**
      ^^^
      New features and migration guide for PyTorch 2.8 on Neuron





