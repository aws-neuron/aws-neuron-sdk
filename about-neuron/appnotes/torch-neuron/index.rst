.. _torch-neuron-appnotes:

PyTorch Neuron Application Notes
=================================

.. toctree::
   :maxdepth: 1
   :hidden:

   bucketing-app-note
   rcnn-app-note
   torch-neuron-dataparallel-app-note

This section contains application notes specific to PyTorch Neuron (``torch-neuron``) for ``Inf1`` instances. These guides cover advanced optimization techniques, implementation patterns, and best practices for deploying PyTorch models on AWS Inferentia.

Application Notes
-----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: bucketing-app-note
      :link-type: doc

      **Dynamic Batching with Bucketing**
      ^^^
      Optimize inference performance using dynamic batching and bucketing strategies

   .. grid-item-card::
      :link: rcnn-app-note
      :link-type: doc

      **R-CNN Implementation Guide**
      ^^^
      Comprehensive guide for implementing and optimizing R-CNN models on Inferentia

   .. grid-item-card::
      :link: torch-neuron-dataparallel-app-note
      :link-type: doc

      **Data Parallel Inference**
      ^^^
      Scale inference workloads using ``torch.neuron.DataParallel`` for multi-core execution