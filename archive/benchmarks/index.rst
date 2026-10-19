.. meta::
   :robots: noindex, nofollow
   :description: Archived AWS Neuron performance benchmarks for Inf1, Inf2, and Trn1 instances, measured with retired or legacy Neuron components.
   :date-modified: 2026-10-13

:nosearch:

.. _benchmark:

Neuron performance benchmarks (archived)
========================================

.. warning::

   These benchmark pages are archived and are no longer maintained. They were measured on Inf1, Inf2, and Trn1 instances with components that are retired or legacy, such as TensorFlow Neuron, torch-neuron, transformers-neuronx, PyTorch/XLA, NxD Core, and NxD Training. They do not reflect the performance of the current Neuron stack. For current inference and training, see :doc:`Inference on Neuron </inference>` and :doc:`Training on Neuron </training>`.

Inference performance
---------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: inf2-performance
      :link-type: ref

      **Inf2 inference performance**
      ^^^
      Inference benchmarks for ``Inf2`` instances (Neuron SDK 2.14 through 2.28)

   .. grid-item-card::
      :link: trn1-inference-performance
      :link-type: ref

      **Trn1 inference performance**
      ^^^
      Inference benchmarks for ``Trn1`` and ``Trn1n`` instances

   .. grid-item-card::
      :link: appnote-performance-benchmark
      :link-type: ref

      **Inf1 inference performance**
      ^^^
      Inference benchmarks for ``Inf1`` instances (Neuron SDK 2.12 through 2.20)

Training performance
--------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: trn1-training-performance
      :link-type: ref

      **Trn1 training performance**
      ^^^
      PyTorch/XLA and NxD Training benchmarks for ``Trn1`` and ``Trn1n`` instances (Neuron SDK 2.24 through 2.28)

.. toctree::
   :maxdepth: 1
   :hidden:

   inf2/inf2-performance
   trn1/trn1-inference-performance
   inf1/index
   trn1/trn1-training-performance
