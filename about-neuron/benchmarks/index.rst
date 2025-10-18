.. _benchmark:

.. meta::
   :description: Explore AWS Neuron performance benchmarks for Inf1, Inf2, and Trn1 instances. Find detailed inference and training performance data across NLP, CV, and recommender models to optimize your machine learning workloads.
   :date-modified: 2025-10-03

Neuron performance
==================

The Neuron performance pages provide comprehensive benchmarks and performance data for AWS Neuron SDK across different Trainium and Inferentia instance types. These benchmarks cover various open-source models for Natural Language Processing (NLP), Computer Vision (CV), and Recommender systems. Each benchmark includes detailed setup instructions and reproducible test configurations to help you evaluate performance for your specific use cases.

Inference performance
---------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: appnote-performance-benchmark
      :link-type: ref

      **Inf1 Inference Performance**
      ^^^
      Comprehensive inference benchmarks for ``Inf1`` instances across NLP, CV, and recommender models

   .. grid-item-card::
      :link: inf2-performance
      :link-type: ref

      **Inf2 Inference Performance**
      ^^^
      Latest inference performance data for ``Inf2`` instances with improved throughput and latency metrics

   .. grid-item-card::
      :link: trn1-inference-performance
      :link-type: ref

      **Trn1 Inference Performance**
      ^^^
      Inference benchmarks for ``Trn1`` instances showcasing versatile training and inference capabilities

Training performance
--------------------

.. grid:: 1
   :gutter: 2

   .. grid-item-card::
      :link: trn1-training-performance
      :link-type: ref

      **Trn1 Training Performance**
      ^^^
      Training performance benchmarks for ``Trn1`` instances with distributed training metrics and scalability data

.. toctree::
   :maxdepth: 1
   :hidden:

   inf1/index
   inf2/inf2-performance
   trn1/trn1-inference-performance
   trn1/trn1-training-performance


