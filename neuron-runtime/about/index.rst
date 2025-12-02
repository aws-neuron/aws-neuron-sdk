.. _neuron-runtime-about:

.. meta::
   :description: Learn about the AWS NeuronX Runtime, its features, and capabilities.
   :date-modified: 11/03/2025

About the NeuronX Runtime
==========================

This section provides information about the AWS Neuron Runtime, its features, and capabilities. Learn about core dumps, debugging techniques, and other important aspects of the Neuron Runtime.

What is the NeuronX Runtime?
--------------------------------

The NeuronX Runtime consists of a kernel driver and C/C++ libraries which provides APIs to access Inferentia and Trainium Neuron devices. The Neuron ML frameworks plugins for TensorFlow, PyTorch and Apache MXNet use the Neuron runtime to load and run models on the NeuronCores. Neuron runtime loads compiled deep learning models, also referred to as Neuron Executable File Format (NEFF) to the Neuron devices and is optimized for high-throughput and low-latency.

What are Neuron Collectives?
-----------------------------

Neuron Collectives are distributed communication primitives that coordinate data exchange among multiple NeuronCores in distributed machine learning workloads. Each rank represents a physical or logical NeuronCore that participates in collective operations such as AllGather, AllReduce, ReduceScatter, and AllToAll.

These operations enable efficient gradient aggregation during distributed training and parameter sharing during distributed inference. Collectives operate at two levels: intra-node communication uses high-bandwidth NeuronLink interconnects between chips within a node, while inter-node communication leverages EFA (Elastic Fabric Adapter) networks to coordinate across multiple physical nodes. The runtime automatically selects optimal algorithms based on message size, cluster topology, and latency requirements.

Get Started
------------  

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: Quickstart: Generate a Neuron Runtime Core Dump
      :link: runtime-core-dump-quickstart
      :link-type: ref
      :class-header: sd-bg-primary sd-text-white

      Learn how to generate a Neuron runtime core dump for debugging runtime failures and analyzing device state.

Neuron Runtime Collectives
---------------------------

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: About Neuron Runtime Collectives
      :link: collectives
      :link-type: doc
      :class-header: sd-bg-primary sd-text-white

      Learn about "Collectives", distributed communication primitives that enable efficient data exchange between NeuronCores.
   


