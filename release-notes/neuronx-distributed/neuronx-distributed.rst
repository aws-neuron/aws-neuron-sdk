.. _neuronx-distributed-rn:


Neuron Distributed Release Notes (``neuronx-distributed``)
==========================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for Neuronx-Distributed library.

Neuron Distributed [0.3.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 8/28/2023

New in this release
-------------------

* Added Zero1 Optimizer support that works with tensor-parallelism
* Added support for sequence-parallel that works with tensor-parallelism
* Added IO aliasing feature in parallel_trace api, which can allow marking certains tensors as state tensors
* Fixed hangs when tracing models using parallel_trace for higher TP degree

Known Issues and Limitations
----------------------------

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.

Neuron Distributed [0.2.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 7/19/2023

New in this release
-------------------

* Added parallel cross entropy loss function.

Known Issues and Limitations
----------------------------

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.

Date: 6/14/2023

New in this release
-------------------

* Releasing the Neuron Distributed (``neuronx-distributed``) library for enabling large language model training/inference.
* Added support for tensor-parallelism training/inference.

Known Issues and Limitations
----------------------------

* Currently the model checkpointing saves a sharded checkpoint, and users have to write a script to combine the shards.
