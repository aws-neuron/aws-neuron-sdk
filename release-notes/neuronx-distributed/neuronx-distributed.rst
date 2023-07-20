.. _neuronx-distributed-rn:


Neuron Distributed Release Notes (``neuronx-distributed``)
==========================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for Neuronx-Distributed library.

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
