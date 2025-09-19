.. _neuron-2-25-0-nxd-core:

.. meta::
   :description: The official release notes for the AWS Neuron SDK NxD Core component, version 2.25.0. Release date: 7/31/2025.

AWS Neuron SDK 2.25.0: NxD Core release notes
=============================================

**Date of release**: July 31, 2025

**Version**: 0.14.18461

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.25.0 release notes home <neuron-2-25-0-whatsnew>`

Improvements
------------

*Improvements are significant new or improved features and solutions introduced with release 2.25.0 of the AWS Neuron SDK. Read on to learn about them!*

Inference
^^^^^^^^^

ModelBuilder V2
"""""""""

ModelBuilder V2 provides a simplified version of the ModelBuilder API that is more flexible and extensible.
This API includes basic building blocks that you can use to trace, compile, and load modules to Neuron.
For more information, see :ref:`nxd-core-model-builder-v2` and the updated
`Llama-3.2-1B reference inference sample <https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/llama>`__. 

Training
^^^^^^^^

Support for Shared Experts
"""""""""

Shared Experts allow multiple model components to utilize the same expert neural networks. This release adds full support for Shared Experts in training workloads.
  
