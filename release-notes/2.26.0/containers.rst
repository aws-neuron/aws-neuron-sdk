.. _neuron-2-26-0-dlc:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Deep Learning Containers (DLC) component, version 2.26.0. Release date: 9/18/2025.

AWS Neuron SDK 2.26.0: Neuron Deep Learning Containers release notes
====================================================================

**Date of release**:  September 18, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.26.0 release notes home <neuron-2-26-0-whatsnew>`

.. important::
   All Neuron packages and their dependencies have been upgraded to support version ``2.26.0`` of the AWS Neuron SDK.

Improvements
------------

We've added the following improvements for Deep Learning Container support in this release of the AWS Neuron SDK:

* Both `pytorch-training-neuronx` and `pytorch-inference-neuronx` DLCs have been upgraded to version ``2.8.0`` along with their related dependencies.
* Upgraded Python version to 3.11 in all Deep Learning Containers.

Behavioral changes
------------------

* End-of-support for the Transformers NeuronX library starts with the 2.26.0 release of the AWS Neuron SDK. With this support ended, the PyTorch inference Deep Learning Container (DLC) will no longer include the ``transformers-neuronx`` package. For more details, see :ref:`announce-eos-tnx`.

Previous release notes
----------------------

* :ref:`neuron-containers-release-notes`
* :ref:`neuron-dlc-release-notes`
* :ref:`neuron-k8-rn`