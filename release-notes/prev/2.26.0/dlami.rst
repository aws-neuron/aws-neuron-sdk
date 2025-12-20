.. _neuron-2-26-0-dlami:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Deep Learning AWS Machine Images (DLAMIs) component, version 2.26.0. Release date: 9/18/2025.

AWS Neuron SDK 2.26.0: Neuron Deep Learning AWS Machine Images release notes
============================================================================

**Date of release**:  September 18, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.26.0 release notes home <neuron-2-26-0-whatsnew>`

Improvements
------------

We've added the following improvements for DLAMI support in this release of the AWS Neuron SDK:

* Support for PyTorch 2.8 (Amazon Linux 2023, Ubuntu 22.04) single-framework DLAMI
* Updates multi-framework DLAMI virtual environments to support PyTorch 2.8
* All Neuron packages and their dependencies have been upgraded to support version 2.26.0 of the AWS Neuron SDK

Behavioral changes
------------------
* End-of-support for the Transformers NeuronX library starts with the 2.26.0 release of the AWS Neuron SDK. As a result, the PyTorch inference Deep Learning Container (DLC) will no longer provide the ``transformers-neuronx`` virtual environment in both single and multi-framework DLAMIs. For more details, see :ref:`announce-eos-tnx`.

Previous release notes
----------------------

* :ref:`neuron-2-25-0-dlami`
* :ref:`neuron-dlami-release-notes`
