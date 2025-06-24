.. _neuron-dlc-release-notes:

Neuron DLC Release Notes
===============================

.. contents:: Table of contents
   :local:
   :depth: 1


Known Issues
------------
Updated : 6/23/2025

- pytorch-training-neuronx 2.7.0 DLC has two HIGH CVEs related to `sagemaker-python-sdk` package. We are actively working to resolve these high CVEs:
- * `CVE-2024-34072 <https://nvd.nist.gov/vuln/detail/CVE-2024-34072>`_
- * `CVE-2024-34073 <https://nvd.nist.gov/vuln/detail/CVE-2024-34073>`_


Neuron 2.24.0
-------------
Date: 06/23/2025

- Added new pytorch-inference-vllm-neuronx 0.7.2 DLC that contains all dependencies including drivers, tools, NxDI and other packages to run :ref:`nxdi-vllm-user-guide` out of the box
- Upgraded pytorch-training-neuronx DLC to 2.7 version along with its related dependencies
- Upgraded pytorch-inference-neuronx DLC to 2.7 version along with its related dependencies
- Upgraded jax-training-neuronx DLC to 0.6 version along with its related dependencies
- Updated Neuron SDK to latest 2.24.0 release for all Neuron DLCs


Neuron 2.23.0
-------------
Date: 05/19/2025

- Upgraded pytorch-training-neuronx DLC to 2.6 version along with its related dependencies
- Upgraded pytorch-inference-neuronx DLC to 2.6 version along with its related dependencies
- Updated Neuron SDK to latest 2.23.0 release for all Neuron DLCs


Neuron 2.22.0
-------------
Date: 04/04/2025

- Upgraded jax-training-neuronx DLC to 0.5 version
- Updated Neuron SDK to latest 2.22.0 release for all Neuron DLCs
- Restructure all Dockerfiles by combining RUN commands for faster build time


Neuron 2.21.1
-------------
Date: 01/14/2025

- Minor improvements and bug fixes.


Neuron 2.21.0
-------------
Date: 12/19/2024

- Added new jax-training-neuronx 0.4 Training DLC that contains all dependencies including drivers, tools and other packages to run :ref:`jax-neuronx-setup` out of the box.
- Added new pytorch-inference-neuronx 2.5.1 and pytorch-training-neuronx 2.5.1 DLCs
- PyTorch 1.13.1 and 2.1.2 DLCs reached end of support phase, We now recommend customers to use PyTorch 2.5.1 DLCs by default.
- All Neuron supported DLCs to use latest Neuron SDK 2.21.0 version.
- All Neuron supported DLCs are now updated to Ubuntu 22. Here is the list:
 * pytorch-inference-neuron 2.5.1 with Ubuntu 22
 * pytorch-training-neuron 2.5.1 with Ubuntu 22
 * jax-training-neuronx 0.4 with Ubuntu 22
- pytorch-inference-neuronx now supports both NxD Inference and Transformers NeuronX libraries for inference.


Neuron 2.20.2
-------------
Date: 11/20/2024

- Neuron 2.20.2 DLC fixes dependency bug for NxDT use case by pinning the correct torch version. 


Neuron 2.20.1
-------------

Date: 10/25/2024

- Neuron 2.20.1 DLC includes prerequisites for :ref:`nxdt_installation_guide`. Customers can expect to use NxDT out of the box.


Neuron 2.20.0
-------------

Date: 09/16/2024

- Updated Neuron SDK to latest 2.20.0 release for PyTorch Neuron DLCs.
- Added new NxD Training package to `pytorch-training-neuronx DLCs <https://github.com/aws-neuron/deep-learning-containers/tree/main?tab=readme-ov-file#pytorch-training-neuronx>`_.
- See `here <https://github.com/aws-neuron/deep-learning-containers/tree/2.20.0>`_ for the new DLC details.


Neuron 2.19.0
-------------

Date: 07/03/2024

- Updated Neuron SDK to latest 2.19.0 release for PyTorch Neuron DLCs.
- Updated TorchServe to 0.11.0 for PyTorch Neuron DLCs.
- See `here <https://github.com/aws-neuron/deep-learning-containers/tree/2.19.0>`_ for the new DLC details.
