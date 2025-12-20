.. _neuron-2-27-0-dlami:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Deep Learning AWS Machine Images (DLAMIs) component, version 2.27.0. Release date: 12/19/2025.

AWS Neuron SDK 2.27.0: Neuron Deep Learning AWS Machine Images release notes
============================================================================

**Date of release**: December 19, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.27.0 release notes home <neuron-2-27-0-whatsnew>`

What's New
----------

**Ubuntu 24.04 Support** — This release adds support for Ubuntu 24.04 base, single framework, and multi-framework DLAMIs with Python 3.12, providing customers with the latest Ubuntu LTS version for their machine learning workloads.

**vLLM V1 with vLLM-Neuron Plugin** — Published new vLLM V1 with the `vLLM-Neuron Plugin <https://github.com/vllm-project/vllm-neuron>`_ single framework DLAMI and added virtual environment to multi-framework DLAMIs (Amazon Linux 2023, Ubuntu 24.04).

**PyTorch 2.9 Support** — Added PyTorch 2.9 support for single framework DLAMIs and virtual environment to multi-framework DLAMIs (Amazon Linux 2023, Ubuntu 24.04).

**JAX 0.7 Support** — Published JAX 0.7 single framework DLAMI and updated multi-framework DLAMI virtual environments to JAX 0.7 (Amazon Linux 2023, Ubuntu 24.04).

**Neuron SDK Updates** — Upgraded all Neuron packages and dependencies to support AWS Neuron SDK version 2.27.

End of Support
--------------

**TensorFlow 2.10 End of Support** — The ``tensorflow_2_10`` single framework DLAMI and virtual environment in multi-framework DLAMIs will reach end of support in a future release. Customers are advised to use previously released DLAMIs for TensorFlow support.

**Ubuntu 22.04 Single Framework End of Support** — Ubuntu 22.04 single framework DLAMIs for PyTorch and JAX will reach end of support in a future release. Customers are advised to use multi-framework or previously released DLAMIs for Ubuntu 22.04.

**Inf1 virtual environments End of Support** — Inf1 virtual environments and AMIs have reached end of support. Use Neuron DLAMIs released up to SDK version 2.26 for Inf1 support.


