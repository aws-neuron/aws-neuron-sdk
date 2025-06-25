.. _neuron-dlami-release-notes:

Neuron DLAMI Release Notes
===============================

.. contents:: Table of contents
   :local:
   :depth: 1

Neuron 2.24.0
-------------

Date: 06/24/2025

- Adding support for PyTorch 2.7 (Amazon Linux 2023, Ubuntu 22.04) single framework DLAMI.
- Adding support for JAX 0.6 (Amazon Linux 2023, Ubuntu 22.05) single framework DLAMI.
- Update multi framework DLAMI's virtual environments to use PyTorch 2.7 and JAX 0.6

See :ref:`neuron-dlami-overview`.


Neuron 2.23.0
-------------

Date: 05/19/2025

- Adding support for PyTorch 2.6 (Amazon Linux 2023, Ubuntu 22.04) single framework DLAMI.
- Adding support JAX 0.5 (Amazon Linux 2023, Ubuntu 22.05) single framework DLAMI.
- Update multi framework DLAMI's virtual environments to use PyTorch 2.6 and JAX 0.5
- Security improvements: Bump Linux kernel to 6.8.1027 for Ubuntu 22 DLAMIs.
- Security improvements: Bump Linux kernel to 6.1.134 for Amazon Linux 2023 DLAMIs.
- Added a setup script within neuronx-distributed-training virtual environment to automate the installation of required dependencies. See :ref: `nxdt_installation_guide`.

See :ref:`neuron-dlami-overview`.


Neuron 2.22.0
-------------

Date: 4/3/2025

- Adding PyTorch 2.5 (Amazon Linux 2023, Ubuntu 22.04) and PyTorch 1.13 Inf1 (Ubuntu 22.04) single framework DLAMIs.
- Adding PyTorch 1.13 Inf1 virtual environments within the Neuron Multi Framework DLAMIs. (Amazon Linux 2023, Ubuntu 22.04)  
- Adding Tensorflow 2.10 Inf1 virtual environments within Multi Framework DLAMI and Tensorflow singleframework DLAMI.
- Adding support for Amazon Linux 2023 in the Base Neuron DLAMI.
- Security improvements: Bump Linux kernel to 5.19.0-1024-aws for Ubuntu 22 DLAMIs.
- Bug fix: Update venv paths in message of the day (MOTD) launch screens for Neuron DLAMIs.
- Optimization: Reduce EBS storage size for all DLAMIs such that the virtual environments and dependencies consume 80% of available block storage. This results in reduced cost and time to launch for the DLAMIs. Customers can always request more storage if needed.

See :ref:`neuron-dlami-overview`.

Neuron 2.21.1
-------------

Date: 01/14/2025

- No changes to DLAMI. Incompatibility issue reported for Tensorflow 2.10 (inf1) on v2.21.1. See :ref:`neuron-dlami-overview`.

Neuron 2.21.0
-------------

Date: 12/20/2024

- Added support for Trainium2 chips within the Neuron Multi Framework DLAMI.
- Added support for JAX 0.4 to Neuron Multi Framework DLAMI.
- Added NxD Training (NxDT), NxD Inference (NxDI) and NxD Core PyTorch 2.5 support within the Neuron Multi Framework DLAMI.
- Added Single Framework DLAMI for TensorFlow 2.10 on U22 and corresponding SSM Parameter support.
- Removing virtual environments for PyTorch 1.13 and 2.1 within Neuron Multi Framework DLAMI.
- Removing PyTorch 1.13 inf1 virtual environment from Neuron Multi Framework DLAMI.
- Removing Single Framework DLAMI and corresponding SSM Parameters for PyTorch 1.13 and 2.1.
- Removing SSM Parameters for AL2 Base DLAMI, PyTorch 1.13 and 2.1 Neuron DLAMI.
  
See :ref:`neuron-dlami-overview`

Neuron 2.20.1
-------------

Date: 10/25/2024

- Added support for Amazon Linux 2023 to Neuron Multi Framework DLAMI. Customers will have two operating system options when using the multi framework DLAMI. See :ref:`neuron-dlami-overview`.

Neuron 2.20.0
-------------

Date: 09/16/2024

- Add neuronx-distributed-training library to PyTorch virtual enviornments. See :ref:`neuron-dlami-overview`
- Updated existing Neuron supported DLAMIs with Neuron 2.20 SDK release.

Neuron 2.19.0
-------------

Date: 07/03/2024

- New Neuron PyTorch-2.1, PyTorch-1.13 and Base Deep Learning AMIs (DLAMI) for Ubuntu 22. See :ref:`neuron-dlami-overview`
- Updated Existing Neuron supported DLAMIs with Neuron 2.19 SDK release.
- End of support for Amazon Linux 2 DLAMIs :ref:`announce-eos-al2`





