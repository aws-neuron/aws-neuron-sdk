.. meta::
    :description: Complete release notes for the Neuron DLAMI component across all AWS Neuron SDK versions.
    :keywords: neuron dlami, deep learning ami, release notes, aws neuron sdk
    :date-modified: 02/26/2026

.. _dlamis_rn:

Component Release Notes for Neuron DLAMI
=========================================

The release notes for the Neuron DLAMI component. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _dlami-2-28-0-rn:   

Neuron DLAMIs [(Neuron 2.28.0 Release)
------------------------------------------------------------------------

Date of Release: 02/26/2026


Known Issues
~~~~~~~~~~~~

- AL2023-based DLAMIs released alongside version 2.28.0 do not include PyTorch 2.9+ or Multi-Framework environments due to an incompatibility with the default GLIBC version installed on AL2023.


.. _dlami-2-27-1-rn:

Neuron DLAMI [2.27.1] (Neuron 2.27.1 Release)
-----------------------------------------------

Date of Release: 01/14/2026

Improvements
~~~~~~~~~~~~~~~

* Support for NKI has been added to all DLAMI virtual environments.


----

.. _dlami-2-27-0-rn:

Neuron DLAMI [2.27.0] (Neuron 2.27.0 Release)
-----------------------------------------------

Date of Release: 12/19/2025

Improvements
~~~~~~~~~~~~~~~

* Ubuntu 24.04 Support: This release adds support for Ubuntu 24.04 base, single framework, and multi-framework DLAMIs with Python 3.12, providing customers with the latest Ubuntu LTS version for their machine learning workloads.
* vLLM V1 with vLLM-Neuron Plugin: Published new vLLM V1 with the vLLM-Neuron Plugin single framework DLAMI and added virtual environment to multi-framework DLAMIs (Amazon Linux 2023, Ubuntu 24.04).
* PyTorch 2.9 Support: Added PyTorch 2.9 support for single framework DLAMIs and virtual environment to multi-framework DLAMIs (Amazon Linux 2023, Ubuntu 24.04).
* JAX 0.7 Support: Published JAX 0.7 single framework DLAMI and updated multi-framework DLAMI virtual environments to JAX 0.7 (Amazon Linux 2023, Ubuntu 24.04).
* Neuron SDK Updates: Upgraded all Neuron packages and dependencies to support AWS Neuron SDK version 2.27.

Breaking Changes
~~~~~~~~~~~~~~~~

* TensorFlow 2.10 End of Support: The tensorflow_2_10 single framework DLAMI and virtual environment in multi-framework DLAMIs will reach end of support in a future release. Customers are advised to use previously released DLAMIs for TensorFlow support.
* Ubuntu 22.04 Single Framework End of Support: Ubuntu 22.04 single framework DLAMIs for PyTorch and JAX will reach end of support in a future release. Customers are advised to use multi-framework or previously released DLAMIs for Ubuntu 22.04.
* Inf1 virtual environments End of Support: Inf1 virtual environments and AMIs have reached end of support. Use Neuron DLAMIs released up to SDK version 2.26 for Inf1 support.


----

.. _dlami-2-26-0-rn:

Neuron DLAMI [2.26.0] (Neuron 2.26.0 Release)
-----------------------------------------------

Date of Release: 09/18/2025

Improvements
~~~~~~~~~~~~~~~

* Support for PyTorch 2.8 (Amazon Linux 2023, Ubuntu 22.04) single-framework DLAMI.
* Updates multi-framework DLAMI virtual environments to support PyTorch 2.8.
* All Neuron packages and their dependencies have been upgraded to support version 2.26.0 of the AWS Neuron SDK.

Breaking Changes
~~~~~~~~~~~~~~~~

* End-of-support for the Transformers NeuronX library starts with the 2.26.0 release of the AWS Neuron SDK. As a result, the PyTorch inference Deep Learning Container (DLC) will no longer provide the transformers-neuronx virtual environment in both single and multi-framework DLAMIs.


----

.. _dlami-2-25-0-rn:

Neuron DLAMI [2.25.0] (Neuron 2.25.0 Release)
-----------------------------------------------

Date of Release: 07/31/2025

Improvements
~~~~~~~~~~~~~~~

* All multi-framework virtual environments for the Deep Learning AMIs have been upgraded with the latest Neuron packages to support the AWS Neuron SDK version 2.25.0.


----

.. _dlami-2-24-0-rn:

Neuron DLAMI [2.24.0] (Neuron 2.24.0 Release)
-----------------------------------------------

Date of Release: 06/24/2025

Improvements
~~~~~~~~~~~~~~~

* Adding support for PyTorch 2.7 (Amazon Linux 2023, Ubuntu 22.04) single framework DLAMI.
* Adding support for JAX 0.6 (Amazon Linux 2023, Ubuntu 22.05) single framework DLAMI.
* Update multi framework DLAMI's virtual environments to use PyTorch 2.7 and JAX 0.6.


----

.. _dlami-2-23-0-rn:

Neuron DLAMI [2.23.0] (Neuron 2.23.0 Release)
-----------------------------------------------

Date of Release: 05/19/2025

Improvements
~~~~~~~~~~~~~~~

* Adding support for PyTorch 2.6 (Amazon Linux 2023, Ubuntu 22.04) single framework DLAMI.
* Adding support JAX 0.5 (Amazon Linux 2023, Ubuntu 22.05) single framework DLAMI.
* Update multi framework DLAMI's virtual environments to use PyTorch 2.6 and JAX 0.5.
* Security improvements: Bump Linux kernel to 6.8.1027 for Ubuntu 22 DLAMIs.
* Security improvements: Bump Linux kernel to 6.1.134 for Amazon Linux 2023 DLAMIs.
* Added a setup script within neuronx-distributed-training virtual environment to automate the installation of required dependencies.


----

.. _dlami-2-22-0-rn:

Neuron DLAMI [2.22.0] (Neuron 2.22.0 Release)
-----------------------------------------------

Date of Release: 04/03/2025

Improvements
~~~~~~~~~~~~~~~

* Adding PyTorch 2.5 (Amazon Linux 2023, Ubuntu 22.04) and PyTorch 1.13 Inf1 (Ubuntu 22.04) single framework DLAMIs.
* Adding PyTorch 1.13 Inf1 virtual environments within the Neuron Multi Framework DLAMIs. (Amazon Linux 2023, Ubuntu 22.04).
* Adding Tensorflow 2.10 Inf1 virtual environments within Multi Framework DLAMI and Tensorflow singleframework DLAMI.
* Adding support for Amazon Linux 2023 in the Base Neuron DLAMI.
* Security improvements: Bump Linux kernel to 5.19.0-1024-aws for Ubuntu 22 DLAMIs.
* Optimization: Reduce EBS storage size for all DLAMIs such that the virtual environments and dependencies consume 80% of available block storage. This results in reduced cost and time to launch for the DLAMIs. Customers can always request more storage if needed.

Bug Fixes
~~~~~~~~~

* Update venv paths in message of the day (MOTD) launch screens for Neuron DLAMIs.


----

.. _dlami-2-21-1-rn:

Neuron DLAMI [2.21.1] (Neuron 2.21.1 Release)
-----------------------------------------------

Date of Release: 01/14/2025

Improvements
~~~~~~~~~~~~~~~

* No changes to DLAMI.

Known Issues
~~~~~~~~~~~~

* Incompatibility issue reported for Tensorflow 2.10 (inf1) on v2.21.1.


----

.. _dlami-2-21-0-rn:

Neuron DLAMI [2.21.0] (Neuron 2.21.0 Release)
-----------------------------------------------

Date of Release: 12/20/2024

Improvements
~~~~~~~~~~~~~~~

* Added support for Trainium2 chips within the Neuron Multi Framework DLAMI.
* Added support for JAX 0.4 to Neuron Multi Framework DLAMI.
* Added NxD Training (NxDT), NxD Inference (NxDI) and NxD Core PyTorch 2.5 support within the Neuron Multi Framework DLAMI.
* Added Single Framework DLAMI for TensorFlow 2.10 on U22 and corresponding SSM Parameter support.

Breaking Changes
~~~~~~~~~~~~~~~~

* Removing virtual environments for PyTorch 1.13 and 2.1 within Neuron Multi Framework DLAMI.
* Removing PyTorch 1.13 inf1 virtual environment from Neuron Multi Framework DLAMI.
* Removing Single Framework DLAMI and corresponding SSM Parameters for PyTorch 1.13 and 2.1.
* Removing SSM Parameters for AL2 Base DLAMI, PyTorch 1.13 and 2.1 Neuron DLAMI.


----

.. _dlami-2-20-1-rn:

Neuron DLAMI [2.20.1] (Neuron 2.20.1 Release)
-----------------------------------------------

Date of Release: 10/25/2024

Improvements
~~~~~~~~~~~~~~~

* Added support for Amazon Linux 2023 to Neuron Multi Framework DLAMI. Customers will have two operating system options when using the multi framework DLAMI.


----

.. _dlami-2-20-0-rn:

Neuron DLAMI [2.20.0] (Neuron 2.20.0 Release)
-----------------------------------------------

Date of Release: 09/16/2024

Improvements
~~~~~~~~~~~~~~~

* Add neuronx-distributed-training library to PyTorch virtual environments.
* Updated existing Neuron supported DLAMIs with Neuron 2.20 SDK release.


----

.. _dlami-2-19-0-rn:

Neuron DLAMI [2.19.0] (Neuron 2.19.0 Release)
-----------------------------------------------

Date of Release: 07/03/2024

Improvements
~~~~~~~~~~~~~~~

* New Neuron PyTorch-2.1, PyTorch-1.13 and Base Deep Learning AMIs (DLAMI) for Ubuntu 22.
* Updated Existing Neuron supported DLAMIs with Neuron 2.19 SDK release.

Breaking Changes
~~~~~~~~~~~~~~~~

* End of support for Amazon Linux 2 DLAMIs.

Known Issues
~~~~~~~~~~~~

* None reported for this release.