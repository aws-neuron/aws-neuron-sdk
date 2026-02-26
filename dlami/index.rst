.. meta::
   :description: Neuron Deep Learning AMIs (DLAMIs) are pre-configured Amazon Machine Images with the Neuron SDK for easy deployment on AWS Inferentia and Trainium instances.
   :keywords: Neuron DLAMI, Deep Learning AMI, AWS Neuron SDK, Inferentia, Trainium, PyTorch, JAX, TensorFlow, vLLM, SSM Parameters
   :date-modified: 01/22/2026

.. _neuron-dlami-overview:

Neuron DLAMI User Guide
=======================

This guide helps you select, configure, and deploy AWS Neuron Deep Learning AMIs (DLAMIs) for running machine learning workloads on AWS Inferentia and Trainium instances. Learn about the different DLAMI types available, pre-installed virtual environments for popular ML frameworks like PyTorch and JAX, and how to automate DLAMI deployment.

.. contents:: Table of Contents
   :local:
   :depth: 2

What are Neuron DLAMIs?
------------------------

Neuron Deep Learning AMIs (DLAMIs) are pre-configured Amazon Machine Images that provide the easiest way to get started with the AWS Neuron SDK. Each DLAMI comes with Neuron drivers, frameworks, and libraries pre-installed, enabling you to quickly launch and run deep learning workloads on AWS Inferentia and Trainium instances without manual setup.

Neuron currently supports three types of DLAMIs to meet different deployment needs:

* **Multi-Framework DLAMIs**: Support multiple ML frameworks (PyTorch, JAX, vLLM) with separate virtual environments for each
* **Single Framework DLAMIs**: Optimized for a specific framework version with focused virtual environments
* **Base DLAMIs**: Include only Neuron drivers, EFA, and tools - ideal for containerized applications and custom builds

All Neuron DLAMIs support automated discovery through AWS Systems Manager (SSM) parameters, making them easy to integrate into cloud automation workflows and infrastructure-as-code deployments.

.. note::
  Starting with version 2.26.1, Neuron DLAMIs no longer support ``Inf1`` instance types due to an incompatibility with the Neuron driver.  
  If you'd like to run ``Inf1`` workloads, use previous DLAMIs released up to SDK version 2.26.

----

Neuron Multi Framework DLAMI
----------------------------

Neuron Multi-Framework DLAMIs provide the most comprehensive environment, supporting multiple ML frameworks and libraries in isolated virtual environments. Each DLAMI is pre-installed with Neuron drivers and supports all current Neuron instance types (Inf2, Trn1, Trn1n, Trn2, Trn3). This is the recommended option for teams working with multiple frameworks or exploring different ML libraries.

.. note::
  Starting with version 2.27.1, AL2023 DLAMIs no longer support ``PyTorch 2.9+`` due to an incompatibility issue with the default GLIB.c installed on AL2023.
  PyTorch requires GLIB.c 2.35+ and upgrading the version within AL2023 can break other system dependencies. This is the error message:
  
  ``ImportError: /lib64/libm.so.6: version `GLIBC_2.35' not found``

  Since the latest vLLM version depends on 
  PyTorch 2.9 we have also removed that environment from the DLAMI.
  
  For a workaround, use the latest Ubuntu based AMIs instead.

.. note::
  AL2023 DLAMIs released with version 2.27.1 no longer support ``PyTorch 2.9+`` due to an incompatibility issue with the default GLIB.c installed on AL2023.
  PyTorch requires GLIB.c 2.35+ and upgrading the version within AL2023 can break other system dependencies. This is the error message:
  
  ``ImportError: /lib64/libm.so.6: version `GLIBC_2.35' not found``

  Since the latest vLLM version depends on 
  PyTorch 2.9 we have also removed that environment from the DLAMI.
  
  For a workaround, use the latest Ubuntu based AMIs instead.

Multi Framework DLAMIs supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left
    :class: table-smaller-font-size

    * - Operating System
      - Neuron Instances Supported
      - DLAMI Name

    * - Ubuntu 24.04
      - Inf2, Trn1, Trn1n, Trn2, Trn3
      - Deep Learning AMI Neuron (Ubuntu 24.04)

    * - Ubuntu 22.04
      - Inf2, Trn1, Trn1n, Trn2, Trn3
      - Deep Learning AMI Neuron (Ubuntu 22.04)

.. _neuron-dlami-multifw-venvs:


Virtual Environments pre-installed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left
    :class: table-smaller-font-size

    * - Neuron Framework/Libraries supported
      - Virtual Environment

    * - PyTorch 2.9 Torch NeuronX, NxD Core (Ubuntu 24.04)
      - /opt/aws_neuronx_venv_pytorch_2_9

    * - PyTorch 2.9 NxD Training, Torch NeuronX (Ubuntu 24.04)
      - /opt/aws_neuronx_venv_pytorch_2_9_nxd_training

    * - PyTorch 2.9 NxD Inference, Torch NeuronX (Ubuntu 24.04)
      - /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference

    * - JAX 0.7 NeuronX (Ubuntu 22.04 / Ubuntu 24.04 / Amazon Linux 2023)
      - /opt/aws_neuronx_venv_jax_0_7

    * - vLLM 0.13.0 NxD Inference, Torch NeuronX (Ubuntu 24.04)
      - /opt/aws_neuronx_venv_pytorch_inference_vllm

    * - PyTorch 2.8 Torch NeuronX, NxD Core (Ubuntu 22.04)
      - /opt/aws_neuronx_venv_pytorch_2_8

    * - PyTorch 2.8 NxD Training, Torch NeuronX (Ubuntu 22.04)
      - /opt/aws_neuronx_venv_pytorch_2_8_nxd_training

    * - PyTorch 2.8 NxD Inference, Torch NeuronX (Ubuntu 22.04)
      - /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference


We have included a setup script that installs required dependencies for the package within the PyTorch 2.8 and 2.9 NxD Training virtual environment. To run this script,
activate the virtual environment and run ``setup_nxdt.sh`` and this will run :ref:`the setup steps here <nxdt_installation_guide>`.

You can easily get started with the multi-framework DLAMI through AWS console by following this :ref:`setup guide <setup-ubuntu22-multi-framework-dlami>`. If you are looking to 
use the Neuron DLAMI in your cloud automation flows, Neuron also supports :ref:`SSM parameters <ssm-parameter-neuron-dlami>` to easily retrieve the latest DLAMI id.

----

Neuron Single Framework DLAMI
-----------------------------

Neuron Single Framework DLAMIs are optimized for specific framework versions, providing a streamlined environment when you know exactly which framework you'll be using. Each DLAMI is pre-installed with Neuron drivers and supports all Neuron instance types. These DLAMIs are ideal for production deployments where you want a focused, framework-specific environment. 


Single Framework DLAMIs supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left
    :class: table-smaller-font-size

    * - Framework
      - Operating System
      - Neuron Instances Supported
      - DLAMI Name

    * - PyTorch 2.9
      - Ubuntu 24.04
      - Inf2, Trn1, Trn1n, Trn2, Trn3
      - Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)

    * - JAX 0.7
      - Amazon Linux 2023
      - Inf2, Trn1, Trn1n, Trn2, Trn3
      - Deep Learning AMI Neuron JAX 0.7 (Amazon Linux 2023)

    * - JAX 0.7
      - Ubuntu 24.04
      - Inf2, Trn1, Trn1n, Trn2, Trn3
      - Deep Learning AMI Neuron JAX 0.7 (Ubuntu 24.04)

    * - vLLM 0.13.0
      - Ubuntu 24.04
      - Inf2, Trn1, Trn1n, Trn2, Trn3
      - Deep Learning AMI Neuron PyTorch Inference vLLM (Ubuntu 24.04)


Virtual Environments pre-installed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left
    :class: table-smaller-font-size

    * - DLAMI Name
      - Neuron Libraries supported
      - Virtual Environment
  
    * - Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04) 
      - PyTorch 2.9 Torch NeuronX, NxD Core
      - /opt/aws_neuronx_venv_pytorch_2_9

    * - Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04) 
      - PyTorch 2.9 NxD Training, Torch NeuronX
      - /opt/aws_neuronx_venv_pytorch_2_9_nxd_training

    * - Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04) 
      - PyTorch 2.9 NxD Inference, Torch NeuronX
      - /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference

    * - Deep Learning AMI Neuron JAX 0.7 (Ubuntu 24.04, Amazon Linux 2023) 
      - JAX NeuronX 0.7
      - /opt/aws_neuronx_venv_jax_0_7

    * - Deep Learning AMI Neuron PyTorch Inference vLLM (Ubuntu 24.04) 
      - vLLM NeuronX 0.13.0
      - /opt/aws_neuronx_venv_pytorch_inference_vllm


Get started with the single framework DLAMI through AWS console by following one of the corresponding setup guides. If you want to
use the Neuron DLAMI in your cloud automation flows, Neuron also supports :ref:`SSM parameters <ssm-parameter-neuron-dlami>` to retrieve the latest DLAMI id.

----

Neuron Base DLAMI
-----------------

Neuron Base DLAMIs provide a minimal foundation with only the essential components: Neuron driver, EFA (Elastic Fabric Adapter), and Neuron tools. These DLAMIs are designed for advanced users who want to build custom environments, create containerized applications, or have specific framework version requirements not covered by the pre-configured DLAMIs.


Base DLAMIs supported
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left
    :class: table-smaller-font-size

    * - Operating System
      - Neuron Instances Supported
      - DLAMI Name

    * - Amazon Linux 2023
      - Inf2, Trn1, Trn1n, Trn2, Trn3 
      - Deep Learning Base Neuron AMI (Amazon Linux 2023)

    * - Ubuntu 24.04
      - Inf2, Trn1, Trn1n, Trn2, Trn3
      - Deep Learning Base Neuron AMI (Ubuntu 24.04)

    * - Ubuntu 22.04
      - Inf2, Trn1, Trn1n, Trn2, Trn3
      - Deep Learning Base Neuron AMI (Ubuntu 22.04)


.. _ssm-parameter-neuron-dlami:

----

Using SSM Parameters for Cloud Automation
------------------------------------------

Neuron DLAMIs support AWS Systems Manager (SSM) parameters for automated DLAMI discovery and deployment. This enables you to always use the latest Neuron SDK release in your infrastructure-as-code templates, CI/CD pipelines, and auto-scaling configurations without hardcoding AMI IDs.

SSM parameters provide several key benefits:

* **Always up-to-date**: Automatically reference the latest DLAMI with the newest Neuron SDK release
* **Infrastructure-as-code friendly**: Use in CloudFormation, Terraform, and other IaC tools
* **Auto Scaling integration**: Update Auto Scaling groups without modifying launch templates
* **Multi-region support**: Available across all AWS regions where Neuron instances are supported

Currently, SSM parameters support finding the latest DLAMI ID for each DLAMI type. Support for finding specific Neuron SDK version DLAMIs will be added in future releases.


Finding specific DLAMI image id with the latest neuron release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can find the DLAMI that supports latest Neuron SDK by using the SSM get-parameter.


.. code-block::

    aws ssm get-parameter \
    --region us-east-1 \
    --name <dlami-ssm-parameter-prefix>/latest/image_id \
    --query "Parameter.Value" \
    --output text


The SSM parameter prefix for each DLAMI can be seen below


SSM Parameter Prefix
""""""""""""""""""""
.. list-table::
    :widths: 20 39
    :header-rows: 1
    :align: left
    :class: table-smaller-font-size

    * - AMI Name
      - SSM parameter Prefix

    * - Deep Learning AMI Neuron (Ubuntu 24.04)
      - /aws/service/neuron/dlami/multi-framework/ubuntu-24.04

    * - Deep Learning AMI Neuron (Ubuntu 22.04)
      - /aws/service/neuron/dlami/multi-framework/ubuntu-22.04

    * - Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)
      - /aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04

    * - Deep Learning AMI Neuron JAX 0.7 (Amazon Linux 2023)
      - /aws/service/neuron/dlami/jax-0.7/amazon-linux-2023

    * - Deep Learning AMI Neuron JAX 0.7 (Ubuntu 24.04)
      - /aws/service/neuron/dlami/jax-0.7/ubuntu-24.04

    * - Deep Learning AMI Neuron PyTorch Inference vLLM (Ubuntu 24.04)
      - /aws/service/neuron/dlami/pytorch-inference-vllm/ubuntu-24.04

    * - Deep Learning AMI Neuron PyTorch 2.8 (Ubuntu 22.04)
      - /aws/service/neuron/dlami/pytorch-2.8/ubuntu-22.04

    * - Deep Learning AMI Neuron TensorFlow 2.10 (Ubuntu 22.04)
      - /aws/service/neuron/dlami/tensorflow-2.10/ubuntu-22.04

    * - Deep Learning Base Neuron AMI (Amazon Linux 2023)
      - /aws/service/neuron/dlami/base/amazon-linux-2023

    * - Deep Learning Base Neuron AMI (Ubuntu 24.04)
      - /aws/service/neuron/dlami/base/ubuntu-24.04

    * - Deep Learning Base Neuron AMI (Ubuntu 22.04)
      - /aws/service/neuron/dlami/base/ubuntu-22.04


For example to find the latest DLAMI id for Multi-Framework DLAMI (Ubuntu 22.04) you can use the following:

.. code-block::

    aws ssm get-parameter \
    --region us-east-1 \
    --name /aws/service/neuron/dlami/multi-framework/ubuntu-22.04/latest/image_id \
    --query "Parameter.Value" \
    --output text


You can find all available parameters supported in Neuron DLAMis via CLI

.. code-block::

    aws ssm get-parameters-by-path \
    --region us-east-1 \
    --path /aws/service/neuron \
    --recursive


You can also view the SSM parameters supported in Neuron through AWS parameter store by selecting the "Neuron" service.



Use SSM Parameter to launch instance directly via CLI
"""""""""""""""""""""""""""""""""""""""""""""""""""""

You can use the AWS CLI to resolve the latest DLAMI ID and launch an instance in a single command. This is particularly useful for scripting and automation workflows.

Below is an example of launching an Inf2 instance using the TensorFlow 2.10 single-framework DLAMI: 


.. code-block::

    aws ec2 run-instances \
    --region us-east-1 \
    --image-id resolve:ssm:/aws/service/neuron/dlami/tensorflow-2.10/ubuntu-22.04/latest/image_id \
    --count 1 \
    --instance-type inf2.48xlarge \
    --key-name <my-key-pair> \
    --security-groups <my-security-group>



Use SSM alias in EC2 launch templates
"""""""""""""""""""""""""""""""""""""

SSM Parameters can be used directly in EC2 launch templates, enabling your Auto Scaling groups to automatically use the latest AMI IDs without requiring updates to launch templates or creating new versions each time an AMI ID changes. This significantly simplifies AMI lifecycle management in production environments.

For more information, see: https://docs.aws.amazon.com/autoscaling/ec2/userguide/using-systems-manager-parameters.html

----

Other Resources
---------------

Learn more about AWS Deep Learning AMIs and Systems Manager:

* `AWS Deep Learning AMI Developer Guide <https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html>`_
* `AWS DLAMI Release Notes <https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html>`_
* `AWS Systems Manager Parameter Store <https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html>`_
* :doc:`Neuron DLAMI Release Notes </release-notes/components/dlamis>`
