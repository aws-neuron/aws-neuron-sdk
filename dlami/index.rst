.. _neuron-dlami-overview:

Neuron DLAMI User Guide
=======================


.. contents:: Table of Contents
   :local:
   :depth: 2

Neuron DLAMI Overview
---------------------
Neuron DLAMIs are an easy way to get started on Neuron SDK as they come pre-installed with Neuron SDK. Neuron currently supports 3 types of DLAMIs, multi-framework DLAMIs , single framework DLAMIs and base DLAMIs
to easily get started on single Neuron instance. Below sections describe the supported Neuron DLAMIs, corresponding virtual environments and easy way to retrieve the DLAMI id using SSM parameters.


Neuron Multi Framework DLAMI
----------------------------
Neuron Deep Learning AMI (DLAMI) is a multi-framework DLAMI that supports multiple Neuron framework/libraries. Each DLAMI is pre-installed with Neuron drivers and support all Neuron instance types. Each virtual environment that corresponds to a specific Neuron framework/library 
comes pre-installed with all the Neuron libraries including Neuron compiler and Neuron runtime needed for you to easily get started. 

.. note::
  Neuron DLAMIs released with version 2.26.1 no longer support ``Inf1`` instance types due to an incompatibility with the Neuron driver.
  If you'd like to run ``Inf1`` workloads, use previous DLAMIs released up to SDK version 2.26.

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

    * - Amazon Linux 2023
      - Inf2, Trn1, Trn1n, Trn2, Trn3
      - Deep Learning AMI Neuron (Amazon Linux 2023)

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

    * - PyTorch 2.9 Torch NeuronX, NxD Core (Ubuntu 24.04 / Amazon Linux 2023)
      - /opt/aws_neuronx_venv_pytorch_2_9

    * - PyTorch 2.9 NxD Training, Torch NeuronX (Ubuntu 24.04 / Amazon Linux 2023)
      - /opt/aws_neuronx_venv_pytorch_2_9_nxd_training

    * - PyTorch 2.9 NxD Inference, Torch NeuronX (Ubuntu 24.04 / Amazon Linux 2023)
      - /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference

    * - JAX 0.7 NeuronX (Ubuntu 22.04 / Ubuntu 24.04 / Amazon Linux 2023)
      - /opt/aws_neuronx_venv_jax_0_7

    * - vLLM 0.11.0 NxD Inference, Torch NeuronX (Ubuntu 24.04 / Amazon Linux 2023)
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

Neuron Single Framework DLAMI
-----------------------------

Neuron supports single framework DLAMIs that correspond to a single framework version (ex:- TensorFlow 2.10). Each DLAMI is pre-installed with Neuron drivers and supports all Neuron instance types. Each virtual environment corresponding to a specific
Neuron framework/library comes pre-installed with all the relevant Neuron libraries including Neuron compiler and Neuron run-time.


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
      - Amazon Linux 2023
      - Inf2, Trn1, Trn1n, Trn2, Trn3
      - Deep Learning AMI Neuron PyTorch 2.9 (Amazon Linux 2023)

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

    * - vLLM 0.11.0
      - Amazon Linux 2023
      - Inf2, Trn1, Trn1n, Trn2, Trn3
      - Deep Learning AMI Neuron PyTorch Inference vLLM (Amazon Linux 2023)

    * - vLLM 0.11.0
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
  
    * - Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04, Amazon Linux 2023) 
      - PyTorch 2.9 Torch NeuronX, NxD Core
      - /opt/aws_neuronx_venv_pytorch_2_9

    * - Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04, Amazon Linux 2023) 
      - PyTorch 2.9 NxD Training, Torch NeuronX
      - /opt/aws_neuronx_venv_pytorch_2_9_nxd_training

    * - Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04, Amazon Linux 2023) 
      - PyTorch 2.9 NxD Inference, Torch NeuronX
      - /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference

    * - Deep Learning AMI Neuron JAX 0.7 (Ubuntu 24.04, Amazon Linux 2023) 
      - JAX NeuronX 0.7
      - /opt/aws_neuronx_venv_jax_0_7

    * - Deep Learning AMI Neuron PyTorch Inference vLLM (Ubuntu 24.04, Amazon Linux 2023) 
      - vLLM NeuronX 0.11.0
      - /opt/aws_neuronx_venv_pytorch_inference_vllm


Get started with the single framework DLAMI through AWS console by following one of the corresponding setup guides. If you want to
use the Neuron DLAMI in your cloud automation flows, Neuron also supports :ref:`SSM parameters <ssm-parameter-neuron-dlami>` to retrieve the latest DLAMI id.

Neuron Base DLAMI
-----------------
Neuron Base DLAMIs comes pre-installed with Neuron driver, EFA, and Neuron tools. Base DLAMIs might be relevant if you are extending the DLAMI for containerized applications.


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


Using SSM parameters to find DLAMI id and trigger Cloud Automation flows
------------------------------------------------------------------------

Neuron DLAMIs support AWS SSM parameters to easily find the Neuron DLAMI id.  Currently we only support finding the latest DLAMI id that corresponds to latest Neuron SDK release with SSM parameter support.
In the future releases, we will add support for finding the DLAMI id using SSM parameters for a specific Neuron release.


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

    * - Deep Learning AMI Neuron (Amazon Linux 2023)
      - /aws/service/neuron/dlami/multi-framework/amazon-linux-2023

    * - Deep Learning AMI Neuron (Ubuntu 24.04)
      - /aws/service/neuron/dlami/multi-framework/ubuntu-24.04

    * - Deep Learning AMI Neuron (Ubuntu 22.04)
      - /aws/service/neuron/dlami/multi-framework/ubuntu-22.04

    * - Deep Learning AMI Neuron PyTorch 2.9 (Amazon Linux 2023)
      - /aws/service/neuron/dlami/pytorch-2.9/amazon-linux-2023

    * - Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)
      - /aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04

    * - Deep Learning AMI Neuron JAX 0.7 (Amazon Linux 2023)
      - /aws/service/neuron/dlami/jax-0.7/amazon-linux-2023

    * - Deep Learning AMI Neuron JAX 0.7 (Ubuntu 24.04)
      - /aws/service/neuron/dlami/jax-0.7/ubuntu-24.04

    * - Deep Learning AMI Neuron PyTorch Inference vLLM (Amazon Linux 2023)
      - /aws/service/neuron/dlami/pytorch-inference-vllm/amazon-linux-2023

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


For example to find the latest DLAMI id for Multi-Framework DLAMI (Ubuntu 22) you can use the following

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

You can use CLI to find the latest DLAMI id and also launch the instance simulataneuosly.
Below code snippet shows an example of launching inf2 instance using multi-framework DLAMI


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


SSM Parameters can also be used directly in launch templates. So, you can update your Auto Scaling groups to use new AMI IDs without needing to create new launch templates or new versions of launch templates each time an AMI ID changes.
Ref: https://docs.aws.amazon.com/autoscaling/ec2/userguide/using-systems-manager-parameters.html



Other Resources
---------------

https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html

https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html

https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html
