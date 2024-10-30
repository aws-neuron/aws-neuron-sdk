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
comes pre-installed with all the Neuron libraries including Neuron compiler and Neuron run-time needed for you to easily get started. 


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

    * - Ubuntu 22.04
      - Inf1, Inf2, Trn1, Trn1n 
      - Deep Learning AMI Neuron (Ubuntu 22.04)
    * - Amazon Linux 2023
      - Inf1, Inf2, Trn1, Trn1n 
      - Deep Learning AMI Neuron (Amazon Linux 2023)



Virtual Environments pre-installed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left
    :class: table-smaller-font-size

    * - Neuron Framework/Libraries supported
      - Virtual Environment 

    * - PyTorch Neuron 2.1 (Torch NeuronX , NeuronX Distributed)
      - /opt/aws_neuronx_venv_pytorch_2_1

    * - PyTorch Neuron 1.13.1 (Torch NeuronX , NeuronX Distributed)
      - /opt/aws_neuronx_venv_pytorch_1_13

    * - Transformers NeuronX (PyTorch 2.1)
      - /opt/aws_neuronx_venv_transformers_neuronx

    * - Tensorflow Neuron 2.10 (Tensorflow NeuronX)
      - /opt/aws_neuronx_venv_tensorflow_2_10

    * - PyTorch Neuron 1.13.1 (Inf1) (Torch Neuron) 
      - /opt/aws_neuron_venv_pytorch_1_13_inf1

    * - Tensorflow 2.10 (Inf1) (Tensorflow Neuron) 
      - /opt/aws_neuron_venv_tensorflow_2_10_inf1


You can easily get started with the multi-framework DLAMI through AWS console by following this :ref:`setup guide <setup-ubuntu22-multi-framework-dlami>` . If you are looking to 
use the Neuron DLAMI in your cloud automation flows , Neuron also supports :ref:`SSM parameters <ssm-parameter-neuron-dlami>` to easily retrieve the latest DLAMI id.



Neuron Single Framework DLAMI
-----------------------------

Neuron supports single framework DLAMIs that correspond to a single framework version (ex:- PyTorch 1.13). Each DLAMI is pre-installed with Neuron drivers and supports all Neuron instance types. Each virtual environment corresponding to a specific
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

    * - PyTorch 2.1
      - Ubuntu 22.04
      - Inf2, Trn1, Trn1n 
      - Deep Learning AMI Neuron PyTorch 2.1 (Ubuntu 22.04)

    * - PyTorch 1.13
      - Ubuntu 22.04
      - Inf1, Inf2, Trn1, Trn1n 
      - Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 22.04)

    * - PyTorch 1.13
      - Ubuntu 20.04
      - Inf1, Inf2, Trn1, Trn1n 
      - Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04)

    * - Tensorflow 2.10
      - Ubuntu 20.04
      - Inf2, Trn1, Trn1n 
      - Deep Learning AMI Neuron TensorFlow 2.10 (Ubuntu 20.04) 





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

    * - Deep Learning AMI Neuron PyTorch 2.1 (Ubuntu 22.04)
      - torch-neuronx, neuronx-distributed
      - /opt/aws_neuron_venv_pytorch

    * - Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 22.04)
      - torch-neuronx, neuronx-distributed
      - /opt/aws_neuron_venv_pytorch

    * - Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 22.04)
      - torch-neuron
      - /opt/aws_neuron_venv_pytorch_inf1

    * - Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04)
      - torch-neuronx, neuronx-distributed
      - /opt/aws_neuron_venv_pytorch

    * - Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04)
      - torch-neuron
      - /opt/aws_neuron_venv_pytorch_inf1

    * - Deep Learning AMI Neuron TensorFlow 2.10 (Ubuntu 20.04) 
      - tensorflow-neuronx
      - /opt/aws_neuron_venv_tensorflow

You can easily get started with the single framework DLAMI through AWS console by following one of the corresponding setup guides . If you are looking to 
use the Neuron DLAMI in your cloud automation flows , Neuron also supports :ref:`SSM parameters <ssm-parameter-neuron-dlami>` to easily retrieve the latest DLAMI id.




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

    * - Ubuntu 22.04
      - Inf1, Inf2, Trn1, Trn1n 
      - Deep Learning Base Neuron AMI (Ubuntu 22.04)

    * - Ubuntu 20.04
      - Inf1, Inf2, Trn1, Trn1n 
      - Deep Learning Base Neuron AMI (Ubuntu 20.04)


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

    * - Deep Learning AMI Neuron (Ubuntu 22.04)
      - /aws/service/neuron/dlami/multi-framework/ubuntu-22.04
    
    * - Deep Learning AMI Neuron (Amazon Linux 2023)
      - /aws/service/neuron/dlami/multi-framework/amazon-linux-2023

    * - Deep Learning AMI Neuron PyTorch 2.1 (Ubuntu 22.04) 
      - /aws/service/neuron/dlami/pytorch-2.1/ubuntu-22.04

    * - Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 22.04) 
      - /aws/service/neuron/dlami/pytorch-1.13/ubuntu-22.04

    * - Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04) 
      - /aws/service/neuron/dlami/pytorch-1.13/ubuntu-20.04

    * - Deep Learning AMI Neuron TensorFlow 2.10 (Ubuntu 20.04)
      - /aws/service/neuron/dlami/tensorflow-2.10/ubuntu-20.04

    * - Deep Learning Base Neuron AMI (Ubuntu 22.04)
      - /aws/service/neuron/dlami/base/ubuntu-22.04

    * - Deep Learning Base Neuron AMI (Ubuntu 20.04)
      - /aws/service/neuron/dlami/base/ubuntu-20.04


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
    --image-id resolve:ssm:/aws/service/neuron/dlami/pytorch-1.13/amazon-linux-2/latest/image_id \
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
