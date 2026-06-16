.. _locate-neuron-dlc-image:

Neuron Deep Learning Containers
===============================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

AWS Deep Learning Containers (DLCs) provide a set of Docker images that are pre-installed with deep learning frameworks.
The containers are optimized for performance and available in Amazon Elastic Container Registry (Amazon ECR).
DLCs make it straightforward to deploy custom ML environments in a containerized manner,
while taking advantage of the portability and reproducibility benefits of containers.

AWS Neuron DLCs are a set of Docker images for training and serving models on AWS Trainium and Inferentia instances using AWS Neuron SDK.
The sections below list all of the AWS Neuron DLCs, as well as the AWS DLCs that come pre-installed with the Neuron SDK.


Inference Containers
--------------------

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left
    :class: table-smaller-font-size

    * - DLC Name
      - DLC Link(s)
      - Tutorial(s)

    * - Neuron Inference Containers
      - | `Neuron PyTorch Inference Containers <https://github.com/aws-neuron/deep-learning-containers#pytorch-inference-neuron>`_
        | `Neuronx PyTorch Inference Containers <https://github.com/aws-neuron/deep-learning-containers#pytorch-inference-neuronx>`_
        | `Neuronx PyTorch vLLM Inference Containers <https://github.com/aws-neuron/deep-learning-containers#vllm-inference-neuronx>`_
      - | :ref:`tutorial-infer`
        | :ref:`torchserve-neuron`
        | :ref:`quickstart_vllm_dlc_deploy`

    * - Large Model Inference (LMI)/Deep Java Library (DJL) Containers
      - `LMI Containers <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers>`_
      -

    * - HuggingFace Inference Containers
      - | `HuggingFace PyTorch Inference Neuronx Containers <https://aws.github.io/deep-learning-containers/reference/available_images/#huggingface-pytorch-inference-neuronx>`_
        | `HuggingFace vLLM Inference Neuronx Containers <https://aws.github.io/deep-learning-containers/reference/available_images/#huggingface-vllm-inference-neuronx>`_
      -

    * - Triton Inference Containers
      - `NVIDIA Triton Inference Containers <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#nvidia-triton-inference-containers-sm-support-only>`_
      -


Training Containers
-------------------

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left
    :class: table-smaller-font-size

    * - DLC Name
      - DLC Link(s)
      - Tutorial(s)

    * - Neuron Training Containers
      - | `Neuronx PyTorch Training Containers <https://github.com/aws-neuron/deep-learning-containers#pytorch-training-neuronx>`_
        | `Neuronx Jax Training Containers <https://github.com/aws-neuron/deep-learning-containers#jax-training-neuronx>`_
      - :ref:`tutorial-training`

    * - HuggingFace Training Containers
      - `HuggingFace PyTorch Training Neuronx Containers <https://aws.github.io/deep-learning-containers/reference/available_images/#huggingface-pytorch-training-neuronx>`_
      -

.. note::
   Latest HuggingFace Neuron containers are also available on the `HuggingFace Optimum website <https://huggingface.co/docs/optimum-neuron/en/containers#available-optimum-neuron-containers>`_.


Getting started with Neuron DLC using Docker
----------------------------------------------

:doc:`/deploy/environments/docker-setup`


Using containers on AWS services
----------------------------------

* :doc:`Amazon EKS </deploy/eks/index>` — Deploy DLCs on Kubernetes with device plugins and DRA
* :doc:`Amazon ECS </deploy/ecs/index>` — Run DLC tasks on ECS with node problem detection
* :doc:`Amazon SageMaker </deploy/sagemaker/index>` — Use managed ML services with Neuron
* :doc:`AWS Batch </deploy/batch/index>` — Execute batch training jobs with DLCs
* :doc:`Amazon EC2 </deploy/ec2/inference-dlc>` — Run a DLC directly on EC2 with Docker


Customizing Neuron Deep Learning Containers
-------------------------------------------
Deep Learning Containers can be customized to fit your specific project needs.
To read more, visit :doc:`/deploy/environments/customize-dlc`.
