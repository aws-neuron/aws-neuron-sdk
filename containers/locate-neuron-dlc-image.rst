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
      - | :ref:`tutorial-infer`
        | :ref:`torchserve-neuron`

    * - Large Model Inference (LMI)/Deep Java Library (DJL) Containers
      - `LMI Containers <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers>`_
      -

    * - HuggingFace Inference Containers
      - | `HuggingFace Text Generation Inference (TGI) Containers <https:https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-text-generation-inference-tgi-containers>`_
        | `HuggingFace Neuron Inference Containers <https:https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-neuron-inference-containers>`_
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
      - `Neuronx PyTorch Training Containers <https://github.com/aws-neuron/deep-learning-containers#pytorch-training-neuronx>`_
      - :ref:`tutorial-training`

    * - HuggingFace Training Containers
      - `HuggingFace Neuron Training Containers <https:https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-neuron-training-containers>`_
      -


Getting started with Neuron DLC using Docker
----------------------------------------------

:ref:`containers-getting-started`


Using containers on AWS services
----------------------------------

:ref:`Amazon EKS<eks_flow>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`Amazon ECS<ecs_flow>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`Amazon SageMaker<sagemaker_flow>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`AWS Batch<aws_batch_flow>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Customizing Neuron Deep Learning Containers
-------------------------------------------
Deep Learning Containers can be customized to fit your specific project needs.
To read more, visit :ref:`containers-dlc-then-customize-devflow`.
