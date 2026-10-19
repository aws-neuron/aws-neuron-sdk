.. meta::
   :description: Install PyTorch Neuron using AWS Deep Learning Containers on Trn2 and Trn3
   :keywords: pytorch, neuron, dlc, deep learning containers, docker, installation, vllm, inference, training
   :framework: pytorch
   :installation-method: dlc
   :instance-types: trn2, trn3
   :content-type: installation-guide
   :estimated-time: 10 minutes
   :date-modified: 2026-10-16

Install PyTorch via Deep Learning Container
=============================================

Deploy PyTorch with Neuron support using pre-configured Docker images from AWS ECR.

⏱️ **Estimated time**: 10 minutes

.. note::
   For a non-containerized setup, consider the :doc:`DLAMI-based installation <dlami>` or
   :doc:`manual installation <manual>` instead.

----

What are Neuron DLCs?
---------------------

AWS Neuron Deep Learning Containers (DLCs) are pre-configured Docker images with the Neuron SDK
and ML frameworks pre-installed. They provide Docker-based isolation, reproducibility, and
portability across deployment platforms including EC2, EKS, ECS, and SageMaker.

Available PyTorch Neuron DLC images:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Container Type
     - Use Case
     - Links
   * - PyTorch Inference vLLM (NeuronX)
     - LLM serving with vLLM
     - `vLLM images <https://github.com/aws-neuron/deep-learning-containers#vllm-inference-neuronx>`__
   * - PyTorch Training (TorchNeuron)
     - Native PyTorch training with ``torch-neuronx`` 2.14 and TorchTitan
     - `torch-neuronx images <https://github.com/aws-neuron/deep-learning-containers#torch-neuronx>`__

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Requirement
     - Details
   * - Instance Type
     - Trn2 or Trn3
   * - Docker
     - Docker Engine installed and running
   * - AWS CLI
     - Configured with ECR access permissions
   * - Neuron Driver
     - ``aws-neuronx-dkms`` installed on the host

Quick Start: vLLM Inference Container
--------------------------------------

The fastest way to get started with LLM inference on Neuron:

.. code-block:: bash

   # Pull the vLLM inference container
   docker pull public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04

   # Run with Neuron device access
   docker run -it --device=/dev/neuron0 \
     public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04

For the latest image tags and a step-by-step walkthrough, see
:doc:`/deploy/environments/quickstart-deploy-dlc`.

.. note::
   The image tags above are examples. For the latest available images, see the
   `Neuron DLC repository <https://github.com/aws-neuron/deep-learning-containers>`__.

.. note::
   Starting with Neuron 2.33.0, Neuron DLC images are published on Ubuntu 26.04 only.
   Image tags carry a ``-ubuntu26.04`` suffix (for example,
   ``...-neuronx-py314-sdk2.33.0-ubuntu26.04``). If you need an Ubuntu 24.04 image, use a
   tag from Neuron 2.32.0 or earlier. Check the
   `Neuron DLC repository <https://github.com/aws-neuron/deep-learning-containers>`__
   for the exact published tags.

Customizing a DLC
-----------------

You can extend a Neuron DLC with additional packages by creating a custom Dockerfile:

.. code-block:: dockerfile

   FROM public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04

   # Install additional packages
   RUN pip install transformers datasets

   # Copy your application code
   COPY app/ /app/

For more details, see :doc:`/deploy/environments/customize-dlc`.

Deployment Platforms
--------------------

Neuron DLCs can be deployed across multiple AWS services:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Amazon EC2
      :link: /deploy/ec2/inference-dlc
      :link-type: doc
      :class-card: sd-rounded-3

      Deploy containers directly on EC2 instances with Neuron devices.

   .. grid-item-card:: Amazon EKS
      :link: /deploy/eks/inference
      :link-type: doc
      :class-card: sd-rounded-3

      Run containers on managed Kubernetes with the Neuron device plugin.

   .. grid-item-card:: Amazon ECS
      :link: /deploy/ecs/inference
      :link-type: doc
      :class-card: sd-rounded-3

      Deploy containers using Amazon Elastic Container Service.

Next Steps
----------

- :doc:`/deploy/environments/quickstart-deploy-dlc` - Full vLLM DLC deployment walkthrough
- :doc:`/deploy/environments/dlc-images` - Find the right DLC image for your workload
- :doc:`/deploy/index` - Full containers documentation
- :doc:`TorchNeuron tutorials </torch/tutorials/index>` - Training and inference with native PyTorch on Trainium
- :doc:`vLLM Neuron online serving quickstart </vllm-neuron/docs/getting-started/quickstart-online-serving>` - Serve an LLM with the vLLM DLC
