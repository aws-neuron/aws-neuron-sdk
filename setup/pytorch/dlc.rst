.. meta::
   :description: Install PyTorch Neuron using AWS Deep Learning Containers on Inf2, Trn1, Trn2, Trn3
   :keywords: pytorch, neuron, dlc, deep learning containers, docker, installation, vllm, inference, training
   :framework: pytorch
   :installation-method: dlc
   :instance-types: inf2, trn1, trn2, trn3
   :content-type: installation-guide
   :estimated-time: 10 minutes
   :date-modified: 2026-03-30

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
   * - PyTorch Inference (NeuronX)
     - Model serving on Inf2/Trn1/Trn2/Trn3
     - `Inference images <https://github.com/aws-neuron/deep-learning-containers#pytorch-inference-neuronx>`__
   * - PyTorch Inference vLLM (NeuronX)
     - LLM serving with vLLM
     - `vLLM images <https://github.com/aws-neuron/deep-learning-containers#vllm-inference-neuronx>`__
   * - PyTorch Training (NeuronX)
     - Model training on Trn1/Trn2/Trn3
     - `Training images <https://github.com/aws-neuron/deep-learning-containers#pytorch-training-neuronx>`__
   * - PyTorch Inference (Neuron)
     - Legacy inference on Inf1
     - `Inf1 images <https://github.com/aws-neuron/deep-learning-containers#pytorch-inference-neuron>`__

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Requirement
     - Details
   * - Instance Type
     - Inf2, Trn1, Trn2, or Trn3
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

   # Authenticate with ECR
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

   # Pull the vLLM inference container
   docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.20.2-ubuntu20.04

   # Run with Neuron device access
   docker run -it --device=/dev/neuron0 \
     763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.20.2-ubuntu20.04

For the latest image tags and a step-by-step walkthrough, see
:doc:`/deploy/environments/quickstart-deploy-dlc`.

Quick Start: Training Container
--------------------------------

.. code-block:: bash

   # Authenticate with ECR
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

   # Pull the training container
   docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-neuronx:2.1.2-neuronx-py310-sdk2.20.2-ubuntu20.04

   # Run with all Neuron devices
   docker run -it --device=/dev/neuron0 --device=/dev/neuron1 \
     763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-neuronx:2.1.2-neuronx-py310-sdk2.20.2-ubuntu20.04

.. note::
   The image tags above are examples. For the latest available images, see the
   `Neuron DLC repository <https://github.com/aws-neuron/deep-learning-containers>`__.

Customizing a DLC
-----------------

You can extend a Neuron DLC with additional packages by creating a custom Dockerfile:

.. code-block:: dockerfile

   FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.20.2-ubuntu20.04

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
      :link: /deploy/ec2/inference
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
- :doc:`/frameworks/torch/training-torch-neuronx` - Training tutorials
- :doc:`/frameworks/torch/inference-torch-neuronx` - Inference tutorials
