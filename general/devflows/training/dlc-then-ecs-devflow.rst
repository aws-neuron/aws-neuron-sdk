.. _training-dlc-then-ecs-devflow:

Deploy Neuron Container on Elastic Container Service (ECS) for Training
=======================================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

   
Description
-----------

|image|
 
.. |image| image:: /images/dlc-on-ecs-dev-flow.png
   :width: 750
   :alt: Neuron developer flow for DLC on ECS
   :align: middle

You can use the Neuron version of the `AWS Deep Learning Containers <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs-tutorials-training.html>`_ to run training on Amazon Elastic Container Service (ECS). In this developer flow, you set up an ECS cluster with trn1 instances, create a task description for your training container and deploy it to your cluster. This developer flow assumes:

1. The model has already been compiled through :ref:`Compilation with Framework API on EC2 instance <ec2-training>` or through :ref:`Compilation with Sagemaker Neo <neo-then-hosting-devflow>`.

2. You already set up your container to retrieve it from storage.

.. _training-dlc-then-ecs-setenv:

Setup Environment
-----------------


1. Set up an Amazon ECS cluster:
	Follow the instructions on `Setting up Amazon ECS for Deep Learning Containers <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs-setting-up-ecs.html>`_

2. Define a Training Task:
	Use the instruction on the `DLC Training on ECS Tutorial <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs-tutorials-training.html>`_ to define a task and create a service for the appropriate framework.

	When creating tasks for trn1 instances on ECS, be aware of the considerations and requirements listed in `Working with training workloads on Amazon ECS <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-inference.html>`_.


3. Use the container image created using :ref:`how-to-build-neuron-container` as the ``image`` in your task definition.

   .. _training_push_to_ecr_note:

   .. note::

       Before deploying your task definition to your ECS cluster, make sure to push the image to ECR. Refer to `Pushing a Docker image <https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html>`_ for more information.
