.. _dlc-then-ecs-devflow:

Deploy with DLC on Elastic Container Service (ECS)
==================================================

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

You can use the Neuron version of the `AWS Deep Learning Containers <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs-tutorials-inference.html>`_ to run inference on Amazon Elastic Container Service (ECS). In this developer flow, you set up an ECS cluster with inf1 instances, create a task description for your inference service and deploy it to your cluster. This developer flow assumes:

1. The model has already been compiled through :ref:`Compilation with Framework API on EC2 instance <ec2-then-ec2-devflow>` or through :ref:`Compilation with Sagemaker Neo <neo-then-hosting-devflow>`. 

2. You already set up your container to retrieve it from storage.

.. _dlc-then-ecs-setenv:

Setup Environment
-----------------


1. Set up an Amazon ECS cluster:
	Follow the instructions on `Setting up Amazon ECS for Deep Learning Containers <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs-setting-up-ecs.html>`_

2. Define an Inference Task:
	Use the instruction on the `DLC Inference on ECS Tutorial <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs-tutorials-inference.html>`_ to define a task and create a service for the appropriate framework.

	When creating tasks for inf1 instances on ECS, be aware of the considerations and requirements listed in `Working with inference workloads on Amazon ECS <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-inference.html>`_. 


