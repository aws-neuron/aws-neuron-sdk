.. _dlc-then-eks-devflow:

Deploy  Neuron Container on Elastic Kubernetes Service (EKS)
============================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

   
Description
-----------

|image|
 
.. |image| image:: /images/dlc-on-eks-dev-flow.png
   :width: 750
   :alt: Neuron developer flow for DLC on ECS
   :align: middle

You can use the Neuron version of the `AWS Deep Learning Containers <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs-tutorials-inference.html>`_ to run inference on Amazon Elastic Kubernetes Service (EKS). In this developer flow, you set up an EKS cluster with Inf1 instances, create a Kubernetes manifest for your inference service and deploy it to your cluster. This developer flow assumes:

1. The model has already been compiled through :ref:`Compilation with Framework API on EC2 instance <ec2-then-ec2-devflow>` or through :ref:`Compilation with Sagemaker Neo <neo-then-hosting-devflow>`. 

2. You already set up your container to retrieve it from storage.

.. _dlc-then-eks-setenv:

Setup Environment
-----------------

1. Install pre-requisits:
	Follow `these instruction <https://docs.aws.amazon.com/eks/latest/userguide/eksctl.html>`_ to install or upgrade the *eksctl* command line utility on your local computer.

	Follow `these instruction <https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html>`_ to install *kubectl* in the same computer. *kubectl* is a command line tool for working with Kubernetes clusters.


2. Follow the instructions in this `EKS documentation link <https://docs.aws.amazon.com/eks/latest/userguide/inferentia-support.html>`_ to set up AWS Inferentia on your EKS cluster.
	Using the YML deployment manifest shown `in the same link <https://docs.aws.amazon.com/eks/latest/userguide/inferentia-support.html#deploy-tensorflow-serving-application>`_, replace the `image` in the `containers` specification with the one you built using :ref:`how-to-build-neuron-container` above.

	.. note::

    	Before deploying your task definition to your EKS cluster, make sure to push the image to ECR. Refer to `Pushing a Docker image <https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html>`_ for more information.


Self-managed Kubernetes
~~~~~~~~~~~~~~~~~~~~~~~

Please refer to :ref:`tutorial-k8s-env-setup-for-neuron`. In :ref:`example-deploy-rn50-as-k8s-service`, the
container image referenced in the YML manifest is created using :ref:`how-to-build-neuron-container`.
