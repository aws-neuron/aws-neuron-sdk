.. _dlc-then-ec2-devflow:

Customize Neuron DLC
==============================

.. contents:: Table of Contents
   :local:
   :depth: 2


Description
-----------

This guide covers how to customize and extend the Neuron Deep Learning Container (DLC) to fit your specific project needs. You can customize the DLC either by using the DLC as a base image in your Dockerfile or by modifying published Dockerfiles on GitHub.

Method 1: Using DLC as a Base Image
-----------------

1. Create a New Dockerfile. In your Dockerfile, specify the Neuron DLC as your base image using the FROM directive.

2. Complete the Dockerfile. You can add additional packages, change the base environment, or any other modifications that suit your project. `AWS Batch Training <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/devflows/training/batch/batch-training.html#batch-training>`_ is a good example which needs customize Neuron DLC by using it as the base image. From its `Dockerfile <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/aws-batch/llama2/docker/Dockerfile>`_, we can find the customized container copies llama_batch_training.sh to the container and runs it.

3. Navigate to the directory containing your Dockerfile and build your custom container.

Method 2: Modifying Published Dockerfiles
-----------------

1. Visit the `Neuron DLC Github repo <https://github.com/aws-neuron/deep-learning-containers>`_ and locate the Dockerfile for the container you wish to customize.

2. Modify the Dockerfile as needed. You can add additional packages, change the base environment, or any other modifications that suit your project. For example, if you do not need to use Neuron tools in your scenario and want to make the container smaller, you can remove aws-neuronx-tools at this `line <https://github.com/aws-neuron/deep-learning-containers/blob/a969c77fdba17ff8d35f411b39ce3a9bc6368730/docker/pytorch/inference/2.1.1/Dockerfile.neuronx#L64>`_.

3. Navigate to the directory containing your Dockerfile and build your custom container.
