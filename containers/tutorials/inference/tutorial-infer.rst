.. _tutorial-infer:

Run inference in pytorch neuron container
==========================================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

This tutorial demonstrates how to run a pytorch DLC on an inferentia instance.

By the end of this tutorial you will be able to run the inference using the container

You will use an inf1.2xlarge to test your Docker configuration for Inferentia.

To find out the available neuron devices on your instance, use the command ``ls /dev/neuron*``.

Setup Environment
-----------------

1. Launch an Inf1 Instance
	.. include:: /neuron-intro/install-templates/launch-inf1-ami.rst

2. Set up docker environment according to :ref:`tutorial-docker-env-setup`

3. A sample Dockerfile for for torch-neuron can be found here :ref:`inference-dlc-dockerfile`.
This dockerfile needs the torchserve entrypoint found here :ref:`torchserve-neuron` and torchserve
config.properties found here :ref:`torchserve-config-properties`. 

With the files in a dir, build the image with the following command:

.. code:: bash

   docker build . -f Dockerfile.pt -t neuron-container:pytorch

Run the following command to start the container

.. code:: bash

   docker run -itd --name pt-cont -p 80:8080 -p 8081:8081 --device=/dev/neuron0 neuron-container:pytorch /usr/local/bin/entrypoint.sh -m 'pytorch-resnet-neuron=https://aws-dlc-sample-models.s3.amazonaws.com/pytorch/Resnet50-neuron.mar' -t /home/model-server/config.properties






