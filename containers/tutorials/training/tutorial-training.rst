.. _tutorial-training:

Run training in Pytorch Neuron container
========================================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

This tutorial demonstrates how to run a pytorch container on an trainium instance.

By the end of this tutorial you will be able to run simple mlp training using the container

You will use an trn1.2xlarge to test your Docker configuration for Trainium.

To find out the available neuron devices on your instance, use the command ``ls /dev/neuron*``.

Setup Environment
-----------------

1. Launch an Trn1 Instance
	.. include:: /neuron-intro/install-templates/launch-trn1-ami.rst

2. Set up docker environment according to :ref:`tutorial-docker-env-setup`

3. A sample Dockerfile for for torch-neuron can be found here :ref:`trainium-dlc-dockerfile`.
This dockerfile needs the mlp train script found here  :ref:`mlp-train` 

With the files in a dir, build the image with the following command:

.. code:: bash

   docker build . -f Dockerfile.pt -t neuron-container:pytorch

Run the following command to start the container

.. code:: bash

   docker run -it --name pt-cont --net=host --device=/dev/neuron0 neuron-container:pytorch python3 /opt/ml/mlp_train.py






