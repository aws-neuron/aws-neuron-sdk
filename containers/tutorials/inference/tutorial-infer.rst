.. _tutorial-infer:

Run Inference in PyTorch Neuron Container
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

2. Set up docker environment according to :ref:`tutorial-docker-env-setup`

3. Clone the `aws-neuron/deep-learning-containers<https://github.com/aws-neuron/deep-learning-containers>`_ GitHub repository and use one of the PyTorch inference Dockerfiles found in the folders of the repo:

.. code:: bash

   git clone https://github.com/aws-neuron/deep-learning-containers.git
   cd deep-learning-containers/docker/pytorch/inference/2.7.0

For additional prerequisites and setup requirements, see the `docker build prerequisites <https://github.com/aws-neuron/deep-learning-containers/blob/main/README.md#prerequisites>`_.

This tutorial requires the `torchserve entrypoint <https://github.com/aws-neuron/deep-learning-containers/blob/main/docker/common/torchserve-neuron.sh>`_ and `torchserve config.properties <https://github.com/aws-neuron/deep-learning-containers/blob/main/docker/common/config.properties>`_ which are copied over to the same parent folder as part of prerequisites.

With the files in a local directory, build the image with the following command:

.. code:: bash

   docker build . -f Dockerfile.neuronx -t neuron-container:pytorch

Run the following command to start the container

.. code:: bash

   docker run -itd --name pt-cont -p 80:8080 -p 8081:8081 --device=/dev/neuron0 neuron-container:pytorch /usr/local/bin/entrypoint.sh -m 'pytorch-resnet-neuron=https://aws-dlc-sample-models.s3.amazonaws.com/pytorch/Resnet50-neuron.mar' -t /home/model-server/config.properties