.. _tutorial-docker-env-setup-for-neuron:

Docker environment setup for Neuron on EC2
==========================================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

This tutorial demonstrates how to configure `Docker on an EC2 instance <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html>`_ to expose Inferentia devices to containers.

By the end of this tutorial you will be able to run the Neuron Runtime inside a container and specify the desired set of Inferentia devices using the Docker flag ``--device=/dev/neuron#``, where # is a available Neuron Device.

You will use an inf1.2xlarge to test your Docker configuration for Inferentia.

To find out the available neuron devices on your instance, use the command ``ls /dev/neuron*``.

When running neuron-ls inside a container, you will only see the set of
exposed Inferentias. For example:

.. code:: bash

   docker run --device=/dev/neuron0 neuron-test neuron-ls

Would produce the following output:

::

   +--------+--------+--------+-----------+--------------+---------+---------+---------+
   | NEURON | NEURON | NEURON | CONNECTED |     PCI      | RUNTIME | RUNTIME | RUNTIME |
   | DEVICE | CORES  | MEMORY |  DEVICES  |     BDF      | ADDRESS |   PID   | VERSION |
   +--------+--------+--------+-----------+--------------+---------+---------+---------+
   | 0      | 4      | 8 GB   | 1         | 0000:00:1c.0 | NA      | 6       |  NA     |
   +--------+--------+--------+-----------+--------------+---------+---------+---------+

Run the tutorial
----------------

Start by setting up an environment as described in :ref:`ec2-then-ec2-setenv` section. While selecting the instance AMI you can select an `Amazon Linux AMI <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/amazon-linux-ami-basics.html>`_, instead of the AWS Deep Learning AMI. 

Step 1: Set up Neuron on the host instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you are able to connect to your instance, start by installing the Neuron Kernel Modules as described in :ref:`pytorch-pip-al2`. 

The neuron-rtd will run on the hostafter installation. Stop the neuron-rtd service before starting a containerized neuron-rtd. This is needed to allow assignment of devices to containers:

.. code:: bash

   sudo service neuron-rtd stop

Step 2: Install and run docker daemon
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install Docker in your Amazon Linux 2 instance using the command:

.. code:: bash

   sudo yum -y install docker
   sudo usermod -aG docker $USER

If running on a Ubuntu AMI, refer to the official `Docker installation documentation. <https://docs.docker.com/engine/install/ubuntu/>`_

Logout and log back in to refresh membership. To verify your docker installation, run a simple ``hello-work`` Docker container with:

.. code:: bash

   docker run hello-world

Expected result:

::

   Hello from Docker!
   This message shows that your installation appears to be working correctly.

   To generate this message, Docker took the following steps:
   1. The Docker client contacted the Docker daemon.
   2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
   (amd64)
   3. The Docker daemon created a new container from that image which runs the
   executable that produces the output you are currently reading.
   4. The Docker daemon streamed that output to the Docker client, which sent it
   to your terminal.

   To try something more ambitious, you can run an Ubuntu container with:
   $ docker run -it ubuntu bash

   Share images, automate workflows, and more with a free Docker ID:
   https://hub.docker.com/

   For more examples and ideas, visit:
   https://docs.docker.com/get-started/


Step 3: Build and Run a Neuron Docker image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using DockerFile for the rutime only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build a docker image using provided dockerfile :ref:`neuron-runtime-dockerfile` and use to
verify whitelisting:

.. code:: bash
   wget https://raw.githubusercontent.com/aws/aws-neuron-sdk/master/neuron-deploy/docker-example/Dockerfile.neuron-rtd
   docker build . -f Dockerfile.neuron-rtd -t neuron-test

Then run:

.. code:: bash

   docker run --device=/dev/neuron0  neuron-test neuron-ls

Expected result:

::

   +--------+--------+--------+-----------+--------------+---------+---------+---------+
   | NEURON | NEURON | NEURON | CONNECTED |     PCI      | RUNTIME | RUNTIME | RUNTIME |
   | DEVICE | CORES  | MEMORY |  DEVICES  |     BDF      | ADDRESS |   PID   | VERSION |
   +--------+--------+--------+-----------+--------------+---------+---------+---------+
   | 0      | 4      | 8 GB   | 1         | 0000:00:1c.0 | NA      | 6       |  NA     |
   +--------+--------+--------+-----------+--------------+---------+---------+---------+

Using DLC Neuron Image
~~~~~~~~~~~~~~~~~~~~~~
Login to DLC repo with the following command:

.. code:: bash

        aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

Pull the docker image for your framework of choice. Images can be found `here <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-inference-containers>`_. The following examples pulls a TensorFlow Neuron inference image:

.. code:: bash

        docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-neuron:1.15.5-neuron-py37-ubuntu18.04

After download finishes, the expected result for the command ``docker images`` is:

::
        
       763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-neuron 1.15.5-neuron-py37-ubuntu18.04  44c7584a6115 5 days ago 3.03GB

With the image available locally, you can Tag it and run it. The bellow example shows the ``docker run...`` command for the models used of :ref:`tensorflow-serving` tutorial.

.. code:: bash

        docker tag 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-neuron:1.15.5-neuron-py37-ubuntu18.04 tf-dlc

        docker run -it --name tf  -p 8500:8500 --device=/dev/neuron0 --net=host  --cap-add IPC_LOCK --mount type=bind,source=<saved_model_location>,target=/models/<model_name> -e -e MODEL_NAME=<model_name> tf-dlc

Building your own container
~~~~~~~~~~~~~~~~~~~~~~~~~~~
A sample Dockerfile for for torch-neuron can be found here :ref:`torch-neuron-dockerfile`. This Dockerfile requires an entrypoint sript, to start the runtime with the ``docker run ...`` command. 

Download the Dockerfile and the entripoint script first, then build the image with the following command:

.. code:: bash

   wget https://raw.githubusercontent.com/aws/aws-neuron-sdk/master/neuron-deploy/docker-example/Dockerfile.torch-neuron
   wget https://raw.githubusercontent.com/aws/aws-neuron-sdk/master/neuron-deploy/docker-example/dockerd-entrypoint.sh

   docker build . -f Dockerfile.torch-neuron -t torch-neuron

You can change the Neuron framework installation on the container and add your own application code, by modifying lines 35 to 40 on the :ref:`torch-neuron-dockerfile`. 

Run the following command to execute ``neuron-top`` on the container: 

.. code:: bash

   docker run -it --device=/dev/neuron0 --cap-add IPC_LOCK torch-neuron neuron-top






