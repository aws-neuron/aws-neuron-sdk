.. _tutorial-docker-env-setup-for-neuron:

Tutorial: Docker environment setup for Neuron
=============================================

Introduction
------------

A Neuron application can be deployed using docker containers. This
tutorial describes how to configure docker to expose Inferentia devices
to containers.

Once the environment is setup, a container can be started with
--device=/dev/neuron# to specify desired set of Inferentia devices to be
exposed to the container. To find out the available neuron devices on
your instance, use the command ``ls /dev/neuron*``.

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

Steps:
------

This tutorial starts from a fresh Ubuntu Server 16.04 LTS AMI
"ami-08bc77a2c7eb2b1da".

Step 1: Make sure that the neuron-rtd service is not running
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If neuron-rtd is running on the host, stop the neuron-rtd service before
starting the containerized neuron-rtd. This is needed to allow
assignment of devices to containers:

.. code:: bash

   sudo service neuron-rtd stop

Step 2: Install and run docker daemon
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   sudo apt install -y docker.io
   sudo usermod -aG docker $USER

Logout and log back in to refresh membership. Place daemon.json Docker
configuration file supplied by Neuron SDK in default location. This file
specifies oci-neuron as default docker runtime:

.. code:: bash

   sudo service docker restart

If the docker restart command fails, make sure to check if the docker
systemd service is not masked. More information on this can be found
here:
`https://stackoverflow.com/a/37640824 <https://stackoverflow.com/a/37640824>`__

Verify docker:

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

Step 3: Run Neuron Docker image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using DockerFile
~~~~~~~~~~~~~~~~
Build a docker image using provided dockerfile :ref:`neuron-runtime-dockerfile` and use to
verify whitelisting:

.. code:: bash

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
Login to DLC repo

.. code:: bash

        aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

Pull the framework docker image. Images can be found `here <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-inference-containers>`_

.. code:: bash

        docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-neuron:1.15.5-neuron-py37-ubuntu18.04

Expected result for cmd ``docker images``

::
        
       763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-neuron 1.15.5-neuron-py37-ubuntu18.04  44c7584a6115 5 days ago 3.03GB

Tag the docker image

.. code:: bash

        docker tag 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-neuron:1.15.5-neuron-py37-ubuntu18.04 tf-dlc

Run the docker -

.. code:: bash

        sudo docker run -it --name tf  -p 8500:8500 --device=/dev/neuron0 --net=host  --cap-add IPC_LOCK --mount type=bind,source=<saved_model_location>,target=/models/<model_name> -e -e MODEL_NAME=<model_name> tf-dlc
