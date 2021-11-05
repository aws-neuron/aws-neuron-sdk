.. _tutorial-docker-env-setup:

Docker environment setup
========================

Introduction
------------

A Neuron application can be deployed using docker containers. This
tutorial describes how to configure docker to expose Inferentia devices
to containers.

Once the environment is setup, a container can be started with
*AWS_NEURON_VISIBLE_DEVICES* environment variable to specify desired set
of Inferentia devices to be exposed to the container.
AWS_NEURON_VISIBLE_DEVICES is a set of contiguous comma-seperated
inferentia logical ids. To find out the available logical ids on your
instance, run the neuron-ls tool. For example, on inf1.6xlarge instance
with 4 inferentia devices, you may set AWS_NEURON_VISIBLE_DEVICES="2,3"
to expose the last two devices to a container. When running neuron-ls
inside a container, you will only see the set of exposed Inferentias.
For example:

.. code:: bash

   docker run --env AWS_NEURON_VISIBLE_DEVICES="0" neuron-test neuron-ls

Would produce the following output:

::

   +--------------+---------+--------+-----------+-----------+------+------+
   |   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   | EAST | WEST |
   |              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 |      |      |
   +--------------+---------+--------+-----------+-----------+------+------+
   | 0000:00:1f.0 |       0 |      4 | 4096 MB   | 4096 MB   |    0 |    0 |
   +--------------+---------+--------+-----------+-----------+------+------+

Steps:
------

This tutorial starts from a fresh Ubuntu 18

Step 1: Install Neuron driver and aws-neuron-runtime-base on the Linux host
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important::

   This step should run on the Linux host and not inside the container.

Configure Linux for Neuron repository updates, install Neuron driver and the aws-neuron-runtime-base
package.

.. code:: bash

   # Configure Linux for Neuron repository updates
   . /etc/os-release
   sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
   deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
   EOF
   wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

   # Update OS packages
   sudo apt-get update -y

   ###############################################################################################################
   # Before installing or updating aws-neuron-dkms:
   # - Stop any existing Neuron runtime 1.0 daemon (neuron-rtd) by calling: 'sudo systemctl stop neuron-rtd'
   ###############################################################################################################

   ################################################################################################################
   # To install or update to Neuron versions 1.16.0 and newer from previous releases:
   # - DO NOT skip 'aws-neuron-dkms' install or upgrade step, you MUST install or upgrade to latest Neuron driver
   ################################################################################################################

   # Install OS headers
   sudo apt-get install linux-headers-$(uname -r) -y

   # Install Neuron Driver
   sudo apt-get install aws-neuron-dkms -y

   sudo apt-get install aws-neuron-runtime-base -y


Step 2: install oci-add-hooks dependency on the Linux host
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important::

   This step should run on the Linux host and not inside the container.
   

`oci-add-hooks <https://github.com/awslabs/oci-add-hooks>`__ is an OCI
runtime with the sole purpose of injecting OCI prestart, poststart, and
poststop hooks into a container config.json before passing along to an
OCI compatable runtime. oci-add-hooks is used to inject a hook that
exposes Inferentia devices to the container.

.. code:: bash

   sudo apt install -y golang && \
       export GOPATH=$HOME/go && \
       go get github.com/joeshaw/json-lossless && \
       cd /tmp/ && \
       git clone https://github.com/awslabs/oci-add-hooks && \
       cd /tmp/oci-add-hooks && \
       make build && \
       sudo cp /tmp/oci-add-hooks/oci-add-hooks /usr/local/bin/

.. _step-3-setup-docker-to-use-oci-neuron-oci-runtime:

Step 3: setup Docker to use oci-neuron OCI runtime.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

oci-neuron is a script representing OCI compatible runtime. It wraps
oci-add-hooks, which wraps runc. In this step, we configure docker to
point at oci-neuron OCI runtime. Install dockerIO:

.. code:: bash

   sudo apt install -y docker.io
   sudo usermod -aG docker $USER

Logout and log back in to refresh membership. Place daemon.json Docker
configuration file supplied by Neuron SDK in default location. This file
specifies oci-neuron as default docker runtime:

.. code:: bash

   sudo cp /opt/aws/neuron/share/docker-daemon.json /etc/docker/daemon.json
   sudo service docker restart

If the docker restart command fails, make sure to check if the docker
systemd service is not masked. More information on this can be found
here: https://stackoverflow.com/a/37640824

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

Build a docker image using provided dockerfile :ref:`libmode-dockerfile`, and use to
verify whitelisting:

.. code:: bash

   docker build . -f Dockerfile.app -t neuron-test

Then run:

.. code:: bash

   docker run --env AWS_NEURON_VISIBLE_DEVICES="0"  neuron-test neuron-ls

Expected result:

::

   +--------------+---------+--------+-----------+-----------+------+------+
   |   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   | EAST | WEST |
   |              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 |      |      |
   +--------------+---------+--------+-----------+-----------+------+------+
   | 0000:00:1f.0 |       0 |      4 | 4096 MB   | 4096 MB   |    0 |    0 |
   +--------------+---------+--------+-----------+-----------+------+------+
