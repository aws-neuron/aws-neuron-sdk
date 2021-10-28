.. _how-to-build-neuron-container:

How to Build a Neuron Container
===============================

Introduction
------------

This document explains how to build a Neuron Container using an existing Dockerfile.

Pre-requisites
--------------
#. Docker version 18 or newer is configured according to :ref:`tutorial-docker-env-setup`
#. Inf1 instance with available :ref:`Neuron Devices<container-devices>`
#. If running a serving application such as tensorflow-model-server, torchserve or multi-model-server, make sure the appropriate ports that the server listens to are exposed using EXPOSE in the Dockerfile or the arguments ``-p 80:8080`` on the ``docker run`` command.

.. _running-application-container:

Build and Run the Application Container
---------------------------------------
Follow the steps below for creating neuron application containers. If there were already existing containers that are packaged as per :ref:`packaging-neuron-rt-containers.rst` refer the :ref:`containers-migration-to-runtime2`

#. Build the container using :ref:`libmode-dockerfile`
#. Run the container locally:

.. code:: bash

   docker run -it --name pt17 -p 80:8080 -e "AWS_NEURON_VISIBLE_DEVICES=ALL"  neuron-container:pytorch neuron-top

Important to know
-----------------

.. _container-devices:

Devices
^^^^^^^

There are currently two ways to specify Neuron Devices to a container.

#. The docker native way is to use --device /dev/neuron# for each of the Neuron Devices intended to be passed. When using --device option ALL/all is not supported.

    .. code:: bash

        docker run --device=/dev/neuron0 --device=/dev/neuron1

#. If you install the aws-neuron-runtime-base package, you will have an OCI hook that also supports use of a container environment variable AWS_NEURON_VISIBLE_DEVICES=<ALL | csv of devices>, which intends to make things easier for multi device scenarios. Following are some examples

    .. code:: bash

        docker run -e “AWS_NEURON_VISIBLE_DEVICES=0,1”
        docker run -e “AWS_NEURON_VISIBLE_DEVICES=ALL”

#. Multiple container applications running in the same host can share the devices but the cores cannot be shared. This is similar to running multiple applications in the host.
