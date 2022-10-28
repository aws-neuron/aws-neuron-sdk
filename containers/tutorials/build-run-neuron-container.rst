.. _how-to-build-neuron-container:

Tutorial How to Build and Run a Neuron Container
================================================

Introduction
------------

This document explains how to build a Neuron Container using an existing Dockerfile.

Pre-requisites
--------------
#. Docker version 18 or newer is configured according to :ref:`tutorial-docker-env-setup`
#. Inf1/Trn1 instance with available :ref:`Neuron Devices<container-devices>`
#. If running a serving application such as tensorflow-model-server, torchserve or multi-model-server, make sure the appropriate ports that the server listens to are exposed using EXPOSE in the Dockerfile or the arguments ``-p 80:8080`` on the ``docker run`` command.

.. _running-application-container:

Build and Run the Application Container
---------------------------------------
Follow the steps below for creating neuron application containers.

- Build a docker image using provided dockerfile :ref:`libmode-dockerfile` for Inf1 and :ref:`trainium-dlc-dockerfile` for Trn1 (also for Trn1 the dockerfile needs mlp train script found here at :ref:`mlp-train`

.. code:: bash

   docker build . -f Dockerfile.pt -t neuron-container:pytorch

- Run the container locally:

.. code:: bash

   docker run -it --name pt17 --device=/dev/neuron0 neuron-container:pytorch neuron-ls

Expected result for Inf1:

::

   +--------------+---------+--------+-----------+-----------+------+------+
   |   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   | EAST | WEST |
   |              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 |      |      |
   +--------------+---------+--------+-----------+-----------+------+------+
   | 0000:00:1f.0 |       0 |      4 | 4096 MB   | 4096 MB   |    0 |    0 |
   +--------------+---------+--------+-----------+-----------+------+------+

Expected result for Trn1:

::

   +--------+--------+--------+-----------+---------+
   | NEURON | NEURON | NEURON | CONNECTED |   PCI   |
   | DEVICE | CORES  | MEMORY |  DEVICES  |   BDF   |
   +--------+--------+--------+-----------+---------+
   | 0      | 4      | 8 GB   | 1         | 00:1f.0 |
   +--------+--------+--------+-----------+---------+


.. note::

   If instead of the --device option above if the env variable AWS_NEURON_VISIBLE_DEVICES
   is to be used then the oci hook needs to installed by following instructions in :ref:`tutorial-oci-hook`


Important to know
-----------------

.. _container-devices:

Devices
^^^^^^^

- The docker native way is to use --device /dev/neuron# for each of the Neuron Devices intended to be passed. When using --device option ALL/all is not supported.

    .. code:: bash

        docker run --device=/dev/neuron0 --device=/dev/neuron1

- If you install the aws-neuronx-oci-hook package, you will have an OCI hook that also supports use of a container environment variable AWS_NEURON_VISIBLE_DEVICES=<ALL | csv of devices>, which intends to make things easier for multi device scenarios. Following are some examples. For setting up oci hook please refer :ref:`oci neuron hook <tutorial-oci-hook>`

    .. code:: bash

        docker run -e “AWS_NEURON_VISIBLE_DEVICES=0,1”
        docker run -e “AWS_NEURON_VISIBLE_DEVICES=ALL”

- In kubernetes environment, the neuron device plugin is used for exposing the neuron device to the containers in the pod. The number of devices can be adjusted using the *aws.amazon.com/neurondevice* resource in the pod specification. Refer :ref:`K8s setup <tutorial-k8s-env-setup-for-neuron>` for more details

    .. code:: bash

         resources:
            limits:
            aws.amazon.com/neurondevice: 1

   .. note::

      Only the number of devices can be specfied.
      When only the neuron device plugin is running that does not guaratee the devices to be
      contiguous. Make sure to run the neuron scheduler extension :ref:`neuron-k8-scheduler-ext`
      so that it makes sure that contigiuous devices are allocated to the containers


- Multiple container applications running in the same host can share the devices but the cores cannot be shared. This is similar to running multiple applications in the host. 
- In the kubernetes environment the devices cannot be shared by multiple containers in the pod

.. _container-cores:

Cores
^^^^^
Each neuron device has multiple cores. The cores allocated to process/container can be controlled by
the environment variable NEURON_RT_VISIBLE_CORES and NEURON_RT_NUM_CORES. Please refer :ref:`nrt-configuration` for more details.

- The docker native way is to use --device /dev/neuron# for each of the Neuron Devices intended to be passed. Add --env NEURON_RT_VISIBLE_CORES-1,2 to use cores 1 and 2 to this container. For example in inf1.24xlarge with 64 cores, if we want to use cores 51 & 52, the appropriate device and NEURON_RT_VISIBLE_CORES needs to be used. With 4 cores in each device, core 51 is in device 12 and 52 is in device 13

    .. code:: bash

        docker run --device=/dev/neuron12 --device=/dev/neuron13 --env NEURON_RT_VISIBLE_CORES=51,52

- In kubernetes environment, the neuron device plugin is used for exposing the neuron cores to the containers in the pod. The number of cores can be adjusted using the *aws.amazon.com/neuroncore* resource in the pod specification. Refer :ref:`K8s setup <tutorial-k8s-env-setup-for-neuron>` for more details.

    .. code:: bash

         resources:
            limits:
            aws.amazon.com/neuroncore: 1

   .. note::

      Only the number of cores can be specfied.
      When only the neuron device plugin is running that does not guaratee the cores to be
      contiguous. Make sure to run the neuron scheduler extension :ref:`neuron-k8-scheduler-ext`
      so that it makes sure that contigiuous cores are allocated to the containers

- Multiple container applications running in the same host cannot share the cores. This is similar to running multiple applications in the host.
- In the kubernetes environment the cores cannot be shared by multiple containers in the pod