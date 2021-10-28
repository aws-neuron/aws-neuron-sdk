.. _packaging-neuron-rt-containers.rst:

Packaging Container Applications using Neuron Runtime 1.x
=========================================================

This document describes three ways to configure the Neuron SDK and run Neuron applications using containers. The goal is to ensure that the Neuron Runtime is able to control the Neuron Devices on the host instance and communicate with your application running inside a container.

The three different configurations enable you to chose between:

* Packaging your application and the runtime in a single container [Recommended].
* Packaging the application and runtime in separate containers.
* Packaging your application in a container and using the runtime from the host OS.

All three configurations are visualized below and can be implemented using docker alone on an EC2 Inf1 instance, or via a container orchestration services such as EKS and ECS.

|image|

.. |image| image:: /images/ContainerPackagingImages.png
   :width: 850
   :align: middle

Pre-requisites
--------------
#. Docker version 18 or newer is configured according to :ref:`tutorial-docker-env-setup-for-neuron`
#. Inf1 instance with available Neuron :ref:`Devices<container-devices>`
#. If running Runtime inside a container, system capability (docker run --cap-add IPC_LOCK). Refer :ref:`tutorial-docker-env-setup-for-neuron`
#. The :ref:`UDS<container-uds>` file that must be mounted and appropriate directories are setup
#. If running tensorflow-model-server/multi-model-server/torchserve etc then make sure appropriate ports that these servers are listening to are exposed using the EXPOSE in dockerfile or docker run -p 80:8080

Recommended - Packaging Application and Neuron Runtime in the same container
-----------------------------------------------------------------------

This is **recommended packaging mode**, as you benefit from

    * Not having to mount the UDS file from the host on to the container. The application and runtime use the default UDS local to the container.
    * Full control of the Neuron Devices used by the application in the container.

#. Build the container using :ref:`app-rt-same-dockerfile`
#. The above docker file copies :ref:`dockerd-entrypoint-app-rt-same`
#. Run the containers

.. code:: bash

   docker run -it --name pt17 -p 80:8080 --cap-add IPC_LOCK -e "AWS_NEURON_VISIBLE_DEVICES=ALL"  neuron-container:pytorch neuron-top

.. note::
   Since runtime is running inside the container the Neuron Devices needs to be mounted inside the container with the argument -e "AWS_NEURON_VISIBLE_DEVICES=ALL"


Alternative 1 - Packaging Application and Neuron Runtime in different container
------------------------------------------------------------------------

This is **alternative packaging mode**. Should be used only when the recommended mode is not achievable, due to reasons such as:

    * There are already application only/runtime only containers running.
    * Would not want to mount devices to application container and localize them to the runtime container.
    * Separate out the resources for application and runtime.

#. Build the runtime docker image using :ref:`neuron-runtime-dockerfile`
#. Run the runtime container

    .. code:: bash

       docker run --device=/dev/neuron0 --cap-add IPC_LOCK -v /run/:/run neuron-rtd

    .. note::

       Since runtime is running inside the container the Neuron Devices needs to be mounted inside the container with the argument \--device=/dev/neuron0

#. Build the container using :ref:`app-rt-diff-dockerfile`
#. Run the application containers

    .. code:: bash

       docker run -it -v /run/:/run neuron-container:pytorch neuron-top

    .. note::
       Since runtime is not part of this container no need to mount Neuron Devices in this container


Alternative 2 - Packaging Application in container and Neuron Runtime directly on host
-------------------------------------------------------------------------------

This is **alternative packaging mode**. Should be used only when the recommended mode is not achievable, due to reasons such as:

    * Runtime is already running on the host
    * Require multiple applications process and containers to access the runtime

#. Run the runtime software - refer :ref:`rtd-getting-started`
#. Build the container using :ref:`app-rt-diff-dockerfile`
#. Run the application containers

    .. code:: bash

       docker run -it -v /run/:/run neuron-container:pytorch neuron-top

    .. note::

       Since runtime is not part of this container no need to mount neuron devices in this container

Important to know
-----------------

.. _container-devices:

Devices
#######
There are currently two ways to specify Neuron Devices to a container.

#. The docker native way is to use --device /dev/neuron# for each of the Neuron Devices intended to be passed. When using --device option ALL/all is not supported.

    .. code:: bash

        docker run --device=/dev/neuron0 --device=/dev/neuron1

#. If you install the aws-neuron-runtime-base package, you will have an OCI hook that also supports use of a container environment variable AWS_NEURON_VISIBLE_DEVICES=<ALL | csv of devices>, which intends to make things easier for multi device scenarios. Following are some examples

    .. code:: bash

        docker run -e “AWS_NEURON_VISIBLE_DEVICES=0,1”
        docker run -e “AWS_NEURON_VISIBLE_DEVICES=ALL”

.. _container-uds:

UDS
###
#. The aws-neuron-runtime software is a grpc server (neuron-rtd) that listens on ``unix:/run/neuron.sock`` by default.

    Please refer the :ref:`neuron-runtime` that shows how the default can be changed.
#. The framework/app also by default sends grpc requests to uds ``unix:/run/neuron.sock``. This can be changed by the environment variable NEURON_RTD_ADDRESS.
#. The docker run command below assumes the defaults are used. If using non-default uds then make the appropriate changes in the mount.

    * Default UDS

        .. code:: bash

           docker run -it neuron-container:pytorch

    * Non-default UDS - Mount /run in host to /tmp in container.

        .. code:: bash

           docker run -it -v /run/my-new-uds.sock:/tmp/custom/path/to/my-new-uds.sock neuron-container:pytorch
