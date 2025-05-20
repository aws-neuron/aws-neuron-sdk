.. _tutorial-oci-hook:

Tutorial Docker Neuron OCI Hook Setup
=====================================

Introduction
------------

A Neuron application can be deployed using docker containers. Neuron devices
are exposed to the containers using the --device option in the docker run command.
Docker runtime (runc) does not yet support the ALL option to expose all neuron
devices to the container. In order to do that an environment variable,
“AWS_NEURON_VISIBLE_DEVICES=ALL" can be used.

For the above environment variable to be used, the oci neuron hook has to be
installed/configured.

.. important::

    The Neuron OCI Hook is currently NOT supported with AL2023 on ECS. For workarounds,
    see the :ref:`oci-hook-workarounds` section below.

Install oci-add-hooks dependency on the Linux host
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Install the package that has oci hook software
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important::

    This step should run on the Linux host and not inside the container.

For Inf1 install the following package

.. code:: bash

    sudo apt-get install aws-neuron-runtime-base -y

For Trn1 install the following package

.. code:: bash

    sudo apt-get install aws-neuronx-oci-hook -y

For docker runtime setup Docker to use oci-neuron OCI runtime.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

oci-neuron is a script representing OCI compatible runtime. It wraps
oci-add-hooks, which wraps runc. In this step, we configure docker to
point at oci-neuron OCI runtime. Install dockerIO:

.. code:: bash

    sudo cp /opt/aws/neuron/share/docker-daemon.json /etc/docker/daemon.json
    sudo service docker restart

If the docker restart command fails, make sure to check if the docker
systemd service is not masked. More information on this can be found
here: https://stackoverflow.com/a/37640824

For containerd runtime, setup containerd to use oci-neuron OCI runtime.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Update the following fields in the /etc/containerd/config.toml to configure
containerd to use the neuron oci hook

.. code:: bash

    default_runtime_name = "neuron"
    [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.neuron]
        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.neuron.options]
            BinaryName = "/opt/aws/neuron/bin/oci_neuron_hook_wrapper.sh"


After that restart the containerd daemon

.. code:: bash

    sudo systemctl restart containerd

For cri-o runtime, setup cri-o to use oci-neuron OCI runtime.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Update the following fields in the /etc/crio/crio.conf to configure
cri-o to use the neuron oci hook

.. code:: bash

    default_runtime_name = "neuron"
    [crio.runtime.runtimes.neuron]
    runtime_path = "/opt/aws/neuron/bin/oci_neuron_hook_wrapper.sh"

After that restart the containerd daemon

.. code:: bash

    sudo systemctl restart cri-o

.. _oci-hook-workarounds:

OCI hook workarounds
^^^^^^^^^^^^^^^^^^^^

**ECS (EC2)**

Add the following to your ECS task definition:

.. code:: json

    "linuxParameters": {
        "devices": [
            {
                "containerPath": "/dev/neuron0",
                "hostPath": "/dev/neuron0",
                "permissions": [
                    "read",
                    "write"
                ]
            },
            {
                "containerPath": "/dev/neuron1",
                "hostPath": "/dev/neuron1",
                "permissions": [
                    "read",
                    "write"
                ]
            },
            ...,
        ],
    },

The linuxParameters parameter can be found under containerDefinition. More information can be found here:
https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definition_parameters.html#container_definition_linuxparameters.
Expose as many Neuron devices as needed, up to the max number of devices for the specified instance.
For example, the trn1.32xlarge instance type contains 16 neuron devices, so the devices that can be exposed are
/dev/neuron0, /dev/neuron1, up to /dev/neuron15.
To see an example of an ECS task definition exposing Neuron devices,
see https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-inference-task-def.html.
