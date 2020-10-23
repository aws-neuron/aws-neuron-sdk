# Tutorial: Docker environment setup for Neuron Runtime 1.0

## Introduction

A Neuron application can be deployed using docker containers. This tutorial 
describes how to configure docker to expose Inferentia devices to containers.  

Once the environment is setup, a container can be started with 
*AWS_NEURON_VISIBLE_DEVICES* environment variable to specify desired set 
of Inferentia devices to be exposed to the container. AWS_NEURON_VISIBLE_DEVICES 
is a set of contiguous comma-seperated inferentia logical ids. To find out the 
available logical ids on your instance, run the neuron-ls tool.
For example, on inf1.6xlarge instance with 4 inferentia devices, you may 
set AWS_NEURON_VISIBLE_DEVICES="2,3" to expose the last two devices to a 
container.
When running neuron-ls inside a container, you will only see the set of exposed Inferentias.
For example:
```bash
docker run --env AWS_NEURON_VISIBLE_DEVICES="0" neuron-test neuron-ls
```
Would produce the following output:
```
+--------------+---------+--------+-----------+-----------+------+------+
|   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   | EAST | WEST |
|              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 |      |      |
+--------------+---------+--------+-----------+-----------+------+------+
| 0000:00:1f.0 |       0 |      4 | 4096 MB   | 4096 MB   |    0 |    0 |
+--------------+---------+--------+-----------+-----------+------+------+
```

##  Steps:

This tutorial starts from a fresh Ubuntu Server 16.04 LTS AMI "ami-08bc77a2c7eb2b1da".

#### Step 1: install aws-neuron-runtime-base  package
Follow the [Neuron Install Guide](../neuron-install-guide.md) to setup access to Neuron repos.
Then, install the aws-neuron-runtime-base package. 
```bash
sudo apt-get install aws-neuron-runtime-base
```

#### Step 2: Make sure that the neuron-rtd service is not running
If neuron-rtd is running on the host, stop the neuron-rtd service before starting the containerized neuron-rtd. This is needed to allow assignment of devices to containers:

```bash
sudo service neuron-rtd stop
```

#### Step 3: install oci-add-hooks dependency 

[oci-add-hooks](https://github.com/awslabs/oci-add-hooks) is an OCI runtime with the sole purpose of injecting OCI prestart, poststart, and poststop hooks into a container config.json before passing along to an OCI compatable runtime.
oci-add-hooks is used to inject a hook that exposes Inferentia devices to the container.
```bash
sudo apt install -y golang && \
    export GOPATH=$HOME/go && \
    go get github.com/joeshaw/json-lossless && \
    cd /tmp/ && \
    git clone https://github.com/awslabs/oci-add-hooks && \
    cd /tmp/oci-add-hooks && \
    make build && \
    sudo cp /tmp/oci-add-hooks/oci-add-hooks /usr/local/bin/
```


#### Step 4: setup Docker to use oci-neuron OCI runtime.
oci-neuron is a script representing OCI compatible runtime. It wraps oci-add-hooks, which wraps runc. In this step, we configure docker to point at oci-neuron OCI runtime.
Install dockerIO:

```bash
sudo apt install -y docker.io
sudo usermod -aG docker $USER
```

Logout and log back in to refresh membership. 
Place daemon.json Docker configuration file supplied by Neuron SDK in default location. This file specifies oci-neuron as default docker runtime:

```bash
sudo cp /opt/aws/neuron/share/docker-daemon.json /etc/docker/daemon.json
sudo service docker restart
```

If the docker restart command fails, make sure to check if the docker systemd service is not masked. More information on this can be found here: https://stackoverflow.com/a/37640824

Verify docker:

```bash
docker run hello-world
```

Expected result:
```
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
```


Build a docker image using provided dockerfile [Neuron Runtime Dockerfile](./docker-example/Dockerfile.neuron-rtd), and use to verify whitelisting:
```bash
docker build . -f Dockerfile.neuron-rtd -t neuron-test
```

Then run:
```bash
docker run --env AWS_NEURON_VISIBLE_DEVICES="0"  neuron-test neuron-ls
```
Expected result:
```
+--------------+---------+--------+-----------+-----------+------+------+
|   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   | EAST | WEST |
|              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 |      |      |
+--------------+---------+--------+-----------+-----------+------+------+
| 0000:00:1f.0 |       0 |      4 | 4096 MB   | 4096 MB   |    0 |    0 |
+--------------+---------+--------+-----------+-----------+------+------+

```
