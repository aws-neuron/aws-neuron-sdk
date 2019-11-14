# Tutorial: Container Configurations for Neuron-RTD on an Inf1 instance 

##  Steps Overview:

1. Prerequisite
2. 4 x Inferentia with 4 x Docker containers (1 for each Neuron-RTD, 1 Neuron-RTD per Inferentia)

## Step1: Prerequisite

[Getting started:  Installing and Configuring Neuron-RTD on an Inf1 instance](./getting-started-neuron-rtd.md)

## Step2 : Configure Neuron-RTD

### 4 X Neuron-RTD in Docker Containers:

This will configure 1 Neuron-RTD per Inferentia and place each into its own Docker container. 

Steps Overview:

Step 1: install the container package  and the tools package
Step 2: enable the oci hooks
Step 3: setup docker json config
Step 4: run the containers

Install container support:

```bash
sudo apt-get install aws-neuron-runtime-base
```

Add oci hooks:

```bash
sudo apt install -y golang && \
        go get github.com/joeshaw/json-lossless && \
        cd /tmp/ && \
        git clone https://github.com/awslabs/oci-add-hooks && \
        cd /tmp/oci-add-hooks && \
        make build && \
        sudo cp /tmp/oci-add-hooks/oci-add-hooks /usr/local/bin/
```

Install dockerIO:

```bash
sudo apt install -y docker.io
sudo usermod -aG docker $USER
```

Place daemon.json Docker configuration file in default location. This file specifies oci-add-hooks as default docker runtime:

```bash
sudo cp /opt/aws/neuron/share/docker-daemon.json /etc/docker/daemon.json
sudo service docker restart
```

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

Create a container that holds Neuron-RTD and any other services (like an ML framework such as Tensorflow). This example creates a dockerfile for this:

```bash
# stop all other krtd if they have been previously setup or run:
sudo systemctl stop neuron-rtd

export BUILD_PATH=/tmp/neuron_docker_build
rm -rf $BUILD_PATH
mkdir -p $BUILD_PATH

tee $BUILD_PATH/Dockerfile > /dev/null << EOF

FROM amazonlinux:2

RUN echo $'[neuron] \n\
name=Neuron YUM Repository \n\
baseurl=https://yum.repos.neuron.amazonaws.com \n\
enabled=1' > /etc/yum.repos.d/neuron.repo

RUN rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
RUN yum install -y aws-neuron-tools
RUN yum install -y aws-neuron-runtime
RUN yum install -y tar gzip jq

ENV PATH="/opt/aws/neuron/bin:${PATH}"

# Optional: Configure neuron-rtd to drop privileges during initialization
RUN mv /opt/aws/neuron/config/neuron-rtd.config /opt/aws/neuron/config/neuron-rtd.config.bk
RUN cat /opt/aws/neuron/config/neuron-rtd.config.bk  | jq '.init_config.drop_all_capabilities = true' > /opt/aws/neuron/config/neuron-rtd.config


CMD neuron-rtd -g unix:/sock/neuron.sock

EOF

cd $BUILD_PATH && docker build . -t neuron-test

```

Now start and verify that the container will start and that the desired Inferentia devices will be mapped to it. The AWS_MLA_VISIBLE_DEVICES environment variable will map the chosen (set of) Inferentia devices to this container. The default configuration for a Neuron-RTD is always to be configured for all present Inferentias, and thus this container will have one Neuron-RTD to manage all present Inferentias in the container.


* Note: the container must start with root level privileges in order to map the memory needed from the Infernetia devices, but this will be dropped following initialization and is not needed during use.

Run neuron-ls in the container to verify device whitelisting works as expected:

```bash
docker run --env AWS_MLA_VISIBLE_DEVICES="0" --cap-add SYS_ADMIN -v /tmp/sock:/sock neuron-test neuron-ls
```
Expected result:
```
+--------------+---------+--------+-----------+-----------+---------+------+------+
|   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   |   DMA   | EAST | WEST |
|              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 | ENGINES |      |      |
+--------------+---------+--------+-----------+-----------+---------+------+------+
| 0000:00:1f.0 |       0 |      4 | 4096 MB   | 4096 MB   |      12 |    0 |    1 |
+--------------+---------+--------+-----------+-----------+---------+------+------+ 

```

Run neuron-rtd. A volume must be mounted to :/sock where neuron-rtd will open a UDS socket. Framework can interact with runtime using this socket.

```bash
docker run --env AWS_MLA_VISIBLE_DEVICES="0" --cap-add SYS_ADMIN -v /tmp/sock:/sock neuron-test
```

