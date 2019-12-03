# Tutorial: Container Configurations for Neuron-RTD

## Introduction

In other tutorials, we used tensorflow serving to compile a model and run inference. To do that, Neuron runtime daemon (Neuron-RTD) would be running in the background as a service, and TensorFlow serving would use a default socket to interact with neuron-rtd.

For containerized applications, it is recommended to use the neuron-rtd container. It is also recommended that framework-serving is ran in its own container because neuron-rtd requires higher privileges, which may not be desired for application code.

Neuron-rtd container is available over ECR repositories and can be used directly. Customers may also build their own using Neuron packages. Guidance on building your own custom framework serving container is outlined in Appendix A.

Neuron-rtd container: [790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:latest]()

[DL framework containers](https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-images.html)


##  Steps:

This example will configure a single Neuron-RTD per Inferentia and place each into its own Docker container.

#### Step 1: install/upgrade host base package

```bash
sudo apt-get install aws-neuron-runtime-base
```

Stop neuron-rtd service if running
```bash
sudo service neuron-rtd stop
```

#### Step 2: install oci-add-hooks depdency

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


#### Step 3: setup Docker to use oci-neuron runtime.

Install dockerIO:

```bash
sudo apt install -y docker.io
sudo usermod -aG docker $USER
```

Logout and log back in to refresh membership. 
Place daemon.json Docker configuration file in default location. This file specifies oci-add-hooks as default docker runtime:

```bash
sudo cp /opt/aws/neuron/share/docker-daemon.json /etc/docker/daemon.json
sudo service docker restart
```

If the docker restart command fails, make sure to check if the docker systemd service is not masked. More information on this can be found here: https://stackoverflow.com/a/37640824

This can be checked using:
```bash
sudo systemctl status docker
```

To fix:
```bash
sudo systemctl unmask docker.service
sudo systemctl unmask docker.socket
sudo systemctl start docker.service
sudo systemctl status docker
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


#### Step 4: run neuron-rtd container:

You can choose to build your own neuron-rtd image as shown in Appendix A, or just use: 790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:latest

Run neuron-rtd. A volume must be mounted to :/sock where neuron-rtd will open a UDS socket. Framework can interact with runtime using this socket.

```bash
# Note: Make sure to have some AWS credentials setup for the "aws ecr login" command below to work
$(aws ecr get-login --no-include-email --region us-east-1 --registry-ids 790709498068)
docker pull 790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:latest
docker tag 790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:latest neuron-rtd
mkdir /tmp/neuron_rtd_sock/
docker run --env AWS_NEURON_VISIBLE_DEVICES="0" --cap-add SYS_ADMIN --cap-add IPC_LOCK -v /tmp/neuron_rtd_sock/:/sock neuron-rtd
```


#### Step 5: run framework serving image:

Run tensorflow-model-server-neuron image:
Guide:
[Neuron TensorFlow Serving](./../tensorflow-neuron/tutorial-tensorflow-serving.md)

In this step, we will use tensorflow-model-server-neuron image built in appendix A.

Assuming a compiled saved model was stored in s3://my_magical_bucket/my_magical_model/

```bash

# Note: the neuron-rtd socket directory must be mounted and pointed at using environment variable.
#       Tensorflow serving will use that socket to talk to Neuron-rtd
docker run --env NEURON_RTD_ADDRESS=/sock/neuron.sock \
           -v /tmp/neuron_rtd_sock/:/sock \
           -p 8501:8501 \
           -p 8500:8500 \
           --env MODEL_BASE_PATH=s3://my_magical_bucket/my_magical_model/ \
           --env MODEL_NAME=my_magical_model
           tensorflow-model-server-neuron

```

P.S. You can run multiple instances of the model serving containers to run
more models and fully utilize the instance.


#### Step 6: Verify by running an inference!



## Appendix A: building your own tensorflow-model-server-neuron image

Use the following example dockerfile, and build it as shown. It will install
tensorflow-model-server-neuron from neuron yum repo. You can either modify
the Dockerfile to copy your compiled saved model into the MODEL_BASE_PATH directory,
or point tensorflow_model_server_neuron to s3:// location.

```bash
cat Dockerfile
```
Result:
```
FROM amazonlinux:2

# Expose ports for gRPC and REST
EXPOSE 8500 8501

ENV MODEL_BASE_PATH=/models \
    MODEL_NAME=model

RUN echo $'[neuron] \n\
name=Neuron YUM Repository \n\
baseurl=https://yum.repos.neuron.amazonaws.com \n\
enabled=1' > /etc/yum.repos.d/neuron.repo

RUN rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB

RUN yum install -y tensorflow-model-server-neuron
RUN mkdir -p $MODEL_BASE_PATH

CMD ["/bin/sh", "-c", "/usr/local/bin/tensorflow_model_server_neuron --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/"]

```
Docker build:
```bash
docker build . -t tensorflow-model-server-neuron
```


## Appendix B: building your own neuron-rtd image


You can choose to build your own neuron-rtd image using the following example dockerfile, or just use:
790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:latest

To create your own, use the following example dockerfile:

```bash
cat Dockerfile
```
Result:
```
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
```
Docker build:
```bash
docker build . -t neuron-test
```

Now start and verify that the container will start and that the desired Inferentia devices will be mapped to it. The AWS_NEURON_VISIBLE_DEVICES environment variable will map the chosen (set of) Inferentia devices to this container. The default configuration for a Neuron-RTD is always to be configured for all present Inferentias, and thus this container will have one Neuron-RTD to manage all present Inferentias in the container.


* Note: the container must start with root level privileges in order to map the memory needed from the Infernetia devices, but this will be dropped following initialization and is not needed during use.

Run neuron-ls in the container to verify device whitelisting works as expected:

```bash
docker run --env AWS_NEURON_VISIBLE_DEVICES="0" --cap-add SYS_ADMIN --cap-add IPC_LOCK  neuron-test neuron-ls
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
