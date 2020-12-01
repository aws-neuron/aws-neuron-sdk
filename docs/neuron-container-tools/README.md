</br>
</br>

Please view our documentation at **[https://awsdocs-neuron.readthedocs-hosted.com/](https://awsdocs-neuron.readthedocs-hosted.com/)** 

**Please note this file will be deprecated.**

</br>
</br>



# Tutorial: Neuron container tools

## Introduction

Neuron provides tools to support deploying containarized applications to Inf1 instances.

An containerized inference application would consist of two parts
1. Application container: running the application such as tensorflow-neuron or tensorflow-model-server-neuron. 
2. Neuron-rtd container: running neuron-rtd. This is a side-car container that requires elevated privileges, in order for neuron-rtd to access 
Inferentia devices. However, neuron-rtd drops all capabilities after initialization, before opening a gRPC socket. 


We recommend the application and neuron-rtd to run in separate containers to prevent accidental privilege escalation 
of users application. In order for the application and neuron-rtd to communicate, 
a shared volume can be mounted to place a UDS socket. It is also possible to use port-forwarding.


A neuron-rtd container is available over ECR repositories and can be used directly. Customers may also build their own using Neuron packages, as shown in [Neuron Runtime Dockerfile](./docker-example/Dockerfile.neuron-rtd)

Official neuron-rtd container: 790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:latest



#### Tutorials
The following tutorials detail the steps needed to configure Neuron into a Docker or Kubernetes environment.

* [Docker environment setup for Neuron](./tutorial-docker.md)
* [Kubernetes environment setup for Neuron](./tutorial-k8s.md)


#### Examples

* [Run containerized Neuron application](./docker-example/README.md)
* [Deploy BERT as a k8s service](./../../src/examples/tensorflow/k8s_bert_demo/README.md)
* [Neuron Runtime Dockerfile](./docker-example/Dockerfile.neuron-rtd)
* [tensorflow-model-server-neuron Dockerfile](./docker-example/Dockerfile.tf-serving)
