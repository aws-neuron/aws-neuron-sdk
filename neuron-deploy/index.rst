.. _neuron-containers:

Neuron Containers
=================

Neuron provides tools to support deploying containarized applications to
Inf1 instances.

An containerized inference application would consist of two parts

1. Application container: running the application such as
   tensorflow-neuron or tensorflow-model-server-neuron.
2. Neuron-rtd container: running neuron-rtd. This is a side-car
   container that requires elevated privileges, in order for neuron-rtd
   to access Inferentia devices. However, neuron-rtd drops all
   capabilities after initialization, before opening a gRPC socket.

We recommend the application and neuron-rtd to run in separate
containers to prevent accidental privilege escalation of users
application. In order for the application and neuron-rtd to communicate,
a shared volume can be mounted to place a UDS socket. It is also
possible to use port-forwarding.

A neuron-rtd container is available over ECR repositories and can be
used directly. Customers may also build their own using Neuron packages,
as shown in :ref:`neuron-runtime-dockerfile`

Official neuron-rtd container:
790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-rtd:latest

