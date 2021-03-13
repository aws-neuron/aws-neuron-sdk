.. _neuron-sdk-containers:

Neuron SDK Containers
=====================

Neuron provides several components in ECR that are updated on each
release of the relevant Neuron component.  When using these containers, you MUST specify
the version you are interested in using.  Avoid use of any "latest" tag as it will move
without your approval.  The latest tag is only provided in some cases for convenience.

Each ECR is maintained in only the us-east-1 and us-west-2 regions.

* Neuron Kubernetes Device Plugin:
790709498068.dkr.ecr.<region>.amazonaws.com/neuron-device-plugin:<version>

* Neuron Kubernetes Scheduler Extension:
790709498068.dkr.ecr.<region>.amazonaws.com/neuron-scheduler:<version>

* (deprecated) Neuron Runtime Deamon container, aka "the sidecar":
790709498068.dkr.ecr.<region>.amazonaws.com/neuron-rtd:<version>
