![neuron](./misc/images/Site-Merch_Neuron-ML-SDK_Editorial.png)

# AWS Neuron  

## Table of Contents

1. AWS Neuron Overview
2. Getting started

## AWS Neuron overview

AWS Neuron is a software development kit (SDK) enabling high-performance deep learning inference using AWS Inferentia custom designed machine learning chips. With Neuron, you can develop, profile, and deploy high-performance inference predictions on top of Inferentia based EC2 Inf1 instances.

Neuron is pre-integrated into popular machine learning frameworks like TensorFlow, MXNet and Pytorch to provide a seamless training-to-inference workflow. It includes a compiler, runtime driver, as well as debug and profiling utilities with a TensorBoard plugin for visualization.


### Neuron developer flow

Since Neuron is pre-integrated with popular frameworks, it can be easily incorporated into ML applications to provide high-performance inference predictions. Neuron is built to enable the above steps to be done from within an ML framework with the addition of the compilation step and load the model to the Inferentia chips. Neuron allows customers to keep training in 32-bit floating point for best accuracy and auto-convert the 32-bit trained model to run at speed of 16-bit using bfloat16 model.

![image devflow](./misc/images/devflow.png)

Once a model is trained to the required accuracy, it is compiled to an optimized binary form, referred to as a Neuron Executable File Format (NEFF), which is in turn loaded by the Neuron runtime driver to execute inference input requests on the Inferentia chips. The compilation step may be performed on any EC2 instance or on-premises.



### Neuron roadmap
We know ML models constantly evolve, so we’ve designed Neuron to give builders a future-proof development environment, utilizing an ahead-of-time compiler that ensures Neuron will optimally utilize the hardware as new operators and neural-net models are developed.

The AWS Neuron feature roadmap provides visibility onto what functures and performance enhancements we are working on in the near future. We invite you to view the roadmap [here](roadmap-readme.md) and let us know what you think.


# Getting started:


## Start using one of the supported frameworks:

TensorFlow-Neuron [TensorFlow-Neuron readme](./docs/tensorflow-neuron/readme.md) provides useful pointers to install and use Neuron from within the TensorFlow framework.

MXNet-Neuron [MXNet-Neuron readme ](./docs/mxnet-neuron/readme.md) provides useful pointers to install and use Neuron from within the MXNet framework.

Pytorch-Neuron [Pytorch-Neuron readme ](./docs/pytorch-neuron/README.md) provides useful pointers to install and use Neuron from within the Pytorch framework



## Performance optimizations

Neuron provides developers with various performance optimization options. Two of the most widely used ones are Batching and NeuronCore-Pipeline. Both techniques aim to keep the data close to the compute engines to improve hardware utilization, but achieve that in different ways. In batching it is achieved by loading the data into an on-chip cache and reusing it multiple times for multiple different model-inputs, while in pipelining this is achieved by caching all model parameters into the on-chip cache across multiple NeuronCores and streaming the calculation across them. For more details on the NeuronCore Pipeline checkout the tech note [here](./docs/technotes/neuroncore-pipeline.md), and for more details on Neuron Batching, please read the tech note [here](./docs/technotes/neuroncore-batching.md).

Another capability, called NeuronCore Groups allows developers to assign different models to separate NeuronCores, and run the same or multiple models in parallel. NeuronCore Groups may be useful for increasing accuracy through majority-vote, or when different models need to run as a pipeline. For more details please read more [here](./docs/tensorflow-neuron/tutorial-NeuronCore-Group.md).


## Installing Neuron
To use Neuron you can use a pre-built Amazon Machine Images (DLAMI) or DL containers or install Neuron software on your own instances. To ensure you have the latest Neuron version we recommend to either install it on your own instance, or to check for the installed version when using DLAMI or DL containers.

### Install Neuron in your own AMI
You can [Install Neuron in your own AMI](./docs/neuron-install-guide.md#user-guide-configuring-linux-for-repository-updates) if you already have an environment you'd like to continue using.

### DLAMI
Refer to the [AWS DLAMI Getting Started](https://docs.aws.amazon.com/dlami/latest/devguide/gs.html) guide to learn how to use the DLAMI with Neuron. When first using a released DLAMI, there may be additional updates to the Neuron packages installed in it.

NOTE: Only DLAMI versions 26.0 and newer have Neuron support included.

### DL Containers
For containerized applications, it is recommended to use the neuron-rtd container, more details [here](./docs/neuron-container-tools/README.md).
Inferentia support for [AWS DL Containers](https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-ec2.html) is coming soon.



## Tutorials
Neuron github provides detailed tutorials for each of the supported frameworks, we advise you watch the repo, as we add more tutorials and guides frequently. A few examples:

### Run ResNet50 inference examples
* [Using TensorFlow](./docs/tensorflow-neuron/tutorial-compile-infer.md)
* [Using PyTorch](./docs/pytorch-neuron/tutorial-compile-infer.md)
* [Using MXNet](./docs/mxnet-neuron/tutorial-compile-infer.md)
* Hands-on Neuron lab [Inference with Amazon EC2 Inf1 Instance](https://github.com/awshlabs/reinvent19Inf1Lab)

### Run BERT inference examples
* [TensorFlow BERT-Large implementation](./src/examples/tensorflow/bert_demo/README.md)


## Profiling and debugging

Neuron includes a set of tools and capabilities to help developers monitor and optimize their Neuron based inference applications. Neuron tools can be incorporated into scripts to automate Neuron devices operation and health monitoring, and include discover and usage utilities, data-path profiling tools, and visualization utilities. Using a TensorBoard plugin you can inspect and profile graphs execution.

* [Getting started: Neuron TensorBoard profiling](./docs/neuron-tools/getting-started-tensorboard-neuron.md)
* [Neuron utilities](./docs/neuron-tools/Readme.md)

## Support
If none of the github and online resources have an answer to your question, checkout the AWS Neuron [support forum](https://forums.aws.amazon.com/forum.jspa?forumID=355).

## Application and Technical Notes
* [Tech Notes Readme](./docs/technotes/README.md)
