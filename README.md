# Neuron SDK readme

## Table of Contents

1. AWS Neuron Overview
2. Getting started

## AWS Neuron overview

AWS Neuron is a software development kit (SDK) enabling high-performance deep learning inference using AWS Inferentia custom designed machine learning chips. With Neuron, you can develop, deploy and run high-performance inference predictions with complex neural networks on top of Inferentia based EC2 Inf1 instances. Neuron is designed from the ground-up to allow for maximum scalability in optimizing both for highest throughput and lowest latency for a wide variety of use-cases.

Neuron is pre-integrated into popular machine learning frameworks like TensorFlow, MXNet and Pytorch to provide a seamless training-to-inference workflow. It includes a compiler, runtime library, as well as debug and profiling utilities with a TensorBoard plugin for visualization. In most cases developers will only have to change a few lines of code to use Neuron from within a framework. Developers can integrate Neuron to their own custom frameworks/environments as well.


### Neuron Developer Flow

Since Neuron is pre-integrated with popular frameworks, it can be easily incorporated into modern applications to provide high-performance inference predictions. Neuron is built to enable the above steps to be done from within an ML framework with the addition of the compilation step and load the model. Neuron allows customers to keep training in 32-bit floating point for best accuracy and auto-convert the 32-bit trained model to run at speed of 16-bit using bfloat16 model.

![image devflow](./misc/images/devflow.png)

Since ML models constantly evolve, we’ve designed AWS Neuron to give builders of ML solutions a future-proof development environment, utilizing an ahead-of-time compiler that ensures Neuron will optimally utilize the hardware when new operators and neural-net models are developed.

Once a model is trained to the required accuracy, the model should be compiled to an optimized binary form, referred to as a Neuron Executable File Format (NEFF), which is in turn loaded by the Neuron runtime to execute inference input requests on Inferentia chips. The compilation step may be performed on any EC2 instance or on-premises. We recommend using a high-performance compute server of choice (C/M/R/Z instance types), for the fastest compile times and ease-of-use with a prebuilt DLAMI. Developers can also install Neuron in their own environments; this approach may work well for example when building a large fleet for inference, allowing the model creation, training and compilation to be done in the training fleet, with the NEFF files being distributed by a configuration management application to the inference fleet. More details [here](./docs/getting-started-neuron-rtd.md)

### Deployment 

AWS Neuron provides developers flexibility to deploy their inference workloads, and optimize for both throughput and low-latency, to meet specific application constraints. For large models utilizing the large on-chip cache built in the Inferentia devices may be useful to run at super low latencies. Enable NeuronCore Pipeline to stores the model directly in cache for maximum efficiency. For models that require increased cache capacity, Neuron will split them across multiple chips, utilizing a high-speed chip to chip interconnect. Once deployed, NeuronCore Pipeline appears as one Neuron virtual device to the framework/application. 

Neuron also enables developers to assign different models to separate NeuronCore Groups in a flexible and scalable way. This allows to run the same or multiple models in parallel. NeuronCore Groups may be useful for increasing accuracy through majority-vote, or when different models need to run as a pipeline. With NeuronCore Groups developers can maximize the hardware utilization by controlling the NeuronCore compute resources allocated to each NeuronCore Group to ensure it fits their specific application requirements.
 

### Profiling and debugging

Neuron includes a set of tools and capabilities to help developers monitor and optimize their Neuron based inference applications. Neuron tools can be incorporated into scripts to automate Neuron devices operation and health monitoring, and include discover and usage utilities, data-path profiling tools, and visualization utilities. Using a tensorboard plugin you can inspect and profile graphs execution. For more details refer to: [Getting started: Neuron Tensorboard profiling](./docs/getting-started-tensorboard-neuron.md)


# Getting started and More Information:

To use Neuron you can use a pre-built Amazon Machine Images (DLAMI) or DL containers or install Neuron software on your own instances. To use AMI/Containers these documents may help.

* [AWS DLAMI Getting Started](https://docs.aws.amazon.com/dlami/latest/devguide/gs.html)
* [AWS DL Containers](https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-ec2.html)
* [Neuron Repository Package Manager Configurations](./docs/guide-repo-config.md)

## Neuron Runtime
* [Neuron Runtime Getting Started](./docs/getting-started-neuron-rtd.md)
* [Tutorial: Advanced Neuron Runtime Configurations](./docs/tutorial-advanced-neuron-rtd-configs.md)
* [Tutorial: Container Configurations for Neuron Runtime](./docs/tutorial-containers-neuron-rtd.md)


## Start Using one of the supported frameworks:

### TensorFlow-Neuron:
* [Tutorial: Tensorflow-Neuron and Neuron Compiler](./docs/tutorial-tensorflow-neuron-compile-infer.md)
* [Tutorial: Data Parallel Tensorflow-Neuron and Neuron Compiler](./docs/tutorial-tensorflow-neuron-compile-infer-data-parallel.md)
* [Reference: TensorFlow-Neuron Compilation API](./docs/api-tensorflow-neuron-compilation-python-api.md)
* [Tutorial: Tensorflow Model Server](./docs/tutorial-tensorflow-serving.md)
* [Tutorial: Data Parallel Tensorflow Model Server](./docs/tutorial-tensorflow-serving-data-parallel.md) 


### MXNet-Neuron:
* [Tutorial: MXNet-Neuron and Neuron Compiler](./docs/tutorial-mxnet-neuron-compile-infer.md)
* [Reference: MXNet-Neuron Python API](./docs/api-mxnet-neuron-compilation-python-api.md)
* [Tutorial: MXNet-Neuron Model Server](./docs/tutorial-mxnet-neuron-model-serving.md)
* [Tutorial: MXNet Configurations of NeuronCore Groups](./docs/tutorial-mxnet-neuroncore-groups.md
)

## Debugging, Profiling and other tools:
* [Getting started: Neuron Tensorboard profiling](./docs/getting-started-tensorboard-neuron.md)
* [Tutorial: Neuron utilities](./docs/tutorial-advanced-neuron-operational-tools.md)




