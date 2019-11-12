# Neuron SDK readme

## Table of Contents

1. AWS Neuron Overview
2. Getting started

## AWS Neuron overview

AWS Neuron is a software development kit (SDK) enabling high-performance deep learning inference using AWS Inferentia custom designed machine learning chips. With Neuron, you can develop, deploy and run high-performance inference predictions with complex neural networks on top of Inferentia based EC2 Inf1 instances. Neuron is designed from the ground-up to allow for maximum scalability in optimizing both for highest throughput and lowest latency for a wide variety of use-cases.

Neuron is pre-integrated into popular machine learning frameworks like TensorFlow, MXNet and Pytorch to provide a seamless training-to-inference workflow. It includes a compiler, runtime library, as well as debug and profiling utilities with a TensorBoard plugin for visualization. In most cases developers will only have to change a few lines of code to use Neuron from within a framework. Developers can integrate Neuron to their own custom frameworks/environments as well.


### Neuron Developer Flow

Since Neuron is pre-integrated with popular frameworks, it can be easily incorporated into modern applications to provide high-performance inference predictions. Neuron is built to enable the above steps to be done from within an ML framework with the addition of the compilation step and load the model. Neuron allows customers to keep training in 32-bit floating point for best accuracy and auto-convert the 32-bit trained model to run at speed of 16-bit using bfloat16 model.
[Image: image.png]
Since ML models constantly evolve, we’ve designed AWS Neuron to give builders of ML solutions a future-proof development environment, utilizing an ahead-of-time compiler that ensures Neuron will optimally utilize the hardware when new operators and neural-net models are developed.

Once a model is trained to the required accuracy, the model should be compiled to an optimized binary form , referred to as a Neuron Executable File Format (NEFF), which is in turn loaded by the Neuron runtime to execute inference input requests on Inferentia chips. The compilation step may be performed on any EC2 instance or on premises. We recommend using a high-performance compute server of choice (C/M/R/Z instance types), for the fastest compile times and ease-of-use with a prebuilt DLAMI (TODO: add link).  Developers can also install Neuron in their own environments; this approach may work well for example when building a large fleet for inference, allowing the model creation, training and compilation to be done in the training fleet, with the NEFF files being distributed by a configuration management application to the inference fleet.

### deployment 

AWS Neuron provides developers flexibility to deploy their inference workloads, and optimize for both bandwidth and low-latency, to meet specific application constraints. Utilizing the large on-chip cache built in the Inferentia devices, developers can leverage the Neuron Core Pipeline capability, and store all of the entire model directly cache for maximum memory efficiency. For increased cache capacity, the model can also be split across multiple chips, utilizing a high-speed chip to chip interconnect. Once deployed, Neuron Core Pipeline appears as one Neuron virtual device to the framework/application This allows developers to guarantee best-in-class latency, while maintaining high throughput.

Another option available for builders, is to allocate different models to a Neuron Core Groups, and run multiple models in parallel. This mode can be useful for increasing accuracy through majority-vote, or when different models need to run as a pipeline. With Neuron Core Groups builders can maximize the hardware utilization by controlling the Neuron Core compute resources allocated to each Neuron Core Group to fit  their specific application requirements.
 

### Profiling and debugging

Neuron includes a set of tools and capabilities to help developers monitor and optimize their Neuron based inference applications. Neuron tools can be incorporated into scripts to automate Neuron devices operation and health monitoring, and include discover and usage utilities, data-path profiling tools, and visualization utilities. Using a tensorboard plugin you can inspect and profile graphs execution. For more details refer to: TODO LINK
‘

# Getting started

To use Neuron you can use a pre-built Amazon Machine Images (DLAMI) or DL containers or install Neuron software on your own instances. To use AMI/Containers these documents may help.

* [AWS DLAMI Getting Started](https://docs.aws.amazon.com/dlami/latest/devguide/gs.html)
* [AWS DL Containers](https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-ec2.html)
* TODO: Neuron installation 

Using one of the supported frameworks:

* [Getting Started with Tensorflow and AWS Neuron](https://quip-amazon.com/52iNAJrZgZPe)
* [Getting Started: MXNet-Neuron](https://quip-amazon.com/JCw4AYinp0ve)
* [Getting started with Keras and AWS Neuron](https://quip-amazon.com/OIiFAM6hCEyo)

* Getting started with Pytorch and Neuron

Getting started with a Neuron:

* [Getting started:  Installing and Configuring Neuron-RTD on an Inf1 instance](https://quip-amazon.com/OtKfA3NkbaO7)
* TODO Getting started: Neuron profiling
* [Getting started with Neuron Compiler](https://quip-amazon.com/u4lvAHIB6OSB)



