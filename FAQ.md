# AWS Neuron: Frequently Asked Questions 

[Getting Started FAQs](#getting-started)

[General Neuron FAQs](#general)

[Neuron Compiler FAQs](#compiler)

[Neuron Runtime FAQs](#runtime)

[Troubleshooting FAQs](#troubleshooting)


<a name="getting-started"></a>
## Getting started with Neuron FAQs

**Q: How can I get started?**

You can start your workflow by training your model in one of the popular ML frameworks using EC2 GPU instances such as P3 or P3dn, or alternativelly download a pre-training model. Once the model is trained to your required accuracy, you can use the ML frameworks' API to invoke Neuron, the software development kit for Inferentia, to re-target(compile) the model for execution on Inferentia. The later step could be done once, and the developer doesnt need to do it as long as the model is not changing. Once compiler, the Inferentia binary can be loaded into one or more Inferentia, and can service inference calls. In order to get started quickly, you can use [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/) that come pre-installed with ML frameworks and the Neuron SDK. For a fully managed experience, you will be able to use Amazon SageMaker which will enable you to seamlessly deploy your trained models on Inf1 instances. 

For customers who use popular frameworks like Tensorflow, MXNet and PyTorch, a guide to help you get started with frameworks 
is available at [TODO](). I you wish to deploy without a framework, you can install Neuron using pip 
install and use Neuron to compile and deploy your models to Inf1 instances to run inference. More details on using Neuron without a framework are here [TODO]().


**Q: How would I select which Inf1 instance to use?**

The decision as to which Inf instance size to use is based upon the application and the performance/cost targets and may differ for each workload. To assist, the Neuron compiler provides guidance on expected latency and throughput for various Inf1 instance sizes, and TensorBoard profiling will show actual results when executed on a given instance. A guide to this process is available here: [TODO]()

Ultimately, AWS encourages developers try out all the Inf1 instance sizes (which can be done at low cost and quickly in a cloud environment), with their specific models, and choose the right instance for them.


<a name="general"></a>
## General Neuron FAQs

**Q: What ML models types and operators are supported by AWS Neuron?**

AWS Neuron includes a compiler that converts your trained machine learning model to Inferentia binary code for execution. The Neuron compiler supports many commonly used machine learning models such as single shot detector (SSD) and ResNet for object detection and image recognition, and Transformer and BERT for natural language processing and translation. Neuron list of supported ML operators and supported inputs are in [release notes](./RELEASE_NOTES.md). AWS continues to grow the list based on customers' feedback. 

**Q: Why is a compiler needed, and how do I use it?**

The Neuron compiler converts from a framework level Neural Network graph, with operators like convolution and pooling, into the hardware-specific instruction set of Inferentia, build the schedule for execution these instructions, and convers the model parameters into format that Inferentia can consume.  The supported input formats include TensorFlow, PyTorch, MXNet, or ONNX. The output from the compiler is Inferentia program binary, referred to as a Neuron Executable File Format (NEFF). NEFF contains a combination of these binary code, the model parameters, and additional meta-data needed by the runtime and profiler. 

**Q: I am using a ML framework today – what will change for me to use this?**

The likely (but not only) workflow for developers is a hand-off of pre-trained model to large scale of inference fleet.
To use Inferentia and Inf1 instances, the developer need to perform one-time compilation of the pre-trained model to generate NEFF, and use this as the inference model in fleet of Inf1 instances.

TODO [A guide to compile and deploy inference models using TensorFlow interface support](http://github.com/aws/aws-neuron-sdk/docs/tensorflow-neuron/readme.md)
TODO [A guide to compile and deploy inference models using MXNet interface support](http://github.com/aws/aws-neuron-sdk/docs/mxnet-neuron/readme.md)


**Q: What is Inferentia's NeuronCore Pipeline ? and How do I take advantage of it?**

Inferentia NeuronCore Pipeline is a unique technique to shard a specific Neural Network across multiple Inferentia accelerators, to take advantage of the large on-chip cache, that would typically increase throughput and reduce latency at low batch size. Inf1 instances with multiple Inferentia accelerators, such as Inf1.6xlarge or Inf1.24xlarge, support NeuronCore Pipeline thanks to fast chip-to-chip interconnect. 

Developers can choose to use NeuronCore Pipeline mode during compile stage, with an opt-in flag. [NeuronCore Pipeline guide](TODO) provides further details. 

**Q: Can I use TensorFlow networks from tfhub.dev as-is ? if not, what should I do?**
TODO
Yes. Models format can  be imported into Tensorflow, either as a standard model-server, in which case it appears as a simple command line utility, or via the Python based Tensorflow environment.  The primary additional step needed is to compile the model into Inferentia NEFF format. 


<a name="compiler"></a>
## Neuron compiler FAQs

**Q: Where can I compile to Neuron?** 

The one-time compilation step from the standard Framework-level model to Inferentia binary may be performed on any EC2 instance or on-premises. 

We recommend using a high-performance compute server of choice (C5 or z1d instance types), for the fastest compile times and 
ease-of-use with a prebuilt [DLAMI](. Developers can also install Neuron in their own environments; this approach may work well 
for example when building a large fleet for inference, allowing the model creation, training and compilation to be done in the 
training fleet, with the NEFF files being distributed by a configuration management application to the inference fleet.

**Q: My current Neural Network is based on FP32, how can I use it with Neuron?**

Inferentia chips support FP16, BFloat16 mixed-precision data-types and INT8 (coming soon). It is common for Neural Networks to be trained in FP32, in which case the trained graph needs to be converted to one of these data types for execution on Inferentia. Neuron can compile and execute FP32 neural nets by automatically converting them to BFloat16. Given an input using FP32, the compiler output will ensure that the executed graph can accept input inference requests in FP32. 

**Q: What are some of the important compiler defaults I should be aware of?**
TODO
The compiler compiles the input graph for a single NeuronCore by default.  Using the The “`num-neuroncores`” option directs compiler to direct compiled graph to run on a specified number of NeuronCores. This number can be less than the total available NeuronCores on an instance. See performance tuning application note [link]TODO for more information. 

**Q: Which operators does Neuron support?**
TODO

**Q: Any operators that Neuron doesn't support?**
Models with control-flow and dynamic shapes are not supported. You will need to partition the model using the framework prior to compilation.TODO

**Q: Will I need to recompile again if I updated runtime/driver version?**
TODO
The compiler and runtime are committed to maintaining compatibility for major version releases with each other. The versoning is defined as major.minor, with compatibility for all versions with the same major number. If the versions mismatch, an error notification is logged and the load will fail. This will then require the model to be recompiled.

**Q: I have a NEFF binary, how can I tell which compiler version generated it?**
TODO

**Q: How long does it take to compile?**
TODO

<a name="runtime"></a>
## Neuron runtime FAQs

**Q: How does Neuron connect to all the Inferentia chips in an Inf1 instance?**

By default, a single runtime process will manage all assigned Inferentias, including running the Neuron Core Pipeline mode. In some cases, user can configure multiple KRT processes each managing a fraction of assigned Inferentias. TODO 


**Q: Where can I get logging and other telemetry information?**
TODO

**Q: What about RedHat or other versions of Linux?**
TODO

**Q: What about Windows?**

Windows is not supported at this time.

**Q: How can I use Kaena in a container based environment? Does Kaena work with ECS and EKS?**
TODO


**Q: How can I take advantage of multiple TPBs and run multiple inferences in parallel?**
TODO


<a name="troubleshooting"></a>
## Troubleshooting FAQs

**Q: Performance is not what I expect it to be, what's the next step?**
TODO

**Q: do I need to worry about size of model and size of inferentia memory ? what problems can i expect to have?**
TODO

**Q: How can I  debug / profile my inference request?**
TODO

