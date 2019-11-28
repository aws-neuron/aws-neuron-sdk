# AWS Neuron: Frequently Asked Questions (WIP)

[Getting Started FAQs](#getting-started)

[General Neuron FAQs](#general)

[Neuron Compiler FAQs](#compiler)

[Neuron Runtime FAQs](#runtime)

[Troubleshooting FAQs](#troubleshooting)


<a name="getting-started"></a>
## Getting started with Neuron FAQs

**Q: How can I get started?**

You can start your workflow by building and training your model in one of the popular ML frameworks using GPU compute instances such as P3 or P3dn. Once the model is trained to your required accuracy, you can use the ML framework’s API to invoke Neuron, a software development kit for Inferentia, to compile the model for execution on Inferentia, load it in to Inferentia’s memory and then execute inference calls. In order to get started quickly, you can use AWS Deep Learning AMIs (https://aws.amazon.com/machine-learning/amis/) that come pre-installed with ML frameworks and the Neuron SDK. For a fully managed experience you will be able to use Amazon SageMaker which will enable you to seamlessly deploy your trained models on Inf1 instances. 

For customers who use popular frameworks like Tensorflow, MXNet and PyTorch a guide to help you get started with frameworks 
is available at [TODO](). I you wish to deploy without a framework, you can install Neuron using pip 
install and use Neuron to compile and deploy your models to Inf1 instances to run inference. More details on using Neuron without a framework are here [TODO]().


**Q: How would I select which Inf1 instance to use?**

The decision as to which Inf instance size to use is based upon the application and the performance/cost targets and may differ for each workload. To assist, the Neuron compiler provides guidance on expected latency and throughput for various Inf1 instance sizes, and TensorBoard profiling will show actual results when executed on a given instance. A guide to this process is available here: [TODO]()


<a name="general"></a>
## General Neuron FAQs

**Q: What ML models types and operators are supported by AWS Neuron?**

AWS Neuron includes a compiler that converts your trained machine learning model to Inferentia specific operators for execution. The Neuron compiler supports many commonly used machine learning models such as single shot detector (SSD) and ResNet for image recognition/classification, and Transformer and BERT for natural language processing and translation. Neuron supported operator list can be found in the release notes folders. We will continue to grow the list based on customer feedback, to allow data scientists to create new novel operator types to be supported by Inferentia. 

**Q: Why is a compiler needed and how do I use it?**

The Neuron compiler converts from a neural net graph consisting of operators like convolution and pool into a specific set of 
instructions that can be executed using the unique instruction-set of Inferentia.  The formats of these input graphs may be 
from TensorFlow or MXNet or ONNX.  The output from the compiler is the specific instructions for Inferentia encapsulated 
into a binary form, referred to as a Neuron Executable File Format (NEFF). NEFF contains a combination of these instructions, 
the weights and parameters of the pre-trained model and additional meta-data needed by the runtime. 

**Q: I am using a ML framework today – what will change for me to use this?**

The likely (but not only) workflow for customers is to build and train models in their training fleet, then partition and compile on a compute server, then distribute and execute their inference on Inf1 fleet. The distribution of the artifact may be via services like S3 for example.[TODO] 

TODO For full details on TensorFlow interface support, please refer:  http://github.com/aws/aws-mla/kaena/docs/tensor_flow_interfaces.md.
TODO For full details on MXnet interface support, please refer:  http://github.com/aws/aws-mla/kaena/docs/mxnet_flow_interfaces.md.


**Q: How do I take advantage of Inferentia’s NeuronCore Pipeline capability to lower latency?**

Inf1 instances with multiple Inferentia chips, such as Inf1.6xlarge or Inf1.24xlarge, support a fast chip-to-chip interconnect. Using the Neuron Processing Pipeline capability, you can split your model and load it to local cache memory across multiple chips. The Neuron compiler uses ahead-of-time (AOT) compilation technique to analyze the input model and compile it to fit across the on-chip memory of single or multiple Inferentia chips. Doing so enables the NeuronCores to have high-speed access to models and not require access to off-chip memory, keeping latency bounded while increasing the overall inference throughput. For more details, read the [TODO]().

**Q: Can I use TensorFlow networks from tfhub.dev as-is ? if not, what should I do?**
TODO
Yes. Models format can  be imported into Tensorflow, either as a standard model-server, in which case it appears as a simple command line utility, or via the Python based Tensorflow environment.  The primary additional step needed is to compile it. 


<a name="compiler"></a>
## Neuron compiler FAQs

**Q: In what environments can I use the compiler in?** 
The compilation step may be performed on any EC2 instance or on-premises. 
We recommend using a high-performance compute server of choice (C/M/R/Z instance types), for the fastest compile times and 
ease-of-use with a prebuilt DLAMI. Developers can also install Neuron in their own environments; this approach may work well 
for example when building a large fleet for inference, allowing the model creation, training and compilation to be done in the 
training fleet, with the NEFF files being distributed by a configuration management application to the inference fleet.

**Q: My current Neural Network is based on FP32, how can I use it with Neuron?**

Inferentia chips support FP16, BFloat16 mixed-precision data-types and INT8 (coming soon). It is common for Neural Networks to be trained in FP32, in which case the trained graph needs to be converted to one of these data types for execution on Inferentia. Neuron can compile and execute FP32 neural nets by automatically converting them to BFloat16. Given an input using FP32, the compiler output will ensure that the executed graph can accept input inference requests in FP32. 

**Q: What are some of the important compiler defaults I should be aware of?**
TODO
The compiler compiles the input graph for a single NeuronCore by default.  Using the The “`num-neuroncores`” option directs compiler to direct compiled graph to run on a specified number of NeuronCores. This number can be less than the total available NeuronCores on an instance. See performance tuning application note [link]TODO for more information. 

**Q: Which operators does Neuron support?**
TODO

**Q: Will I need to recompile again if I updated runtime/driver version?**
TODO
The compiler and runtime are committed to maintaining compatibility for major version releases with each other. The versoning is defined as major.minor, with compatibility for all versions with the same major number. If the versions mismatch, an error notification is logged and the load will fail. This will then require the model to be recompiled.

**Q: I have a NEFF binary, how can I tell which compiler version generated it?**
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

