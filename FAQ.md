# AWS Neuron: Frequently Asked Questions 

* [Getting Started FAQs](#getting-started)
* [General Neuron FAQs](#general)
* [Neuron Compiler FAQs](#compiler)
* [Neuron Runtime FAQs](#runtime)
* [Troubleshooting FAQs](#troubleshooting)


<a name="getting-started"></a>
## Getting started with Neuron FAQs

**Q: How can I get started?**

You can start your workflow by training your model in one of the popular ML frameworks using EC2 GPU instances such as P3 or P3dn, or alternativelly download a pre-training model. Once the model is trained to your required accuracy, you can use the ML frameworks' API to invoke Neuron, the software development kit for Inferentia, to re-target(compile) the model for execution on Inferentia. This latter step is done once and the developer doesnt need to redo it as long as the model is not changing. Once compiled, the Inferentia binary can be loaded into one or more Inferentia, and can service inference calls. In order to get started quickly, you can use [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/) that come pre-installed with ML frameworks and the Neuron SDK. For a fully managed experience, you will soon be able to use Amazon SageMaker which will enable you to seamlessly deploy your trained models on Inf1 instances. 

For customers who use popular frameworks like TensorFlow, MXNet and PyTorch, a guide to help you get started with frameworks 
is available at [MXNet Neuron](./docs/mxnet-neuron/readme.md) and  [TensorFlow Neuron](./docs/tensorflow-neuron/readme.md) and [Pytorch Neuron](./docs/pytorch-neuron/README.md). 

**Q: How do I select which Inf1 instance to use?**

The decision as to which Inf1 instance size to use is based upon the application and the performance/cost targets and may differ for each workload. To assist, the Neuron compiler provides guidance on expected performance for various Inf1 instance sizes, and TensorBoard profiling will show actual results when executed on a given instance. A guide to this process is available here: [TensorBoard Neuron](./docs/neuron-tools/getting-started-tensorboard-neuron.md).

Ultimately, AWS encourages developers try out all the Inf1 instance sizes (which can be done at low cost and quickly in the cloud environment), with their specific models, and choose the right instance for them.


<a name="general"></a>
## General Neuron FAQs

**Q: What ML models types and operators are supported by AWS Neuron?**

AWS Neuron includes a compiler that converts your trained machine learning model to Inferentia binary object for execution. The Neuron compiler supports many commonly used machine learning models such as ResNet for image recognition, and Transformer and BERT for natural language processing and translation. A list of supported ML operators and supported inputs are in [release notes](./release-notes/). AWS continues to grow these lists based on customers' feedback. 

**Q: Why is a compiler needed, and how do I use it?**

The Neuron compiler converts from a framework level Neural Network graph, with operators like convolution and pooling, into the hardware-specific instruction set of Inferentia, build the schedule for execution these instructions, and convers the model parameters into format that Inferentia can consume.  The supported input formats include TensorFlow, PyTorch (shortly), MXNet, or ONNX. The output from the compiler is a Neuron Executable File Format (NEFF) artifact. NEFF contains a combination of binary code, the model parameters, and additional meta-data needed by the runtime and profiler. 

**Q: I am using a ML framework today – what will change for me to use this?**

The likely (but not only) workflow for developers is a hand-off of pre-trained model to large scale of inference fleet.
To use Inferentia and Inf1 instances, the developer need to perform one-time compilation of the pre-trained model to generate NEFF, and use this as the inference model in fleet of Inf1 instances.

[TensorFlow interface support](./docs/tensorflow-neuron/readme.md)

[MXNet interface support](./docs/mxnet-neuron/readme.md)


**Q: What is a NeuronCore Pipeline ? and How do I take advantage of it?**

A NeuronCore Pipeline is a unique technique to shard a specific Neural Network across multiple NeuronCores, to take advantage of the large on-chip cache that will typically increase throughput and reduce latency at low batch sizes. All Inf1 instances support it, and the Inf1 instances with multiple Inferentia accelerators, such as inf1.6xlarge or inf1.24xlarge support it thanks to the fast chip-to-chip interconnect. 

Developers can choose to use NeuronCore Pipeline mode during compile stage, with an opt-in flag. [Neuron Compiler](./docs/neuron-cc/readme.md) provides further details. 

**Q: NeuronCores, NeuronCore Groups and NeuronCore Pipelines: What do they do?**

Each Inferentia chip has four compute engines called NeuronCores. A NeuronCore Group is a way to aggregate NeuronCores to improve hardware utilization and assign models with the right compute sizing for a specific application. If you want to run mutliple models in parallel, you can assign different models to separate NeuronCore Groups. A model compiled to use multiple NeuronCores in a NeuronCorePipeline can be assigned to a NeuronCore Group with enough NeuronCores to load it. Finally- it is also possible for sets of Inferentia devices to be mapped to separate Neuron Runtimes. The documents in the [docs](./docs) folder has more information and examples.

**Q: Can I use TensorFlow networks from tfhub.dev as-is ? if not, what should I do?**

Yes. Models format can  be imported into TensorFlow, either as a standard model-server, in which case it appears as a simple command line utility, or via the Python based TensorFlow environment.  The primary additional step needed is to compile the model into Inferentia NEFF format. 


<a name="compiler"></a>
## Neuron compiler FAQs

**Q: Where can I compile to Neuron?** 

The one-time compilation step from the standard framework-level model to Inferentia binary may be performed on any EC2 instance or even on-premises. 

We recommend using a high-performance compute server of choice (C5 or z1d instance types), for the fastest compile times and 
ease of use with a prebuilt [DLAMI](https://aws.amazon.com/machine-learning/amis/). Developers can also install Neuron in their own environments; this approach may work well for example when building a large fleet for inference, allowing the model creation, training and compilation to be done in the training fleet, with the NEFF files being distributed by a configuration management application to the inference fleet.

**Q: My current Neural Network is based on FP32, how can I use it with Neuron?**

Inferentia chips support FP16, BFloat16 mixed-precision data-types and INT8. It is common for Neural Networks to be trained in FP32, in which case the trained graph needs to be converted to one of these data types for execution on Inferentia. Neuron can compile and execute FP32 neural nets by automatically converting them to BFloat16. Given an input using FP32, the compiler output will ensure that the executed graph can accept input inference requests in FP32. Also see this [Tech Note](./docs/technotes/data-types.md).

**Q: What are some of the important compiler defaults I should be aware of?**

The compiler compiles the input graph for a single NeuronCore by default.  Using the The “`num-neuroncores`” option directs compiler to direct compiled graph to run on a specified number of NeuronCores. This number can be less than the total available NeuronCores on an instance. See performance tuning application note [link](.) for more information (TODO). 

**Q: Which operators does Neuron support?**
* [Neuron-cc TensorFlow Operators](./release-notes/neuron-cc-ops/neuron-cc-ops-tensorflow.md)
* [Neuron-cc MXNet Operators](./release-notes/neuron-cc-ops/neuron-cc-ops-mxnet.md)
* [Neuron-cc ONNX Operators](./release-notes/neuron-cc-ops/neuron-cc-ops-onnx.md)


If your model contains operators missing from the above list, please post a message on the Neuron developer forum to let us know.

**Q: Any operators that Neuron doesn't support?**
Models with control-flow and dynamic shapes are not supported. You will need to partition the model using the framework prior to compilation. See the [Neuron compiler](./docs/neuron-cc/readme.md). 

**Q: Will I need to recompile again if I updated runtime/driver version?**

The compiler and runtime are committed to maintaining compatibility for major version releases with each other. The versoning is defined as major.minor, with compatibility for all versions with the same major number. If the versions mismatch, an error notification is logged and the load will fail. This will then require the model to be recompiled.

**Q: I have a NEFF binary, how can I tell which compiler version generated it?**
We will bring a utility out to help with this soon.

**Q: How long does it take to compile?**
It depends on the model and its size and complexity, but this generally takes a few minutes. 

<a name="runtime"></a>
## Neuron runtime FAQs

**Q: How does Neuron connect to all the Inferentia chips in an Inf1 instance?**

By default, a single runtime process will manage all assigned Inferentias, including running the Neuron Core Pipeline mode. if needed, you can configure multiple KRT processes each managing a separate group of Inferentia chips. For more details please refer to [Neuron Runtime readme](./docs/neuron-runtime/README.md) 


**Q: Where can I get logging and other telemetry information?**
See this document on how to collect logs: [Neuron log collector](./docs/neuron-tools/tutorial-neuron-gatherinfo.md)

**Q: What about RedHat or other versions of Linux?**
We dont officially support it yet. 

**Q: What about Windows?**

Windows is not supported at this time.

**Q: How can I use Neuron in a container based environment? Does Neuron work with ECS and EKS?**
ECS and EKS support is coming soon. Containers can be configured as shown [here](./docs/neuron-container-tools/README.md)


**Q: How can I take advantage of multiple NeuronCores to run multiple inferences in parallel?**
Examples of this for TensorFlow are found [here](./docs/tensorflow-neuron/tutorial-NeuronCore-Group.md) as well as for MXNet  [here](./docs/mxnet-neuron/tutorial-neuroncore-groups.md)


<a name="troubleshooting"></a>
## Troubleshooting FAQs

**Q: Performance is not what I expect it to be, what's the next step?**
Please check our [Tech Notes](./docs/technotes/README.md) section on performance tuning and other notes on how to use pipelining and batching to improve performance!

**Q: Do I need to worry about size of model and size of inferentia memory? what problems can I expect to have?**
Errors like this wil be logged and can be found as shown [here](./docs/neuron-tools/tutorial-neuron-gatherinfo.md)

**Q: How can I  debug / profile my inference request?**
See [Neuron Tensorboard](./docs/neuron-tools/getting-started-tensorboard-neuron.md)

