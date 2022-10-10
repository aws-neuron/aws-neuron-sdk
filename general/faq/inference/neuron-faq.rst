.. _neuron-f1-faq:

Inference with Neuron - FAQ
---------------------------

.. contents:: Table of contents
   :local:
   :depth: 1

What ML model types and operators are supported by AWS Neuron?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AWS Neuron includes a compiler that converts your trained machine
learning models to a binary object for execution. The Neuron
compiler supports many commonly used machine learning operators used in computer vision, natural language processing, recommender engines and more. A list of supported ML operators and supported inputs are in :ref:`neuron-supported-operators` .

It's important to mention that to get good performance doesn't require all of the model operators to run on the chip. In many cases, some of the operators will continue to run on the instance CPUs, like the case of embeddings or image pre-processing, and will still provide a compelling end to end performance. We call this approach auto-partitioning, where the Neuron compiler optimizes the model execution based on operators that are most suitable to run on the CPU or the chip.

For the latest model architecture support, please refer to the model architecuture fit and performance pages.

Why is a compiler needed, and how do I use it?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Neuron compiler converts a model from a framework level Neural Network
graph, with operators like convolution and pooling, into a
Neuron Device-specific instruction set, builds the schedule for
execution of these instructions, and converts the model parameters into
format that the neuron device can consume. The supported input formats include
TensorFlow, PyTorch, and MXNet. The output from the
compiler is a Neuron Executable File Format (NEFF) artifact. The NEFF
contains a combination of binary code, the model parameters, and
additional meta-data needed by the Neuron runtime and profiler.

I am using a ML framework today – what will change for me to use this?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use Inferentia within the Inf1 instances, the developer needs to perform one-time compilation
of the pre-trained model to generate a NEFF, and use this as the inference
model in fleet of Inf1 instances.

-  :ref:`tensorflow-neuron`
-  :ref:`neuron-pytorch`
-  :ref:`neuron-mxnet`

What is a NeuronCore Pipeline? How do I take advantage of it?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A NeuronCore Pipeline is a unique technique to shard a specific Neural
Network across multiple NeuronCores, to take advantage of the large
on-chip cache instead of moving data in and out of external memory. The result is an increased throughput and reduce latency
typically important for real-time inference applications. All Inf1 instances support it, and the Inf1
instances with multiple Inferentia accelerators, such as inf1.6xlarge or
inf1.24xlarge support it thanks to the fast chip-to-chip interconnect.

Developers can choose to use NeuronCore Pipeline mode during compile
stage, with an opt-in flag. :ref:`neuron-cc` provides further details.

NeuronCores, NeuronCore Groups and NeuronCore Pipelines: What do they do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each Inferentia chip has four compute engines called NeuronCores. A
NeuronCore Group is a way to aggregate NeuronCores to increase hardware
utilization and assign models with the right compute sizing for a
specific application. If you want to run mutiple models in parallel,
you can assign different models to separate NeuronCore Groups. A model
compiled to use multiple NeuronCores in a NeuronCore Pipeline can be
assigned to a NeuronCore Group with enough NeuronCores to load into.
Finally- it is also possible for sets of Inferentia devices to be mapped
to separate Neuron Runtimes. :ref:`neuron-features-index` section has more
information and examples.

Can I use TensorFlow networks from tfhub.dev as-is ? if not, what should I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. Models format can be imported into TensorFlow, either as a standard
model-server, in which case it appears as a simple command line utility,
or via the Python based TensorFlow environment. The primary additional
step needed is to compile the model into Inferentia NEFF format.
