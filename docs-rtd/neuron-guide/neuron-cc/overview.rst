.. _neuron-cc-overview:

Overview
========

The Neuron Compiler is an Ahead-of-Time (AoT) compiler that accepts
Machine Learning models in various formats (TensorFlow, MXNet, PyTorch)
and optimizes them to run on AWS Inferentia chips.

AoT compilation requires that dynamic tensor shapes (dimension sizes) of
all tensors in the compute-graph are known at compilation time, in order
for the compiler to make informed decisions. Model compilation with
shapes that cannot be determined at compile time will fail.

It is common for developers to train models in FP32 to avoid the
challenges of low-precision training (e.g. loss-scaling, etc). However,
during inference, developers typically look for the most performant and cost-effective
methods. In order to address these two requirements, Neuron supports FP32
auto-casting, which takes trained FP32 models as input and then runs
them at speed of 16-bit using BFloat16 model, using Neuron's FP32 to
BF16 auto conversion.

The Neuron compiler is used within an integrated framework,
such as TensorFlow, PyTorch or MXNet. From that framework, ML models are sent to
the compiler and the results then sent back. This compile step will
often be done on a compilation server. The resulting compiler artifact is called
a NEFF file (Neuron Executable File Format). NEFF files are loaded into an
Inferentia device, using the Neuron Runtime. The NEFF is actually encapsulated
inside a saved version of that model, which the framework can use. 
