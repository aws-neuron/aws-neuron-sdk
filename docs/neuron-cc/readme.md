# Neuron Compiler

The Neuron Compiler is an Ahead-of-Time (AoT) compiler that accepts Machine Learning Models in various formats (Tensorflow, MXnet, ONNX) and converts and optimizes them to run on the AWS Inferentia Machine Learning Accelerator.

The Neuron compiler analyzes the user-provided compute-graph, and performs various optimizations such as loop-fusion, tensorization, scheduling, and memory management, which significantly improves inference throughput and memory usage.

AoT compilation requires that dynamic tensor shapes (dimension sizes) of all tensors in the compute-graph are known at compilation time, in order for the compiler to make sound decisions. If any shape cannot be determined at compile time compilation will fail.

It is common for developers to train in FP32, for avoiding the challenges of low-precision training (e.g. loss-scaling, etc). However, during inference, developers typically look for the most cost-effective target. In order to address these two requirements, Neuron supports auto-conversion, which takes FP32 models as input, and then runs them at speed of 16-bit using BFloat16 model, using our FP32 to BF16 auto conversion.
The Neuron compiler is most often used within an integrated framework, such as Tensorflow-Neuron. From that framework, 
ML models are sent to the compiler and the results then sent back. This compile step will often be done on a compiler server 
during the build steps. The resulting compiler artifacts - a NEFF (Neuron Executable File Format) - can then be loaded into an Inferntia device, using the Neuron Runtime. If being done from an integrated framework, the NEFF is actually encapsulated inside a saved version of that model, which the framework can use. Once compiled, the execution of the model on an Inf1 instance will also typically be done from within the integrated framework: the ML model will be executed from the framework, with the sections of the ML model that we compiled for acceleration being executed on the Inferentia via the Neuron Runtime.

## Further Information:

[Neuron Compiler Command Reference](./command-line-reference.md)
