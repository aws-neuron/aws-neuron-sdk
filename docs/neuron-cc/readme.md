# Neuron Compiler

The Neuron Compiler is an Ahead-of-Time compiler that accepts Machine Learning Models in various formats (Tensorflow, MXnet, ONNX) and 
converts and optimizes them to run on the AWS Inferentia Machine Learning Accelerator.

The Neuron compiler analyzes user neural network and performs optimizations such as loop fusion, scheduling, 
and memory management, which significantly improves inference throughput and memory usage.

As an Ahead of Time compiler, dynamic tensor shapes (dimension sizes) of all tensors in the neural network must be known at compilation time in 
order for the compiler to make sound decisions. If any shape cannot be determined at compile time, the Neuron 
compilation fails with an error.

We heard from many developers that although training models in FP32 is time consuming, the 
advantage it provides is best accuracy. However, 32-bit compute is expensive and high power
and its really hard to move to lower 16-bit floating point or Integers to optimize.
Neuron and Inferentia is the first ML accelerator in the AWS cloud that can take a 32-bit trained model 
and run them at speed of 16-bit using BFloat16 model, using our FP32 to BF16 auto casting.

The Neuron compiler is most often used within an integrated framework, such as Tensorflow-Neuron. From that framework, 
ML models are sent to the compiler and the results then sent back. This compile step will often be done on a compiler server 
during the build steps. The resulting compiler artifacts - a NEFF (Neuron Executable File Format) - can then be loaded into an Inferntia
using the Neuron Runtime. If being done from an integrated framework, the NEFF is actually encapsulated inside a saved version of that model
that the framework can use. Once compiled, the execution of the model on an Inf1 instance  will also typically be done from 
within the integrated framework: the ML model will be executed from the framework, with the sections of the ML model that we compiled for 
acceleration being executed on the Inferentia via the Neuron Runtime.

## Further Information:

[Neuron Compiler Command Reference](./command-line-reference.md)
