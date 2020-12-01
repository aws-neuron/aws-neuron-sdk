</br>
</br>

Please view our documentation at **[https://awsdocs-neuron.readthedocs-hosted.com/](https://awsdocs-neuron.readthedocs-hosted.com/)** 

**Please note this file will be deprecated.**

</br>
</br>



# Neuron Compiler

The Neuron Compiler is an Ahead-of-Time (AoT) compiler that accepts Machine Learning models in various formats (TensorFlow, MXNet, PyTorch, ONNX) and optimizes them to run on AWS Inferentia chips.

AoT compilation requires that dynamic tensor shapes (dimension sizes) of all tensors in the compute-graph are known at compilation time, in order for the compiler to make informed decisions. Models compilation with shapes that cannot be determined at compile time will fail.

It is common for developers to train models in FP32 to avoid the challenges of low-precision training (e.g. loss-scaling, etc). However, during inference, developers typically look for the most cost-effective method. In order to address these two requirements, Neuron supports FP32 auto-casting, which takes trained FP32 models as input, and then runs them at speed of 16-bit using BFloat16 model, using Neuron's FP32 to BF16 auto conversion.

The Neuron compiler is most often used within an integrated framework, such as TensorFlow-Neuron. From that framework, 
ML models are sent to the compiler and the results then sent back. This compile step will often be done on a compilatoin server 
during the build steps. The resulting compiler artifact is called a NEFF file (Neuron Executable File Format); NEFF files are loaded into an Inferntia device, using the Neuron Runtime. If being done from an integrated framework, the NEFF is actually encapsulated inside a saved version of that model, which the framework can use. Once compiled, the execution of the model on an Inf1 instance will also typically be done from within the integrated framework: the ML model will be executed from the framework, with the sections of the ML model that we compiled for acceleration being executed on the Inferentia via the Neuron Runtime.

## Further Information:

[Neuron Compiler Command Reference](./command-line-reference.md)
