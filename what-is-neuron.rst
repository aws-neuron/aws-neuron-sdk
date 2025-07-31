. meta::
   :description: AWS Neuron is a software development kit for high-performance machine learning on AWS Inferentia and Trainium, enabling developers to compile, optimize, and deploy deep learning models at scale.

What is AWS Neuron?
===================

AWS Neuron is a software development kit (SDK) designed to optimize machine learning (ML) model performance for AWS Inferentia and Trainium accelerators. Neuron enables developers to efficiently deploy, scale, and manage deep learning workloads in production environments with high-performance, cost-effective custom silicon.

Key Features of AWS Neuron
--------------------------

- **Optimized Model Compilation**
  AWS Neuron Compiler translates popular frameworks (such as PyTorch and TensorFlow) into instructions for Inferentia and Trainium chips. This provides accelerated inference and training without requiring model redevelopment.

- **Seamless Framework Integration**
  Neuron offers deep integration with ML frameworks like PyTorch, TensorFlow, and Transformers through AWS-maintained plugins and libraries. Developers can access Neuron-optimized kernels, distributed training modules, and advanced configuration options.

- **Neuron Runtime for Production Deployment**
  The Neuron Runtime manages executable models on Neuron devices, handling resource allocation, execution, and monitoring for maximum throughput and minimal latency.

- **Support for Advanced Model Types**
  Neuron supports a wide range of model architectures, including large language models, computer vision networks, and custom operator workflows. Latest features enable efficient deployment of state-of-the-art (SOTA) and “frontier” models.

- **Distributed and Scalable Training/Inference**
  With NeuronX Distributed Libraries, users can leverage data, tensor, and pipeline parallelism to accelerate model training and large-scale inference across multiple instances.

- **Extensive Tools for Debugging, Profiling, and Monitoring**
  A comprehensive set of tools empowers developers to profile model performance, debug execution, and monitor resource utilization.

Typical Use Cases for Neuron Developer Customers
------------------------------------------------

- **Production-Scale ML Inference**
  Deploy deep learning models for high-throughput, low-latency inference in applications such as search, recommendation, natural language processing (NLP), and computer vision.

- **Distributed Training of Large Models**
  Accelerate the training of large and complex models, leveraging multiple Neuron devices and distributed training libraries.

- **Cost Optimization for AI Workloads**
  Reduce infrastructure costs without sacrificing speed by taking advantage of Neuron-optimized hardware and software stacks.

- **Custom Model Optimization**
  Extend or customize model support with Neuron’s Custom Operator API or the Neuron Kernel Interface.

Learn More
----------

- `Getting Started with Neuron <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/get-started.html>`_
- `Supported Frameworks & Model Types <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/supported-sw.html>`_
- `Neuron Compiler Overview <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-compiler/index.html>`_
- `Neuron Runtime System <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-runtime/index.html>`_
- `Distributed Training with Neuron <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuronx-distributed/index.html>`_
- `Profiling and Debugging Tools <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/debug/index.html>`_
