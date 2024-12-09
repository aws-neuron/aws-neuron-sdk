Welcome to AWS Neuron
=====================
  
.. meta::
   :description: Neuron SDK is powering AWS Inferentia and Trainium based instances, natively integrated into PyTorch and TensorFlow. Enabling you to quicky start with Amazon EC2, AWS Sagemaker, ECS and EKS.

AWS Neuron is the software development kit (SDK) used to run deep learning and generative AI workloads on `AWS Inferentia <https://aws.amazon.com/ai/machine-learning/inferentia/>`_ and `AWS Trainium <https://aws.amazon.com/ai/machine-learning/trainium/>`_ powered Amazon EC2 instances and UltraServers (`Inf1 <https://aws.amazon.com/ec2/instance-types/inf1/>`_, `Inf2 <https://aws.amazon.com/ec2/instance-types/inf2/>`_, `Trn1 <https://aws.amazon.com/ec2/instance-types/trn1/>`_ , `Trn2 <https://aws.amazon.com/ec2/instance-types/trn2/>`_ and `Trn2 UltraServer <https://aws.amazon.com/ec2/ultraservers/>`_). It includes a compiler, runtime, training and inference libraries, and profiling tools. Neuron supports customers in their end-to-end ML development lifecycle including building and deploying deep learning and AI models.

* **ML Frameworks and Libraries** - Neuron integrates with :ref:`PyTorch  <pytorch-neuronx-main>` and :ref:`JAX <jax-neuron-main>`, and offers :ref:`NxD Training <nxdt>` and :ref:`NxD Inference <nxdi-index>` PyTorch libraries for distributed workflows. It also supports :ref:`third party libraries <third-party-libraries>` such as Hugging Face Optimum Neuron, PyTorch Lightning, and AXLearn library for JAX model training.

* **Frontier Models Support**  - Neuron supports frontier models such as Llama3.3-70b and Llama Llama3.1-405b.

* **Developer Tools** - Neuron provides :ref:`health monitoring, observability <monitoring_tools>`, and :ref:`profiling tools <profiling-tools>` for AWS Inferentia and Trainium instances. It tracks hardware utilization, model execution metrics, and device information. The Neuron Profiler identifies performance bottlenecks. Neuron also integrates with :ref:`third-party  <third-party-tool-solutions>`  monitoring tools like Datadog and Weights and Biases.

* **Compute Kernels** - :ref:`Neuron Kernel Interface (NKI) <neuron-nki>` provides direct hardware access on AWS Trainium and Inferentia, enabling customer to write optimized kernel. NKI provides a Python-based environment with Triton-like syntax. Neuron supports custom C++ operators, allowing developers to extend functionality and enhance deep learning models.


* **Workloads Orchestrations and Managed Services**  - Neuron enables you to use Trainium and Inferentia-based instances with Amazon services such as SageMaker, EKS, ECS, ParallelCluster, and Batch. and :ref:`third-party solutions <third-party-devflow-solutions>` like Ray (Anyscale) and Domino Data Lab.


* **Architecture**  - To understand the architecture of AWS AI Chips, Trn/Inf instances, and NeuronCores visit :ref:`neuroninstances-arch`, :ref:`neurondevices-arch` and :ref:`neuroncores-arch`.


For more information about the latest AWS Neuron release, see :ref:`latest-neuron-release` and check the :ref:`announcements-main` page.

For list of AWS Neuron model samples and tutorials on Amazon EC2 ``Inf1``, ``Inf2``, ``Trn1``, and ``Trn2`` instances, 
see :ref:`model_samples_tutorials`.


.. card:: Get Started with Neuron
      :link: neuron-quickstart
      :link-type: ref


.. card:: Neuron Quick Links
      :link: docs-quick-links
      :link-type: ref


Overview
========

.. toctree::
   :maxdepth: 1
   :caption: Overview
   
   Quick Links </general/quick-start/docs-quicklinks>
   Ask Q Developer </general/amazonq-getstarted>
   Get Started with Neuron </general/quick-start/index>
   Samples and Tutorials </general/models/index>
   Performance </general/benchmarks/index>
   What’s New </release-notes/index>
   Announcements </general/announcements/index>


ML Frameworks
=============

.. toctree::
   :maxdepth: 1
   :caption: ML Frameworks

   PyTorch Neuron </frameworks/torch/index>
   JAX Neuron </frameworks/jax/index>
   TensorFlow Neuron </frameworks/tensorflow/index>

NeuronX Distributed (NxD)
=========================

.. toctree::
   :maxdepth: 1
   :caption: NeuronX Distributed (NxD)

   NxD Training (Beta) </libraries/nxd-training/index>
   NxD Inference (Beta) </libraries/nxd-inference/index>
   NxD Core </libraries/neuronx-distributed/index>
   

Additional ML Libraries
=======================

.. toctree::
   :maxdepth: 1
   :caption: Additional ML Libraries

   Third Party Libraries </libraries/third-party-libraries/third-party-libraries>
   Transformers Neuron </libraries/transformers-neuronx/index>
   AWS Neuron reference for NeMo Megatron </libraries/nemo-megatron/index>

Developer Flows
===============

.. toctree::
   :maxdepth: 1
   :caption: Developer Flows

   Neuron DLAMI </dlami/index>
   Neuron Containers </containers/index>
   AWS Workload Orchestration </general/devflows/index>
   Amazon SageMaker </general/devflows/sagemaker-flows>
   Third-party Solutions <general/devflows/third-party-solutions>
   Setup Guide </general/setup/index>


Runtime & Tools
===============


.. toctree::
   :maxdepth: 1
   :caption: Runtime & Tools

   Neuron Runtime </neuron-runtime/index>
   Monitoring Tools </general/monitoring-tools>
   Profiling Tools </general/profiling-tools>
   Third-party Solutions <tools/third-party-solutions>
   Other Tools </general/other-tools>

Compiler
========

.. toctree::
   :maxdepth: 1
   :caption: Compiler

   Neuron Compiler </compiler/index>
   Neuron Kernel Interface (Beta) <general/nki/index>
   Neuron C++ Custom Operators </neuron-customops/index>

Learning Neuron
===============

.. toctree::
   :maxdepth: 1
   :caption: Learning Neuron

   Architecture </general/arch/index>
   Features </general/arch/neuron-features/index>
   Application notes </general/appnotes/index>
   FAQ </general/faq>
   Troubleshooting </general/troubleshooting>
   Neuron Glossary </general/arch/glossary>


Legacy Software
===============

.. toctree::
   :maxdepth: 1
   :caption: Legacy Software

   Apache MXNet </frameworks/mxnet-neuron/index>

About Neuron
============

.. toctree::
   :maxdepth: 1
   :caption: About Neuron

   Release Details </release-notes/release>
   Roadmap </general/roadmap-readme>
   Support </general/support>


