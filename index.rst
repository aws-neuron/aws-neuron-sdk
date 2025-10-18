.. meta::
   :description: AWS Neuron SDK enables high-performance deep learning and generative AI on AWS Inferentia and Trainium instances. Get started with PyTorch, JAX, and distributed training.
   :date-modified: 2025-10-03

.. _neuron_home:

AWS Neuron SDK Documentation
============================

:ref:`AWS Neuron <what-is-neuron>` is a software development kit (SDK) that enables high-performance deep learning and generative AI workloads on `AWS Inferentia <https://aws.amazon.com/ai/machine-learning/inferentia/>`_ and `AWS Trainium <https://aws.amazon.com/ai/machine-learning/trainium/>`_ instances. Neuron provides a complete machine learning development experience with compiler optimization, runtime efficiency, and comprehensive tooling.

**Key Features:**

* **Native Framework Integration** - Seamlessly integrated with PyTorch and JAX, with distributed training libraries for large-scale workloads
* **Frontier Model Support** - Optimized for large language models including Llama 3.3-70B and Llama 3.1-405B
* **Performance Optimization** - Advanced compiler, profiling tools, and custom kernel support for maximum efficiency
* **Enterprise Ready** - Full integration with AWS services including SageMaker, EKS, ECS, and third-party platforms

**Supported Instance Types:** ``Inf1``, ``Inf2``, ``Trn1``, ``Trn2``, and ``Trn2`` UltraServer

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 
      :link: about-neuron/index
      :link-type: doc
      :class-card: sd-text-center
 
      **About the AWS Neuron SDK**
      ^^^
      Learn about the AWS Neuron SDK, its components, and supported hardware

   .. grid-item-card:: 
      :link: setup/index
      :link-type: doc
      :class-card: sd-text-center

      **Install the AWS Neuron SDK**
      ^^^
      Step-by-step guides for installing the AWS Neuron SDK

   .. grid-item-card:: 
      :link: about-neuron/quick-start/index
      :link-type: doc
      :class-card: sd-text-center

      **Get started with the Neuron SDK**
      ^^^
      Start building with step-by-step tutorials

   .. grid-item-card:: 
      :link: release-notes/index
      :link-type: doc
      :class-card: sd-text-center

      **Release notes**
      ^^^
      Latest updates and changes to the AWS Neuron SDK
   

Contents
========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: **Learn about AWS Neuron**
      :class-card: sd-border-1

      * :doc:`About AWS Neuron <about-neuron/index>`
      * :doc:`What's New <release-notes/index>`
      * :doc:`Announcements <about-neuron/announcements/index>`
      * :doc:`AWS Neuron features <about-neuron/arch/neuron-features/index>`
      * :doc:`AWS Neuron architecture <about-neuron/arch/index>`
      * :doc:`Ask Q Developer <about-neuron/amazonq-getstarted>`
      * :doc:`Performance <about-neuron/benchmarks/index>`
      * :doc:`Troubleshooting <about-neuron/troubleshooting>`
      * :doc:`Application Notes <about-neuron/appnotes/index>`
      * :doc:`FAQ <about-neuron/faq>`
      * :doc:`AWS Neuron term glossary <about-neuron/arch/glossary>`

   .. grid-item-card:: **Get started with AWS Neuron**
      :class-card: sd-border-1

      * :ref:`Neuron quickstarts <neuron-quickstart>`
      * :doc:`Samples and tutorials <about-neuron/models/index>`
  
   .. grid-item-card:: **Develop with AWS Neuron**
      :class-card: sd-border-1

      * :doc:`Setup guides <setup/index>`

      * :doc:`Developer tools </tools/index>`

   .. grid-item-card:: **AWS Neuron-supported ML frameworks**
      :class-card: sd-border-1

      * :doc:`PyTorch Neuron <frameworks/torch/index>`
      * :doc:`JAX NeuronX <frameworks/jax/index>`

   .. grid-item-card:: **NeuronX Distributed (NxD) libraries**
      :class-card: sd-border-1

      * :doc:`NxD libraries overview <libraries/index>`
      * :doc:`NxD Training <libraries/nxd-training/index>`
      * :doc:`NxD Inference <libraries/nxd-inference/index>`
      * :doc:`NxD Core <libraries/neuronx-distributed/index>`

   .. grid-item-card:: **Additional ML libraries**
      :class-card: sd-border-1

      * :doc:`Third-party libraries <libraries/third-party-libraries//third-party-libraries>`

   .. grid-item-card:: **Developer workloads**
      :class-card: sd-border-1

      * :doc:`Workload orchestration </devflows/index>`
      * :doc:`AWS Neuron Deep Learning Machine Images (DLAMIs) <dlami/index>`
      * :doc:`AWS Neuron Deep Learning Containers (DLCs) <containers/index>`

   .. grid-item-card:: **Runtime & Collectives**
      :class-card: sd-border-1

      * :doc:`Neuron Runtime <neuron-runtime/index>`
  
   .. grid-item-card:: **Neuron Kernel Interface (NKI)**
      :class-card: sd-border-1

      * :doc:`Neuron Kernel Interface (NKI) <nki/index>`
      * :doc:`NKI developer guide <nki/developer_guide>`

   .. grid-item-card:: **Neuron Compiler**
      :class-card: sd-border-1

      * :doc:`Neuron Compiler <compiler/index>`
      * :doc:`Neuron C++ Custom Operators <neuron-customops/index>`

.. toctree::
   :maxdepth: 1
   :hidden:
   
   About Neuron </about-neuron/index>
   Neuron architecture </about-neuron/arch/index>
   What's New </release-notes/index>
   Announcements </about-neuron/announcements/index>

.. toctree::
    :maxdepth: 1
    :caption: Get Started
    :hidden:

    Quickstarts </about-neuron/quick-start/index>
    Setup Guides </setup/index>
    Developer tools </tools/index>
    Models and Tutorials </about-neuron/models/index>
    Ask Amazon Q </about-neuron/amazonq-getstarted>

.. toctree::
   :maxdepth: 1
   :caption: Orchestrate and Deploy
   :hidden:

   Developer workloads </devflows/index>
   Neuron DLAMI </dlami/index>
   Neuron Containers </containers/index>
   Amazon SageMaker </devflows/sagemaker-flows>
   Third-party Solutions </devflows/third-party-solutions>

.. toctree::
   :maxdepth: 2
   :caption: Use ML Frameworks
   :hidden:

   About Neuron frameworks </frameworks/index>
   PyTorch Neuron </frameworks/torch/index>
   JAX NeuronX </frameworks/jax/index>
   TensorFlow Neuron </frameworks/tensorflow/index>

.. toctree::
   :maxdepth: 2
   :caption: Work with Neuron Libraries
   :hidden:

   About NxD </libraries/index>
   Training </libraries/nxd-training/index>
   Inference </libraries/nxd-inference/index>
   Core libraries </libraries/neuronx-distributed/index>
   Third-party </libraries/third-party-libraries/third-party-libraries>

.. toctree::
   :maxdepth: 1
   :caption: Runtime & Collectives
   :hidden:

   Neuron Runtime </neuron-runtime/index>
   
.. toctree::
   :maxdepth: 1
   :caption: Neuron Kernel Interface (NKI)
   :hidden:

   Neuron Kernel Interface (NKI) </nki/index>
   NKI Samples and Tutorials </nki/tutorials/index>

.. toctree::
   :maxdepth: 1
   :caption: Compiler
   :hidden:

   Neuron Compiler </compiler/index>
   Neuron C++ Custom Operators </neuron-customops/index>

.. toctree::
   :maxdepth: 1
   :caption: Legacy software and docs
   :hidden:

   Apache MXNet </frameworks/mxnet-neuron/index>
   Archived content </archive/index>


*AWS and the AWS logo are trademarks of Amazon Web Services, Inc. or its affiliates. All rights reserved.*
