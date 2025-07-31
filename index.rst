AWS Neuron SDK Documentation
============================
  
.. meta::
   :description: AWS Neuron SDK enables high-performance deep learning and generative AI on AWS Inferentia and Trainium instances. Get started with PyTorch, JAX, and distributed training.

AWS Neuron is a software development kit (SDK) that enables high-performance deep learning and generative AI workloads on `AWS Inferentia <https://aws.amazon.com/ai/machine-learning/inferentia/>`_ and `AWS Trainium <https://aws.amazon.com/ai/machine-learning/trainium/>`_ instances. Neuron provides a complete machine learning development experience with compiler optimization, runtime efficiency, and comprehensive tooling.

**Key Features:**

* **Native Framework Integration** - Seamlessly integrated with PyTorch and JAX, with distributed training libraries for large-scale workloads
* **Frontier Model Support** - Optimized for large language models including Llama 3.3-70B and Llama 3.1-405B
* **Performance Optimization** - Advanced compiler, profiling tools, and custom kernel support for maximum efficiency
* **Enterprise Ready** - Full integration with AWS services including SageMaker, EKS, ECS, and third-party platforms

**Supported Instance Types:** ``Inf1``, ``Inf2``, ``Trn1``, ``Trn2``, and ``Trn2`` UltraServer

.. grid:: 1 1 2 2
        :gutter: 2

        .. grid-item-card:: 
                :link: general/setup/index
                :link-type: doc
                :class-card: sd-text-center

                **Install the SDK**
                ^^^
                Step-by-step guide to installing the AWS Neuron SDK

        .. grid-item-card:: 
                :link: neuron-quickstart
                :link-type: ref
                :class-card: sd-text-center

                **Get Started with Neuron**
                ^^^
                Start building with step-by-step tutorials

        .. grid-item-card:: 
                :link: docs-quick-links
                :link-type: ref
                :class-card: sd-text-center
 
                **Quick Reference**
                ^^^
                Essential links and resources

        .. grid-item-card:: 
                :link: release-notes/index
                :link-type: doc
                :class-card: sd-text-center

                **Release Notes**
                ^^^
                Latest updates and changes to the AWS Neuron SDK


Contents
========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: **Get Started**
      :class-card: sd-border-1

      * :doc:`Quick Links <general/quick-start/docs-quicklinks>`
      * :doc:`Ask Q Developer <general/amazonq-getstarted>`
      * :doc:`Get Started with Neuron <general/quick-start/index>`
      * :doc:`Samples and Tutorials <general/models/index>`
      * :doc:`Performance <general/benchmarks/index>`
      * :doc:`What's New <release-notes/index>`
      * :doc:`Announcements <general/announcements/index>`

   .. grid-item-card:: **ML Frameworks**
      :class-card: sd-border-1

      * :doc:`PyTorch Neuron <frameworks/torch/index>`
      * :doc:`JAX NeuronX <frameworks/jax/index>`
      * :doc:`TensorFlow Neuron <frameworks/tensorflow/index>`

   .. grid-item-card:: **NeuronX Distributed (NxD)**
      :class-card: sd-border-1

      * :doc:`NxD Training <libraries/nxd-training/index>`
      * :doc:`NxD Inference <libraries/nxd-inference/index>`
      * :doc:`NxD Core <libraries/neuronx-distributed/index>`

   .. grid-item-card:: **Additional ML Libraries**
      :class-card: sd-border-1

      * :doc:`Third Party Libraries <libraries/third-party-libraries/third-party-libraries>`
      * :doc:`Transformers Neuron <libraries/transformers-neuronx/index>`
      * :doc:`AWS Neuron Reference for NeMo Megatron <libraries/nemo-megatron/index>`

   .. grid-item-card:: **Developer Tools**
      :class-card: sd-border-1

      * :doc:`Neuron DLAMI <dlami/index>`
      * :doc:`Neuron Containers <containers/index>`
      * :doc:`AWS Workload Orchestration <general/devflows/index>`
      * :doc:`Amazon SageMaker <general/devflows/sagemaker-flows>`
      * :doc:`Third-party Solutions <general/devflows/third-party-solutions>`
      * :doc:`Setup Guide <general/setup/index>`

   .. grid-item-card:: **Runtime & Tools**
      :class-card: sd-border-1

      * :doc:`Neuron Runtime <neuron-runtime/index>`
      * :doc:`Monitoring Tools <general/monitoring-tools>`
      * :doc:`Profiling Tools <general/profiling-tools>`
      * :doc:`Third-party Solutions <tools/third-party-solutions>`
      * :doc:`Other Tools <general/other-tools>`

   .. grid-item-card:: **Compiler**
      :class-card: sd-border-1

      * :doc:`Neuron Compiler <compiler/index>`
      * :doc:`Neuron Kernel Interface (Beta) <general/nki/index>`
      * :doc:`Neuron C++ Custom Operators <neuron-customops/index>`

   .. grid-item-card:: **Learning Neuron**
      :class-card: sd-border-1

      * :doc:`Architecture <general/arch/index>`
      * :doc:`Features <general/arch/neuron-features/index>`
      * :doc:`Application Notes <general/appnotes/index>`
      * :doc:`FAQ <general/faq>`
      * :doc:`Troubleshooting <general/troubleshooting>`
      * :doc:`Neuron Glossary <general/arch/glossary>`

.. toctree::
   :maxdepth: 1
   :caption: Overview
   :hidden:
   
   Quick Links </general/quick-start/docs-quicklinks>
   Ask Q Developer </general/amazonq-getstarted>
   Get Started with Neuron </general/quick-start/index>
   Samples and Tutorials </general/models/index>
   Performance </general/benchmarks/index>
   What's New </release-notes/index>
   Announcements </general/announcements/index>

.. toctree::
   :maxdepth: 1
   :caption: ML Frameworks
   :hidden:

   PyTorch Neuron </frameworks/torch/index>
   JAX NeuronX </frameworks/jax/index>
   TensorFlow Neuron </frameworks/tensorflow/index>

.. toctree::
   :maxdepth: 1
   :caption: NeuronX Distributed (NxD)
   :hidden:

   NxD Training </libraries/nxd-training/index>
   NxD Inference </libraries/nxd-inference/index>
   NxD Core </libraries/neuronx-distributed/index>

.. toctree::
   :maxdepth: 1
   :caption: Additional ML Libraries
   :hidden:

   Third Party Libraries </libraries/third-party-libraries/third-party-libraries>
   Transformers Neuron </libraries/transformers-neuronx/index>
   AWS Neuron reference for NeMo Megatron </libraries/nemo-megatron/index>

.. toctree::
   :maxdepth: 1
   :caption: Developer Flows
   :hidden:

   Neuron DLAMI </dlami/index>
   Neuron Containers </containers/index>
   AWS Workload Orchestration </general/devflows/index>
   Amazon SageMaker </general/devflows/sagemaker-flows>
   Third-party Solutions <general/devflows/third-party-solutions>
   Setup Guide </general/setup/index>

.. toctree::
   :maxdepth: 1
   :caption: Runtime & Tools
   :hidden:

   Neuron Runtime </neuron-runtime/index>
   Monitoring Tools </general/monitoring-tools>
   Profiling Tools </general/profiling-tools>
   Third-party Solutions <tools/third-party-solutions>
   Other Tools </general/other-tools>

.. toctree::
   :maxdepth: 1
   :caption: Compiler
   :hidden:

   Neuron Compiler </compiler/index>
   Neuron Kernel Interface (Beta) <general/nki/index>
   Neuron C++ Custom Operators </neuron-customops/index>

.. toctree::
   :maxdepth: 1
   :caption: Learning Neuron
   :hidden:

   Architecture </general/arch/index>
   Features </general/arch/neuron-features/index>
   Application notes </general/appnotes/index>
   FAQ </general/faq>
   Troubleshooting </general/troubleshooting>
   Neuron Glossary </general/arch/glossary>

.. toctree::
   :maxdepth: 1
   :caption: Legacy Software
   :hidden:

   Apache MXNet </frameworks/mxnet-neuron/index>

.. toctree::
   :maxdepth: 1
   :caption: Other Resources
   :hidden:

   Release Details </release-notes/release>
   Roadmap </general/roadmap-readme>
   Support </general/support>
   Archived content </archive/index>



*AWS and the AWS logo are trademarks of Amazon Web Services, Inc. or its affiliates. All rights reserved.*
