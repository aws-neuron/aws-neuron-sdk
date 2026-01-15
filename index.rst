.. meta::
   :description: AWS Neuron SDK enables high-performance deep learning and generative AI on AWS Inferentia and Trainium instances. Get started with PyTorch, JAX, and distributed training.
   :date-modified: 2025-12-02

.. _neuron_home:

AWS Neuron Documentation
=========================

:ref:`AWS Neuron <what-is-neuron>` is a software stack that enables high-performance deep learning and generative AI workloads on `AWS Inferentia <https://aws.amazon.com/ai/machine-learning/inferentia/>`_ and `AWS Trainium <https://aws.amazon.com/ai/machine-learning/trainium/>`_ instances. Neuron provides a complete machine learning development experience with compiler optimization, runtime efficiency, and comprehensive tooling.

* **For more details, see** :doc:`What is AWS Neuron? </about-neuron/what-is-neuron>` and :doc:`What's New in AWS Neuron? </about-neuron/whats-new>`

* **For the latest release notes, see** :doc:`AWS Neuron Release Notes </release-notes/index>`. The current release is :doc:`version 2.27.1 </release-notes/2.27.1>`, released on January 14, 2026.

.. admonition:: Join our Beta program

   Get early access to new Neuron features and tools! `Fill out this form and apply to join our Beta program <https://pulse.aws/survey/NZU6MQGW?p=0>`__.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 
      :class-card: sd-border-2

      **Looking to dive into Neuron development? Follow these links:**
      ^^^
      * :doc:`Learn about Neuron's support for native PyTorch </frameworks/torch/pytorch-native-overview>`
      * :doc:`Get started with vLLM </libraries/nxd-inference/vllm/index>` for :doc:`Offline </libraries/nxd-inference/vllm/quickstart-vllm-offline-serving>` or :doc:`Online </libraries/nxd-inference/vllm/quickstart-vllm-online-serving>` inference model serving
      * :doc:`Implement and run your first NKI kernel </nki/get-started/quickstart-implement-run-kernel>`
      * :doc:`Optimize model performance with Neuron Explorer </tools/neuron-explorer/index>`
      * :doc:`Launch a Inf/Trn instance on Amazon EC2 </devflows/ec2-flows>`
      * :doc:`Deploy a DLC </containers/get-started/quickstart-configure-deploy-dlc>`

Learn more about AWS Neuron
----------------------------

**Select a card below to read more about these features**:

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 
      :link: /frameworks/torch/pytorch-native-overview
      :link-type: doc
      :class-card: sd-border-2
 
      **Native PyTorch**
      ^^^
      Learn about native PyTorch support in AWS Neuron.

   .. grid-item-card:: 
      :link: /libraries/nxd-inference/vllm/index
      :link-type: doc
      :class-card: sd-border-2

      **vLLM on Neuron**
      ^^^
      High-performance inference serving for large language models with OpenAI-compatible APIs on Trainium and Inferentia.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 
      :class-card: sd-border-2

      **Developer Tools**
      ^^^
      Profile and monitor your models as you develop, build, test, and deploy them with Neuron's developer tools.

      * :doc:`Neuron Explorer </tools/neuron-explorer/index>`
      * :doc:`Neuron Profiler </tools/profiler/neuron-profile-user-guide>`
      * :doc:`Neuron Profiler 2.0 </tools/profiler/neuron-profiler-2-0-beta-user-guide>`
      * :doc:`Neuron System tools </tools/neuron-sys-tools/index>`

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 
      :class-card: sd-border-2

      **Neuron Kernel Interface**
      ^^^
      Low-level programming interface for custom kernel development on Trainium and Inferentia with direct hardware access.

      * :doc:`Set up your developer environment </nki/get-started/setup-env>`
      * :doc:`NKI Library  </nki/library/index>`
      * :doc:`NKI Language Guide </nki/get-started/nki-language-guide>`
      * :doc:`NKI Tutorials </nki/guides/tutorials/index>`
      * :doc:`NKI API Reference </nki/api/index>`
      * :doc:`NKI Compiler </nki/deep-dives/nki-compiler>`

**Other Neuron features:**

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: **Orchestration and Deployment on AWS EC2 and EKS**
      :link: /devflows/index
      :link-type: doc
      :class-card: sd-border-1

      Configure and run AWS Deep Learning Images (DLAMIs) and Containers (DLCs) to test and deploy your models with AWS EC2 and EKS.

   .. grid-item-card::  **AWS Neuron Open Source**
      :link: /about-neuron/oss/index
      :link-type: doc
      :class-card: sd-border-1

      Interested in contributing to Neuron source code and samples? Review this documentation and learn about our public GitHub repos and how to contribute to the code and samples in them.  

   .. grid-item-card:: **AWS Neuron-supported ML frameworks**
      :class-card: sd-border-1

      * :doc:`PyTorch NeuronX (torch-neuronx) <frameworks/torch/index>`
      * :doc:`JAX NeuronX <frameworks/jax/index>`

   .. grid-item-card:: **NeuronX Distributed (NxD) libraries**
      :class-card: sd-border-1

      * :doc:`NxD Libraries Overview <libraries/index>`
      * :doc:`NxD Training <libraries/nxd-training/index>`
      * :doc:`NxD Inference <libraries/nxd-inference/index>`
      * :doc:`NxD Core <libraries/index>`

   .. grid-item-card:: **Workloads**
      :class-card: sd-border-1

      * :doc:`Workload orchestration </devflows/index>`
      * :doc:`AWS Neuron Deep Learning Machine Images (DLAMIs) <dlami/index>`
      * :doc:`AWS Neuron Deep Learning Containers (DLCs) <containers/index>`

   .. grid-item-card:: **Runtime & Collectives**
      :class-card: sd-border-1

      * :doc:`Neuron Runtime <neuron-runtime/index>`
      * :doc:`Neuron Collectives <neuron-runtime/about/collectives>`
      * :doc:`Neuron C++ Custom Operators <neuron-customops/index>`

   .. grid-item-card:: **Compilers**
      :class-card: sd-border-1

      * :doc:`Neuron Graph Compiler <compiler/index>`
      * :doc:`Neuron Compiler Error Codes <compiler/error-codes/index>`

   .. grid-item-card:: **Legacy Documentation and Samples**
      :class-card: sd-border-1

      * :doc:`Apache MXNet </frameworks/mxnet-neuron/index>`
      * :doc:`Archived content </archive/index>`

.. toctree::
   :maxdepth: 1
   :hidden:
   
   About Neuron </about-neuron/index>
   Neuron Architecture </about-neuron/arch/index>
   What's New </about-neuron/whats-new>
   Announcements </about-neuron/announcements/index>
   Release Notes </release-notes/index>
   Contribute </about-neuron/oss/index>

.. toctree::
    :maxdepth: 1
    :caption: Get Started
    :hidden:

    Quickstarts </about-neuron/quick-start/index>
    Setup Guides </setup/index>
    Developer Flows </devflows/index>

.. toctree::
   :maxdepth: 1
   :caption: Use ML Frameworks
   :hidden:

   Home </frameworks/index>
   Native PyTorch </frameworks/torch/pytorch-native-overview>
   PyTorch NeuronX </frameworks/torch/index>
   JAX NeuronX</frameworks/jax/index>
   TensorFlow NeuronX</frameworks/tensorflow/index>

.. toctree::
   :maxdepth: 1
   :caption: Training Libraries
   :hidden:

   NxD Training </libraries/nxd-training/index>
   NxD Core (Training) </libraries/neuronx-distributed/index-training>

.. toctree::
   :maxdepth: 1
   :caption: Inference Libraries
   :hidden:

   Overview </libraries/nxd-inference/neuron-inference-overview>
   vLLM </libraries/nxd-inference/vllm/index>
   NxD Inference </libraries/nxd-inference/index>
   NxD Core (Inference) </libraries/neuronx-distributed/index-inference>

.. toctree::
   :maxdepth: 1
   :caption: NxD Core Libraries
   :hidden:

   Overview </libraries/index>

.. toctree::
   :maxdepth: 1
   :caption: Developer Tools
   :hidden:

   Home </tools/index>
   Neuron Explorer </tools/neuron-explorer/index>
   Neuron Profiler 2.0 </tools/profiler/neuron-profiler-2-0-beta-user-guide>
   Neuron Profiler </tools/profiler/neuron-profile-user-guide>
   System Tools </tools/neuron-sys-tools/index>

.. toctree::
   :maxdepth: 1
   :caption: Orchestrate and Deploy
   :hidden:

   AWS Workload Orchestration </devflows/index>
   Neuron DLAMI </dlami/index>
   Neuron Containers </containers/index>

.. toctree::
   :maxdepth: 1
   :caption: Runtime & Collectives
   :hidden:

   Neuron Runtime </neuron-runtime/index>
   Collectives </neuron-runtime/about/collectives>
   Neuron C++ Custom Operators </neuron-customops/index>

.. toctree::
   :maxdepth: 1
   :caption: Compilers
   :hidden:

   Graph Compiler </compiler/index>
   Compiler Error Codes </compiler/error-codes/index>

.. toctree::
   :maxdepth: 1
   :caption: Neuron Kernel Interface (NKI)
   :hidden:

   Home </nki/index>
   Get Started </nki/get-started/index>
   Guides </nki/guides/index>
   Deep Dives </nki/deep-dives/index>
   NKI API Reference </nki/api/index>
   NKI Library </nki/library/index>

.. toctree::
   :maxdepth: 1
   :caption: Archive
   :hidden:

   Archived content </archive/index>
   
*AWS and the AWS logo are trademarks of Amazon Web Services, Inc. or its affiliates. All rights reserved.*
