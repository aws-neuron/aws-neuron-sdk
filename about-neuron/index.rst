.. _about-neuron:

About the AWS Neuron SDK
========================

AWS Neuron is a software development kit (SDK) enabling high-performance deep learning acceleration using AWS Inferentia and Trainium, AWS's custom designed machine learning accelerators. It enables you to develop, profile, and deploy high-performance machine learning workloads on AWS Inferentia and Trainium instances. 

The AWS Neuron SDK includes:

* **Neuron Compiler** - Compiles high-level, framework-based models for optimal performance on Neuron devices
* **Neuron Kernel Interface (NKI)** - Provides direct compiler access to Neuron device capabilities
* **Neuron Runtime** - Executes compiled models on Neuron devices
* **ML Framework integration** - Deep support for PyTorch and JAX
* **Training and inference libraries** - Distributable training and inference libraries for large-scale models
* **Deployment support** - Integration with AWS services like SageMaker, EC2, EKS, and ECS
* **Developer tools** - Profiling, monitoring, and debugging utilities

For a full list of AWS Neuron features, see :ref:`what-is-neuron`

What is "NeuronX"?
------------------

"NeuronX" refers to the next-generation AWS Neuron SDK, which provides enhanced capabilities for both inference and training on AWS Inferentia and Trainium instances. NeuronX includes:

* Support for the latest versions of PyTorch and JAX
* Advanced compiler optimizations for improved performance
* Enhanced distributed training libraries for large-scale models
* Improved profiling and debugging tools
* Ongoing feature development and support for new instance types

Learn about AWS Neuron
----------------------

.. grid:: 1
   :gutter: 2

   .. grid-item-card::
      :link: what-is-neuron
      :link-type: ref

      **What is AWS Neuron?**
      ^^^
      Short overview of the AWS Neuron SDK and its components
   
   .. grid-item-card::
      :link: neuron-architecture-index
      :link-type: ref

      **Neuron architecture**
      ^^^
      Understand the Neuron hardware and software architecture

   .. grid-item-card::
      :link: frameworks-neuron-sdk
      :link-type: ref

      **Supported ML frameworks**
      ^^^
      Neuron support for popular ML frameworks including PyTorch and JAX

   .. grid-item-card::
      :link: libraries-neuron-sdk
      :link-type: ref

      **NeuronX distributed (NxD) libraries**
      ^^^
      NeuronX distributed libraries for training and inference

   .. grid-item-card::
      :link: neuron-nki
      :link-type: ref

      **Neuron Kernel Interface (NKI)**
      ^^^
      NKI is a low-level interface for custom, bare-metal kernel development

   .. grid-item-card::
      :link: neuron_cc
      :link-type: ref

      **Neuron Compiler**
      ^^^
      The Neuron compiler optimizes models for Neuron hardware

   .. grid-item-card::
      :link: neuron_runtime
      :link-type: ref

      **Neuron Runtime**
      ^^^
      Runtime for executing compiled models on Neuron devices

   .. grid-item-card::
      :link: neuron-tools
      :link-type: ref

      **Neuron developer tools**
      ^^^
      Tools for profiling, debugging, and monitoring Neuron applications

   .. grid-item-card::
      :link: neuron-dlami-overview
      :link-type: ref

      **Neuron AWS Neuron Deep Learning AMIs**
      ^^^
      Deploy the Neuron SDK on EC2 instances with pre-installed Amazon Machine Images (AMIs)

   .. grid-item-card::
      :link: neuron-containers
      :link-type: ref

      **Neuron AWS Neuron Deep Learning Containers**
      ^^^
      Deploy the Neuron SDK using pre-built Docker deep learning containers (DLCs)

Resources
---------

* :ref:`Setup Guide <setup-guide-index>`
* :ref:`Release Notes <neuron_release_notes>`
* :ref:`FAQ <neuron_faq>`

Support
-------

* :ref:`AWS Neuron SDK maintenance policy <sdk-maintenance-policy>`
* :ref:`AWS Neuron Support <neuron_support>`

.. _contact-us:

Contact us
----------

For support, submit a request with AWS Neuron `Github issues <https://github.com/aws/aws-neuron-sdk/issues>`_ or visit the `Neuron AWS forums <https://forums.aws.amazon.com/forum.jspa?forumID=355>`_ for an answer. 

If you want to request a feature or report a critical issue, you can contact us directly at ``aws-neuron-support@amazon.com``.

.. toctree::
   :maxdepth: 1
   :hidden:

   What is AWS Neuron? <what-is-neuron>
   Architecture <arch/index>
   Benchmarks </about-neuron/benchmarks/index>
   App notes <appnotes/index>
   Troubleshooting <troubleshooting>
   Security <security>
   Neuron FAQ <faq>