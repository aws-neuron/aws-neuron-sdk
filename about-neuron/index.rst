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

For a full list of AWS Neuron features, see :ref:`what-is-neuron`.

.. admonition:: Join our Beta program

   Get early access to new Neuron features and tools! `Fill out this form and apply to join our Beta program <https://pulse.aws/survey/NZU6MQGW?p=0>`__.

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
      :link: /about-neuron/what-is-neuron
      :link-type: doc
      :class-card: sd-border-1

      **What is AWS Neuron?**
      ^^^
      Short overview of the AWS Neuron SDK and its components

.. grid:: 1 1 2 2
   :gutter: 2
   
   .. grid-item-card::
      :link: /about-neuron/arch/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron architecture**
      ^^^
      Understand the Neuron hardware and software architecture

   .. grid-item-card::
      :link: /about-neuron/arch/neuron-features/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron features**
      ^^^
      Overviews of model development features provided by Neuron

   .. grid-item-card::
      :link: /frameworks/index
      :link-type: doc
      :class-card: sd-border-1

      **Supported ML frameworks**
      ^^^
      Neuron support for popular ML frameworks including PyTorch and JAX

   .. grid-item-card::
      :link: /libraries/index
      :link-type: doc
      :class-card: sd-border-1

      **NeuronX distributed (NxD) libraries**
      ^^^
      NeuronX distributed libraries for training and inference

   .. grid-item-card::
      :link: /nki/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron Kernel Interface (NKI)**
      ^^^
      NKI is a low-level interface for custom, bare-metal kernel development

   .. grid-item-card::
      :link: /compiler/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron Compiler**
      ^^^
      The Neuron compiler optimizes models for Neuron hardware

   .. grid-item-card::
      :link: /neuron-runtime/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron Runtime**
      ^^^
      Runtime for executing compiled models on Neuron devices

   .. grid-item-card::
      :link: /tools/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron developer tools**
      ^^^
      Tools for profiling, debugging, and monitoring Neuron applications

   .. grid-item-card::
      :link: /dlami/index
      :link-type: doc
      :class-card: sd-border-1

      **Neuron AWS Neuron Deep Learning AMIs**
      ^^^
      Deploy the Neuron SDK on EC2 instances with pre-installed Amazon Machine Images (AMIs)

   .. grid-item-card::
      :link: /containers/index
      :link-type: doc
      :class-card: sd-border-1

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

* :doc:`Neuron Open Source GitHub Repos </about-neuron/oss/index>`
* :ref:`AWS Neuron SDK maintenance policy <sdk-maintenance-policy>`

.. _contact-us:

Contact us
----------

For support, submit a request with AWS Neuron `Github issues <https://github.com/aws/aws-neuron-sdk/issues>`_ or visit the `Neuron AWS forums <https://forums.aws.amazon.com/forum.jspa?forumID=355>`_ for an answer. 

If you want to request a feature or report a critical issue, you can contact us directly at ``aws-neuron-support@amazon.com``.

.. toctree::
   :maxdepth: 1
   :hidden:

   App Notes <appnotes/index>
   Ask Amazon Q </about-neuron/amazonq-getstarted>
   Benchmarks </about-neuron/benchmarks/index>
   Beta Participation </about-neuron/beta-participation>
   Model Samples </about-neuron/models/index>
   Neuron FAQ <faq>
   Neuron Features </about-neuron/arch/neuron-features/index>
   Open Source </about-neuron/oss/index>
   SDK Maintenance Policy <sdk-policy>
   Security <security>
   Term Glossary <arch/glossary>
   Troubleshooting <troubleshooting>
   What is AWS Neuron? <what-is-neuron>
