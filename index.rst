Welcome to AWS Neuron
=====================

.. meta::
   :description: Neuron SDK is powering AWS Inferentia and Trainium based instances, natively integrated into PyTorch and TensorFlow. Enabling you to quicky start with Amazon EC2, AWS Sagemaker, ECS and EKS.

AWS Neuron is the SDK used to run deep learning workloads on AWS Inferentia and AWS Trainium based instances. It supports customers in their end-to-end ML development lifecycle to build new models, train and optimize these models, and then deploy them for production. To learn about the model architectures currently supported on Inf1, Inf2, Trn1 and Trn1n instances, please see :ref:`model_architecture_fit`. To learn about upcoming capabilities, please view the :ref:`neuron_roadmap`.

AWS Neuron includes a deep learning compiler, runtime, and tools that are natively integrated into TensorFlow, PyTorch and Apache MXNet (incubating). The EC2 Trn1/Trn1n instances are optimized for the highest performance and best price-performance training in AWS. The EC2 Inf2 instances are designed for high-performance deep learning inference applications. With Neuron, customers can quickly start using Inf/Trn instances through services like Amazon Sagemaker, Amazon Elastic Container Service (ECS), Amazon Elastic Kubernetes Service (EKS), AWS Batch, and AWS Parallel Cluster. 

Check :ref:`announcements-main` and check :ref:`latest-neuron-release` for latest release.


.. grid:: 2


   .. card:: Get Started with PyTorch Neuron
         :link: torch_quick_start
         :link-type: ref


   .. card:: Get Started with TensorFlow Neuron
         :link: tensorflow_quick_start
         :link-type: ref


   .. card:: Neuron Quick Links
         :link: docs-quick-links
         :link-type: ref


.. toctree::
   :maxdepth: 1
   :caption: Overview
   
   Quick Links </general/quick-start/docs-quicklinks>
   Get Started with PyTorch </general/quick-start/torch-neuron>
   Get Started with TensorFlow </general/quick-start/tensorflow-neuron>
   Get Started with MXNet  </general/quick-start/mxnet-neuron>
   GitHub Samples </general/quick-start/github-samples>
   Performance </general/benchmarks/index>
   What’s New </release-notes/index>
   /general/announcements/blogs
   Announcements </general/announcements/index>

.. toctree::
   :maxdepth: 1
   :caption: ML Frameworks

   PyTorch Neuron </frameworks/torch/index>
   TensorFlow Neuron </frameworks/tensorflow/index>
   Apache MXNet (Incubating) </frameworks/mxnet-neuron/index>

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   Neuron Runtime </neuron-runtime/index>
   Neuron Compiler </compiler/index>
   Neuron C++ Custom Operators </neuron-customops/index>
   Neuron Tools </tools/index>
   Neuron Calculator </general/calculator/neuron-calculator>
   Setup Guide </general/setup/index>
   Containers Deployment </containers/index>
   Developer Flows </general/devflows/index>

.. toctree::
   :maxdepth: 1
   :caption: Learning Neuron

   Architecture </general/arch/index>
   Features </general/arch/neuron-features/index>
   Application Notes </general/appnotes/index>
   FAQ </general/faq>
   Troubleshooting </general/troubleshooting>


.. toctree::
   :maxdepth: 1
   :caption: About Neuron
   
   /release-notes/release
   Roadmap </general/roadmap-readme>
   Support </general/support>


