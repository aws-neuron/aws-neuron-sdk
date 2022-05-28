
Welcome to AWS Neuron
=====================

.. meta::
   :description: Neuron SDK is powering AWS Inferentia and Trainium based instances, natively integrated into PyTorch and TensorFlow and can quickly start using with AWS Sagemaker, ECS and EKS.
   
AWS Neuron is the SDK for `AWS Inferentia <https://aws.amazon.com/machine-learning/inferentia/>`__, the custom designed machine learning chips enabling high-performance deep learning inference applications on `EC2 Inf1 instances <https://aws.amazon.com/ec2/instance-types/inf1/>`__. Neuron includes a deep learning compiler, runtime and tools that are natively integrated into TensorFlow, PyTorch and Apache MXNet (Incubating). With Neuron, you can develop, profile, and deploy high-performance inference applications on top of `EC2 Inf1 instances <https://aws.amazon.com/ec2/instance-types/inf1/>`__.

Check :ref:`neuron-release-content`, :ref:`Neuron Performance page <appnote-performance-benchmark>` and :ref:`neuron-whatsnew` in :ref:`latest-neuron-release` release.


|image|


.. |image| image:: /images/neuron-devflow.jpg
   :width: 600
   :alt: Neuron developer flow


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   QuickStart </neuron-intro/get-started>
   PyTorch </neuron-guide/neuron-frameworks/pytorch-neuron/index>
   TensorFlow </neuron-guide/neuron-frameworks/tensorflow-neuron/index>
   Apache MXNet (Incubating) </neuron-guide/neuron-frameworks/mxnet-neuron/index>
   Tutorials </neuron-intro/tutorials>
   Performance </neuron-guide/benchmark/index>
   What’s New </release-notes/index>

.. toctree::
   :maxdepth: 1
   :caption: Learning Neuron

   Neuron Features </neuron-guide/technotes/index>
   Neuron Developer Flows </neuron-intro/devflows/dev-flows>
   Containers </neuron-deploy/index>
   Application Notes </neuron-guide/appnotes>
   Neuron FAQ </faq>

.. toctree::
   :maxdepth: 1
   :caption: Neuron SDK

   Setup Guide </neuron-intro/neuron-install-guide>
   Neuron Compiler </neuron-guide/neuron-cc/index>
   Neuron Runtime </neuron-guide/neuron-runtime/index>
   Neuron Tools </neuron-guide/neuron-tools/index> 
   NeuronPerf (Beta) </neuron-guide/neuronperf/index>
   Release Details </release-notes/releasecontent>
   Roadmap </neuron-intro/roadmap-readme>
   Support </neuron-intro/releaseinfo>

