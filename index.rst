
Welcome to AWS Neuron
=====================

AWS Neuron is the SDK for `AWS Inferentia <https://aws.amazon.com/machine-learning/inferentia/>`__, the custom designed machine learning chips enabling high-performance deep learning inference applications on `EC2 Inf1 instances <https://aws.amazon.com/ec2/instance-types/inf1/>`__. Neuron includes a deep learning compiler, runtime and tools that are natively integrated into TensorFlow, PyTorch and Apache MXNet (Incubating). With Neuron, you can develop, profile, and deploy high-performance inference applications on top of `EC2 Inf1 instances <https://aws.amazon.com/ec2/instance-types/inf1/>`__.

Check :ref:`neuron-release-content` and :ref:`neuron-whatsnew` in latest Neuron release.


|image|


.. |image| image:: /images/neuron-devflow.jpg
   :width: 600
   :alt: Neuron developer flow



.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   /neuron-intro/get-started
   Start with PyTorch <neuron-guide/neuron-frameworks/pytorch-neuron/index>
   Start with TensorFlow <neuron-guide/neuron-frameworks/tensorflow-neuron/index>
   Start with Apache MXNet (Incubating) <neuron-guide/neuron-frameworks/mxnet-neuron/index>
   Tutorials <neuron-intro/tutorials>
   /release-notes/index
   /release-notes/releasecontent
   /neuron-intro/releaseinfo

.. toctree::
   :maxdepth: 1
   :caption: Learning Neuron

   Neuron Developer Flows </neuron-intro/devflows/dev-flows>
   neuron-guide/technotes/index
   Neuron Compiler <neuron-guide/neuron-cc/index>
   Neuron Runtime <neuron-guide/neuron-runtime/index>
   Neuron Tools <neuron-guide/neuron-tools/index>
   Containers <neuron-deploy/index>
   Application Notes <neuron-guide/appnotes>
   /neuron-intro/neuron-install-guide   

.. toctree::
   :maxdepth: 1
   :caption: Neuron Frameworks
   
   .. neuron-guide/neuron-frameworks/pytorch-neuron/index
   .. neuron-guide/neuron-frameworks/tensorflow-neuron/index
   .. neuron-guide/neuron-frameworks/mxnet-neuron/index