.. _neuron-gettingstarted:

Getting started
===============

This Getting Started Guide provides the beginning point to
start developing and deploying your ML inference applications, whether
you are a first time user or if you are looking for specific topic documentation.

.. _setup-neuron-env:

Setup Neuron environment
-----------------------

A typical workflow with the Neuron SDK will be to compile trained ML models on
a compute instance **(compilation instance)** and then distribute the artifacts to
a fleet of inf1 instances **(deployment instance)** , for execution and deployment.

|image|



.. note::

  `AWS Deep Learning AMI (DLAMI) <https://docs.aws.amazon.com/dlami/index.html>`_ is 
  the recommended AMI to use with Neuron SDK.


.. _compilation-instance:

Step 1 - Compilation Instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended to choose c5.4xlarge or larger for compilation instance, however the
user can choose to compile and deploy on the same instance, when choosing the same instance
for compilation and deployment it is recommend to use an inf1.6xlarge instance or larger.



#. `Launch compilation instance with DLAMI <https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html>`_ , see :ref:`dlami` for more information, If you choose other AMI `launch EC2 instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ and choose your AMI of choice.
#. :ref:`Install Neuron SDK <neuron-install-guide>`

.. _deployment-instance:

Step 2 - Deployment Instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deployment instance is the `inf1 instance <https://aws.amazon.com/ec2/instance-types/inf1/>`_ 
chosen to deploy and execute the user trained model.

#. `Launch inf1 instance with DLAMI <https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html>`_ , see :ref:`dlami` for more information, If you choose other AMI `launch an inf1 instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ and choose your AMI of choice.
#. :ref:`Install Neuron SDK <neuron-install-guide>`


Start with ML Framework
-----------------------

Start with PyTorch
~~~~~~~~~~~~~~~~~~

#. :ref:`install-neuron-pytorch`
#. Run :ref:`pytorch-getting-started`
#. Visit :ref:`neuron-pytorch` for more resources.

Start with Tensorflow
~~~~~~~~~~~~~~~~~~~~~

#. :ref:`install-neuron-tensorflow`
#. Run :ref:`tensorflow-getting-started`
#. Visit :ref:`tensorflow-neuron` for more resources.

Start with MXNet
~~~~~~~~~~~~~~~~

#. :ref:`install-neuron-mxnet`
#. Run :ref:`mxnet-resnet50`
#. Visit :ref:`neuron-mxnet` for more resources.

Run Tutorials & Examples
------------------------

ML Framework
~~~~~~~~~~~~

  -  :ref:`tensorflow-tutorials`

  -  :ref:`pytorch-tutorials`

  -  :ref:`mxnet-tutorials`

Containers
~~~~~~~~~~

  - :ref:`Containers Tutorials <containers-tutorials>`


Learn Neuron Fundamentals
-------------------------

Get familiar with Neuron fundamentals and tools:

-  Learn :ref:`neuron-fundamentals` : such as :ref:`neuron-data-types`, :ref:`neuron-batching` and :ref:`neuroncore-pipeline`,  which will help you utilize Neuron to develop a highly optimized ML application.

-  Get familiar with :ref:`neuron-cc`,\ :ref:`neuron-runtime` and :ref:`neuron-tools` by reviewing the overview sections and reading about the supported features and capabilities of the Neuron Compiler, Runtime and Tools.


Performance optimization
------------------------

The following steps are recommended for you to build highly optimized
Neuron applications:

#. Get familiar with Neuron fundamentals and tools:

   -  Learn :ref:`neuron-fundamentals` : such as
      :ref:`neuron-data-types`, :ref:`neuron-batching` and
      :ref:`neuroncore-pipeline`
   -  Get familiar with :ref:`neuron-cc`, \ :ref:`neuron-runtime` and
      :ref:`neuron-tools` by reviewing the overview sections and reading about the supported features
      and capabilities

#. Learn how to optimize your application by reviewing the HowTo guides
   at :ref:`performance-optimization` .


.. |image| image:: /images/devflow.png
