.. _neuron-gettingstarted:

Getting started
===============

This Getting Started Guide provides the beginning point to
start developing and deploying your ML inference applications, whether
you are a first time user or if you are looking for specific topic documentation.

First time user
---------------

If you are a first-time-user, the following steps are
recommended to get started with Neuron:

#. Run your first Neuron ML application by following the instructions in
   one of the following Getting Started tutorials to get familiar with
   Neuron development flow of your ML framework of choice:

   -  :ref:`tensorflow-resnet50`
   -  :ref:`pytorch-resnet50`
   -  :ref:`mxnet-resnet50`

#. Get familiar with Neuron fundamentals and tools:

   -  Learn :ref:`neuron-fundamentals` : such as
      :ref:`neuron-data-types`, :ref:`neuron-batching` and
      :ref:`neuroncore-pipeline`,  which will help
      you utilize Neuron to develop a highly optimized ML application.
   -  Get familiar with :ref:`neuron-cc`,\ :ref:`neuron-runtime` and
      :ref:`neuron-tools` by reviewing the overview sections and reading about
      the supported features and capabilities of
      the Neuron Compiler, Runtime and Tools.

#. Deploy Neuron ML applications at scale by learning how to tune,
   optimize and deploy your ML application by following the
   instructions of one of the HowTo guides at
   :ref:`deploy-ml-application`.


Navigate documentation
----------------------

Tutorials
~~~~~~~~~

Explore more Tutorials and examples here:

-  :ref:`tensorflow-tutorials`
-  :ref:`pytorch-tutorials`
-  :ref:`mxnet-tutorials`
-  :ref:`Neuron Containers Tutorials and Examples <containers-tutorials>`

ML Frameworks
~~~~~~~~~~~~~

You can find Neuron supported ML Frameworks here:

-  :ref:`tensorflow-neuron`
-  :ref:`neuron-pytorch`
-  :ref:`neuron-mxnet`

ML Inference Models
~~~~~~~~~~~~~~~~~~~

You can find ML Inference model tutorials here:

-  Computer Vision

   -  :ref:`Tensor Flow <tensorflow-computervision>`
   -  :ref:`PyTorch <pytorch-computervision>`
   -  :ref:`MXNet <mxnet-computervision>`

-  Natural Language Processing

   -  :ref:`Tensor Flow <tensorflow-nlp>`
   -  :ref:`PyTorch <pytorch-nlp>`
   -  :ref:`MXNet <mxnet-nlp>`

Performance optimization
~~~~~~~~~~~~~~~~~~~~~~~~

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

Container Support
~~~~~~~~~~~~~~~~~~

Visit :ref:`neuron-containers` for more information about Neuron
suport for containers and more :ref:`containers-tutorials`.


.. _install-neuron:

Installing Neuron
-----------------

To use Neuron, you can use a pre-built Amazon Machine Images
(the Deep Learning AMI - DLAMI), use the pre built DL containers or install
Neuron software into your own instances and AMIs. To
ensure you have the latest Neuron version we recommend to either install
it on your own instance, or to check for the installed version when
using DLAMI or DL containers.

Follow :ref:`neuron-install-guide` if you already have an environment
you'd like to continue using.


.. toctree::
   :maxdepth: 1

   neuron-install-guide
   dlami
   dlcontainers
