.. _mxnet-tutorials:

Neuron Apache MXNet (Incubating) Tutorials
==========================================

Before running a tutorial
-------------------------

You will run the tutorials on an inf1.6xlarge instance running Deep Learning AMI (DLAMI) to enable both compilation and deployment (inference) on the same instance. In a production environment we encourage you to try different instance sizes to optimize to your specific deployment needs. 

Follow instructions at :ref:`mxnet-tutorial-setup` before running an MXNet tutorial on Inferentia.


.. toctree::
   :hidden:

   /neuron-guide/neuron-frameworks/mxnet-neuron/tutorials/mxnet-tutorial-setup

.. _mxnet-computervision:

Computer Vision
---------------

* ResNet-50 tutorial :ref:`[html] </src/examples/mxnet/resnet50/resnet50.ipynb>` :mxnet-neuron-src:`[notebook] <resnet50/resnet50.ipynb>`
* Model Serving tutorial :ref:`[html] <mxnet-neuron-model-serving>`
* Getting started with Gluon tutorial :ref:`[html] <mxnet-gluon-tutorial>`


.. toctree::
   :hidden:

   /src/examples/mxnet/resnet50/resnet50.ipynb
   /neuron-guide/neuron-frameworks/mxnet-neuron/tutorials/tutorial-model-serving
   /neuron-guide/neuron-frameworks/mxnet-neuron/tutorials/tutorial-gluon

.. _mxnet-nlp:

Natural Language Processing
---------------------------

* BERT tutorial :ref:`[html] <mxnet-bert-tutorial>`
* MXNet 1.8: Using data parallel mode tutorial :ref:`[html] </src/examples/mxnet/data_parallel/data_parallel_tutorial.ipynb>` :mxnet-neuron-src:`[notebook] <data_parallel/data_parallel_tutorial.ipynb>`

.. toctree::
   :hidden:

   /neuron-guide/neuron-frameworks/mxnet-neuron/tutorials/bert_mxnet/index
   /src/examples/mxnet/data_parallel/data_parallel_tutorial.ipynb

   

.. _mxnet-utilize-neuron:

Utilizing Neuron Capabilities
-----------------------------

* NeuronCore Groups tutorial :ref:`[html] </src/examples/mxnet/resnet50_neuroncore_groups.ipynb>` :mxnet-neuron-src:`[notebook] <resnet50_neuroncore_groups.ipynb>`


.. toctree::
   :hidden:

   /src/examples/mxnet/resnet50_neuroncore_groups.ipynb
