.. _tensorflow-tutorials:

TensorFlow Tutorials
====================

Before running a tutorial
-------------------------

You will run the tutorials on an inf1.6xlarge instance running Deep Learning AMI (DLAMI) to enable both compilation and deployment (inference) on the same instance. In a production environment we encourage you to try different instance sizes to optimize to your specific deployment needs.

Follow instructions at :ref:`tensorflow-tutorial-setup` before running a TensorFlow tutorial on Inferentia. We recommend new users start with the ResNet-50 tutorial.


.. toctree::
   :hidden:

   /frameworks/tensorflow/tensorflow-neuron/tutorials/tensorflow-tutorial-setup

.. _tensorflow-nlp:

Natural Language Processing
---------------------------

*  Tensorflow 2.x - HuggingFace DistilBERT with Tensorflow2 Neuron :ref:`[html] </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>` :github:`[notebook] </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>`

.. toctree::
   :hidden:

   /frameworks/tensorflow/tensorflow-neuron/tutorials/bert_demo/bert_demo
   /src/examples/tensorflow/huggingface_bert/huggingface_bert

.. _tensorflow-utilize-neuron:

Utilizing Neuron Capabilities
-----------------------------

*  Tensorflow 2.x - Using NEURON_RT_VISIBLE_CORES with TensorFlow Serving :ref:`[html] </src/examples/tensorflow/tensorflow_serving_tutorial.rst>`

.. toctree::
   :hidden:

   /src/examples/tensorflow/tensorflow_serving_tutorial.rst
