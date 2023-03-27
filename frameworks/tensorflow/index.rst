.. _tensorflow-neuron-main:
.. _tensorflow-neuron:

TensorFlow Neuron
=================
TensorFlow Neuron unlocks high-performance and cost-effective deep learning acceleration on AWS Trainium-based and Inferentia-based Amazon EC2 instances.

TensorFlow Neuron enables native TensorFlow models to be accelerated on Neuron devices, so you can use your existing framework application and get started easily with minimal code changes.


.. toctree::
    :maxdepth: 1
    :hidden:
    
    /frameworks/tensorflow/tensorflow-setup

 
.. toctree::
    :maxdepth: 1
    :hidden:
    
    /frameworks/tensorflow/inference


.. toctree::
    :maxdepth: 1
    :hidden:
    
    /frameworks/tensorflow/training




.. dropdown::  Tensorflow Neuron Setup
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
    
    .. include:: tensorflow-setup.txt


.. tab-set::

    .. tab-item:: Inference

        .. tab-set::

            .. tab-item:: Inference on Inf2 & Trn1/Trn1n (``tensorflow-neuronx``)

                .. include:: tensorflow-neuronx-inference.txt

            .. tab-item:: Inference on Inf1 (``tensorflow-neuron``)

                .. include:: tensorflow-neuron-inference.txt
    
    .. tab-item:: Training 

        .. include:: training.txt