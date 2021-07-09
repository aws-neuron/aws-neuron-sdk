.. _models-inferentia:

Inferentia Model Architecture Fit
==================================

.. contents::
   :local:
   :depth: 2

Introduction
------------

This section describes what types of deep learning Architectures perform well out of the box on Inferentia. It provides guidance on how Neuron maps operations to `Inferentia <https://aws.amazon.com/machine-learning/inferentia/>`_, and discuss techniques you can use to optimize your deep learning models for Inferentia.

`AWS Neuron <https://aws.amazon.com/machine-learning/neuron/>`_, the SDK of Inferentia, enables you to deploy a wide range of petrained deep learning models on AWS machine learning (ML) chips. Neuron includes a deep learning compiler, a runtime and tools natively integrated into popular ML frameworks like TensorFlow, PyTorch and Apache MXNet (Incubating). 

Model architectures that run well on Inferentia
-----------------------------------------------

Many popular models used in today’s leading AI applications run out-of-the box on Inferentia. The following models are examples of model types that perform well on Inferentia:

* Language Models: 

    * Transformers based Natural Language Processing/Understanding (NLP/NLU) such as `HuggingFace Transformers <https://huggingface.co/transformers/>`_ BERT, distilBERT, XLM-BERT, Roberta and BioBert. To get started with NLP models you can refer to Neuron :ref:`PyTorch <pytorch-nlp>`, :ref:`TensorFlow <tensorflow-nlp>` and :ref:`MXNet <mxnet-nlp>` NLP tutorials.
    * Generative language models like :ref:`MarianMT <pytorch-tutorials-marianmt>`, Pegasus and Bart.
    
* Computer Vision Models

    * Image classification models like :ref:`Resnet <tensorflow-Resnet50>`, Resnext and VGG
    * Object detection models like Yolo :ref:`v3 <tensorflow-yolo_v3>`/:ref:`v4 <tensorflow-yolo4>` and v5, and :ref:`SSD <tensorflow-ssd300>`

* Recommender engines models that include Embeddings and MLP layers.

Model enablement guidelines
---------------------------

The following points provide guidelines for deploying a model that doesn't fit into one of the above categories or when deploying your own custom models. We encourage you to compile and run the model on Inferentia and :ref:`contact us <neuron-support>` for support, if needed.

Operator coverage
^^^^^^^^^^^^^^^^^

Neuron has wide support for operator types for popular model types. That said, with Neuron Auto partition feature it is not required that all operators are supported by Neuron to successfully deploy a model on Inferentia. 

Prior to compilation, the Neuron extension in the given Framework will examine the supported operators in the model and then partition the model graph, creating subgraph(s) that contain the unsupported operators that will execute within the framework on the CPU instance, or subgraph(s) that contain the supported operators that will execute within the accelerator on Inferentia.

While many models perform very well with subgraphs running on CPU, especially if the operations map well to CPU execution, it is possible that the performance will not meet your application needs. In such cases, we encourage you to contact us for further optimization.



Variable input size
^^^^^^^^^^^^^^^^^^^

With Neuron, the input size shape is fixed at compile time. If your application requires multiple input sizes, we recommend using padding or bucketing techniques.  Padding requires you to compile your models to the largest expected input size, and test your application performance. If performance is not within your targets, you can consider implementing a bucketing scheme. With bucketing, you compile your model to a few input size categories that represent the range of possible input sizes. with some applications, bucketing will help optimize compute utilization compared to padding, especially if small input sizes are more frequent than large input sizes. If the varying input dimension is the batch size.

Control Flow
^^^^^^^^^^^^

Models that contain control flow operators (see :ref:`pytorch-tutorials-marianmt`) may require specific handling to ensure successful compilation with Neuron.

Dynamic shapes
^^^^^^^^^^^^^^

Currently it is required that all tensor shapes (dimension sizes) in the compute-graph are known at compilation time. Model compilation with shapes that cannot be determined at compile time will fail.

For additional resources see:

* `Neuron public roadmap <https://github.com/aws/aws-neuron-sdk/projects/2>`_
* :ref:`Getting Started <neuron-gettingstarted>`
* List of supported operators:

  * :ref:`PyTorch supported operators <neuron-cc-ops-pytorch>`
  * :ref:`TensorFlow supported operators <neuron-cc-ops-tensorflow>`
  * :ref:`MXNet supported operators <neuron-cc-ops-mxnet>`


