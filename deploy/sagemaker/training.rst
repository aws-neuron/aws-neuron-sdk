.. _sm-training-devflow:

Train your model on SageMaker
===================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Description
------------

SageMaker Training helps you manage cloud computing resources in Amazon EC2, data storage services
such as S3, EFS, and FSx, and security management services such as IAM and VPC. SageMaker Training 
provides you a complete end-to-end experience of training classical ML and state-of-the-art DL models. 

You can use SageMaker to train models using Trn1 instances (ml.trn1 instance types). 
In this developer flow, you provision a SageMaker Notebook instance or SageMaker Studio to train 
your model using the `SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/index.html>`_.

The Amazon SageMaker Python SDK lets you launch training jobs in just a few lines of code with ease. 
As shown in the below diagram Amazon SageMaker launches Trn1 instances, copies both data and code 
onto the instance. It then runs the training script to generate model artifacts. The trained model 
artifacts are then uploaded to S3 and finally SageMaker will terminate the provisioned instances. 
In order to speed up the training process for successive runs you can copy the `Neuron Persistent Cache
<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/neuron-caching.html>`_
to S3 and then copied by future training jobs as they will leverage the cached artifacts. 
(See `Hugging Face fine tuning BERT base model on Amazon SageMaker Tutorial 
<https://github.com/aws-neuron/aws-neuron-sagemaker-samples/tree/main/training/trn1-bert-fine-tuning-on-sagemaker>`_
for an example on how to reuse the compiled cache.)

.. image:: /images/trn1-on-sm-dev-flow.png


Setup environment
-----------------

1. Create an Amazon SageMaker Notebook Instance

   Follow the instructions in `Get Started with Notebook Instances 
   <https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html>`_ or 
   `Use Amazon SageMaker Studio Notebooks <https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks.html>`_.
   The Notebook instance provides the required Python SDK for training models with Amazon SageMaker.
   Please make sure SageMaker Python SDK version is 2.116.0 or later.

2. Train a model using the Amazon SageMaker SDK

   Follow the instructions in `Distributed Training with PyTorch Neuron on Trn1 instances
   <https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#distributed-training-with-pytorch-neuron-on-trn1-instances>`_.
   You’ll be able to follow the `Hugging Face fine tuning BERT base model on Amazon SageMaker Tutorial
   <https://github.com/aws-neuron/aws-neuron-sagemaker-samples/tree/main/training/trn1-bert-fine-tuning-on-sagemaker>`_.

   .. note::
     SageMaker support for EC2 Trn1 instance is currently available only for PyTorch Estimator. 
     HuggingFace Estimator will be available in future release.
