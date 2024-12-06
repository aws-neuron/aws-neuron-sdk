.. _third-party-libraries:

Third-party libraries
=====================

Third-party partner libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AWS Neuron integrates with multiple third-party partner products that alow you to run deep learning workloads on Amazon EC2 
instances powered by AWS Trainium and AWS Inferentia chips. The following list gives an overview of the third-party libraries 
working with AWS Neuron.

.. contents:: Table of contents
   :local:
   :depth: 1


Hugging Face Optimum Neuron
""""""""""""""""""""""""""""

Optimum Neuron bridges Hugging Face Transformers and the AWS Neuron SDK, providing standard Hugging Face APIs for 
`AWS Trainium <https://aws.amazon.com/ai/machine-learning/trainium/>`_ and `AWS Inferentia <https://aws.amazon.com/ai/machine-learning/inferentia/>`_. 
It offers solutions for both training and inference, including support for large-scale model training and deployment for AI workflows. 
Supporting Amazon SageMaker and pre-built Deep Learning Containers, Optimum Neuron simplifies the use of Trainium and Inferentia 
for machine learning. This integration allows developers to work with familiar Hugging Face interfaces while leveraging Trainium 
and Inferentia for their transformer-based projects.

`Optimum Neuron documentation <https://huggingface.co/docs/optimum-neuron/en/index>`_

PyTorch Lightning
"""""""""""""""""

PyTorch Lightning is a deep learning framework for professional AI researchers and machine learning engineers who need maximal 
flexibility without sacrificing performance at scale. Lightning organizes PyTorch code to remove boilerplate and unlock scalability.

`Get Started with Lightning  <https://lightning.ai/lightning-ai/studios/finetune-llama-90-cheaper-on-aws-trainium~01hh3kj60fs8b8x91rv9n9fn2j?section=featured>`_

Use PyTorch Lightning Trainer with :ref:`NxD <pytorch-lightning>`. 


AXLearn
""""""""

AXLearn is an open-source JAX-based library used by AWS Neuron for training deep learning models on AWS Trainium. Integrates with JAX ecosystem and supports distributed training.

Check `AXLearn Github repository <https://github.com/apple/axlearn>`_


Additional third-party libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NeMo 
""""
:ref:`NxD Training <nxd-training-overview>` offers a `NeMo <https://github.com/NVIDIA/NeMo>`_-compatible YAML interface for training 
PyTorch models on AWS Trainium chips. The library supports both Megatron-LM and HuggingFace model classes through its model hub. 
NxD Training leverages key NeMo components, including Experiment Manager for tracking ML experiments and data loaders for efficient 
data processing. This library simplifies the process of training deep learning models on AWS Trainium while providing compatibility 
with familiar NeMo YAML Interface.

