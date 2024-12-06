.. _sagemaker_flow:

Amazon SageMaker
================

Amazon SageMaker is a fully managed machine learning (ML) platform that streamlines the end-to-end ML workflow at scale. AWS Neuron integrates 
with Amazon SageMaker to provide optimized performance for ML workloads on AWS Inferentia and AWS Trainium chips.

.. contents:: Table of contents
   :local:
   :depth: 1

SageMaker JumpStart
"""""""""""""""""""
Use `Amazon SageMaker JumpStart <https://aws.amazon.com/sagemaker/jumpstart/>`_ to train and deploy models using Neuron.  SageMaker JumpStart is an ML hub that accelerates model 
selection and deployment. It provides support for fine-tuning and deploying popular models such as Meta’s Llama family of models. 
Users can customize pre-trained models with their data and easily deploy them.

SageMaker HyperPod
""""""""""""""""""
Use `Amazon SageMaker HyperPod <https://aws.amazon.com/sagemaker/hyperpod/>`_ to streamline ML infrastructure setup and optimization with AWS Neuron. SageMaker HyperPod leverages 
pre-configured distributed training libraries to split workloads across numerous AI accelerators, enhancing model performance. 
HyperPod ensures uninterrupted training through automatic checkpointing, fault detection, and recovery.

SageMaker Training
""""""""""""""""""
`Amazon SageMaker Model Training <https://aws.amazon.com/sagemaker/train/>`_ reduces the time and cost to train and tune ML models at scale without the need to manage infrastructure.

SageMaker Inference
"""""""""""""""""""
With `Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html>`_ , you can start getting predictions, or inferences, from your trained ML models. SageMaker 
provides a broad selection of ML infrastructure and model deployment options to help meet all your ML inference needs.