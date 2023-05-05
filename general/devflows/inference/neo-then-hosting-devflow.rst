.. _neo-then-hosting-devflow:

Compile with Sagemaker Neo and Deploy on Sagemaker Hosting (inf1)
==========================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

   
Description
-----------

|image|
 
.. |image| image:: /images/neo-then-hosting-dev-flow.png
   :width: 700
   :alt: Neuron developer flow on SageMaker Neo
   :align: middle

You can use SageMaker Neo to compile models for deployment on SageMaker Hosting using ml.inf1 instances. In this developer flow, you provision a Sagemaker Notebook instance to train, compile and deploy your model using the SageMaker Python SDK. Follow the steps bellow to setup your environment. 

.. _neo-then-hosting-setenv:

Setup Environment
-----------------

1. Create an Amazon SageMaker Notebook Instance:

	Follow the instructions in `Get Started with Notebook Instances <https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html>`_

	The Notebook instance created provides the required Python SDK for training, compiling and deploying models with Amazon SageMaker.

2. Compile a model using the Amazon SageMaker SDK:

	Refer to `Supported Instances Types and Frameworks <https://docs.aws.amazon.com/sagemaker/latest/dg/neo-supported-cloud.html>`_ for information on the framework versions currently supported by Amazon SageMaker Neo on AWS Inferentia. 

	More information about compiling and deploying models with Amazon SageMaker Neo can be found on `Use Neo to Compile a Model <https://docs.aws.amazon.com/sagemaker/latest/dg/neo-job-compilation.html>`_






