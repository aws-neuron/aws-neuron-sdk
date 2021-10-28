.. _byoc-hosting-devflow:

Bring Your Own Neuron Container to Sagemaker Hosting
====================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

   
Description
-----------

|image|
 
.. |image| image:: /images/byoc-then-hosting-dev-flow.png
   :width: 850
   :alt: Neuron developer flow on SageMaker Neo
   :align: middle

You can use a SageMaker Notebook or an EC2 instance to compile models and build your own containers for deployment on SageMaker Hosting using ml.inf1 instances. In this developer flow, you provision a Sagemaker Notebook or an EC2 instance to train and compile your model to Inferentia. Then you deploy your model to SageMaker Hosting using the SageMaker Python SDK. Follow the steps bellow to setup your environment. Once your environment is set you'll be able to follow the :ref:`BYOC HuggingFace pretrained BERT container to Sagemaker Tutorial </src/examples/pytorch/byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb>` .

.. _byoc-hosting-setenv:

Setup Environment
-----------------

1. Create a Compilation Instance:
	If using an **EC2 instance for compilation** you can use an Inf1 instance to compile and test a model. Follow these steps to launch an Inf1 instance:
		
		.. include:: /neuron-intro/install-templates/launch-inf1-ami.rst
	

	If using an **SageMaker Notebook for compilation**, follow the instructions in `Get Started with Notebook Instances <https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html>`_ to provision the environment. 

	It is recommended that you start with an ml.c5.4xlarge instance for the compilation. Also, increase the volume size of you SageMaker notebook instance, to accomodate the models and containers built locally. A volume of 10GB is sufficient.
	
		.. note::
			
			To compile the model in the SageMaker Notebook instance, you'll need to update the conda environments to include the Neuron Compiler and Neuron Framework Extensions. Follow the installation guide on the section :ref:`how-to-update-to-latest-Neuron-Conda-Env` to update the environments.  


2. Set up the environment to compile a model, build your own container and deploy:
    To compile your model on EC2 or SageMaker Notebook, follow the *Set up a development environment* section on the EC2 :ref:`ec2-then-ec2-setenv` documentation.

    Refer to `Adapting Your Own Inference Container <https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html>`_ documentation for information on how to bring your own containers to SageMaker Hosting.

    Make sure to add the **AmazonEC2ContainerRegistryPowerUser** role to your IAM role ARN, so you're able to build and push containers from your SageMaker Notebook instance.

    .. note::
        The container image can be created using :ref:`how-to-build-neuron-container`.
