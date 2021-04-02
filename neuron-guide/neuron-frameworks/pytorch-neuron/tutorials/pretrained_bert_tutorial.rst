.. _pytorch-tutorials-HuggingFace Pretrained BERT:

PyTorch - HuggingFace Pretrained BERT Tutorial
====================================

.. contents:: Table of Contents
   :local:
   :depth: 2



Overview
--------

In this tutorial we will compile and deploy HuggingFace Pretrained BERT model on an Inf1 instance. To enable faster enviroment setup, you will run the tutorial on an inf1.6xlarge instance to enable both compilation and deployment (inference) on the same instance.

.. note::
 
  Model compilation can be executed on an inf1 instance. Follow the same EC2 Developer Flow Setup using other instance families and leverage Amazon Simple Storage Service (S3) to share the compiled models between different instances.

If you already have Inf1 environment ready, you can skip to :ref:`pytorch-HuggingFace Pretrained BERT-run-tutorial`.
 
.. _pytorch-HuggingFace Pretrained BERT-env-setup:

Setup The Environment 
---------------------

Launch Inf1 instance by following the below steps, please make sure to choose an inf1.6xlarge instance.

.. include:: /neuron-intro/install-templates/launch-inf1-dlami.rst


.. _pytorch-HuggingFace Pretrained BERT-run-tutorial:

Run The Tutorial
----------------

After connecting to the instance from the terminal, clone the Neuron Github repository to the EC2 instance and then change the working directory to the tutorial directory:

.. code::

  git clone https://github.com/aws/aws-neuron-sdk.git
  cd aws-neuron-sdk/src/examples/pytorch/bert_tutorial
  


The Jupyter notebook is available as a file with the name :pytorch-neuron-src:`tutorial_pretrained_bert.ipynb <bert_tutorial/tutorial_pretrained_bert.ipynb>`, you can either run the Jupyter notebook from a browser or run it as a script from terminal:


* **Running tutorial from browser**

  * First setup and launch the Jupyter notebook on your local browser by following instructions at :ref:`Running Jupyter Notebook Browser`
  * Open the Jupyter notebook from the menu and follow the instructions
  
  
You can also view the Jupyter notebook at:

.. toctree::
   :maxdepth: 1

   /src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.ipynb


.. _pytorch-HuggingFace Pretrained BERT-cleanup-instances:

Clean up your instance(s)
------------------------

After you've finished with the instance(s) that you created for this tutorial, you should clean up by terminating the instance(s), please follow instructions at `Clean up your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-clean-up-your-instance>`_.
