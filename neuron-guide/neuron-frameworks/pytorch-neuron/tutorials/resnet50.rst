.. _pytorch-tutorials-resnet-50:

PyTorch Resnet-50 Tutorial
==========================

.. contents:: Table of Contents
   :local:
   :depth: 2



Overview
--------

In this tutorial we will compile and deploy Resnet-50 model on an Inf1 instance. To enable faster enviroment setup, you will run the tutorial on an inf1.6xlarge instance to enable both compilation and deployment (inference) on the same instance.

.. note::
 
  Model compilation can be executed on a non-inf1 instance for later deployment. Follow the same EC2 Developer Flow Setup using other instance families and leverage Amazon Simple Storage Service (S3) to share the compiled models between different instances.

This tutorial is divided into the following parts:

* :ref:`resnet50 Environment Setup`  - Steps needed to setup the compilation and deployment enviroments that will enable you to run this tutorial. In this tutorial a single inf1 instance will provide both the compilation and deployment enviroments.

 If you already have Inf1 environment ready, you can skip to :ref:`resnet50 Running the tutorial`.

* :ref:`resnet50 Running the tutorial` - The tutorial is available as a Jupyter notebook. You have the option to run the tutorial as a Jupyter notebook or run the tutorial on the EC2 instance terminal as a script. This section will guide you into the two options. 
* :ref:`resnet50-cleanup-instances` - After running the tutorial, make sure to cleanup instance/s used for this tutorial.

.. _resnet50 Environment Setup:

Setup The Environment 
---------------------

Please launch Inf1 instance by following the below steps, please make sure to choose an inf1.6xlarge instance.

.. include:: /neuron-intro/install-templates/launch-inf1-dlami.rst

.. _resnet50 Running the tutorial:

Run The Tutorial
----------------


After connecting to the instance from the terminal, clone the Neuron Github repository to the EC2 instance and then change the working directory to the tutorial directory:

.. code::

  git clone https://github.com/aws/aws-neuron-sdk.git
  cd aws-neuron-sdk/src/examples/pytorch
  

The Jupyter notebook is available as a file with the name :ref:`resnet50.ipynb </src/examples/pytorch/resnet50.ipynb>`, you can either run the Jupyter notebook from a browser or run it as a script from terminal:


* **Running tutorial from browser**

  * First setup and launch the Jupyter notebook on your local browser by following instructions at :ref:`Running Jupyter Notebook Browser`
  * Open the Jupyter notebook from the menu and follow the instructions
  
  
You can also view the Jupyter notebook at:

.. toctree::
   :maxdepth: 1

   /src/examples/pytorch/resnet50.ipynb

.. _resnet50-cleanup-instances:

Clean up your instance/s
------------------------

After you've finished with the instance/s that you created for this tutorial, you should clean up by terminating the instance/s, please follow instructions at `Clean up your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-clean-up-your-instance>`_.
