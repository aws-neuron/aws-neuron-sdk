.. _pytorch-tutorials-YOLOv4:

PyTorch - Evaluate YOLO v4 on Inferentia
========================================

.. contents:: Table of Contents
   :local:
   :depth: 2



Overview
--------

In this tutorial you will compile and deploy the YOLO v4 model on an Inf1 instance. To enable faster enviroment setup, you will run the tutorial on an inf1.6xlarge instance to enable both compilation and deployment (inference) on the same instance.

If you already launched an Inf1 instance and have Neuron Pytorch DLAMI environment ready, tutorial is available as a Jupyter notebook at :pytorch-neuron-src:`yolo_v4.ipynb <yolo_v4.ipynb>` and instructions can be viewed at

.. toctree::
   :maxdepth: 1

   /src/examples/pytorch/yolo_v4.ipynb  

 
.. _pytorch-YOLOv4-env-setup:

Instructions of how to setup Neuron Pytorch environment and run the tutorial as a Jupyter notebook are available in the next sections.

Setup The Environment
---------------------

Launch Inf1 instance by following the below steps, please make sure to choose an inf1.6xlarge instance.

.. include:: /neuron-intro/install-templates/launch-inf1-dlami.rst

.. _pytorch-YOLOv4-run-tutorial:

Run The Tutorial
----------------

After connecting to the instance from the terminal, clone the Neuron Github repository to the EC2 instance and then change the working directory to the tutorial directory:

.. code::

  git clone https://github.com/aws/aws-neuron-sdk.git
  cd aws-neuron-sdk/src/examples/pytorch/
  


The Jupyter notebook is available as a file with the name :pytorch-neuron-src:`yolo_v4.ipynb <yolo_v4.ipynb>`, you can either run the Jupyter notebook from a browser or run it as a script from terminal:


* **Running tutorial from browser**

  * First setup and launch the Jupyter notebook on your local browser by following instructions at :ref:`Running Jupyter Notebook Browser`
  * Open the Jupyter notebook from the menu and follow the instructions
  
  
You can also view the Jupyter notebook at:

.. toctree::
   :maxdepth: 1

   /src/examples/pytorch/yolo_v4.ipynb   

.. _pytorch-YOLOv4-cleanup-instances:

Clean up your instance(s)
------------------------

After you've finished with the instance(s) that you created for this tutorial, you should clean up by terminating the instance(s), please follow instructions at `Clean up your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-clean-up-your-instance>`_.
