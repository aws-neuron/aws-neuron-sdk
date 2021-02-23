.. _dlc-then-ec2-devflow:

Deploy DLC on EC2
=================

.. contents:: Table of Contents
   :local:
   :depth: 2

   
Description
-----------

|image|
 
.. |image| image:: /images/dlc-on-ec2-dev-flow.png
   :width: 500
   :alt: Neuron developer flow for DLC on EC2
   :align: middle

You can use the Neuron version of the `AWS Deep Learning Containers <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2-tutorials-inference.html>`_ to run inference on inf1 instances. In this developer flow, you provision an EC2 inf1 instance using a Deep Learming AMI (DLAMI), pull the container image with the Neuron version of the desired framework, and run the container as a server for the already compiled model. This developer flow assumes the model has already has been compiled through a :ref:`compilation developer flow <compilation-flow-target>` 

.. _dlc-then-ec2-setenv:

Setup Environment
-----------------

1. Launch an Inf1 Instance
	.. include:: /neuron-intro/install-templates/launch-inf1-dlami.rst

	To deploy your container using a Jupyter Notebook see :ref:`setup-jupyter-notebook-steps-troubleshooting`  

2. Deploy an inference container on your inf1 instance:
	Follow the `Getting Started with Deep Learning Containers for Inference on EC2 <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2-tutorials-inference.html>`_. 

.. note:: 

	**Prior to running the container**, make sure that the Neuron runtime on the instance is turned off, by running the command:

	.. code:: bash

		sudo service neuron-rtd stop



