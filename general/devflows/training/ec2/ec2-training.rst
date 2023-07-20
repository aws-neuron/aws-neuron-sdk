.. _ec2-training:

Train your model on EC2
=======================

.. contents:: Table of Contents
   :local:
   :depth: 3
   
Description
-----------

|image|
 
.. |image| image:: /images/trn1-on-ec2-dev-flow.png
   :width: 500
   :alt: Neuron developer flow on EC2
   :align: middle
   
You can use a single Trn1 instance as a development environment to compile and train Neuron models. In this developer flow, you provision an EC2 Trn1 instance using a Deep Learming AMI (DLAMI) and execute the two steps of the development flow in the same instance. The DLAMI comes pre-packaged with the Neuron frameworks, compiler, and required runtimes to complete the flow. Development happens through Jupyter Notebooks or using a secure shell (ssh) connection in terminal. Follow the steps bellow to setup your environment.

Setup Environment
-----------------

1. Launch an Trn1 Instance
^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. include:: /general/setup/install-templates/launch-trn1-dlami.rst

2. Set up a development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
Enable PyTorch-Neuron
~~~~~~~~~~~~~~~~~~~~~

    .. include:: /frameworks/torch/torch-neuronx/setup/install-templates/pytorch-dev-install.txt

3. Set up Jupyter notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^

To develop from a Jupyter notebook see :ref:`setup-jupyter-notebook-steps-troubleshooting`  

You can also run a Jupyter notebook as a script, first enable the ML framework Conda or Python environment of your choice and see :ref:`running-jupyter-notebook-as-script` for instructions. 
