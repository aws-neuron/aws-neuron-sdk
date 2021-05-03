.. _mxnet-resnet50-neuroncore-group:


Neuron Apache MXNet (Incubating) - Configurations for NeuronCore Groups
=======================================================================


.. contents:: Table of Contents
   :local:
   :depth: 2



Overview
--------

In this tutorial you will compile and deploy Resnet-50 model in parallel using the concept of NeuronCore Groups on an Inf1 instance.


A NeuronCore group is a one-to-one mapping from a compiled model to a set of NeuronCores used to load and run that model. At any time, one model will be running in a NeuronCore Group. With NeuronCore groups a user may load independent models in parallel to execute. Additionally, within a NeuronCore Group, loaded models can be dynamically started and stopped, allowing for dynamic context switching from one model to another.


To enable faster environment setup, you will run the tutorial on an Inf1.6xlarge instance to enable both compilation and deployment (inference) on the same instance.

If you already launched an Inf1 instance and have Neuron MXNet DLAMI environment ready, tutorial is available as a Jupyter notebook at :mxnet-neuron-src:`resnet50_neuroncore_groups.ipynb <resnet50_neuroncore_groups.ipynb>` and instructions can be viewed at,

.. toctree::
   :maxdepth: 1

   /src/examples/mxnet/resnet50_neuroncore_groups.ipynb

Instructions of how to setup Neuron Mxnet environment and run the tutorial as a Jupyter notebook are available in the next sections.


.. _mxnet-resnet50-neuroncore-group-env-setup:

Setup The Environment
---------------------

Launch Inf1 instance by following the steps below, making sure to choose an Inf1.6xlarge instance.

.. include:: /neuron-intro/install-templates/launch-inf1-dlami.rst


.. _mxnet-resnet50-neuroncore-group-run-tutorial:



Run The Tutorial
----------------

After connecting to the instance from the terminal, clone the Neuron Github repository to the EC2 instance and then change the working directory to the tutorial directory:

.. code::

  git clone https://github.com/aws/aws-neuron-sdk.git
  cd aws-neuron-sdk/src/examples/mxnet



The Jupyter notebook is available as a file with the name :mxnet-neuron-src:`resnet50_neuroncore_groups.ipynb <resnet50_neuroncore_groups.ipynb>`. You can either run the Jupyter notebook from a browser or run it as a script from a terminal:


* **Running tutorial from browser**

  * First setup and launch the Jupyter notebook on your local browser by following instructions at :ref:`Running Jupyter Notebook Browser`
  * Open the Jupyter notebook from the menu and follow the instructions


You can also view the Jupyter notebook at:

.. toctree::
   :maxdepth: 1

   /src/examples/mxnet/resnet50_neuroncore_groups.ipynb



.. _mxnet-resnet50-neuroncore-group-instances:

Clean up your instance/s
------------------------

After you've finished with the instance/s that you created for this tutorial, you should clean up by terminating the instance/s, please follow instructions at `Clean up your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-clean-up-your-instance>`_.
