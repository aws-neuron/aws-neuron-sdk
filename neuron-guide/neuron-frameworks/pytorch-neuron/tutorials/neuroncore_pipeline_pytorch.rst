.. _pytorch-tutorials-neuroncore-pipeline-pytorch:

Using NeuronCore Pipeline with PyTorch Tutorial
================================================================

.. contents:: Table of Contents
   :local:
   :depth: 2



Overview
--------

In this tutorial we will benchmark latency of a Hugging Face Transformers model deployed in model pipeline paralle mode using the NeuronCore Pipeline feature. We will compare the results with the usual data parallel (multi-worker) deployment. We compile a pretrained BERT base model and run the benchmarking locally.

To enable faster enviroment setup, We will run both compilation and deployment (inference) on an single inf1.6xlarge instance. You can take similar steps to recreate the benchmark on other instance sizes, such as inf1.xlarge.

If you already have an Inf1 instance environment ready, this tutorial is availabe as a Jupyter notebook at :pytorch-neuron-src:`neuroncore_pipeline_pytorch.ipynb <pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb>` and instructions can be viewed at: 

.. toctree::
   :maxdepth: 1

   /src/examples/pytorch/pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb

Instructions of how to setup the environment and run the tutorial are available in the next sections.

.. _pytorch-neuroncore-pipeline-pytorch-env-setup:

Setup The Environment 
---------------------

Launch an Inf1 instance by following the below steps, please make sure to choose an inf1.6xlarge instance.

.. include:: /neuron-intro/install-templates/launch-inf1-dlami.rst


.. _pytorch-neuroncore-pipeline-pytorch-run-tutorial:

Run The Tutorial
----------------

After connecting to the instance from the terminal, clone the Neuron Github repository to the EC2 instance and then change the working directory to the tutorial directory:

.. code::

  git clone https://github.com/aws/aws-neuron-sdk.git
  cd aws-neuron-sdk/src/examples/pytorch
  


The Jupyter notebook is available as a file with the name :pytorch-neuron-src:`neuroncore_pipeline_pytorch.ipynb <pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb>`, you can either run the Jupyter notebook from a browser or run it as a script from terminal:


* **Running tutorial from browser**

  * First setup and launch the Jupyter notebook on your local browser by following instructions at :ref:`Running Jupyter Notebook Browser`
  * Open the Jupyter notebook from the menu and follow the instructions
  
  
You can also view the Jupyter notebook at:

.. toctree::
   :maxdepth: 1

   /src/examples/pytorch/pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb


.. _pytorch-neuroncore-pipeline-pytorch-cleanup-instances:

Clean up your instance/s
------------------------

After you've finished with the instance/s that you created for this tutorial, you should clean up by terminating the instance/s, please follow instructions at `Clean up your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-clean-up-your-instance>`_.
