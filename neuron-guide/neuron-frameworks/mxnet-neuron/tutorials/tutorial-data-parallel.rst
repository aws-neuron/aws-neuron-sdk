MXNet 1.8: Using Data Parallel Mode Tutorial 
==============================================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview:
---------
In this tutorial you will compile and run models in data-parallel (multi-worker) mode using the newly supported MXNet 1.8 on an Inf1 instance. You will benchmark latency and throughput for a Gluon BERT model in data-parallel mode and compare that against a single worker setup. To enable faster environment setup, you will run the tutorial on an Inf1.2xlarge instance to enable both compilation and inference on the same instance.

This tutorial is only supported with MXNet 1.8.

If you already have an Inf1 instance environment ready, this tutorial is availabe as a Jupyter notebook at :mxnet-neuron-src:`data_parallel_tutorial.ipynb <data_parallel/data_parallel_tutorial.ipynb>` and instructions can be viewed at: 

.. toctree::
   :maxdepth: 1

   /src/examples/mxnet/data_parallel/data_parallel_tutorial.ipynb


Setup the Environment
---------------------

- Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to Launch an Inf1 instance, when choosing the instance type at the EC2 console. Please make sure to select the correct instance type. To get more information about Inf1 instances sizes and pricing see `Inf1 web page <https://aws.amazon.com/ec2/instance-types/inf1/>`_.

- When choosing an Amazon Machine Image (AMI) make sure to select `Deep Learning AMI with Conda Options <https://docs.aws.amazon.com/dlami/latest/devguide/conda.html>`_. Please note that Neuron Conda packages are supported only in Ubuntu 16 DLAMI, Ubuntu 18 DLAMI and Amazon Linux2 DLAMI, Neuron Conda packages are not supported in Amazon Linux DLAMI.

- After launcing the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-connect-to-instance-linux>`_ to connect to the instance. 

.. note::

  You can also launch the instance from AWS CLI, please see 
  :ref:`AWS CLI commands to launch inf1 instances <launch-inf1-dlami-aws-cli>`.

- Install MXNet-Neuron and Neuron Compiler On Compilation Instance

    - To install in your own AMI, please see :ref:`neuron-install-guide` to
      setup virtual environment and install MXNet-Neuron (mxnet-neuron) and
      Neuron Compiler (neuron-cc) packages.

Run The Tutorial
----------------

After connecting to the instance from the terminal, clone the Neuron Github repository to the EC2 instance and then change the working directory to the tutorial directory:

.. code::

  git clone https://github.com/aws/aws-neuron-sdk.git
  cd aws-neuron-sdk/src/examples/mxnet



The Jupyter notebook is available as a file with the name :mxnet-neuron-src:`data_parallel_tutorial.ipynb <data_parallel/data_parallel_tutorial.ipynb>`, you can either run the Jupyter notebook from a browser or run it as a script from terminal:


* **Running tutorial from browser**

  * First setup and launch the Jupyter notebook on your local browser by following instructions at :ref:`Running Jupyter Notebook Browser`
  * Open the Jupyter notebook from the menu and follow the instructions


You can also view the Jupyter notebook at:

.. toctree::
   :maxdepth: 1

   /src/examples/mxnet/data_parallel/data_parallel_tutorial.ipynb



Clean up your instance/s
------------------------
After you've finished with the instance/s that you created for this tutorial, you should clean up by terminating the instance/s, please follow instructions at 
`Clean up your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-clean-up-your-instance>`_.