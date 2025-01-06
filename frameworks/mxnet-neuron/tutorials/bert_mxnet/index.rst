.. _mxnet-bert-tutorial:

Tutorial: Apache MXNet BERT in a Jupyter notebook
===================================================

Introduction
------------

BERT (Bidirectional Encoder Representations from Transformers) is a
Google Research project published in 2018 (https://arxiv.org/abs/1810.04805). BERT has a number of practical applications,
it can be used for question answering, sequence prediction and sequence
classification amongst other tasks.

This tutorial is using Jupyter notebooks to adapt the BERT-base model
from https://github.com/dmlc/gluon-nlp, for the purpose of classifying
sentences.

In this tutorial we will use a trained model, an inf1.2xlarge to compile
the model to Inferentia using neuron-cc. We will use the same
inf1.2xlarge to also run inference. The aim is to demonstrate how to
compile, infer and measure performance.

In this tutorial we’ll also leverage the AWS Deep Learning AMI. This
tutorial assumes you know how to configure your AWS CLI
(https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html),
which is required for the notebooks to run.

The tutorial assumes you are operating in us-east-1. It is possible to
run in other regions, but you will need to choose a region where Inf1
instances are supported, and modify the setup script for MRPC or the
training notebook where your S3 bucket is created.

Steps Overview
--------------

These steps will allow you to setup an environment for running Jupyter
Notebooks, and in particular the tutorial on MXNet BERT on Inferentia,
and access it via your notebook.

-  Launch an EC2 Inf1 instance running the DLAMI (recommended instance:
   Inf1.2xlarge)
-  Connect using ssh and local port forwarding
-  Setup a virtual environment for the notebook to use as a kernel
-  Fetch the notebook from github
-  Start Jupyter and select the correct python virtual environment
-  Execute the Notebook to compile a partitioned compute graph

Step 1: Launch EC2 instance
---------------------------

For this task we’ll use a inf1.2xlarge instance. Ensure it has the
latest DLAMI. Refer to the :ref:`install-neuron-mxnet` for details.

Step 2: Connecting to your instance
-----------------------------------

In this tutorial we use a Jupyter notebook that runs via a browser on
port 8888 by default. For simplicity we will use ssh port forwarding
from your machine to the instance. The regular ssh command is:

::

   ssh -i "<pem file>" <user>@<instance DNS name>

We will modify this base for to use:

::

   ssh -i "<pem file>" <user>@<instance DNS name> -L 8888:127.0.0.1:8888

On an Ubuntu AMI the user will be ubuntu@, while on an AL2 the user will
be ec2-user@

This additional argument forwards connections to port 8888 on your
machine to the new Inf1 instance. Now: ssh to the Inf1 instance

.. _step-3-set-up-the-neuron-runtime-environment--create-a-tutorial-directory:

Step 3: Set up the Neuron Runtime environment & create a tutorial directory
---------------------------------------------------------------------------

If using Conda DLAMI version 27 and up, activate pre-installed
MXNet-Neuron environment (using
``source activate aws_neuron_mxnet_p36`` command). Please update
MXNet-Neuron environment by following update steps in :ref:`install-neuron-mxnet`.

To install in your own AMI, please see :ref:`install-neuron-mxnet` to setup virtual environment and
install MXNet-Neuron (mxnet-neuron) and Neuron Compiler (neuron-cc)
packages. In this tutorial we will use a python virtual environment.

::

   # Make sure we are up to date
   sudo apt update
   sudo apt upgrade

Setup a new Python virtual environment:

::

   python3 -m venv test_venv
   source test_venv/bin/activate
   pip install -U pip

Modify Pip repository configurations to point to the Neuron repository:

::

   tee $VIRTUAL_ENV/pip.conf > /dev/null <<EOF
   [global]
   index-url = https://pip.repos.neuron.amazonaws.com
   EOF

Install neuron packages:

::

   pip install mxnet-neuron
   pip install neuron-cc
   pip install wget jupyter

Create a work directory:

::

   mkdir -p notebook_tutorial
   cd notebook_tutorial

Step 4: Fetch the notebook from GitHub
--------------------------------------

Run the following command to fetch the notebook into the current
directory:

::

   wget -O bert_mxnet.ipynb https://raw.githubusercontent.com/aws/aws-neuron-sdk/master/src/examples/mxnet/bert_mxnet.ipynb


Step 5: Start Jupyter
---------------------

From your ssh prompt run

::

   # lets clear the old config
   mv ~/.jupyter ~/.jupyter.old
   mkdir -p ~/.jupyter
   echo "c.NotebookApp.iopub_data_rate_limit = 10000000000" > ~/.jupyter/jupyter_notebook_config.py

   #Start jupyter
   jupyter notebook

You should see logging in your ssh session similar to::

::

   [I 21:53:11.729 NotebookApp] Using EnvironmentKernelSpecManager...
   [I 21:53:11.730 NotebookApp] Started periodic updates of the kernel list (every 3 minutes).
   [I 21:53:11.867 NotebookApp] Loading IPython parallel extension
   [I 21:53:11.884 NotebookApp] JupyterLab beta preview extension loaded from /home/ubuntu/anaconda3/lib/python3.6/site-packages/jupyterlab
   [I 21:53:11.884 NotebookApp] JupyterLab application directory is /home/ubuntu/anaconda3/share/jupyter/lab
   [I 21:53:12.002 NotebookApp] [nb_conda] enabled
   [I 21:53:12.004 NotebookApp] Serving notebooks from local directory: /home/ubuntu/tutorial
   [I 21:53:12.004 NotebookApp] 0 active kernels
   [I 21:53:12.004 NotebookApp] The Jupyter Notebook is running at:
   [I 21:53:12.004 NotebookApp] http://localhost:8888/?token=f9ad4086afd3c91f33d5587781f9fd8143b4cafbbf121a16
   [I 21:53:12.004 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
   [W 21:53:12.004 NotebookApp] No web browser found: could not locate runnable browser.
   [C 21:53:12.004 NotebookApp] 

If you copy and paste the link that looks like
``http://localhost:8888/?token=f9ad4086afd3c91f33d5587781f9fd8143b4cafbbf121a16&token=f9ad4086afd3c91f33d5587781f9fd8143b4cafbbf121a16``
into your local browser the Notebook navigation pane should pop up.

This works because ssh is forwarding you local port 8888 through to the
Inf1 instance port 8888 where the notebook is running. Note that our new
conda environment is visible as “kernel” with the “conda\_” prefix
(highlighted)

.. _step-6-start-the-notebook--select-the-correct-kernel:

Step 6: Start the notebook and select the correct kernel
--------------------------------------------------------

-  In notebook browser select “bert_mxnet.ipynb”
-  This will pop up a new tab. In that tab use the menus:

   -  Kernel → Change Kernel → Environment (my_notebook_env)

-  Start reading through the self documenting notebook tutorial

Step 7: Terminate your instance
-------------------------------

When done, don't forget to terminate your instance through the AWS
console to avoid ongoing charges

Appendix
--------

-  Try installing environment_kernels, if you see the following error
   while launching Jupyter notebook:

::

   [C 06:39:39.153 NotebookApp] Bad config encountered during initialization: 
   [C 06:39:39.153 NotebookApp] The 'kernel_spec_manager_class' trait of <notebook.notebookapp.NotebookApp object at 0x7f21309035c0> instance must be a type, but 'environment_kernels.EnvironmentKernelSpecManager' could not be imported

-  If you do not see your conda enviroment in jupyter kernel list, try
   installing the kernel manually:

::

   python -m ipykernel install --user --name my_notebook_env --display-name "Python (my_notebook_env)"
