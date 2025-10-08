.. _setup-jupyter-notebook-steps-troubleshooting:
.. _Running Jupyter Notebook Browser:

Jupyter Notebook QuickStart
===========================

.. contents:: Table of Contents
   :local:
   :depth: 2

SSH Tunnel to the Inf1/Trn1 instance
------------------------------------
The Jupyter notebook can be run via a browser on port 8888 by default. For simplicity we will use ssh port forwarding from your machine to the instance.

::

   ssh -i "<pem file>" <user>@<instance DNS name> -L 8888:127.0.0.1:8888

On an Ubuntu image the user will be ubuntu@, while on AL2 you should use
ec2-user@

This additional argument forwards connections to port 8888 on your
machine to the new Inf1/Trn1 instance.


Starting the Jupyter Notebook on the instance
---------------------------------------------
From your ssh prompt on the Inf1/Trn1 instance run

::

   jupyter notebook

You should see logging in your ssh session similar to:

.. code:: bash

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


Copy/paste this URL into your browser when you connect for the first
time, to login with a token:
``http://localhost:8888/?token=f9ad4086afd3c91f33d5587781f9fd8143b4cafbbf121a16&token=f9ad4086afd3c91f33d5587781f9fd8143b4cafbbf121a16``

.. code:: bash

   [I 21:53:12.004 NotebookApp] Starting initial scan of virtual environments...
   [I 21:53:13.507 NotebookApp] Found new kernels in environments: conda_tensorflow2_p27, conda_aws_neuron_mxnet_p36, conda_anaconda3, conda_tensorflow_p27, conda_chainer_p27, conda_python3, conda_tensorflow_p36, conda_aws_neuron_tensorflow_p36, conda_mxnet_p27, **conda_my_notebook_env**, conda_tensorflow2_p36, conda_pytorch_p27, conda_python2, conda_chainer_p36, conda_mxnet_p36, conda_pytorch_p36



Running the Jupyter Notebook from your local browser
----------------------------------------------------

If you copy and paste the link that looks like
``http://localhost:8888/?token=f9ad4086afd3c91f33d5587781f9fd8143b4cafbbf121a16&token=f9ad4086afd3c91f33d5587781f9fd8143b4cafbbf121a16``
into your local browser the Notebook navigation pane should pop up.

This works because ssh is forwarding you local port 8888 through to the
Inf1/Trn1 instance port 8888 where the notebook is running. Note that our new
conda environment is visible as “kernel” with the “conda\_” prefix
(highlighted)

1) In notebook browser select the tutorial.
2) This will pop up a new tab. In that tab use the menus:

Kernel → Change Kernel → Environment (conda_my_notebook_env)

3) Start reading through the self documenting notebook tutorial

Troubleshooting
---------------

If your jupyter notebook does not start please try the following:

::

   mv ~/.jupyter ~/.jupyter.old
   mkdir -p ~/.jupyter
   echo "c.NotebookApp.iopub_data_rate_limit = 10000000000" > ~/.jupyter/jupyter_notebook_config.py

   # Instal Jupyter notebook kernel
    pip install ipykernel
    python3 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python Neuronx"
    pip install jupyter notebook
    pip install environment_kernels

   jupyter notebook


