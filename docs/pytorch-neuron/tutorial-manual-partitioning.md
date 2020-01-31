# Tutorial: Manual partitioning in a Jupyter notebook

## Steps Overview:

1. Launch an EC2 compilation instance running the DLAMI (recommended instance: c5.4xlarge or stronger)
2. Connect using ssh and local port forwarding
3. Setup a conda environment for the notebook to use as a kernel
4. Fetch the notebook from github
5. Start Jupyter and select the correct conda environment
6. Execute the Notebook to compile a partitioned compute graph

These steps will allow you to setup an environment for running Jupyter Notebooks, and in particular the tutorial on manual partitioning using Neuron for PyTorch, and access it via your notebook.

## Step 1: Launch EC2 compilation instance

Using the AWS console, select a Deep Learning AMI (DLAMI) of your choice, which may be Ubuntu 16.x, Ubuntu 18.x, Amazon Linux 2 based. For this tutorial we use an inf1.xlarge instance, you may want to experiment with a larger instance.

## Step 2: Connecting to your instance

For this tutorial we will use a Jupyter notebook that runs via a browser on port 8888 by default.  For simplicity we will use ssh port forwarding from your machine to the instance.

The regular ssh instructions is:

```
ssh -i "<pem file>" <user>@<instance DNS name>
```

On an Ubuntu image the user will be ubuntu@, while on AL2 you should use ec2-user@

We will modify this base for to use:

```
ssh -i "<pem file>" <user>@<instance DNS name> -L 8888:127.0.0.1:8888
```

This additional argument forwards connections to port 8888 on your machine to the new inf1 instance.

Now: ssh to the Inf1 instance

## Step 3: Set up the Neuron Runtime conda environment & create a tutorial directory

1) Install the neuron runtime using these instruction:  [Getting started: Installing and Configuring Neuron-RTD](https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-runtime/nrt_start.md).

2) Set up your own conda environment ahead of launching 

```
echo ". /home/ubuntu/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate

conda create --name my_notebook_env python=3.6 -y
conda activate my_notebook_env

pip install torch-neuron --extra-index-url=https://pip.repos.neuron.amazonaws.com
pip install neuron-cc[tensorflow] --extra-index-url=https://pip.repos.neuron.amazonaws.com
pip install pillow==6.2.2
pip install torchvision --no-deps
pip install jupyter

mkdir -p notebook_tutorial
cd notebook_tutorial
```

## Step 4: Fetch the notebook from GitHub

Run the following command to fetch the notebook into the current directory:

```bash
curl -O https://github.com/aws/aws-neuron-sdk/blob/master/src/pytorch/resnet50_partition.ipynb
```


## Step 5: Start Jupyter and connect in your browser:

From your ssh prompt run

```
jupyter notebook
```

You should see logging in your ssh session similar to::

```bash
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
```

 Copy/paste this URL into your browser when you connect for the first time,
 to login with a token:
 `http://localhost:8888/?token=f9ad4086afd3c91f33d5587781f9fd8143b4cafbbf121a16&token=f9ad4086afd3c91f33d5587781f9fd8143b4cafbbf121a16`
 
```bash
[I 21:53:12.004 NotebookApp] Starting initial scan of virtual environments...
[I 21:53:13.507 NotebookApp] Found new kernels in environments: conda_tensorflow2_p27, conda_aws_neuron_mxnet_p36, conda_anaconda3, conda_tensorflow_p27, conda_chainer_p27, conda_python3, conda_tensorflow_p36, conda_aws_neuron_tensorflow_p36, conda_mxnet_p27, **conda_my_notebook_env**, conda_tensorflow2_p36, conda_pytorch_p27, conda_python2, conda_chainer_p36, conda_mxnet_p36, conda_pytorch_p36
```

If you copy and paste the link that looks like `http://localhost:8888/?token=f9ad4086afd3c91f33d5587781f9fd8143b4cafbbf121a16&token=f9ad4086afd3c91f33d5587781f9fd8143b4cafbbf121a16` into your local browser the Notebook navigation pane should pop up.  

This works because ssh is forwarding you local port 8888 through to the Inf1 instance port 8888 where the notebook is running.  Note that our new conda environment is visible as “kernel” with the “conda_” prefix (highlighted)

## Step 6: Start the notebook & select the correct kernel

1) In notebook browser select “[resnet50_partition.ipynb](http://localhost:8888/notebooks/resnet50_partition.ipynb)“
2) This will pop up a new tab.  In that tab use the menus:

Kernel → Change Kernel → Environment (conda_my_notebook_env)

3) Start reading through the self documenting notebook tutorial

## Step 7: Terminate your instance

When done, don' forget to terminate your instance through the AWS console to avoid ongoing charges
