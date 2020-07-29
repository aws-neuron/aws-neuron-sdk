# Adapting, Partitioning and Benchmarking PyTorch BERT

## Overview

BERT (Bidirectional Encoder Representations from Transformers) is a Google Research project published in 2018 (https://arxiv.org/abs/1810.04805).  BERT has a number of practical applications, it can be used for question answering, sequence prediction and sequence classification amongst other tasks.

**Note that automatic partitioning is now released!**.  Use the auto-partitioner on BERT as described [in this Jupyter notebook](tutorial_pretrained_bert.ipynb).  Use of the auto-partitioner is the normal way to compile BERT with PyTorch Neuron.  It's not only easier, but also expected to be performant!  

If you are going to perform manual partitioning, it is important to execute the steps in this guide and not run the notebook without the appropriate installation steps below.  Manual partitioning is intended for advanced users who find that the automated partitioning does not perform, and/or those in need of more control.

This tutorial is using Jupyter notebooks to adapt the BERTlarge model from https://github.com/huggingface/transformers, for the purpose of classifying sentences as having similar or dissimilar meaning, based on the MRPC corpus. You can find more here from Hugging Face: https://github.com/huggingface/transformers#fine-tuning-bert-model-on-the-mrpc-classification-task

In this tutorial we will use a p3.16xlarge instance for adapting the model (optional), a c5n.4xl for compilation and an inf1.2xlarge for running inference with the model.  The aim is to demonstrate the process of adapting a model, manually partitioning a graph in PyTorch, compile, deploy and measure performance.   

In this tutorial we’ll use the AWS Deep Learning AMI. This tutorial assumes you know how to configure your [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html), which is required for the notebooks to run.

The tutorial assumes you are operating in us-east-1. It is possible to run in other regions where inf1 is supported, and modify the setup script for MRPC or the training notebook where your S3 bucket is created. 

## Steps Overview:

1. **STAGE 1:** Adapt the BERT model to the MRPC corpus [Optional]
    1. Start a p3.16xlarge instance, and configure a conda environment for adaptation
    2. Configure the AWS CLI (https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
    3. Download and run the adaptation scripts: 
        1. Create an S3 bucket for these steps
        2. Set Hugging Face BERT
        3. Download MRPC
        4. Adapt the model (this outputs accuracy and F-score metrics)
        5. Upload the model and (optional) GPU benchmark to your S3 location
    4. Terminate your instance
2. **STAGE 2:** Modify the BERT code and compile for 
    1. Start a c5n.4xlarge instance, set up the environment for compilation
    2. Download the manual partitioning and compilation notebook:
        1. Download the adapted BERT model from the previous step
        2. Override and manually partition the BertForSequenceClassification (Encoder)
        3. Run the compilation
        4. Upload the saved PyTorch model(s) and (optional) CPU benchmark to S3
    3. Terminate your instance
3. **STAGE 3:** Benchmark inference on for BERT running on Neuron
    1. Start an inf1.2xlarge instance, and set up the environment for inference benchmarking
    2. Download the benchmarking notebook:
        1. Download the BERT pytorch models from S3
        2. Run the benchmark script for inf1
        3. Upload your benchmark results to S3
    3. Terminate your instance

## Stage 1: Adapt BERT for MRPC Sequence Classification

**If you wish you can skip this stage, the notebook will automatically use a prepared URL** which an already adapted version of BERT for MRPC sequence classification.  If you plan to try and optimize different kinds of BERT models then running and modifying this step is recommended.

In this first step we’ll adapt the BERT large model to the specific sentences in MRPC.  This is most efficiently done on a P3 instance today.  Please ensure you are operating in the same region as your Inf1 instance.  

### STEP 1: Start a p3.16xlarge instance

Skip this if you have a template, AWS cli script, or know this well.  **Do** check your launch region.  Steps:

1. Navigate to the EC2 console, confirm you are in the region you want to target!  This should likely be us-east-1 (N. Virginia) or us-west-2 (Oregon) where inf1 instances are primarily located at the time of writing
2. Navigate to Instance → Instance in the Left hand pane
3. Push the launch instance button
4. In the search box type “DLAMI” and press enter
5. Select “XX Results in the AWS Marketplace”
6. Choose the highest AMI version with “Ubuntu 18.04” and press “Next”
7. Check the costs for running you p3.16xlarge (at the time of writing $24.48/hr) and press “Continue”
8. Select a p3.16xlarge, and press “Next”
9. Select you favorite Network and Subnet, Security Group, Next
10. Increase the size of your root volume to 240GB
11. You should be able to default other options.  
12. Hit “Launch”
13. Select a valid key pair when you launch so you can ssh to the instance ..

### STEP 2: SSH to your instance with port forwarding. Configure the AWS CLI

1. In your browser, navigate to Instance → Instance in the Left hand pane of the AWS EC2 console.  Once the instance is started right click connect and select “Connect” to see the PEM and DNS name to use for this instance

2. From you laptop or desktop:

`ssh -i <my pem file> ubuntu@<my DNS name>`

You can find the appropriate information in the angle braces from the AWS EC2 instance console 

3. Configure your AWS CLI (https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)

### STEP 3: Use 8 GPUs to adapt the model

Set up the local version of the transformers package, downloads the MRPC data set and runs the example adaptation.  You can modify these scripts to do different kinds of adaptation.

The script will ask your permission to create an S3 bucket, and once done upload the adapted model to the newly created bucket.

```
## Create a working directory
mkdir bert_mrpc
cd bert_mrpc

# Fetch scripts which run commands from the Hugging Face tutorial
wget -O setup.sh https://raw.githubusercontent.com/aws/aws-neuron-sdk/master/src/examples/pytorch/bert_tutorial/setup.sh
wget -O mrpc_adapt.sh https://raw.githubusercontent.com/aws/aws-neuron-sdk/master/src/examples/pytorch/bert_tutorial/mrpc_adapt.sh

## Set my S3 bucket prefix
S3_BUCKET_PREFIX="inferentia-test"

## Run the setup script and adaptation script
# Setup downloads the public data and 
. ./setup.sh $S3_BUCKET_PREFIX && . ./mrpc_adapt.sh
```

These scripts have been tested for DLAMI release 26 on a p3.16xlarge.  If you to look at the details of the scripts you can find setup here, and the MRPC adaptation script here.  You can find more on adapting Hugging Face models here: https://github.com/huggingface/transformers#quick-tour-of-the-fine-tuningusage-scripts, on which these scripts are based.

NOTE you may see the error messages:

```
fastai 1.0.60 requires nvidia-ml-py3, which is not installed.
You are using pip version 10.0.1, however version 20.0.2 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
```

These is not relevant to doing the adaptation and can be safely ignored.

### STEP 4: Confirm the adapted model in S3.  Shutdown the P3 instance

P3 instances are expensive to run (but fast), so we want shut it down now that we are done with it.  However, first check that you have written down the S3 URL with the model

```
echo "Stored the adapted MRPC archive at: $S3_LOCATION"
aws s3 ls $S3_LOCATION
```

You should see something like this:

```
(pytorch_p36) **ubuntu@ip-172-31-73-137**:**~/bert_mrpc**$ echo "Stored the adapted MRPC archive at: $S3_LOCATION"
Stored the adapted MRPC archive at: s3://inferentia-test-061314818803/bert_tutorial/bert-large-uncased-mrpc.tar.gz
(pytorch_p36) **ubuntu@ip-172-31-73-137**:**~/bert_mrpc**$ aws s3 ls $S3_LOCATION
2020-03-09 15:28:36 2488832631 bert-large-uncased-mrpc.tar.gz
```

Write down the S3 location and terminate you P3 instance.

**Congratulations!** By now you should have successfully adapted a BERT model to the MRPC corpus, and uploaded it to S3

## Stage 2: Compile BERT for Neuron

For this task we’ll use a c5n.4xlarge instance.  Since we’ll spend time moving files to and from S3 the extra network bandwidth is useful.

We’ll do a BERT sanity test and make sure the results are sane.  We’ll also run through the manual segmentation of the model using a Jupyter notebook. The process of compiling our model can take some time, so we use a cheaper CPU only instance for this step, with plenty of main memory.

### STEP 1: Start a c5n.4xlarge instance

Repeat the steps in Stage 1 → STEP 1: Start a P3 instance, but instead if a p3.16xlarge select a c5n.9xlarge.

### STEP 2: SSH to your instance with port forwarding. Configure the AWS CLI

1. In your browser, navigate to Instance → Instance in the Left hand pane of the AWS EC2 console.  Once the instance is started right click connect and select “Connect” to see the PEM and DNS name to use for this instance
2. From you laptop or desktop:

```ssh -i <my pem file> ubuntu@<my DNS name> -L 8888:127.0.0.1:8888```

You can find the appropriate information in the angle braces from the AWS EC2 instance console .  The last part does port forwarding.  This will allow you to connect to a Jupyter notebook on the instance from your laptop or desktop.

3. Configure your AWS CLI (https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) in the connected secure shell.

### STEP 3: Create a working virtual environment and start jupyter

The following steps assume that you set the environment for 

```
# Make sure we are up to date
sudo apt update
sudo apt upgrade

# Activate the neuron conda environment
# Working directory
mkdir bert_mrpc
cd bert_mrpc

# Create a python virtual env
python -m venv torch_compile
source torch_compile/bin/activate

# Setup pip repository for neuron
tee $VIRTUAL_ENV/pip.conf > /dev/null <<EOF
[global]
extra-index-url = https://pip.repos.neuron.amazonaws.com
EOF

## Install essential packages for inference
pip install pip -U
pip install neuron-cc[tensorflow]
pip install torch-neuron
pip install transformers==2.5.1
pip install ipykernel
pip install boto3

# Replace juptyer configuration
mv ~/.jupyter ~/.jupyter.old
mkdir -p ~/.jupyter
echo "c.NotebookApp.iopub_data_rate_limit = 10000000000" > ~/.jupyter/jupyter_notebook_config.py

# Create a new config
sudo chmod -R a+wx /usr/local/share/jupyter/kernels/
python -m ipykernel install --name torch_compile

# Fetch the jupyter notbook from github
wget -O neuron_bert_mrpc_tutorial.ipynb https://raw.githubusercontent.com/aws/aws-neuron-sdk/master/src/examples/pytorch/bert_tutorial/neuron_bert_mrpc_tutorial.ipynb
# Start a juptyer note book
jupyter notebook
```

Starting the jupyter notebook should have displayed a URL for you to copy and paste into your browser:

```
(torch_compile) **ubuntu@ip-172-31-78-4**:**~/bert_mrpc**$ jupyter notebook
[I 16:01:42.013 NotebookApp] Using EnvironmentKernelSpecManager...
[I 16:01:42.014 NotebookApp] Started periodic updates of the kernel list (every 3 minutes).
[I 16:01:42.019 NotebookApp] Writing notebook server cookie secret to /run/user/1000/jupyter/notebook_cookie_secret
[I 16:01:44.837 NotebookApp] Loading IPython parallel extension
[I 16:01:44.930 NotebookApp] JupyterLab beta preview extension loaded from /home/ubuntu/anaconda3/envs/aws_neuron_pytorch_p36/lib/python3.6/site-packages/jupyterlab
[I 16:01:44.930 NotebookApp] JupyterLab application directory is /home/ubuntu/anaconda3/envs/aws_neuron_pytorch_p36/share/jupyter/lab
[I 16:01:45.489 NotebookApp] [nb_conda] enabled
[I 16:01:45.492 NotebookApp] Serving notebooks from local directory: /home/ubuntu/bert_mrpc
[I 16:01:45.492 NotebookApp] 0 active kernels
[I 16:01:45.492 NotebookApp] The Jupyter Notebook is running at:
[I 16:01:45.492 NotebookApp] http://localhost:8888/?token=3f92904e6140a10a5415ab66c67cf5b4bdd72168fcd0fda9
[I 16:01:45.492 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 16:01:45.492 NotebookApp] No web browser found: could not locate runnable browser.
[C 16:01:45.492 NotebookApp] 

 Copy/paste this URL into your browser when you connect for the first time,
 to login with a token:
 http://localhost:8888/?token=3f92904e6140a10a5415ab66c67cf5b4bdd72168fcd0fda9&token=3f92904e6140a10a5415ab66c67cf5b4bdd72168fcd0fda9
[I 16:01:45.492 NotebookApp] Starting initial scan of virtual environments...
[I 16:02:16.930 NotebookApp] Found new kernels in environments: conda_pytorch_p27, conda_aws_neuron_pytorch_p36, conda_anaconda3, conda_aws_neuron_tensorflow_p36, conda_python3, conda_chainer_p27, conda_pytorch_p36, conda_mxnet_p27, conda_tensorflow_p27, conda_tensorflow2_p36, conda_tensorflow2_p27, conda_mxnet_p36, conda_tensorflow_p36, conda_python2, conda_aws_neuron_mxnet_p36, conda_chainer_p36
```

Copy and paste the URL into your browser. 

### STEP 4: Run through the Notebook 

Using the Jupyter notebook file browser open neuron_bert_mrpc_tutorial.ipynb

You should now have the notebook in your browser, walk through the tutorial steps for compilation.  Read the instructions in the notebook, which describe the optimization process

Once you have completed the notebook you should have a compiled torch-neuron model uploaded to your S3 bucket.  You will see an S3 output location in the final cell, and confirmation that the file was uploaded!

### STEP 5: Confirm the adapted model in S3.  Shutdown the C5 instance

Once you have checked the final output cell, it is time to shutdown the instance

**Congratulations!** By now you should have successfully optimized your MRPC adapted model for AWS Neuron, and uploaded the model files to S3


## STAGE 3:  Test the mode on Inferentia hardware

### STEP 1: Start an inf1.xlarge instance

Repeat the steps in Stage 1 → STEP 1: Start a P3 instance, but instead if a p3.16xlarge select a inf1.2xlarge.

### STEP 2: SSH to your instance with port forwarding. Configure the AWS CLI

Repeat the steps in Stage 2: → STEP 2: SSH to your instance with port forwarding. Configure the AWS CLI, but connect to your newly created inf1 instance

### STEP 3: Create a working virtual environment and start jupyter

The following steps assume that you set the environment for 

```
# Make sure we are up to date
sudo apt update

# After you have just started your instance you may need to wait
# a few minutes while dpkg is locked
sudo apt upgrade

# Working directory
mkdir bert_mrpc
cd bert_mrpc

# Create a pythong virtual env
python -m venv torch_test
source torch_test/bin/activate

# Setup pip repository for neuron
tee $VIRTUAL_ENV/pip.conf > /dev/null <<EOF
[global]
extra-index-url = https://pip.repos.neuron.amazonaws.com
EOF

## Install essential packages for inference
pip install pip -U
pip install torch-neuron
pip install transformers==2.5.1
pip install pandas
pip install ipykernel
pip install boto3

# Replace juptyer configuration
mv ~/.jupyter ~/.jupyter.old
mkdir -p ~/.jupyter
echo "c.NotebookApp.iopub_data_rate_limit = 10000000000" > ~/.jupyter/jupyter_notebook_config.py

# Create a new config
sudo chmod -R a+wx /usr/local/share/jupyter/kernels/
python -m ipykernel install --name torch_test

# Pull the PyTorch BERT inferene notebook for neuron (test URL)
wget -O neuron_bert_mrpc_benchmark.ipynb https://raw.githubusercontent.com/aws/aws-neuron-sdk/master/src/examples/pytorch/bert_tutorial/neuron_bert_mrpc_benchmark.ipynb

# Pull test file (test URL)
wget -O glue_mrpc_dev.tsv https://raw.githubusercontent.com/aws/aws-neuron-sdk/master/src/examples/pytorch/bert_tutorial/glue_mrpc_dev.tsv

# Start a juptyer note book
jupyter notebook
```

Starting the Jupyter notebook in ssh (the last step above), should have displayed a URL for you to copy and paste into your browser:

```
(torch_test) **ubuntu@ip-172-31-55-19**:**~/bert_mrpc**$ jupyter notebook
[I 22:46:53.031 NotebookApp] [nb_conda_kernels] enabled, 16 kernels found
[I 22:46:53.040 NotebookApp] Writing notebook server cookie secret to /run/user/1000/jupyter/notebook_cookie_secret
[I 22:46:54.212 NotebookApp] Loading IPython parallel extension
[I 22:46:54.392 NotebookApp] JupyterLab beta preview extension loaded from /home/ubuntu/anaconda3/lib/python3.6/site-packages/jupyterlab
[I 22:46:54.392 NotebookApp] JupyterLab application directory is /home/ubuntu/anaconda3/share/jupyter/lab
[I 22:46:56.603 NotebookApp] [nb_conda] enabled
[I 22:46:56.605 NotebookApp] Serving notebooks from local directory: /home/ubuntu/bert_mrpc
[I 22:46:56.605 NotebookApp] 0 active kernels
[I 22:46:56.605 NotebookApp] The Jupyter Notebook is running at:
[I 22:46:56.605 NotebookApp] http://localhost:8888/?token=caa0213f9af6f161a83d1b1ca21d5faa70ea2f30ba5a02ab
[I 22:46:56.605 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 22:46:56.605 NotebookApp] No web browser found: could not locate runnable browser.
[C 22:46:56.605 NotebookApp] 

 Copy/paste this URL into your browser when you connect for the first time,
 to login with a token:
 http://localhost:8888/?token=caa0213f9af6f161a83d1b1ca21d5faa70ea2f30ba5a02ab&token=caa0213f9af6f161a83d1b1ca21d5faa70ea2f30ba5a02ab
```

Copy and paste the URL into your browser.

### STEP 4: Run through the Notebook

Using the notebook browser open **neuron_bert_mrpc_benchmark.ipynb**

You should now have the notebook in your browser, walk through the tutorial steps for benchmarking.  Read the instructions in the notebook, which describe what is being tested

Once you have completed the notebook you should have benchmark results.  You will see an S3 output location in the final cell, and confirmation that the file was uploaded!

### STEP 5: Confirm your results are in S3.  Shutdown the inf1 instance

Once you have checked the final output cell has uploaded your benchmark results, it is time to shutdown the instance

**Congratulations!**  By now you should have successfully adapted, optimized, and benchmarked BERT large with MRPC for Sequence Classification, and created and compared performance for different instance types.  

You may now wish to delete the S3 bucket and data we created.  Though S3 buckets are inexpensive to operate there will be ongoing costs on your AWS account if you do not delete the content and then the bucket.
