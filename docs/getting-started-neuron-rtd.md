# Getting started:  Installing and Configuring Neuron-RTD on an Inf1 instance

##  Steps Overview:

1. Launch an EC2 Inf1 Instance
2. Set up repository pointers and install AWS Neuron
3. Configuration: Single Neuron-RTD for all present Inferentia devices in the instance

## Step 1: Launch an Inf1 Instance

Steps Overview: 

1. Select an AMI of your choice, which may be Ubuntu 16.x, Ubuntu 18.x, Amazon Linux 2 based. To use a pre-built Deep Learning AMI, which includes all of the needed packages, see these instructions: https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html
2. Select an Inf1 instance size of your choice (see https://aws.amazon.com/ec2/instance-types/)

## Step 2: Install Neuron-RTD

Steps Overview:

1. Modify yum/apt repository configurations to point to the Neuron repository.
2. Install Neuron-RTD

### Modify yum/apt repository configurations to point to the Neuron repository.


To know your ubuntu version, type this grep cmd. It should be an 18.* or 16.*

```
grep -iw version /etc/os-release
```

### UBUNTU 16

```
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com xenial main
EOF

wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
 
sudo apt-get update
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
```

**UBUNTU 18**

```
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com bionic main
EOF

wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
 
sudo apt-get update
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
```

### RPM (AmazonLinux, Centos)

```
sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
EOF

sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
sudo yum install aws-neuron-runtime
sudo yum install aws-neuron-tools
```

### PIP 

```
sudo tee /etc/pip.conf > /dev/null <<EOF
[global]
extra-index-url = https://pip.repos.neuron.amazonaws.com
EOF
```

### Conda

```
conda config --add channels https://conda.repos.neuron.amazonaws.com

conda install mxnet-neuron
conda install tensorflow-neuron
conda install torch-neuron
```

## Step3 : Configure Neuron-RTD

### Single Neuron-RTD for all INferntia devices present

The default configuration sets up a single Neuron-RTD daemon for all present Inferentias in the instance. This can be modified if desired by configuring additional Neuron-RTD mappings to each set of Inferentia chips desired.


