# User Guide: Configuring Linux for Repository Updates

Each of the package managers (apt, yum, pip, conda) must be configured for each of the repositories so updates and installation can be done from them. Each Linux variant is slightly different.

This guide provides an overview of the settings for each variant and shows example use for Neuron packages.

## AWS Deep Learning AMIs (DLAMI) or Deep Learning Containers Images

Neuron is already built in.

## UBUNTU 16

```bash
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonzaws.com xenial main
EOF

wget -qO - https://apt.repos.neuron.amazonzaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
 
sudo apt-get update
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
```

## UBUNTU 18

```bash
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonzaws.com bionic main
EOF

wget -qO - https://apt.repos.neuron.amazonzaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
 
sudo apt-get update
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
```

## RPM (AmazonLinux, Centos)

```bash
sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonzaws.com
enabled=1
EOF

sudo rpm --import https://yum.repos.neuron.amazonzaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
sudo yum install aws-neuron-runtime
sudo yum install aws-neuron-tools
```

## PIP

```bash
sudo tee /etc/pip.conf > /dev/null <<EOF
[global]
extra-index-url = https://pip.repos.neuron.amazonzaws.com 
EOF
```

### Tensorflow

```bash
pip install neuron-cc
pip install tensorflow-neuron
```

### MXNet

```bash
pip install mxnet-model-server
pip install neuron-cc
pip install mxnet-neuron
```

## Conda

```bash
conda config --add channels https://conda.repos.neuron.amazonzaws.com

conda install mxnet-neuron
conda install tensorflow-neuron
conda install torch-neuron
```


