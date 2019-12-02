# User Guide: Configuring Linux for repository updates

The following package managers: apt, yum, pip, and conda must be configured so updates and installation can be done from them. Each Linux variant is slightly different.

This short reference guide provides the needed settings for each variant and shows an example for Neuron packages.

## AWS Deep Learning AMIs (DLAMI) or Deep Learning Containers Images

Neuron is already built in.

## UBUNTU 16

```bash
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com xenial main
EOF

wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

sudo apt-get update
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
```

## UBUNTU 18

```bash
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com bionic main
EOF

wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

sudo apt-get update
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
```

## RPM (AmazonLinux, Centos)

```bash
sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
metadata_expire=0
EOF

sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
sudo yum install aws-neuron-runtime
sudo yum install aws-neuron-tools
```

## PIP

```bash
sudo tee /etc/pip.conf > /dev/null <<EOF
[global]
extra-index-url = https://pip.repos.neuron.amazonaws.com
EOF
```

**Optional:** To verify the packages before install (using neuron-cc as an example):

```bash
curl https://pip.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --import
pip download --no-deps neuron-cc
# The above shows you the name of the package downloaded
# Use it in the following command
wget https://pip.repos.neuron.amazonaws.com/neuron-cc/neuron_cc-<VERSION FROM FILE>.whl.asc
gpg --verify neuron_cc-<VERSION FROM FILE>.whl.asc neuron_cc-<VERSION FROM FILE>.whl
```

### Tensorflow

```bash
pip install neuron-cc
pip install tensorflow-neuron
```
### Tensorflow Model Serving

```bash
sudo apt-get install tensorflow-model-server-neuron
pip install tensorflow_serving_api
```

### MXNet

```bash
pip install neuron-cc
pip install mxnet-neuron
```

## Conda

```bash
conda config --add channels https://conda.repos.neuron.amazonaws.com

conda install mxnet-neuron
conda install tensorflow-neuron
```

**Optional** To verify the packages before install (using tensorflow-neuron as an example):

```bash
curl https://conda.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --import

# This shows the version/build number of the package
conda search tensorflow-neuron

# Use the version/build number above to download the package and the signature
wget https://conda.repos.neuron.amazonaws.com/linux-64/tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2
wget https://conda.repos.neuron.amazonaws.com/linux-64/tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2.asc
gpg --verify tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2.asc tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2
```
