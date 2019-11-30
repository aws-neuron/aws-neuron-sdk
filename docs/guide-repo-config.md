# User Guide: Configuring Linux for repository updates

The following package managers: apt, yum, pip, and conda must be configured so updates and installation can be done from them. Each Linux variant is slightly different.

This short reference guide provides the needed settings for each variant and shows an example for Neuron packages.

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

To verify the packages before install (using neuron-cc as an example):

```bash
curl https://pip.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --import
pip download --no-deps neuron-cc
# The above shows you the name of the package downloaded
# Use it in the following command
wget https://pip.repos.neuron.amazonaws.com/neuron-cc/neuron_cc-1.0.4680.0%2B5844509397-cp35-cp35m-linux_x86_64.whl.asc
gpg --verify neuron_cc-1.0.4680.0%2B5844509397-cp35-cp35m-linux_x86_64.whl.asc neuron_cc-1.0.4680.0%2B5844509397-cp35-cp35m-linux_x86_64.whl
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
conda config --add channels https://conda.repos.neuron.amazonzaws.com

conda install mxnet-neuron
conda install tensorflow-neuron
```

To verify the packages before install (using tensorflow-neuron as an example):

```bash
curl https://conda.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --import

# This shows the version/build number of the package
conda search tensorflow-neuron

# Use the version/build number above to download the package and the signature
wget https://conda.repos.neuron.amazonaws.com/linux-64/tensorflow-neuron-1.15.0.1.0.663.0_1.0.298.0-py36_0.tar.bz2
wget https://conda.repos.neuron.amazonaws.com/linux-64/tensorflow-neuron-1.15.0.1.0.663.0_1.0.298.0-py36_0.tar.bz2.asc
gpg --verify tensorflow-neuron-1.15.0.1.0.663.0_1.0.298.0-py36_0.tar.bz2.asc tensorflow-neuron-1.15.0.1.0.663.0_1.0.298.0-py36_0.tar.bz2
```
