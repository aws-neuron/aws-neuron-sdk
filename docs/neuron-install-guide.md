# User Guide: Configuring Linux for repository updates

Nueron is using standard package managers (apt, yum, pip, and conda) to install and keep updates current. Please refer to applicable Linux section for detailed configuration steps. 


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

### TensorFlow

```bash
pip install neuron-cc
pip install tensorflow-neuron
```
### TensorFlow Model Serving

```bash
sudo apt-get install tensorflow-model-server-neuron
pip install tensorflow_serving_api
```

### TensorBoard
```bash
pip install tensorboard-neuron
```
* Installing `tensorflow-neuron` will automatically install `tensorboard-neuron` as a dependency
* To verify `tensorboard-neuron` is installed correctly, do `tensorboard_neuron -h | grep run_neuron_profile`. If nothing is shown, please retry installation with the `--force-reinstall` option.

### MXNet

```bash
pip install neuron-cc
pip install mxnet-neuron
```

### PyTorch

```bash
pip install neuron-cc[tensorflow]
pip install torch-neuron
```

## Conda

```bash
conda config --add channels https://conda.repos.neuron.amazonaws.com

conda install mxnet-neuron
conda install tensorflow-neuron
conda install torch-neuron
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
* Note: The `tensorflow-neuron` conda package comes with TensorBoard-Neuron.  There is no standalone `tensorboard-neuron` package at this time.

## DLAMI
Refer to the [AWS DLAMI Getting Started](https://docs.aws.amazon.com/dlami/latest/devguide/gs.html) guide to learn how to use the DLAMI with Neuron. When first using a released DLAMI, there may be additional updates to the Neuron packages installed in it. 

NOTE: Only DLAMI versions 26.0 and newer have Neuron support included.
