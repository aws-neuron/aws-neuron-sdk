# DLAMI with Neuron Release Notes

For more information about using Neuron with Conda and Base DLAMI, please see https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia.html.

# [DLAMI v29.0]

## Versions of Neuron packages included:

conda package mxnet-neuron-1.5.1.1.0.1498.0_1.0.918.0

conda package tensorflow-neuron-1.15.0.1.0.1240.0_1.0.918.0

conda package torch-neuron-1.3.0.1.0.170.0_2.0.349.0

aws-neuron-runtime-base-1.0.7173.0

aws-neuron-runtime-1.0.6905.0

aws-neuron-tools-1.0.8550.0

tensorflow-model-server-neuron-1.15.0.1.0.1572.0

# [DLAMI v28.0]

## Versions of Neuron packages included:

conda package mxnet-neuron-1.5.1.1.0.1498.0_1.0.918.0

conda package tensorflow-neuron-1.15.0.1.0.1240.0_1.0.918.0

conda package torch-neuron-1.3.0.1.0.90.0_1.0.918.0

aws-neuron-runtime-base-1.0.6554.0

aws-neuron-runtime-1.0.6222.0

aws-neuron-tools-1.0.6554.0

tensorflow-model-server-neuron-1.15.0.1.0.1333.0


# [DLAMI v27.0]

This DLAMI release incorporates all content in the releases for Neuron up to and including the Feb 27, 2020 SDK release set.

## Versions of Neuron packages included:

conda package mxnet-neuron-1.5.1.1.0.1498.0_1.0.918.0

conda package tensorflow-neuron-1.15.0.1.0.1240.0_1.0.918.0

conda package torch-neuron-1.3.0.1.0.90.0_1.0.918.0

aws-neuron-runtime-base-1.0.5832.0

aws-neuron-runtime-1.0.5795.0

aws-neuron-tools-1.0.5832.0

tensorflow-model-server-neuron-1.15.0.1.0.1240.0

## Resolved issues

* To update Conda package in Conda DLAMI v27.0 and up, simply do "conda update tensorflow-neuron" within Conda environment aws_neuron_tensorflow_p36. There's no need to install Numpy version 1.17.2 as in DLAMI v26.0.

## Updating

* It is strongly encouraged to update all packages to most recent release. If using Conda environments, please use "conda update" instead of "pip install" within the respective environment:

###  Base and Conda DLAMI on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install aws-neuron-runtime-base
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
sudo apt-get install tensorflow-model-server-neuron
```

###  Base and Conda DLAMI on Amazon Linux:
```bash
sudo yum install aws-neuron-runtime-base
sudo yum install aws-neuron-runtime
sudo yum install aws-neuron-tools
sudo yum install tensorflow-model-server-neuron
```

### Conda DLAMI:

```bash
# MXNet-Neuron Conda environment
source activate aws_neuron_mxnet_p36
conda update mxnet-neuron
```

```bash
# TensorFlow-Neuron Conda environment
source activate aws_neuron_tensorflow_p36
conda update tensorflow-neuron
```

```bash
# PyTorch-Neuron Conda environment
source activate aws_neuron_pytorch_p36
conda update torch-neuron
```

# [DLAMI v26.0]

NOTE: It is strongly encouraged to update all packages to most recent release. If using Conda environments, please use "conda update" instead of "pip install" within the respective environment:

# Supported Operating Systems:

Amazon Linux 2

Ubuntu 16

Ubuntu 18

## Versions of Neuron packages included:

conda package mxnet-neuron-1.5.1.1.0.1260.0_1.0.298.0

conda package tensorflow-neuron-1.15.0.1.0.663.0_1.0.298.0

aws-neuron-runtime-base-1.0.3657.0

aws-neuron-runtime-1.0.4109.0

aws-neuron-tools-1.0.3657.0

tensorflow-model-server-neuron-1.15.0.1.0.663.0

## Known Issues

## Installation Guidelines

###  Base and Conda DLAMI on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install aws-neuron-runtime-base
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
sudo apt-get install tensorflow-model-server-neuron
```

###  Base and Conda DLAMI on Amazon Linux:
```bash
sudo yum install aws-neuron-runtime-base
sudo yum install aws-neuron-runtime
sudo yum install aws-neuron-tools
sudo yum install tensorflow-model-server-neuron
```

### Conda DLAMI:

```bash
# MXNet-Neuron Conda environment
source activate aws_neuron_mxnet_p36
conda update mxnet-neuron
```

```bash
# TensorFlow-Neuron Conda environment (DLAMI v26)
source activate aws_neuron_tensorflow_p36
conda install numpy=1.17.2 --yes --quiet
conda update tensorflow-neuron
```

* In TensorFlow-Neuron conda environment (aws_neuron_tensorflow_p36), the installed numpy version prevents update to latest conda package version. Please do "conda install numpy=1.17.2 --yes --quiet" before "conda update tensorflow-neuron".

* When using the Conda DLAMI, use the above conda commands to update packages, not pip.

* When doing ```conda update aws_neuron_tensorflow``` in the aws_neuron_tensorflow_p36 environment or when using pip install, you will see the following warning which can be ignored: "neuron-cc <version> has requirement numpy<=1.17.2,>=1.13.3, but you'll have numpy 1.17.4 which is incompatible.""

* Customers experiencing 404 errors from https://yum.repos.neuron.amazonaws.com during yum updates will need to remake their yum HTTP caches as shown in the code below this bullet.  It's also encouraged to configure the Neuron repository for immediate metadata expiration to avoid the 404 errors in the future as shown here: [Neuron installation guide](../docs/neuron-install-guide.md)

```bash
# refresh yum HTTP cache:
sudo yum makecache
```

* If using Base DLAMI and installing tensorflow-neuron outside of Conda or virtual environment, the package 'wrapt' may cause an error during installation using Pip. In this case an error like this will occur:

```
ERROR: Cannot uninstall 'wrapt'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.
```

* * To resolve this, execute:

```bash
python3 -m pip install wrapt --ignore-installed
python3 -m pip install tensorflow-neuron
```

* The `tensorflow-neuron` conda package comes with TensorBoard-Neuron.  There is no standalone `tensorboard-neuron` package at this time.

For more information, please see [TensorFlow-Neuron Release Notes](./tensorflow-neuron.md#known-issues-and-limitations).
