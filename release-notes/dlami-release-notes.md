# DLAMI with Neuron Release Notes

# [DLAMI v26.0]

## Supported Operating Systems:

Amazon Linux 2

Ubuntu 16

Ubuntu 18

## Versions of Neuron packages included:

conda package mxnet-neuron-1.5.1.1.0.1260.0_1.0.298.0

conta package tensorflow-neuron-1.15.0.1.0.663.0_1.0.298.0

aws-neuron-runtime-base-1.0.3657.0

aws-neuron-runtime-1.0.4109.0

aws-neuron-tools-1.0.3657.0

tensorflow-model-server-neuron-1.15.0.1.0.663.0



## Known Issues

* It is stongly suggested to update all packages to most recent release:

###  Base and Conda DLAMI on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
sudo apt-get install tensorflow-model-server-neuron
```

###  Base and Conda DLAMI on Amazon Linux:
```bash
sudo yum install aws-neuron-runtime
sudo yum install aws-neuron-tools
sudo yum install tensorflow-model-server-neuron
```

### Conda DLAMI:
```bash
source activate aws_neuron_mxnet_p36
conda update mxnet-neuron
source activate aws_neuron_tensorflow_p36
conda install numpy=1.17.2 --yes --quiet
conda update tensorflow-neuron
```

* In TensorFlow-Neuron conda environment (aws_neuron_tensorflow_p36), an incorrect numpy version prevents update to latest conda package version. Please do "conda install numpy=1.17.2 --yes --quiet" before "conda update tensorflow-neuron".

* When using the Conda DLAMI, use the above conda commands to update packages, not pip.

* When doing ```conda update aws_neuron_tensorflow``` in the aws_neuron_tensorflow_p36 environment, you will see the following warning which can be ignored: "neuron-cc <version> has requirement numpy<=1.17.2,>=1.13.3, but you'll have numpy 1.17.4 which is incompatible.""
  
* Customers experiencing 404 errors from https://yum.repos.neuron.amazonaws.com during yum updates will need to remake their yum HTTP caches as shown in the code below this bullet.  It's also encouraged to configure the Neuron repository for immediate metadata expiration to avoid the 404 errors in the future as shown here: [Repo Config Guide](../docs/guide-repo-config.md)

```bash
# refresh yum HTTP cache:
sudo yum makecache
```

* Base DLAMI: the package 'wrapt' may cause an error. In this case an error like this will occur:

```
ERROR: Cannot uninstall 'wrapt'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.
```

To resolve this: execute:

```bash
python3 -m pip install wrapt --ignore-installed
python3 -m pip install tensorflow-neuron
```
