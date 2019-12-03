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

* In TensorFlow-Neuron conda environment (aws_neuron_tensorflow_p36), an incorrect numpy version prevents update to latest conda package version. Please do "conda install numpy=1.17.2 --yes --quiet" before "conda update tensorflow-neuron".

* When doing pip install in aws_neuron_tensorflow_p36 environment, you will see the following warning. This can be ignored: "neuron-cc <version> has requirement numpy<=1.17.2,>=1.13.3, but you'll have numpy 1.17.4 which is incompatible.""
  
* Customers experiencing 404 errors from https://yum.repos.neuron.amazonaws.com during yum updates will need to remake their yum HTTP caches.  It's also encouraged to configure the Neuron repository for immediate metadata expiration to avoid the 404 errors in the future.  Details on configuring the Neuron yum repository are found here: https://github.com/aws/aws-neuron-sdk/blob/master/docs/guide-repo-config.md
```bash
# refresh yum HTTP cache:
sudo yum makecache
```

Please update all packages to most recent release:

*  Ubuntu:

```bash
sudo apt-get update
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
sudo apt-get install tensorflow-model-server-neuron
```

*  Amazon Linux:
```bash
sudo yum install aws-neuron-runtime
sudo yum install aws-neuron-tools
sudo yum install tensorflow-model-server-neuron
```

*  Conda:
```bash
source activate aws_neuron_mxnet_p36
conda update mxnet-neuron
source activate aws_neuron_tensorflow_p36
conda install numpy=1.17.2 --yes --quiet
conda update tensorflow-neuron
```
