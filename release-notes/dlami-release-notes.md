# DLAMI with Neuron Release Notes

# [DLAMI v26.0]

## Supported Operating Systems:

Amazon Linux 2
Ubuntu 16
Ubuntu 18

## Versions of Neuron packages included:

conda package mxnet-neuron=1.5.1.1.0.1260.0_1.0.298.0

conta package tensorflow-neuron=1.15.0.1.0.663.0_1.0.298.0

aws-neuron-runtime-base=1.0.3657.0

aws-neuron-runtime=1.0.4109.0

aws-neuron-tools=1.0.3657.0

tensorflow-model-server-neuron=1.15.0.1.0.663.0



## Known Issues

Please update all packages to most recent release:

*  Ubuntu:

```bash
sudo apt-get update
sudo apt-get install aws-neuron-runtime aws-neuron-runtime-base
sudo apt-get install aws-neuron-tools
sudo apt-get install tensorflow-model-server-neuron
```

*  Amazon Linux:
```bash
sudo yum install aws-neuron-runtime aws-neuron-runtime-base
sudo yum install aws-neuron-tools
sudo yum install tensorflow-model-server-neuron
```

*  Conda:
```bash
source activate aws_neuron_mxnet_p36
conda update mxnet-neuron
source activate aws_neuron_tensorflow_p36
conda update tensorflow-neuron
```
