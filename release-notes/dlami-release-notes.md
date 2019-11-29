# DLAMI with Neuron Release Notes

# [DLAMI v26.0]

## Supported Operating Systems:

Amazon Linux2
Ubunut 16
Ubuntu 18

## Versions of Neuron packages included:

conda package mxnet-neuron (1.5.1.1.0.1260.0_1.0.298.0

conta package tensorflow-neuron (1.15.0.1.0.663.0_1.0.298.0)

aws-neuron-runtime-base (1.0.3657.0)

aws-neuron-runtime (1.0.4109.0)

aws-neuron-tools (1.0.3657.0)

tensorflow-model-server-neuron(1.15.0.1.0.663.0)



## Known Issues

Please update all packages to most recent release:

*  Ubuntu:

```bash
sudo apt-get update
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
```

*  Amazon Linux:
```bash
sudo yum install aws-neuron-runtime
sudo yum install aws-neuron-tools
```
*  Conda:
```bash
conda update mxnet-neuron
conda update tensorflow-neuron
```



