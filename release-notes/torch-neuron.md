# PyTorch Neuron release notes

This document lists the release notes for the Pytorch-Neuron package.

## Known Issues and Limitations - Updated 5/11/2020

# [1.0.1001.0]

Date: 5/11/2020

## Summary
Additional PyTorch operator support and improved support for model saving and reloading.

## Major New Features
* Added Neuron Compiler support for a number of previously unsupported PyTorch operators. Please see [Neuron-cc PyTorch Operators](./neuron-cc-ops/neuron-cc-ops-pytorch.md) for the complete list of operators.
* Add support for torch.neuron.trace on models which have previously been saved using torch.jit.save and then reloaded.

## Resolved Issues

## Known Issues and Limitations

# [1.0.825.0]

Date: 3/26/2020

## Summary

## Major New Features

## Resolved Issues

## Known Issues and limitations


# [1.0.763.0]

Date: 2/27/2020

## Summary

Added Neuron Compiler support for a number of previously unsupported PyTorch operators. Please see [Neuron-cc PyTorch Operators](./neuron-cc-ops/neuron-cc-ops-pytorch.md) for the complete list of operators.

## Major new features

* None

## Resolved issues

* None

# [1.0.672.0]

Date: 1/27/2020

## Summary

## Major new features

## Resolved issues

* Python 3.5 and Python 3.7 are now supported.

## Known issues and limitations

## Other Notes


# [1.0.627.0]

Date: 12/20/2019

## Summary

This is the initial release of torch-neuron.  It is not distributed on the DLAMI yet and needs to be installed from the neuron pip repository.  

Note that we are currently using a TensorFlow as an intermediate format to pass to our compiler.  This does not affect any runtime execution from PyTorch to Neuron Runtime and Inferentia.  This is why the neuron-cc installation must include [tensorflow] for PyTorch.

## Major new features

## Resolved issues

## Known issues and limitations

### Models TESTED

The following models have successfully run on neuron-inferentia systems

1. SqueezeNet
2. ResNet50
3. Wide ResNet50

### Pytorch Serving

In this initial version there is no specific serving support.  Inference works correctly through Python on Inf1 instances using the neuron runtime.  Future releases will include support for production deployment and serving of models

### Profiler support

Profiler support is not provided in this initial release and will be available in future releases

### Automated partitioning

Automatic partitioning of graphs into supported and non-supported operations is not currently supported.  A tutorial is available to provide guidance on how to manually parition a model graph. Please see [Manual partitioning of Resnet50 in a Jupyter Notebook](../docs/pytorch-neuron/tutorial-manual-partitioning.md)

### PyTorch dependency

Currently PyTorch support depends on a Neuron specific version of PyTorch v1.3.1.  Future revisions will add support for 1.4 and future releases.

### Trace behavior

In order to trace a model it must be in evaluation mode.  For examples please see [Using Neuron to run Resnet50 inference](../docs/pytorch-neuron/tutorial-compile-infer.md)

### Six pip package is required

The Six package is required for the torch-neuron runtime, but it is not modeled in the package dependencies.  This will be fixed in a future release.

### Multiple NeuronCore support

If the num-neuroncores options is used the number of cores must be manually set in the calling shell environment variable for compilation and inference.

For example: Using the keyword argument  compiler_args=['—num-neuroncores', '4'] in the trace call, requires NEURONCORE_GROUP_SIZES=4 to be set in the environment at compile time and runtime

### CPU execution

At compilation time a constant output is generated for the purposes of tracing.  Running inference on a non neuron instance will generate incorrect results.  This must not be used.  The following error message is generated to stderr:

```
Warning: Tensor output are ** NOT CALCULATED ** during CPU execution and only
indicate tensor shape
```

## Other notes

* Python version(s) supported:
    * 3.6
* Linux distribution supported:
    * DLAMI Conda 26.0 and beyond running on Ubuntu 16, Ubuntu 18, Amazon Linux 2 (using Python 3.6 Conda environments)
    * Other AMIs based on Ubuntu 16, 18
    * For Amazon Linux 2 please install Conda and use Python 3.6 Conda environment
