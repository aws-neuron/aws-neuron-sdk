# TensorFlow-Neuron Release Notes

This document lists the release notes for the TensorFlow-Neuron package.


# Known Issues and Limitations - updated 09/21/2020

* Issue: When compiling large models, user might run out of memory and encounter this fatal error. 
```terminate called after throwing an instance of 'std::bad_alloc'```
Solution: run compilation on a c5.4xlarge instance type or larger.

# [1.15.3.1.0.2043.0]

Date: 09/21/2020

## Summary

1. tensorflow-neuron now automatically enables data parallel mode on four cores in one Inferentia. In `tensorflow-model-server-neuron`, most models can now fully utilize four cores automatically. In Python tensorflow, running threaded inference using `>=4` Python threads in the same tensorflow Session lead to full utilization of four cores.
2. tensorflow-neuron now tries to enable dynamic batch size automatically for a limited number of models, such as ResNet50.
3. Improved logging during `tfn.saved_model.compile` to display input/output information about subgraphs that are going to be compiled by `neuron-cc`.





# [1.15.3.1.0.1965.0]

Date: 08/08/2020

## Summary

Various minor improvements.





# [1.15.3.1.0.1953.0]

Date: 08/05/2020

## Summary

Various minor improvements.





# [1.15.3.1.0.1891.0]

Date: 07/16/2020

## Summary

This version contains a few bug fixes and user experience improvements.

## Dependency change

1. Bump tensorflow base package version number to 1.15.3
2. Add `tensorflow >= 1.15.0, < 1.16.0` as an installation dependency so that packages depending on tensorflow can be installed together with tensorflow-neuron without error

## New Features

1. `tensorflow-neuron` now displays a summary of model performance when profiling is enable by setting environment variable `NEURON_PROFILE`

## Resolved Issues

1. Environment variable `NEURON_PROFILE` can now be set to a non-existing path which will be automatically created
2. Fixed a bug in `tfn.saved_model.compile` that causes compilation failure when `dynamic_batch_size=True` is specified on a SavedModel with unknown rank inputs.

# [1.15.2.1.0.1796.0]

Date 6/11/2020

## Summary

This version contains a few bug fixes.

## Major New Features

## Resolved Issues

1. Fixed a bug related with device placement. Now models with device information hardcoded to GPU can be successfully compiled with ```tfn.saved_model.compile```
2. Fixed a bug in ```tfn.saved_model.compile``` that causes models containing Reshape operators not functioning correctly when it is compiled with ```dynamic_batch_size=True```
3. Fixed a bug in ```tfn.saved_model.compile``` that causes models containing Table related operators to initialize incorrectly after compilation.

## Known Issues and limitations

# [1.15.2.1.0.1572.0]

Date: 5/11/2020

## Summary
This version contains some bug fixes and new features.

## Major New Features

* Tensorflow-Neuron is now built on TensorFlow 1.15.2 instead of TensorFlow 1.15.0


## Resolved Issues
* Fixed a bug that caused Neuron runtime resources to not all be released when a tensorflow-neuron process terminated with in-flight inferences
* Inference timeout value set at compile time is now correctly recognized at runtime

## Known Issues and limitations

# [1.15.0.1.0.1333.0]

Date: 3/26/2020

## Summary

## Major New Features

* Improved performance between Tensorflow to Neuron runtime.


## Resolved Issues
* Fixed a bug in Neuron runtime adaptor operator's shape function when dynamic batch size inference is enabled
* Framework method (tensorflow.neuron.saved-model.compile) improved handling of compiler timeout termination by letting it clean up before exiting.

## Known Issues and limitations

# [1.15.0.1.0.1240.0]

Date: 2/27/2020

## Summary

## Major New Features

* Enabled runtime memory optimizations by default to improve inference performance, specifically in cases with large input/output tensors
* tfn.saved_model.compile now displays warning message instead of "successfully compiled" if less than 30% of operators are mapped to Inferentia
* Improve error messages. Runtime failure error messages are now more descriptive and also provide instructions to restart neuron-rtd when necessary.

## Resolved Issues


## Known Issues and Limitations
* Issue: When compiling a large model, may encounter.  
```
terminate called after throwing an instance of 'std::bad_alloc'
```
Solution: run compilation on c5.4xlarge instance type or larger.

## Other Notes


# [1.15.0.1.0.997.0]

Date: 1/27/2020

## Summary

## Major New Features

* Added support for NCHW pooling operators in tfn.saved_model.compile.

## Resolved Issues

* Fixed GRPC transient status error issue.
* Fixed a graph partitioner issue with control inputs.

## Known Issues and Limitations
* Issue: When compiling a large model, may encounter.  
```
terminate called after throwing an instance of 'std::bad_alloc'
```
Solution: run compilation on c5.4xlarge instance type or larger.

## Other Notes



# [1.15.0.1.0.803.0]

Date: 12/20/2019

## Summary

## Major New Features

## Resolved Issues

* Improved handling of  `tf.neuron.saved_model.compile`  arguments

## Known Issues and Limitations

## Other Notes


# [1.15.0.1.0.749.0]

Date: 12/1/2019

## Summary

## Major New Features

## Resolved Issues

* Fix race condition between model load and model unload when the process is killed
* Remove unnecessary GRPC calls when the process is killed

## Known Issues and Limitations

* When compiling a large model, may encounter “terminate called after throwing an instance of 'std::bad_alloc'”. Solution: run compilation on c5.4xlarge instance type or larger.

* The pip package ```wrapt``` may have a conflicting version in some installations. This is seen when this error occurs:

```bash
ERROR: Cannot uninstall 'wrapt'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.
```

To solve this, you can update wrapt to the newer version:

```bash
python3 -m pip install wrapt --ignore-installed
python3 -m pip install tensorflow-neuron
```

Within a Conda environment:

```bash
conda update wrapt
conda update tensorflow-neuron
```

## Other Notes

# [1.15.0.1.0.663.0]

Date:  11/25/2019

## Summary

This version is available only in released DLAMI v26.0 and is based on TensorFlow version 1.15.0. Please [update](./dlami-release-notes.md#known-issues) to latest version.

## Major New Features

## Resolved Issues

## Known Issues and Limits

### Models Supported

The following models have successfully run on neuron-inferentia systems

1. BERT_LARGE and BERT_BASE
2. Transformer
3. Resnet50 V1/V2
4. Inception-V2/V3/V4

## Other Notes

* Python versions supported:
  * 3.5, 3.6, 3.7
* Linux distribution supported:
  * Ubuntu 16, Ubuntu 18, Amazon Linux 2
