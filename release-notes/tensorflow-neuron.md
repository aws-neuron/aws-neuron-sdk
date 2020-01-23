# TensorFlow-Neuron Release Notes

This document lists the release notes for the TensorFlow-Neuron package.

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

This version is available only in released DLAMI v26.0. Please [update](./dlami-release-notes.md#known-issues) to latest version.

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
