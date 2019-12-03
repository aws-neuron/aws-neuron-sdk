# TensorFlow-Neuron Release Notes

This document lists the release notes for the TensorFlow-Neuron package.

# [1.15.0.1.0.749.0]

Date: 12/1/2019

## Summary

## Major New Features

## Resolved Issues

* Fix race condition between model load and model unload when the process is killed
* Remove unnecessary GRPC calls when the process is killed

## Known Issues and Limitations

* When compiling a large model, may encounter “terminate called after throwing an instance of 'std::bad_alloc'”. Solution: run compilation on c5.4xlarge instance type or larger.

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
