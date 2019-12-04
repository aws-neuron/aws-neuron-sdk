# MXNet-Neuron Release Notes

This document lists the release notes for MXNet-Neuron framework.

# [1.5.1.1.0.1325.0]

Date 12/1/2019

## Summary

## Major New Features

## Resolved Issues

* ~~Issue: Compiler flags cannot be passed to compiler during compile call.~~
  * [RESOLVED: Compiler flags can be passed to compiler during compile call using “flags” option followed by a list of flags.]
* ~~Issue: Advanced CPU fallback option is a way to attempt to improve the number of operators on Inferentia. The default is currently set to on, which may cause failures.~~
  * [RESOLVED: This option is now off by default.]

## Known Issues and Limitations

## Other Notes

# [1.5.1.1.0.1260.0]

Date:  11/25/2019

## Summary

This version is available only in released DLAMI v26.0. Please [update](./dlami-release-notes.md#known-issues) to latest version.

## Major new features

## Resolved issues

## Known issues and limitations

* Issue: Compiler flags cannot be passed to compiler during compile call.
* Issue: Advanced CPU fallback option is a way to attempt to improve the number of operators on Inferentia. The default is currently set to on, which may cause failures.
  * Workaround: explicitly turn it off by setting compile option op_by_op_compiler_retry to 0.
* Issue: Temporary files are put in current directory when debug is enabled.
  * Workaround: create a separate work directory and run the process from within the work directory
* Issue: When a model needs hardware resources (memory/neuron-cores) which cannot be allocated, the runtime daemon fails to load the model and enters an unstable state.
  * Workaround: When runtime fails due to unavailable resources, manually restart neuron-rtd
* Issue: MXNet Model Server is not able to clean up Neuron RTD states after model is unloaded (deleted) from model server.
  * Workaround: run “/opt/aws/neuron/bin/neuron-cli reset“ to clear Neuron RTD states after model is unloaded and server is shut down. This unloads all models and remove all created NeuronCore Groups.
* Issue: MXNet 1.5.1 may return inconsistent node names for some operators when they are the primary outputs of a Neuron subgraph. This causes failures during inference.
  * Workaround : Use the `excl_node_names` compilation option to change the partitioning of the graph during compile so that these nodes are not the primary output of a neuron subgraph. See [MXNet-Neuron Compilation API](../docs/mxnet-neuron/api-compilation-python-api.md)
  ```python
  compile_args = { 'excl_node_names': ["node_name_to_exclude"] }
  ```


### Models Supported

The following models have successfully run on neuron-inferentia systems

1. Resnet50 V1/V2
2. Inception-V2/V3/V4
3. Parallel-WaveNet
4. Tacotron 2
5. WaveRNN

## Other Notes
