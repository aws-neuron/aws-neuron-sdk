# MXNet-Neuron Release Notes

This document lists the release notes for MXNet-Neuron framework.

# [1.5.1.1.0.1401.0]

Date 1/27/2020

## Summary

No major changes or fixes. 

## Major New Features

## Resolved Issues

## Known Issues and Limitations

* Latest pip version 20.0.1 breaks installation of MXNet-Neuron pip wheel which has py2.py3 in the wheel name. This breaks all existing released versions. The Error looks like: 

```
Looking in indexes: https://pypi.org/simple, https://pip.repos.neuron.amazonaws.com
ERROR: Could not find a version that satisfies the requirement mxnet-neuron (from versions: none)
ERROR: No matching distribution found for mxnet-neuron
```
   * Work around:  install the older version of pip using "pip install pip==19.3.1".
 
 * Issue: MXNet Model Server is not able to clean up Neuron RTD states after model is unloaded (deleted) from model server and previous workaround "`/opt/aws/neuron/bin/neuron-cli reset`" is unable to clear all Neuron RTD states.
   * Workaround: run “`sudo systemctl restart neuron-rtd`“ to clear Neuron RTD states after all models are unloaded and server is shut down.
 

## Other Notes

# [1.5.1.1.0.1325.0]

Date 12/1/2019

## Summary

## Major New Features

## Resolved Issues

* Issue: Compiler flags cannot be passed to compiler during compile call. The fix: compiler flags can be passed to compiler during compile call using “flags” option followed by a list of flags.

* Issue: Advanced CPU fallback option is a way to attempt to improve the number of operators on Inferentia. The default is currently set to on, which may cause failures. The fix: This option is now off by default.

## Known Issues and Limitations

* Issue: MXNet Model Server is not able to clean up Neuron RTD states after model is unloaded (deleted) from model server and previous workaround "`/opt/aws/neuron/bin/neuron-cli reset`" is unable to clear all Neuron RTD states.
  * Workaround: run “`sudo systemctl restart neuron-rtd`“ to clear Neuron RTD states after all models are unloaded and server is shut down.
  
## Other Notes

# [1.5.1.1.0.1349.0]

Date 12/20/2019

## Summary

No major changes or fixes. Released with other Neuron packages.

# [1.5.1.1.0.1325.0]

Date 12/1/2019

## Summary

## Major New Features

## Resolved Issues

* Issue: Compiler flags cannot be passed to compiler during compile call. The fix: compiler flags can be passed to compiler during compile call using “flags” option followed by a list of flags.

* Issue: Advanced CPU fallback option is a way to attempt to improve the number of operators on Inferentia. The default is currently set to on, which may cause failures. The fix: This option is now off by default.

## Known Issues and Limitations

* Issue: MXNet Model Server is not able to clean up Neuron RTD states after model is unloaded (deleted) from model server and previous workaround "`/opt/aws/neuron/bin/neuron-cli reset`" is unable to clear all Neuron RTD states.
  * Workaround: run “`sudo systemctl restart neuron-rtd`“ to clear Neuron RTD states after all models are unloaded and server is shut down.
  
## Other Notes

# [1.5.1.1.0.1260.0]

Date:  11/25/2019

## Summary

This version is available only in released DLAMI v26.0 and is based on MXNet version 1.5.1. Please [update](./dlami-release-notes.md#known-issues) to latest version.

## Major new features

## Resolved issues

## Known issues and limitations

* Issue: Compiler flags cannot be passed to compiler during compile call.
* Issue: Advanced CPU fallback option is a way to attempt to improve the number of operators on Inferentia. The default is currently set to on, which may cause failures.
  * Workaround: explicitly turn it off by setting compile option op_by_op_compiler_retry to 0.
* Issue: Temporary files are put in current directory when debug is enabled.
  * Workaround: create a separate work directory and run the process from within the work directory
* Issue: MXNet Model Server is not able to clean up Neuron RTD states after model is unloaded (deleted) from model server.
  * Workaround: run “`/opt/aws/neuron/bin/neuron-cli reset`“ to clear Neuron RTD states after all models are unloaded and server is shut down.
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

* Python versions supported:
  * 3.5, 3.6, 3.7
* Linux distribution supported:
  * Ubuntu 16, Ubuntu 18, Amazon Linux 2
