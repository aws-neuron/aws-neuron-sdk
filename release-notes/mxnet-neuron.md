# MXNet-Neuron Release Notes

This document lists the release notes for MXNet-Neuron framework.

# Known Issues 08/05/2020
 
* Issue: MXNet Model Server is not able to clean up Neuron RTD states after model is unloaded (deleted) from model server.
  * Workaround: run “`/opt/aws/neuron/bin/neuron-cli reset`“ to clear Neuron RTD states after all models are unloaded and server is shut down.
  
# [1.5.1.1.0.2101.0]

Date 08/05/2020

## Summary

Various minor improvements.

## Major New Features

## Resolved Issues



# [1.5.1.1.0.2093.0]

Date 07/16/2020

## Summary

This release contains a few bug fixes and user experience improvements.

## Major New Features

## Resolved Issues

* User can specify NEURONCORE_GROUP_SIZES without brackets (for example, "1,1,1,1"), as can be done in TensorFlow-Neuron and PyTorch-Neuron.
* Fixed a memory leak when inferring neuron subgraph properties
* Fixed a bug dealing with multi-input subgraphs
  
# [1.5.1.1.0.2033.0]

Date 6/11/2020

## Summary

* Added support for profiling during inference

## Major New Features

* Profiling can now be enabled by specifying the profiling work directory using NEURON_PROFILE environment variable during inference. For an example of using profiling, see [Getting Started](../docs/neuron-tools/getting-started-tensorboard-neuron.md). (Note that graph view of MXNet graph is not available via TensorBoard).

## Resolved Issues

## Known Issues and Limitations

## Other Notes


# [1.5.1.1.0.1900.0]

Date 5/11/2020

## Summary

Improved support for shared-memory communication with Neuron-Runtime.

## Major New Features
* Added support for the BERT-Base model (base: L-12 H-768 A-12), max sequence length 64 and batch size of 8.
* Improved security for usage of shared-memory for data transfer between framework and Neuron-Runtime
* Improved allocation and cleanup of shared-memory resource
* Improved container support by automatic falling back to GRPC data transfer if shared-memory cannot be allocated by Neuron-Runtime

## Resolved Issues
* User is unable to allocate Neuron-Runtime shared-memory resource when using MXNet-Neuron in a container to communicate with Neuron-Runtime in another container. This is resolved by automatic falling back to GRPC data transfer if shared-memory cannot be allocated by Neuron-Runtime.
* Fixed issue where some large models could not be loaded on inferentia.

## Known Issues and Limitations

## Other Notes

# [1.5.1.1.0.1596.0]

Date 3/26/2020

## Summary

No major changes or fixes

## Major New Features


## Resolved Issues

## Known Issues and Limitations

## Other Notes


# [1.5.1.1.0.1498.0]

Date 2/27/2020

## Summary

No major changes or fixes.

## Major New Features

## Resolved Issues

The issue(s) below are resolved:
* Latest pip version 20.0.1 breaks installation of MXNet-Neuron pip wheel which has py2.py3 in the wheel name.

## Known Issues and Limitations

* User is unable to allocate Neuron-Runtime shared-memory resource when using MXNet-Neuron in a container to communicate with Neuron-Runtime in another container. To work-around, please set environment variable NEURON_RTD_USE_SHM to 0.

## Other Notes

# [1.5.1.1.0.1401.0]

Date 1/27/2020

## Summary

No major changes or fixes.

## Major New Features

## Resolved Issues
* The following issue is resolved when the latest multi-model-server with version >= 1.1.0 is used with MXNet-Neuron. You would still need to use "`/opt/aws/neuron/bin/neuron-cli reset`" to clear all Neuron RTD states after multi-model-server is exited:
  * Issue: MXNet Model Server is not able to clean up Neuron RTD states after model is unloaded (deleted) from model server and previous workaround "`/opt/aws/neuron/bin/neuron-cli reset`" is unable to clear all Neuron RTD states.

## Known Issues and Limitations

* Latest pip version 20.0.1 breaks installation of MXNet-Neuron pip wheel which has py2.py3 in the wheel name. This breaks all existing released versions. The error looks like:
```
Looking in indexes: https://pypi.org/simple, https://pip.repos.neuron.amazonaws.com
ERROR: Could not find a version that satisfies the requirement mxnet-neuron (from versions: none)
ERROR: No matching distribution found for mxnet-neuron
```
  * Work around:  install the older version of pip using "pip install pip==19.3.1".



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
