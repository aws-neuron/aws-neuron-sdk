# Neuron runtime release notes

This document lists the current release notes for AWS Neuron Runtime.  Neuron Runtime software manages runtime aspects of executing inferences on Inferentia chips. It runs on Ubuntu(16/18) and Amazon Linux 2.

# [1.0.4751.0]

Date: 12/20/2019

## Major New Features

N/A

## Improvements

* Improved neuron-rtd startup time on inf1.24xl
* Reduced inference submission overhead (improved inference latency)
* Made the names and the UUIDs of loaded models available to neuron-tools

## Resolved Issues

The following issues have been resolved:

* File I/O errors are not checked during model load
* Memory leak during model unload
* Superfluous error message are logged while reading neuron-rtd configuration file
* neuron-rtd --version command does not work

## Known Issues and Limitations

* A model might fail to load due to insufficient number of huge memory pages made available to Neuron-RTD. A manual reconfiguration and Neuron-RTD restart is required for increasing the amount of huge memory pages available to Neuron-RTD.
  * Workaround: manually increase the amount of huge memory pages available to Neuron runtime by following the [instructions here:](https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-runtime/nrt_start.md#step-3-configure-nr_hugepages) (Requires a restart of the runtime daemon and a possible change to system-wide configs.)
* Neuron-RTD does not return verbose error messages when a model load or an inference fails. Detailed error messages are only available in syslog.
  * Workaround: manually search syslog file for Neuron-RTD error messages.

## Other Notes


# [1.0.4492.0]

Date: 12/1/2019

## Major New Features

N/A

## Resolved Issues

The following issues have been resolved:

* Neuron-RTD fails to initialize all NeuronCores on Inf1.24xl Inferentia instances
* On some instances neuron-discovery requires packages (pciutils)
* An inference request might timeout or return a failure when a NeuronCore Pipeline model is loaded on any instance larger than Inf1.xl or Inf1.2xla
* Loading of a model fails when NeuronCore Pipeline inputs are consumed by NeuronCores beyond the first 4 NeuronCores used by the model
* Neuron-RTD logging to stdout does not work
* Incorrect DMA descriptors validation.  While loading a model; descriptors are allowed to point beyond allocated address ranges.  This could cause the model load failure or produce incorrect numerical results
* NeuronCore statistics are read incorrectly

## Known Issues and Limitations

* A model might fail to load due to insufficient number of huge memory pages made available to Neuron-RTD.  A manual reconfiguration and Neuron-RTD restart is required for increasing the amount of huge memory pages available to Neuron-RTD.
    * Workaround: manually increase the amount of huge memory pages available to Neuron runtime by following the [instructions here:](../docs/neuron-runtime/nrt_start.md#step-3-configure-nr_hugepages) 
    (Requires a restart of the runtime daemon and a possible change to system-wide configs.)
* Neuron-RTD does not return verbose error messages when a model load or an inference fails.  Detailed error messages are only available in syslog.
    * Workaround: manually search syslog file for Neuron-RTD error messages.
* Neuron-RTD takes 6 minutes to start on Inf1.24xl instance.

## Other Notes



# [1.0.4109.0]

Date:  11/25/2019

## Summary

This document lists the current release notes for AWS Neuron runtime.  Neuron runtime software manages runtime aspects of executing inferences on Inferentia chips. It runs on Ubuntu 16, Ubuntu 18 and Amazon Linux 2.

## Major new features

N/A, this is the first release.

## Major Resolved issues

N/A, this is the first release.

## Known issues and limitations

* Neuron-RTD fails to initialize all NeuronCores on Inf1.24xl Inferentia instances.
    * Workarounds: update to next release
* On some instances neuron-discovery requires packages (pciutils) 
    * Workaround: install explicitly
* An inference request might timeout or return a failure when a NeuronCore Pipeline model is loaded on any instance larger than Inf1.xl or Inf1.2xla
    * Workarounds: update to the next release
* Loading of a model fails when NeuronCore Pipeline inputs are consumed by NeuronCores beyond the first 4 NeuronCores used by the model.  
A model can be compiled to run on multiple NeuronCores spread across multiple Inferentias.  The model’s inference inputs (ifmaps) can be 
consumed by one or more NeuronCores, depending on a model.  If a model requires inputs going to NeuronCores beyond the first 4 the loading of the model will fail. 
    * Workarounds: update to the next release
* Neuron-RTD logging to stdout does not work
    * Workarounds: update to the next release
* Incorrect DMA descriptors validation.  While loading a model; descriptors are allowed to point beyond allocated address ranges.  This could cause the model load failure or produce incorrect numerical results.
    * Workarounds: update to the next release
* NeuronCore statistics are read incorrectly
    * Workarounds: update to the next release
* A model might fail to load due to insufficient number of huge memory pages made available to Neuron-RTD.  A manual reconfiguration and Neuron-RTD restart is required for 
increasing the amount of huge memory pages available to Neuron-RTD.
    * Workarounds: manually increase the amount of huge memory pages available to Neuron runtime by [following the instructions here:](../docs/neuron-runtime/nrt_start.md#step-3-configure-nr_hugepages) 
    ** This requires a restart of the runtime daemon.
* Neuron-RTD does not return verbose error messages when a model load or an inference fails.  Detailed error messages are only available in syslog.
    * Workarounds: manually search syslog file for Neuron-RTD error messages.

## Other Notes

* DLAMI v26.0 users are encouraged to update to the latest Neuron release by following these instructions: https://github.com/aws/aws-neuron-sdk/blob/master/release-notes/dlami-release-notes.md



