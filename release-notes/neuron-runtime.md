# Neuron Runtime Release Notes

This document lists the current release notes for AWS Neuron Runtime.  Neuron Runtime software manages runtime aspects of executing inferences on Inferentia chips. Details on the configuration and use is availble in [docs/neuron-runtime/.](../docs/neuron-runtime/nrt_start.md)

### NEFF Support Table:

Use this table to determine the version of Runtime that will support the version of NEFF you are using.  NEFF version is determined by the version of the Neuron Compiler.

| NEFF Version | Runtime Version Range | Notes                                   |
| ---- | ------------- | --------------------------------------------- |
| 0.6  | * | All versions of RT support NEFF 0.6 |
| 1.0  | >= 1.0.6905.0 | Starting support for 1.0 NEFFs |

---

# [1.0.9592.0]

Date: 09/21/2020

## Major New Features
* n/a

## Improvements
* The “handshake” API can be used between a framework, such as TensorFlow, and neuron-rtd.  The API establishes a unique “session-id” (see the next item) and facilitates version exchange between a framework and neuron-rtd.  Version information is used to improve logging and troubleshooting. 
* The API for neural networks loading and for shared memory allocation have been enhanced to allow an optional “session id” to be passed in load/allocate requests.  Session ids are used to associate a framework process with the networks and the shared memory segments used by the process.  Neuron-rtd can optionally monitor framework processes and automatically unload all neural networks loaded by the process and free its shared memory when the process terminates.

## Resolved Issues
* querying Neuron statistics could cause neuron-rtd to crash
* SRAM parity errors are not reported
* Under stress “queue full” error can be returned when submitting an inference request even when neuron-rtd has room for one more request



# [1.0.9197.0]

Date: 08/19/2020

## Summary
Bug fix only.

## Major New Features
* n/a

## Resolved Issues
* get-hw-counters API was returning ECC error counters for only one half of the Inferentia DRAM.


# [1.0.8896.0]

Date: 08/08/2020

## Summary
Bug fix only.

## Major New Features
* n/a

## Resolved Issues
* Fixed a crash in neuron-rtd when multiple clients attempt to load models at the same time.



# [1.0.8813.0]

Date: 08/05/2020

## Summary
Patching a bug from prior versions that could lead to crashes under load.

## Major New Features
* n/a

## Resolved Issues
* Fixed a race condition in the runtime that was leading to crashes in some cases of load testing.  



# [1.0.8444.0]

Date: 07/16/2020

## Major New Features
* n/a

## Improvements
* Improved performance of the Neural Networks with large input tensors.

## Resolved Issues
* neuron-rtd crashes when “Unload All” API is called multiple times.
* In some cases neuron-compiler optimizes access to the input tensors. Because of this optimization inference requests fail with an error message indicating the mismatch between expected and supplied number of input tensors.
* In some cases NEFF can use more DMA rings than is supported by neuron-rtd.  A Neural Network load fails to load with an error message indicating the failure to allocate a DMA ring.

## Other Notes
* Renamed and combined Neuron device memory errors counters.  Four counters - ddr0_ecc_corr, ddr0_ecc_uncorr, ddr1_ecc_corr, ddr1_ecc_uncorr were combined into two counters - mem_ecc_corr and mem_ecc_uncorr.

# [1.0.8032.0]

Date: 6/18/2020

## Major New Features

* n/a

## Improvements

* n/a

## Resolved Issues

*  In the versions of aws-neuron-runtime-base and aws-neuron-runtime, yum downgrade/update removed the service unit files. This results in neuron-discovery and neuron-rtd start failures.

Please update the Neuron Runtime ingredients on AL2 by first removing the old package and installing the latest:
```
# Amazon Linux 2
sudo yum remove aws-neuron-runtime-base
sudo yum remove aws-neuron-runtime
sudo yum install aws-neuron-runtime-base
sudo yum install aws-neuron-runtime
```

# [1.0.7865.0]

Date: 6/11/2020

## Major New Features

* n/a

## Improvements

* Improved Neuron device memory allocation to accommodate Neural Networks that operate on large tensors.
* Log the version of the NEFF file during Neural Network load to aid troubleshooting.

## Resolved Issues

* An inference request with missing IFMAP tensors is allowed to execute and produces undefined results.
* neuron-rtd service is not stopped and is not removed when aws-neuron-runtime package is uninstalled.

## Known Issues and Limitations

* A model might fail to load due to insufficient number of huge memory pages made available to Neuron-RTD.
    * Workaround: manually increase the amount of huge memory pages available to Neuron runtime by following the [instructions here.](https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-runtime/nrt_start.md#step-3-configure-nr_hugepages) (Requires a restart of the runtime daemon and a possible change to system-wide configs.)


# [1.0.6905.0]

Date: 5/11/2020

## Major New Features

* Support is added for NEFF 1.0.

## Improvements

* A new API for unloading all loaded Neural Networks and for freeing all Inferentia resources.  The API is used by ML frameworks in cases when an ML application needs to be restarted to bring Inferentias to their initial state.
* Improved inference error handling and improved verbosity of error notifications.
* Internal changes aimed to improve performance optimization work and debuggability.

## Resolved Issues
* Latency of Neural Networks loading had degraded in 1.0.6222.0 release.  The issue has been resolved.

## Known Issues and Limitations

* A model might fail to load due to insufficient number of huge memory pages made available to Neuron-RTD.
    * Workaround: manually increase the amount of huge memory pages available to Neuron runtime by following the [instructions here.](https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-runtime/nrt_start.md#step-3-configure-nr_hugepages) (Requires a restart of the runtime daemon and a possible change to system-wide configs.)





# [1.0.6222.0]

Date: 3/26/2020

## Major New Features

N/A

## Improvements

* Inferentia memory utilization has improved, allowing larger number of Neural Networks to be loaded simultaneously.  The increased capacity could be up to 25% depending on the networks.
* Added an API to read performance counters for a single Neuron Core.  Used internally by neuron-top, which comes with the aws-neuron-tools package.
* Added Neural Network caching.  Caching of previously loaded Neural Networks in host memory can significantly speed up (up to 10x) the subsequent loading of the same networks, for example when using multiple Neuron Cores in data-parallel mode.

## Resolved Issues

* Occassional neuron-rt service crashes when service was being shutdown.

## Known Issues and Limitations

* A model might fail to load due to insufficient number of huge memory pages made available to Neuron-RTD.
    * Workaround: manually increase the amount of huge memory pages available to Neuron runtime by following the [instructions here.](https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-runtime/nrt_start.md#step-3-configure-nr_hugepages) (Requires a restart of the runtime daemon and a possible change to system-wide configs.)


# [1.0.5795.0]

Date: 2/27/2020

## Major New Features

* Added API to unload all models available via "neuron-cli reset".

## Improvements

* Neural Network Load and Neural Network Infer interfaces return descriptive error messages on failure.
* Throughput of Neural Networks running in NeuronCore Pipeline mode has improved by 10-50% (network dependent) by reducing contention among NeuronCores. 
* Improved CPU utilization of neuron-rt daemon by completely removing one polling thread from neuron-rt. 

## Resolved Issues

* Neural Networks containing CPU partitions only do not load correctly.
* Insufficient logging makes it hard to identify Neural Network loading failure when multiple networks are loaded in parallel.


## Known Issues and Limitations

* A model might fail to load due to insufficient number of huge memory pages made available to Neuron-RTD.
    * Workaround: manually increase the amount of huge memory pages available to Neuron runtime by following the [instructions here.](https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-runtime/nrt_start.md#step-3-configure-nr_hugepages) (Requires a restart of the runtime daemon and a possible change to system-wide configs.)


# [1.0.5236.0]

Date: 1/27/2020

## Major New Features

N/A

## Improvements

* Improved neuron-rtd startup time on inf1.6xl and inf1.24xl.  Neuron-rtd startup now takes the same amount of time on all instance sizes. 
* Improved inference latency for Neural Networks that fully execute on Inferentia (have no on-CPU nodes).  The exact latency improvement is network dependent and is estimated to be 50-100us per inference. 
* Neural Network load GRPC returns descriptive error message when the load fails.
* Changed default behavior of neuron-rtd to drop elevated privileges after runtime initialization.  During initialization elevated priveleges are necessary to allow bus enumeration and shared memory with frameworks.
* Error log is automatically displayed on the console if the installation of aws-neuron-runtime fails.

## Resolved Issues

* minor bug fixes

## Known Issues and Limitations

* A model might fail to load due to insufficient number of huge memory pages made available to Neuron-RTD. A manual reconfiguration and Neuron-RTD restart is required for increasing the amount of huge memory pages available to Neuron-RTD.
    * Workaround: manually increase the amount of huge memory pages available to Neuron runtime by following the [instructions here.](https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-runtime/nrt_start.md#step-3-configure-nr_hugepages) (Requires a restart of the runtime daemon and a possible change to system-wide configs.)
* Neuron-RTD does not return verbose error messages when an inference fails. Detailed error messages are only available in syslog.
    * Workaround: manually search syslog file for Neuron-RTD error messages.




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



