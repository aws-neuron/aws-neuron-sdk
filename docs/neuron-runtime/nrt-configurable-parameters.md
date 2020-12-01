</br>
</br>

Please view our documentation at **[https://awsdocs-neuron.readthedocs-hosted.com/](https://awsdocs-neuron.readthedocs-hosted.com/)** 

**Please note this file will be deprecated.**

</br>
</br>



# Neuron-RTD Configurable Parameters

This guide provides an overview of the different parameters available to configure Neuron runtime behavior.


## Global Runtime Configuration

These parameters are defined in neuron-rtd.config and affect global runtime configuration. Note that Neuron runtime must be restarted after changes to the configuration file for them to take effect.

### Model Directory Caching:

One of the most time consuming stages in model loading is the unpackaging of the NEFF file to a temporary directory for the runtime to digest. To mitigate this cost for repeated loads of the same model, caching can be turned on by giving an integer value to the `model_cache_count` field in neuron-rtd.config to set a threshold on the number of unpacked model directories that the runtime can keep around. Keyed based on NEFF UUID, the runtime will check for an existing mapping to a cached directory and reuse it if found. The cache employs simple LRU eviction when full.


## Per-Model GRPC Load Parameters:

These are optional parameters that can accompany a model `load()` API call to set certain behaviors for that specific model. Note that some of these parameters can also have a default value configurable in neuron-rtd.config that will apply to every model that does not provide that parameter during `load()`.

### Per-inference timeout:

The maximum amount of time in seconds spent waiting for each inference to complete can be configured by passing an integer value to the `timeout` parameter. If the timeout is reached, the runtime will immediately return TIMEOUT (error code 5) regardless of the eventual status of the inference.

The default timeout value is 2 seconds. It can be modified in the neuron-rtd.config file.


### Inference queue size:

More then one inference request could be posted concurrently up to the inference queue size limit.  Having inference requests in the queue allows the runtime to prepare the next set of inputs while the previous inference is running on the hardware thus increasing the overall throughput.


[//]: # (Removed this sentence for now: It is most useful for models running in serial mode, where the inferences can be staggered so that multiple inferences can happen concurrently in hardware.)

Interface queue size can be adjusted for each model by passing an integer value to the 'ninfer' parameter.  A global inference queue size default can be specified in the neuron-rtd.config file. The default value is 4.

### Input staging location:

Inference inputs can be configured to be staged in either host or device memory prior to starting an inference by passing a boolean flag to the `io_data_host` parameter. A value of true stages the data in host memory, while false stages the data in the device. Bandwidth is much higher from the device than the host to the chip, so staging on the device can be beneficial for models with large input loads that would otherwise cause a bottleneck during transfer. Note that this does introduce an extra step during inference posting to transfer the input to the device, so it may negatively affect single-inference latency. This option is most useful when paired with concurrent, pipelined inferences (with an appropriate `ninfer` value) so that inference execution in hardware can hide the extra overhead of staging.

A global default can be specified in the config file with the `io_dma_data_host` flag. Default value on installation is false.
