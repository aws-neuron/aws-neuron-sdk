# NeuronX Runtime Header Files

## Overview

The NeuronX Runtime Library provides C APIs for initializing the Neuron hardware,
loading models and input data, executing iterations on loaded models, and
retrieving output data.

This library is provided to customers via a shared object (libnrt.so) that is installed
through the `aws-neuronx-runtime-lib` package. This directory exposes the header files
that customers can use to write custom applications utilizing the NeuronX Runtime Library.

## File Location

These header files will be installed to the user's system under `/opt/aws/neuron/include`
when installing the `aws-neuronx-runtime-lib` package and the `libnrt.so` library is 
installed under the `/opt/aws/neuron/lib` directory.

## Experimental Headers

The following files contain experimental function declarations and are subject to change in 
future releases.

- nrt_async.h
- nrt_async_sendrecv.h
- nrt_experimental.h

## Documentation

https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-api-guide.html
