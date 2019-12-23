# Neuron Compiler Release Notes

This document lists the release notes for AWS Neuron compiler. The neuron compiler is an ahead-of-time compiler that ensures Neuron will optimally utilize the Inferentia devices.

Operator-support for each input format is provided directly from the compiler:

```
neuron-cc --list-operators --framwork {TENSORFLOW | MXNET | ONNX}
```

The supported operators are also listed here:

* [Neuron-cc Tensorflow Operators](./neuron-cc-ops/neuron-cc-ops-tensorflow.md)
* [Neuron-cc MXNet Operators](./neuron-cc-ops/neuron-cc-ops-mxnet.md)
* [Neuron-cc ONNX Operators](./neuron-cc-ops/neuron-cc-ops-onnx.md)

# [1.0.5939.0]

Date 12/20/2019

## Summary

Bug fixes and some performance enhancement for NeuronCore Pipeline.

## Major New Features

## Resolved Issues

* Fixed pipeline execution on more than 10 NeuronCores
* Improved NeuronCores Pipeline execution by improving data exchange efficiency between NeuronCores
* Added warning for unaligned memory access
* Fixed handling of cast on input fp32 tensor
* Improved handling of data layouts and transpose
* Improved dead-code elimination
* Improved efficiency of compute engine synchronization
* Improved efficiency of data transfers within the Neuron code

## Known issues and limitations

See previous releases. Some tutorials show use of specific compiler options and flags, these are needed to help provide guidance to the compiler to acheive best performance in specific cases. Please do not use in cases other than as shown in the specific tutorial as results may not be defined. These options should be considered experimental and will be removed over time. 

## Other Notes

Dependencies
dmlc_nnvm-1.0.1416.0 
dmlc_topi-1.0.1416.0 
dmlc_tvm-1.0.1416.0 
inferentia_hwm-1.0.720.0 
islpy-2018.2


# [1.0.5301.0]

Date 12/1/2019

## Summary

## Major New Features

## Resolved Issues

* Added warning for unsupported operators and convolution sizes
* Added warning for unsupported layout / upsampling
* Added support for Relu6, AddV2, BatchMatmulV2 operators
* Added support for default MXNet outputs in –io-config
* Improved performance of batched inference for convolutional networks
* Fixed MatMult column size 1
* Fixed bf16 constant loading
* Fixed Conv2D tile accumulation

## Known Issues and Limitations

See Previous releases. Resolved issues are shown in Resolved Issues.

## Other Notes

Please install g++ on AMIs without g++ pre-installed (i.e. server AMIs):

```bash
# Ubuntu
sudo apt-get install -y g++
```

```bash
# Amazon Linux
sudo yum nstall -y gcc-c++
```

Supported Python versions:
  * 3.5, 3.6, 3.7

Supported Linux distributions:
  * Ubuntu 16, Ubuntu 18, Amazon Linux 2


### Dependencies

* dmlc_nnvm-1.0.1328.0
* dmlc_topi-1.0.1328.0
* dmlc_tvm-1.0.1328.0
* inferentia_hwm-1.0.674.0
* islpy-2018.2

# [1.0.4680.0]

Date:  11/25/2019

## Major new features

N/A, this is the first release.

## Resolved issues

N/A, this is the first release.

## Known issues and limitations

1. **Control flow** Inferentia has a limited support for control flow. In general, Neuron can only support control flow operators which are static at compile time, i.e. static length RNN, top-k, sort, ...
2. **Size of neural network** The size of neural network is influenced by a) type of neural network (CNN, LSTM, MLP) , b) number of layers, c) sizes of input (dimension of the tensors, batch size, ...). The current Neuron compiler release has a limitation in terms of the size of neural network it could effectively optimize. As a result, we limit CNN models (e.g. ResNet) to have an input size of up to 480x480 fp/bf16, batch size=4; LSTM models (e.g. GNMT) are limited to a time step limit of up to 900; MLP models (like BERT) are limited up to sequence-length=128, batch=8.
3. **Data layout**  The Neuron compiler supports multiple data layout format (NCHW, NHWC, ...). Non-CNHW input/output data-layouts will require Neuron to insert additional _*transpose*_ operations, causing a degradation in performance.
4. **Object detection models** Computer-vision object detection and segmentation models are not supported by the current release.
5. **Reduce data type** INT8 data type is not currently supported by the Neuron compiler.
6. **Tensor residency** When a sub-graph that is executed on the host is communicating with a sub-graph that is executing on Neuron cores, tensors are copied via the communication queues between the host and Inferentia memory for each inference, which may result in end-to-end performacne degradation.
7. **Primary inputs in NeuronCore Pipeline mode** When a neural network is executed in NeuronCore Pipeline mode, only the first operator in a neural network can receive primary inputs from the host.

## Other Notes

### Dependencies

* nnvm: dmlc_nnvm-1.0.1219.0
* topi: dmlc_topi-1.0.1219.0
* tvm: dmlc_tvm-1.0.1219.0
* hwm: inferentia_hwm-1.0.602.0
* islpy: islpy-2018.2+aws2018.x.73.0
