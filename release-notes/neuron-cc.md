# Neuron Compiler Release Notes

This document lists the release notes for AWS Neuron compiler. The neuron compiler is an ahead-of-time compiler that ensures Neuron will optimally utilize the Inferentia chips.

Operator support for each input format is provided directly from the compiler:

```
neuron-cc --list-operators --framwork {TENSORFLOW | MXNET | ONNX}
```

and

* [Neuron-cc Tensorflow Operators](./neuron-cc-ops-tensorflow.md)
* [Neuron-cc MXNet Operators](./neuron-cc-ops-mxnet.md)
* [Neuron-cc ONNX Operators](./neuron-cc-onnx.md)


# [1.0.5301.0]

Date 12/1/2019

## Summary

## Major New Features

## Resolved Issues

* Added warning for unsupported operators and convolution sizes
* Added warning for unsupported layout / upsampling
* Improved performance of batched inference loading convolutional weights
* Corrected Matmult column size 1
* Added support for Relu6, AddV2, BatchMatmulV2 operators
* Added support for default MXNET outputs in –io-config
* Corrected bf16 constant loading
* Corrected Conv2D tile accumulation 

## Known Issues and Limitations

See Previous releases. Resolved issues are shown in Resolved Issues.

## Other Notes

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
2. **Size of neural network** The size of neural network is influenced by a) type of neural network (CNN, LSTM, MLP) , b) number of layers, c) sizes of input (dimension of the tensors, batch size, ...). The current Neuron compiler release has a limitation in terms of the size of neural network it could effectively optimize. As a result, we limit the sizes of CNN models like ResNet to have an input size limit of 480x480 fp/bf16, batch size=4; LSTM models like GNMT to have a time step limit of 900; MLP models like BERT to have input size limit of sequence length=128, batch=8.
3. **Data layout**  The Neuron compiler supports multiple data layout format (NCHW, NHWC, ...). If the data layout is in native CNHW format, however, the neural network will have a lower latency because the compiler would not insert the necessary _*transpose*_ operators to translate data layout from one to another.
4. **Object detection models** Computer-vision object detection and segmentation models are not supported by the current release.
5. **Reduce data type** INT8 data type is not currently supported by the Neuron compiler.
6. **Tensor residency** When a sub-graph that is executed on the host is communicating with a sub-graph that is executing on Neuron cores, tensors are copied via the communication queues between the host and Inferentia memory for each inference. However, when we implement attention outputs to RNN or Transformers, this could add a significant inefficiency because the communication overhead could dominate the sub-graph executing on the Neuron cores. 
7. **Primary inputs in Neuron Pipeline mode** When a neural network is executed in Neuron Pipeline mode, only the first operator in a neural network can receive primary inputs from the host.

## Other Notes

### Dependencies

* nnvm: dmlc_nnvm-1.0.1219.0
* topi: dmlc_topi-1.0.1219.0
* tvm: dmlc_tvm-1.0.1219.0
* hwm: inferentia_hwm-1.0.602.0
* islpy: islpy-2018.2+aws2018.x.73.0


