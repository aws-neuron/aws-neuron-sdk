# Neuron Compiler Release Notes

This document lists the release notes for AWS Neuron compiler. The Neuron Compiler is an ahead-of-time compiler that ensures Neuron will optimally utilize the Inferentia chips.

Operator-support for each input format is provided directly from the compiler.

```
neuron-cc list-operators --framework {TENSORFLOW | MXNET | ONNX}
```

The supported operators are also listed here:
* [Neuron-cc TensorFlow Operators](./neuron-cc-ops/neuron-cc-ops-tensorflow.md)
* [Neuron-cc PyTorch Operators](./neuron-cc-ops/neuron-cc-ops-pytorch.md)
* [Neuron-cc MXNet Operators](./neuron-cc-ops/neuron-cc-ops-mxnet.md)
* [Neuron-cc ONNX Operators](./neuron-cc-ops/neuron-cc-ops-onnx.md)

## Known issues and limitations - updated 08/08/2020

1. **Control flow** Neuron only supports control flow operators which are static at compile time, i.e. static length RNN, top-k, sort, ...
2. **Size of neural network** The size of neural network is influenced by a) type of neural network (CNN, LSTM, MLP) , b) number of layers, c) sizes of input (dimension of the tensors, batch size, ...). The current Neuron compiler release has a limitation in terms of the size of neural network it could effectively optimize. As a result, we limit CNN models (e.g. ResNet) to have an input size of up to 480x480 fp16/32, batch size=4; LSTM models (e.g. GNMT) are limited to a time step limit of up to 900; MLP models (like BERT) are limited up to sequence-length=128, batch=8. Note: this size limitation on input tensors has been removed by using the "-O2" option - which will be made the default in an upcoming release.
3. **Data layout**  The Neuron compiler supports multiple data layout format (NCHW, NHWC, ...). Non-CNHW input/output data-layouts will require Neuron to insert additional _*transpose*_ operations, causing a degradation in performance.
4. **Object detection models** Computer-vision object detection and segmentation models are not supported by the current release except for SSD-300, more support is coming in future releases.
5. **Tensor residency** When a sub-graph that is executed on the host is communicating with a sub-graph that is executing on Neuron cores, tensors are copied via the communication queues between the host and Inferentia memory for each inference, which may result in end-to-end performance degradation.
6. **Primary inputs in NeuronCore Pipeline mode** When a neural network is executed in NeuronCore Pipeline mode, only the first operator in a neural network can receive primary inputs from the host.
7. On Ubuntu16, ResNet50 FP32 batch 1 compilation fails when "--batch_en" is used. On Ubuntu 18, this is not an issue.
8. **Reduce data type** INT8 data type is not currently supported by the Neuron compiler.
9. The new manually-selected option "-O2" may help address some of these limitations in some cases.



# [1.0.18001.0]

Date 8/08/2020

## Summary

Various performance improvements. 

## Major New Features

Improved performance of BERT base with -O2

## Resolved Issues

* n/a


# [1.0.17937.0]

Date 8/05/2020

## Summary

Various improvements. 





# [1.0.16861.0]

Date 7/16/2020

## Summary

This release has some bug fixes and some functional and performance improvements to support compilation of several neural networks.

## Major New Features
This release
* Supports compilation of PoseNet, tested for images of specific resolutions upto 736.
* Update the -O2 with a new memory allocator to reduce spilling to DRAM
* Improved performance of the '-O2' on BERT base, and openpose pose network.

## Resolved Issues
* Resolved compilation error in Vgg16 batch 1


## Other Notes
- Some versions of Inception network may fail to compile in Tensorflow on Ubuntu 16 in conda environment. The symptom is neuron-cc backend data race error. As a workaround use Ubuntu 18, Amazon Linux 2, or virtual env, or use neuron-cc with flag -O2.


# [1.0.15275.0]

Date 6/11/2020

## Summary

This release has some bug fixes and some functional and performance improvements to support compilation of several neural networks.

## Major New Features
This release
* Supports compilation of PoseNet for images of specific resolutions upto 400x400. 
* Improves performance of resnet152. 
* Supports a new command line option '-O2' that can help with handling of large tensor inputs for certain models.
* increase NEFF versions to 1.0. This means new NEFFs compiled from this release forward are not compatible with older versions of Neuron Runtime prior to May, 2020 (1.0.6905.0) release. Please update the Neuron Runtime when using NEFF version 1.0.

## Resolved Issues
- Compilation issues on prosotron encoder, decoder neural networks.

## Other Notes

### Dependencies
- This version creates NEFF 1.0 thus may require update of neuron-rtd if older than May 2020 release.

dmlc_nnvm==1.0.2574.0
dmlc_topi==1.0.2574.0
dmlc_tvm==1.0.2574.0
inferentia_hwm==1.0.1362.0
islpy==2018.2

# [1.0.12696.0]

Date 5/11/2020

## Summary

Bug fixes and some functional and performance improvements to several neural networks.  

## Major New Features

- This version supports compilation of unmodified Tensorflow BERT with batch size 1, 4, 6 for input sequence 128.
- Improved Tensorflow BERT batch 4 sequence 128 performance to 45% of the accelerator peak (from 34%).
- Support for MXNET BERT base batch 8 compilation
- Support for TF Resnet152 batch 2 compilation
- Most compiler messages are migrated from cout to logging mechanisms with verbosity control

## Resolved Issues
- Fixed failure to compile unmodified Tensorflow BERT model for small batches
- Fixed run-to-run-variability in OneHot operator implementation
- Robustness improvements for ParallelWavenet and transformer decoder networks


## Other Notes

### Dependencies
```
dmlc_nnvm==1.0.2356.0
dmlc_topi==1.0.2356.0
dmlc_tvm==1.0.2356.0
inferentia_hwm==1.0.1294.0
islpy==2018.2
```


# [1.0.9410.0]

Date 3/26/2020

## Summary

Bug fixes and some functional and performance improvements to several neural networks.

## Major New Features

* Support compilation of modified SSD-300 ([Link](../src/examples/tensorflow/ssd300_demo))
* Improved inference performance in natural language processing networks (such as prosotron encoder) by 45%

## Resolved Issues

* Eliminated redundant fp32 to bfloat16 cast on input and output tensors

 
## Known issues and limitations

* See previous releases. 


## Other Notes

* Added support for faster iteration on recurrent networks (aka auto-loop)


### Dependencies
```
dmlc_nnvm==1.0.2049.0 
dmlc_topi==1.0.2049.0 
pip install --upgrade dmlc_tvm==1.0.2049.0
inferentia_hwm==1.0.897.0
islpy==2018.2
```

# [1.0.7878.0]

Date 2/27/2020

## Summary

Bug fixes and minor performance improvements.

## Major New Features

None

## Resolved Issues

 * Corrected image resize operator functionallity
 * Compiler internal enhancements made that will benefit models such as BERT
 
## Known issues and limitations

* See previous releases. 


## Other Notes

### Dependencies
```
dmlc_nnvm-1.0.1826.0
dmlc_topi-1.0.1826.0
dmlc_tvm-1.0.1826.0
inferentia_hwm-1.0.897.0
islpy-2018.2

```

# [1.0.6801.0]

Date 1/27/2020

## Summary

Bug fixes and some performance enhancement related to data movement for BERT-type neural networks.

## Major New Features

None

## Resolved Issues

* Improved throughput for operators processed in the Neuron Runtime CPU. As an example: execution of 4 single NeuronCore NEFF models of ResNet50 v2 float16 batch = 5 in parallel on an inf1.1xlarge sped up by 30%.
* Corrected shape handling in Gather(TensorFlow)/Take(MXNet) operators that are processed by the Neuron Runtime in the Neuron Runtime vCPU, which resolves a possible crash in Neuron Compiler when compiling models with these operators with some shapes.
* Added support for TensorFlow *OneHot* operator (as a Neuron Runtime CPU operator).
* Added more internal checking for compiler correctness with newly defined error messages for this case.
```
      “Internal ERROR: Data race between Op1 'Name1(...) [...]' and Op2 'Name2(...) [...]'”
```
* Fixed out-of-memory issue introduced in 1.0.5939.0 such that some large models (BERT) compiled on instances with insufficient host memory would cause the runtime to crash with an invalid NEFF. This is actually a compiler error, but due to additional script layers wrapping this in the [BERT demo](../src/examples/tensorflow/bert_demo/README.md), this would have likely been seen as a runtime error like this:

```bash
2020-01-09 13:40:26.002594: E tensorflow/core/framework/op_segment.cc:54] Create kernel failed: Invalid argument: neff is invalid
2020-01-09 13:40:26.002637: E tensorflow/core/common_runtime/executor.cc:642] Executor failed to create kernel. Invalid argument: neff is invalid
[[{{node bert/NeuronOp}}]]
```

## Known issues and limitations

See previous release notes. Some tutorials show use of specific compiler options and flags, these are needed to help provide guidance to the compiler to achieve best performance in specific cases. Please do not use in cases other than as shown in the specific tutorial as results may not be defined. These options should be considered experimental and will be removed over time.

## Other Notes

### Dependencies
```
dmlc_nnvm-1.0.1619.0
dmlc_topi-1.0.1619.0
dmlc_tvm-1.0.1619.0
inferentia_hwm-1.0.839.0
islpy-2018.2
```

# [1.0.5939.0]

Date 12/20/2019

## Summary

Bug fixes and some performance enhancement for NeuronCore Pipeline.

## Major New Features

## Resolved Issues

* Fixed pipeline execution on more than 10 NeuronCores
* Improved NeuronCores Pipeline execution by improving data exchange efficiency between NeuronCores
* Added warning for unaligned memory access
* Fixed handling of cast on input FP32 tensor
* Improved handling of data layouts and transpose
* Improved dead-code elimination
* Improved efficiency of compute engine synchronization
* Improved efficiency of data transfers within the Neuron code

## Known issues and limitations

See previous release notes. Some tutorials show use of specific compiler options and flags, these are needed to help provide guidance to the compiler to achieve best performance in specific cases. Please do not use in cases other than as shown in the specific tutorial as results may not be defined. These options should be considered experimental and will be removed over time.

## Other Notes

### Dependencies
* dmlc_nnvm-1.0.1416.0
* dmlc_topi-1.0.1416.0
* dmlc_tvm-1.0.1416.0
* inferentia_hwm-1.0.720.0
* islpy-2018.2


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

See previous release notes. Resolved issues are shown in Resolved Issues.

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
2. **Size of neural network** The size of neural network is influenced by a) type of neural network (CNN, LSTM, MLP) , b) number of layers, c) sizes of input (dimension of the tensors, batch size, ...). The current Neuron compiler release has a limitation in terms of the size of neural network it could effectively optimize. As a result, we limit CNN models (e.g. ResNet) to have an input size of up to 480x480 FP16, batch size of 4; LSTM models (e.g. GNMT) are limited to a time step limit of up to 900; MLP models (like BERT) are limited up to sequence-length equal 128, batch=8.
3. **Data layout**  The Neuron compiler supports multiple data layout format (NCHW, NHWC, ...). Non-CNHW input/output data-layouts will require Neuron to insert additional _*transpose*_ operations, causing a degradation in performance.
4. **Object detection models** Computer-vision object detection and segmentation models are not supported by the current release.
5. **Reduce data type** INT8 data type is not currently supported by the Neuron compiler.
6. **Tensor residency** When a sub-graph that is executed on the host is communicating with a sub-graph that is executing on Neuron cores, tensors are copied via the communication queues between the host and Inferentia memory for each inference, which may result in end-to-end performance degradation.
7. **Primary inputs in NeuronCore Pipeline mode** When a neural network is executed in NeuronCore Pipeline mode, only the first operator in a neural network can receive primary inputs from the host.

## Other Notes

### Dependencies

* nnvm: dmlc_nnvm-1.0.1219.0
* topi: dmlc_topi-1.0.1219.0
* tvm: dmlc_tvm-1.0.1219.0
* hwm: inferentia_hwm-1.0.602.0
* islpy: islpy-2018.2+aws2018.x.73.0
