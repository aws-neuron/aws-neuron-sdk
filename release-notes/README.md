# Neuron Release Notes

## Jan 28, 2020 Release
This release brings significant throughput improvements to running inference on a variety of models; for example Resnet50 throughput is increased by 63% (measured 1800 img/sec on inf1.xlarge up from 1100/sec, and measured 2300/sec on inf1.2xlarge). BERTbase throughput has improved by 36% compared to the re:Invent launch (up to 26100seq/sec  from 19200seq/sec on inf1.24xlarge), and BERTlarge improved by 15% (230 seq/sec, compared to 200 running on inf1.2xlarge). In addition to the performance boost, this release includes various bug fixes as well as additions to the GitHub with [new tech notes](../docs/technotes) diving deep on how Neuron performance features work and overall improved documentation following customer input. 

We continue to work on new features and improving performance further, to stay up to date follow this repository, and watch the [AWS Neuron developer forum](https://forums.aws.amazon.com/forum.jspa?forumID=355).

## Important to know: 
1. Size of neural network. The current Neuron compiler release has a limitation in terms of the size of neural network it could effectively optimize for. The size of neural network is influenced by a number of factors including: a) type of neural network (CNN, LSTM, MLP) , b) number of layers, c) sizes of input (dimension of the tensors, batch size, ...). As a result, we limit the sizes of CNN models like ResNet to have an input size limit of 480x480 fp16/32, batch size=4; LSTM models like GNMT to have a time step limit of 900; MLP models like BERT to have input size limit of sequence length=128, batch=8.

2. Computer-vision object detection and segmentation models are not yet supported.

3. INT8 data type is not currently supported by the Neuron compiler.

4. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.


## Neuron SDK Release Notes Structure
The Neuron SDK is delivered through commonly used package mananagers (e.g. PIP, APT and YUM). These packages are then themselves packaged into Conda packages that are integrated into the AWS DLAMI for minimal developer overhead. 

The Neuron SDK release notes follow a similar structure, with the core improvements and known-issues reported in the release notes of the primary packages (e.g. Neuron-Runtime or Neuron-Compiler release notes), and additional release notes specific to the package-integration are reported through their dedicated release notes (e.g. Conda or DLAMI release notes).

This structure is shown below, with each also linking to the release notes for that package itself as well as showing the consitutuent included packages:


## [DLAMI Release Notes](./dlami-release-notes.md)

+ ### [Conda TensorFlow Release Notes](./conda/conda-tensorflow-neuron.md)

  + #### [TensorFlow-Neuron Release Notes](./tensorflow-neuron.md)
  + #### [Neuron-CC Release Notes](./neuron-cc.md)
  + #### [TensorBoard-Neuron Release Notes](./tensorboard-neuron.md)

+ ### [Conda MXNet Release Notes](./conda/conda-mxnet-neuron.md)

  + #### [MXNet-Neuron Release Notes](./mxnet-neuron.md)
  + #### [Neuron-CC Release Notes](./neuron-cc.md)
  
+ ### [Conda Torch Release Notes](./conda/conda-torch-neuron.md)

  + #### [Torch-Neuron Release Notes](./torch-neuron.md)
  + #### [Neuron-CC Release Notes](./neuron-cc.md)


+ ### [TensorFlow-Model-Server-Neuron Release Notes](./tensorflow-modelserver-neuron.md)
+ ### [Neuron-Runtime Release Notes](./neuron-runtime.md)
+ ### [Neuron-Tools Release Notes](./neuron-tools.md)


## [Neuron-CC Release Notes](./neuron-cc.md)
## [Neuron-Runtime Release Notes](./neuron-runtime.md)
## [Neuron-Tools Release Notes](./neuron-tools.md)
## [MXNet-Neuron Release Notes](./mxnet-neuron.md)
## [TensorFlow-Neuron Release Notes](./tensorflow-neuron.md)
## [Tensorboard-Neuron Release Notes](./tensorboard-neuron.md)
## [Torch-Neuron Release Notes](./torch-neuron.md)
