# Neuron Release Notes

The Neuron SDK is delivered through commonly used package mananagers (e.g. PIP, APT and YUM). These packages are then themselves packaged into Conda packages that are integrated into the AWS DLAMI for minimal developer overhead.

The Neuron SDK release notes follow a similar structure, with the core improvements and known-issues reported in the release notes of the primary packages (e.g. Neuron-Runtime or Neuron-Compiler release notes), and additional release notes specific to the package-integration are reported through their dedicated release notes (e.g. Conda or DLAMI release notes).

This structure is shown below, with each also linking to the release notes for that package itself as well as showing the consitutuent included packages:


## [DLAMI Release Notes](./dlami-release-notes.md)

+ ### [Conda Tensorflow Release Notes](./conda/conda-tensorflow-neuron.md)

  + #### [Tensorflow-Neuron Release Notes](./tensorflow-neuron.md)
  + #### [Neuron-CC Release Notes](./neuron-cc.md)
  + #### [TensorBoard-neuron Release Notes](./tensorboard-neuron.md)

+ ### [Conda MXNet Release Notes](./conda/conda-mxnet-neuron.md)

  + #### [MXNet-neuron Release Notes](./mxnet-neuron.md)
  + #### [Neuron-CC Release Notes](./neuron-cc.md)

+ ### [Tensorflow-Model-Server-Neuron Release Notes](./tensorflow-modelserver-neuron.md)
+ ### [Neuron-Runtime Release Notes](./neuron-runtime.md)
+ ### [Neuron-Tools Release Notes](./neuron-tools.md)


## [Neuron-CC Release Notes](./neuron-cc.md)
## [Neuron-Runtime Release Notes](./neuron-runtime.md)
## [Neuron-Tools Release Notes](./neuron-tools.md)
## [MXNet-Neuron Release Notes](./neuron-cc.md)
## [Tensorflow-Neuron Release Notes](./tensorflow-neuron.md)
## [Tensorboard-Neuron Release Notes](./tensorboard-neuron.md)

## Important to know: 
1. Size of neural network. The current Neuron compiler release has a limitation in terms of the size of neural network it could effectively optimize for. The size of neural network is influenced by a number of factors including: a) type of neural network (CNN, LSTM, MLP) , b) number of layers, c) sizes of input (dimension of the tensors, batch size, ...). As a result, we limit the sizes of CNN models like ResNet to have an input size limit of 480x480 fp/bf16, batch size=4; LSTM models like GNMT to have a time step limit of 900; MLP models like BERT to have input size limit of sequence length=128, batch=8.

2. Computer-vision object detection and segmentation models are not yet supported.

3. INT8 data type is not currently supported by the Neuron compiler.

4. Neuron does not support TensorFlow 2.

5. PyTorch support is coming soon, contact us at aws-neuron-support@amazon.com for more information.
