# Neuron Release Notes

## September 21, 2020 Release
This release improves performance of YOLO v3 and v4, VGG16, SSD300, and BERT.  As part of these improvements, Neuron Compiler doesn’t require any special compilation flags for most models. Details on how to use the prior optimizations are outlined in the neuron-cc [release notes](./neuron-cc.md).  

The release also improves operational deployments of large scale inference applications, with a session management agent incorporated into all supported ML Frameworks. A new Neuron Tool called neuron-monitor, will stream inference metrics helpful with monitoring one or more instances of Neuron Runtime.  Sample script for connecting neuron-monitor to your CloudWatch metrics is provided as well.  Read more about using neuron-monitor [here](../docs/neuron-tools/neuron-monitor-user-guide.md).

## August 19, 2020 Release
Bug fix for an error reporting issue with the Neuron Runtime.  Previous versions of the runtime were only reporting uncorrectable errors on half of the dram per Inferentia.  Other Neuron packages are not changed.

## August 8, 2020 Release
This release of the Neuron SDK delivers performance enhancements for the BERT Base model. Sequence lengths including  128, 256 and 512 were found to have best performance at batch size 6, 3 and 1 respectively using publically available versions of both Pytorch (1.5.x) and Tensorflow-based (1.15.x) models. The compiler option "-O2" was used in all cases. 

A new Kubernetes scheduler extension is included in this release to improve pod scheduling on inf1.6xlarge and inf1.24xlarge instance sizes.  Details on how the scheduler works and how to apply the scheduler can be found [here](../docs/neuron-container-tools/k8s-neuron-scheduler.md).  Check the [Neuron K8 release notes](./neuron-k8.md) for details changes to k8 components going forward.


## August 4, 2020 Release
Bug fix for a latent issue caused by a race condition in Neuron Runtime leading to possible crashes. The crash was observed under stress load conditons.  All customers are encouraged to update the latest Neuron Runtime package (aws-neuron-runtime), version 1.0.8813.0 or newer.  Other Neuron packages are being updated as well, but are to be considered non-critical updates.

## July 16, 2020 Release
This release of Neuron SDK adds support for the OpenPose (posenet) Neural Network. An example of using Openpose for end to end inference is available [here](../src/examples/tensorflow/openpose_demo/).
  
A new PyTorch auto-partitioner feature now automatically builds a Neuron specific graph representation of PyTorch models. The key benefit of this feature is automatic partitioning the model graph to run the supported operators on the NeuronCores and the rest on the host. PyTorch auto-partitioner is enabled by default with ability to disable if a manual partition is needed. More details [here](torch-neuron.md). The release also includes various bug fixes and increased operator support. 

## Important to know: 
1. This update moves the supported version for PyTorch to the current release (PyTorch 1.5.1)
2. This release supports Python 3.7 Conda packages in addition to Python 3.6 Conda packages

## June 18, 2020 Release
Point fix an error related to yum downgrade/update of Neuron Runtime packages.  The prior release fails to successfully downgrade/update Neuron Runtime Base package and Neuron Runtime package when using Yum on Amazon Linux 2.


Please remove and then install both packages on AL2 using these commands:
```
# Amazon Linux 2
sudo yum remove aws-neuron-runtime-base
sudo yum remove aws-neuron-runtime
sudo yum install aws-neuron-runtime-base
sudo yum install aws-neuron-runtime
```


## Jun 11, 2020 Release

This Neuron release provides support for the recent launch of EKS for Inf1 instance types and numerous other improvements.  More details about how to use EKS with the Neuron SDK can be found in AWS documentation [here](https://docs.aws.amazon.com/eks/latest/userguide/inferentia-support.html).

This release adds initial support for OpenPose PoseNet for images with resolutions upto 400x400. 

This release also adds a '-O2' option to the Neuron Compiler. '-O2' can help with handling of large tensor inputs.  

In addition the Neuron Compiler increments the version of the compiled artifacts, called "NEFF", to version 1.0. Neuron Runtime versions earlier than the 1.0.6905.0 release in May 2020 will not be able to execute NEFFs compiled from this release forward. Please see [Neuron Runtime Release Notes](./neuron-runtime.md#neff-support-table) for compatibility.

Stay up to date on future improvements and new features by following the [Neuron SDK Roadmap](https://github.com/aws/aws-neuron-sdk/projects/2).

Refer to the detailed release notes for more information on each Neuron component. 

## Important to know: 
1. Size of neural network. The current Neuron compiler release has a limitation in terms of the size of neural network it could effectively optimize for. The size of neural network is influenced by a number of factors including: a) type of neural network (CNN, LSTM, MLP) , b) number of layers, c) sizes of input (dimension of the tensors, batch size, ...). Using the Neuron Compiler '-O2' option can help with handling of large tensor inputs for some models. If not used, Neuron limits the size of CNN models like ResNet to an input size of 480x480 fp16/32, batch size=4; LSTM models like GNMT to have a time step limit of 900; MLP models like BERT to have input size limit of sequence length=128, batch=8.

2. INT8 data type is not currently supported by the Neuron compiler.

3. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.

## May 15, 2020 Release
Point fix an error related to installation of the Neuron Runtime Base package.  The prior release fails to successfully start Neuron Discovery when the Neuron Runtime package is not also installed.  This scenario of running Neuron Discovery alone is critical to users of Neuron in container environments.  

Please update the aws-neuron-runtime-base package:
```
# Ubuntu 18 or 16:
sudo apt-get update
sudo apt-get install aws-neuron-runtime-base

# Amazon Linux, Centos, RHEL
sudo yum update
sudo yum install aws-neuron-runtime-base
```


## May 11, 2020 Release

This release provides additional throughput improvements to running inference on a variety of models; for example BERTlarge throughput has improved by an additional 35% compared to the previous release and with peak thoughput of 360 seq/second on inf1.xlarge (more details [here](./../src/examples/tensorflow/bert_demo/README.md)).

In addition to the performance boost, this release adds PyTorch, and MXNet framework support for BERT models, as well as expands container support in preparation to an upcoming EKS launch.

We continue to work on new features and improving performance further, to stay up to date follow this repository and our [Neuron roadmap](https://github.com/aws/aws-neuron-sdk/projects/2).

Refer to the detailed release notes for more information for each Neuron component. 

## Important to know: 
1. Size of neural network. The current Neuron compiler release has a limitation in terms of the size of neural network it could effectively optimize for. The size of neural network is influenced by a number of factors including: a) type of neural network (CNN, LSTM, MLP) , b) number of layers, c) sizes of input (dimension of the tensors, batch size, ...). As a result, we limit the sizes of CNN models like ResNet to have an input size limit of 480x480 fp16/32, batch size=4; LSTM models like GNMT to have a time step limit of 900; MLP models like BERT to have input size limit of sequence length=128, batch=8.

2. INT8 data type is not currently supported by the Neuron compiler.

3. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.


## Mar 26, 2020 Release

This release supports a variant of the SSD object detection network, a SSD inference demo is available [here](../src/examples/tensorflow/ssd300_demo) 

This release also enhances our Tensorboard support to enable CPU-node visibility. 

Refer to the detailed release notes for more information for each neuron component. 

## Important to know: 
1. Size of neural network. The current Neuron compiler release has a limitation in terms of the size of neural network it could effectively optimize for. The size of neural network is influenced by a number of factors including: a) type of neural network (CNN, LSTM, MLP) , b) number of layers, c) sizes of input (dimension of the tensors, batch size, ...). As a result, we limit the sizes of CNN models like ResNet to have an input size limit of 480x480 fp16/32, batch size=4; LSTM models like GNMT to have a time step limit of 900; MLP models like BERT to have input size limit of sequence length=128, batch=8.

2. INT8 data type is not currently supported by the Neuron compiler.

3. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.


## Feb 27, 2020 Release

This release improves performance throughput by up to 10%, for example ResNet-50 on inf1.xlarge has increased from 1800 img/sec to 2040 img/sec, Neuron logs include more detailed messages and various bug fixes. Refer to the detailed release notes for more details.

We continue to work on new features and improving performance further, to stay up to date follow this repository, and watch the [AWS Neuron developer forum](https://forums.aws.amazon.com/forum.jspa?forumID=355).

## Important to know: 
1. Size of neural network. The current Neuron compiler release has a limitation in terms of the size of neural network it could effectively optimize for. The size of neural network is influenced by a number of factors including: a) type of neural network (CNN, LSTM, MLP) , b) number of layers, c) sizes of input (dimension of the tensors, batch size, ...). As a result, we limit the sizes of CNN models like ResNet to have an input size limit of 480x480 fp16/32, batch size=4; LSTM models like GNMT to have a time step limit of 900; MLP models like BERT to have input size limit of sequence length=128, batch=8.

2. Computer-vision object detection and segmentation models are not yet supported.

3. INT8 data type is not currently supported by the Neuron compiler.

4. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.


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
