.. _main-rn:

Neuron Release Notes
====================

.. contents:: Table of Contents
   :local:
   :depth: 1

.. _03-04-2021-rn:

March 4, 2021 Release (Patch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This release include bug fixes and minor enhancements to the Neuron Runtime and Tools. 


February 24, 2021 Release (Patch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This release updates all Neuron packages and libraries in response to the Python Secutity issue CVE-2021-3177 as described here: https://nvd.nist.gov/vuln/detail/CVE-2021-3177. This vulnerability potentially exists in multiple versions of Python including 3.5, 3.6, 3.7. Python is used by various components of Neuron, including the Neuron compiler as well as Machine Learning frameworks including TensorFlow, PyTorch and MXNet. It is recommended that the Python interpreters used in any AMIs and containers used with Neuron are also updated. 

Python 3.5 reached `end-of-life <https://devguide.python.org/devcycle/?highlight=python%203.5%20end%20of%20life#end-of-life-branches>`_, from this release Neuron packages will not support Python 3.5.
Users should upgrade to latest DLAMI or upgrade to a newer Python versions if they are using other AMI.


January 30, 2021 Release
^^^^^^^^^^^^^^^^^^^^^^^^

This release continues to improves the NeuronCore Pipeline performance for BERT models. For example, running BERT Base with the the neuroncore-pipeline-cores compile option, at batch=3, seqlen=32 using 16 Neuron Cores, results in throughput of up to  5340 sequences per second and P99 latency of 9ms using Tensorflow Serving. 

This release also adds operator support and performance improvements for the PyTorch based DistilBert model for sequence classification.


December 23, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^^^^

This release introduces a PyTorch 1.7 based torch-neuron package as a part of the Neuron SDK. Support for PyTorch model serving with TorchServe 0.2 is added and will be demonstrated with a tutorial. This release also provides an example tutorial for PyTorch based Yolo v4 model for Inferentia. 

To aid visibility into compiler activity, the Neuron-extended Frameworks TensorFlow and PyTorch will display a new compilation status indicator that prints a dot (.) every 20 seconds to the console as compilation is executing. 

Important to know:
------------------

1. This update continues to support the torch-neuron version of PyTorch 1.5.1 for backwards compatibility.
2. As Python 3.5 reached end-of-life in October 2020, and many packages including TorchVision and Transformers have
stopped support for Python 3.5, we will begin to stop supporting Python 3.5 for frameworks, starting with
PyTorch-Neuron version :ref:`neuron-torch-11170` in this release. You can continue to use older versions with Python 3.5.

November 17, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^^^^

This release improves NeuronCore Pipeline performance. For example,
running BERT Small, batch=4, seqlen=32 using 4 Neuron Cores, results in
throughput of up to 7000 sequences per second and P99 latency of 3ms
using Tensorflow Serving.

Neuron tools updated the NeuronCore utilization metric to include all
inf1 compute engines and DMAs. Added a new neuron-monitor example that
connects to Grafana via Prometheus. We've added a new sample script
which exports most of neuron-monitor's metrics to a Prometheus
monitoring server. Additionally, we also provided a sample Grafana
dashboard. More details at :ref:`neuron-tools`.

ONNX support is limited and from this version onwards we are not
planning to add any additional capabilities to ONNX. We recommend
running models in TensorFlow, PyTorch or MXNet for best performance and
support.

October 22, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^^^

This release adds a Neuron kernel mode driver (KMD). The Neuron KMD
simplifies Neuron Runtime deployments by removing the need for elevated
privileges, improves memory management by removing the need for huge
pages configuration, and eliminates the need for running neuron-rtd as a
sidecar container. Documentation throughout the repo has been updated to
reflect the new support. The new Neuron KMD is backwards compatible with
prior versions of Neuron ML Frameworks and Compilers - no changes are
required to existing application code.

More details in the Neuron Runtime release notes at :ref:`neuron-runtime`.

September 22, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^^^^^

This release improves performance of YOLO v3 and v4, VGG16, SSD300, and
BERT. As part of these improvements, Neuron Compiler doesn’t require any
special compilation flags for most models. Details on how to use the
prior optimizations are outlined in the neuron-cc :ref:`neuron-cc-rn`.

The release also improves operational deployments of large scale
inference applications, with a session management agent incorporated
into all supported ML Frameworks and a new neuron tool called
neuron-monitor allows to easily scale monitoring of large fleets of
Inference applications. A sample script for connecting neuron-monitor to
Amazon CloudWatch metrics is provided as well. Read more about using
neuron-monitor :ref:`neuron-monitor-ug`.

August 19, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^^

Bug fix for an error reporting issue with the Neuron Runtime. Previous
versions of the runtime were only reporting uncorrectable errors on half
of the dram per Inferentia. Other Neuron packages are not changed.

August 8, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^

This release of the Neuron SDK delivers performance enhancements for the
BERT Base model. Sequence lengths including 128, 256 and 512 were found
to have best performance at batch size 6, 3 and 1 respectively using
publically available versions of both Pytorch (1.5.x) and
Tensorflow-based (1.15.x) models. The compiler option "-O2" was used in
all cases.

A new Kubernetes scheduler extension is included in this release to
improve pod scheduling on inf1.6xlarge and inf1.24xlarge instance sizes.
Details on how the scheduler works and how to apply the scheduler can be
found :ref:`neuron-k8-scheduler-ext`.
Check the :ref:`neuron-k8-rn` for details
changes to k8 components going forward.

August 4, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^

Bug fix for a latent issue caused by a race condition in Neuron Runtime
leading to possible crashes. The crash was observed under stress load
conditons. All customers are encouraged to update the latest Neuron
Runtime package (aws-neuron-runtime), version 1.0.8813.0 or newer. Other
Neuron packages are being updated as well, but are to be considered
non-critical updates.

July 16, 2020 Release
^^^^^^^^^^^^^^^^^^^^^

This release of Neuron SDK adds support for the OpenPose (posenet)
Neural Network. An example of using Openpose for end to end inference is
available :ref:`tensorflow-openpose`.

A new PyTorch auto-partitioner feature now automatically builds a Neuron
specific graph representation of PyTorch models. The key benefit of this
feature is automatic partitioning the model graph to run the supported
operators on the NeuronCores and the rest on the host. PyTorch
auto-partitioner is enabled by default with ability to disable if a
manual partition is needed. More details :ref:`neuron-pytorch`. The
release also includes various bug fixes and increased operator support.

Important to know:
------------------

1. This update moves the supported version for PyTorch to the current
   release (PyTorch 1.5.1)
2. This release supports Python 3.7 Conda packages in addition to Python
   3.6 Conda packages

June 18, 2020 Release
^^^^^^^^^^^^^^^^^^^^^

Point fix an error related to yum downgrade/update of Neuron Runtime
packages. The prior release fails to successfully downgrade/update
Neuron Runtime Base package and Neuron Runtime package when using Yum on
Amazon Linux 2.

Please remove and then install both packages on AL2 using these
commands:

::

   # Amazon Linux 2
   sudo yum remove aws-neuron-runtime-base
   sudo yum remove aws-neuron-runtime
   sudo yum install aws-neuron-runtime-base
   sudo yum install aws-neuron-runtime

Jun 11, 2020 Release
^^^^^^^^^^^^^^^^^^^^

This Neuron release provides support for the recent launch of EKS for
Inf1 instance types and numerous other improvements. More details about
how to use EKS with the Neuron SDK can be found in AWS documentation
`here <https://docs.aws.amazon.com/eks/latest/userguide/inferentia-support.html>`__.

This release adds initial support for OpenPose PoseNet for images with
resolutions upto 400x400.

This release also adds a '-O2' option to the Neuron Compiler. '-O2' can
help with handling of large tensor inputs.

In addition the Neuron Compiler increments the version of the compiled
artifacts, called "NEFF", to version 1.0. Neuron Runtime versions
earlier than the 1.0.6905.0 release in May 2020 will not be able to
execute NEFFs compiled from this release forward. Please see :ref:`neff-support-table` for
compatibility.

Stay up to date on future improvements and new features by following the
`Neuron SDK Roadmap <https://github.com/aws/aws-neuron-sdk/projects/2>`__.

Refer to the detailed release notes for more information on each Neuron
component.

.. _important-to-know-1:

Important to know:
------------------

1. Size of neural network. The current Neuron compiler release has a
   limitation in terms of the size of neural network it could
   effectively optimize for. The size of neural network is influenced by
   a number of factors including: a) type of neural network (CNN, LSTM,
   MLP) , b) number of layers, c) sizes of input (dimension of the
   tensors, batch size, ...). Using the Neuron Compiler '-O2' option can
   help with handling of large tensor inputs for some models. If not
   used, Neuron limits the size of CNN models like ResNet to an input
   size of 480x480 fp16/32, batch size=4; LSTM models like GNMT to have
   a time step limit of 900; MLP models like BERT to have input size
   limit of sequence length=128, batch=8.

2. INT8 data type is not currently supported by the Neuron compiler.

3. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.

May 15, 2020 Release
^^^^^^^^^^^^^^^^^^^^

Point fix an error related to installation of the Neuron Runtime Base
package. The prior release fails to successfully start Neuron Discovery
when the Neuron Runtime package is not also installed. This scenario of
running Neuron Discovery alone is critical to users of Neuron in
container environments.

Please update the aws-neuron-runtime-base package:

::

   # Ubuntu 18 or 16:
   sudo apt-get update
   sudo apt-get install aws-neuron-runtime-base

   # Amazon Linux, Centos, RHEL
   sudo yum update
   sudo yum install aws-neuron-runtime-base

May 11, 2020 Release
^^^^^^^^^^^^^^^^^^^^

This release provides additional throughput improvements to running
inference on a variety of models; for example BERTlarge throughput has
improved by an additional 35% compared to the previous release and with
peak thoughput of 360 seq/second on inf1.xlarge (more details :ref:`tensorflow-bert-demo` ).

In addition to the performance boost, this release adds PyTorch, and
MXNet framework support for BERT models, as well as expands container
support in preparation to an upcoming EKS launch.

We continue to work on new features and improving performance further,
to stay up to date follow this repository and our `Neuron roadmap <https://github.com/aws/aws-neuron-sdk/projects/2>`__.

Refer to the detailed release notes for more information for each Neuron
component.

.. _important-to-know-2:

Important to know:
------------------

1. Size of neural network. The current Neuron compiler release has a
   limitation in terms of the size of neural network it could
   effectively optimize for. The size of neural network is influenced by
   a number of factors including: a) type of neural network (CNN, LSTM,
   MLP) , b) number of layers, c) sizes of input (dimension of the
   tensors, batch size, ...). As a result, we limit the sizes of CNN
   models like ResNet to have an input size limit of 480x480 fp16/32,
   batch size=4; LSTM models like GNMT to have a time step limit of 900;
   MLP models like BERT to have input size limit of sequence length=128,
   batch=8.

2. INT8 data type is not currently supported by the Neuron compiler.

3. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.

Mar 26, 2020 Release
^^^^^^^^^^^^^^^^^^^^

This release supports a variant of the SSD object detection network, a
SSD inference demo is available :ref:`tensorflow-ssd300`

This release also enhances our Tensorboard support to enable CPU-node
visibility.

Refer to the detailed release notes for more information for each neuron
component.

.. _important-to-know-3:

Important to know:
------------------

1. Size of neural network. The current Neuron compiler release has a
   limitation in terms of the size of neural network it could
   effectively optimize for. The size of neural network is influenced by
   a number of factors including: a) type of neural network (CNN, LSTM,
   MLP) , b) number of layers, c) sizes of input (dimension of the
   tensors, batch size, ...). As a result, we limit the sizes of CNN
   models like ResNet to have an input size limit of 480x480 fp16/32,
   batch size=4; LSTM models like GNMT to have a time step limit of 900;
   MLP models like BERT to have input size limit of sequence length=128,
   batch=8.

2. INT8 data type is not currently supported by the Neuron compiler.

3. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.

Feb 27, 2020 Release
^^^^^^^^^^^^^^^^^^^^

This release improves performance throughput by up to 10%, for example
ResNet-50 on inf1.xlarge has increased from 1800 img/sec to 2040
img/sec, Neuron logs include more detailed messages and various bug
fixes. Refer to the detailed release notes for more details.

We continue to work on new features and improving performance further,
to stay up to date follow this repository, and watch the `AWS Neuron
developer
forum <https://forums.aws.amazon.com/forum.jspa?forumID=355>`__.

.. _important-to-know-4:

Important to know:
------------------

1. Size of neural network. The current Neuron compiler release has a
   limitation in terms of the size of neural network it could
   effectively optimize for. The size of neural network is influenced by
   a number of factors including: a) type of neural network (CNN, LSTM,
   MLP) , b) number of layers, c) sizes of input (dimension of the
   tensors, batch size, ...). As a result, we limit the sizes of CNN
   models like ResNet to have an input size limit of 480x480 fp16/32,
   batch size=4; LSTM models like GNMT to have a time step limit of 900;
   MLP models like BERT to have input size limit of sequence length=128,
   batch=8.

2. Computer-vision object detection and segmentation models are not yet
   supported.

3. INT8 data type is not currently supported by the Neuron compiler.

4. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.

Jan 28, 2020 Release
^^^^^^^^^^^^^^^^^^^^

This release brings significant throughput improvements to running
inference on a variety of models; for example Resnet50 throughput is
increased by 63% (measured 1800 img/sec on inf1.xlarge up from 1100/sec,
and measured 2300/sec on inf1.2xlarge). BERTbase throughput has improved
by 36% compared to the re:Invent launch (up to 26100seq/sec from
19200seq/sec on inf1.24xlarge), and BERTlarge improved by 15% (230
seq/sec, compared to 200 running on inf1.2xlarge). In addition to the
performance boost, this release includes various bug fixes as well as
additions to the GitHub with  :ref:`neuron-fundamentals`
diving deep on how Neuron performance features work and overall improved
documentation following customer input.

We continue to work on new features and improving performance further,
to stay up to date follow this repository, and watch the `AWS Neuron
developer
forum <https://forums.aws.amazon.com/forum.jspa?forumID=355>`__.

.. _important-to-know-5:

Important to know:
------------------

1. Size of neural network. The current Neuron compiler release has a
   limitation in terms of the size of neural network it could
   effectively optimize for. The size of neural network is influenced by
   a number of factors including: a) type of neural network (CNN, LSTM,
   MLP) , b) number of layers, c) sizes of input (dimension of the
   tensors, batch size, ...). As a result, we limit the sizes of CNN
   models like ResNet to have an input size limit of 480x480 fp16/32,
   batch size=4; LSTM models like GNMT to have a time step limit of 900;
   MLP models like BERT to have input size limit of sequence length=128,
   batch=8.

2. Computer-vision object detection and segmentation models are not yet
   supported.

3. INT8 data type is not currently supported by the Neuron compiler.

4. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.

Neuron SDK Release Notes Structure
----------------------------------

The Neuron SDK is delivered through commonly used package mananagers
(e.g. PIP, APT and YUM). These packages are then themselves packaged
into Conda packages that are integrated into the AWS DLAMI for minimal
developer overhead.

The Neuron SDK release notes follow a similar structure, with the core
improvements and known-issues reported in the release notes of the
primary packages (e.g. Neuron-Runtime or Neuron-Compiler release notes),
and additional release notes specific to the package-integration are
reported through their dedicated release notes (e.g. Conda or DLAMI
release notes).
