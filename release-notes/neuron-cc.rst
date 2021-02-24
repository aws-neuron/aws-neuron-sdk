.. _neuron-cc-rn:

Neuron Compiler Release Notes
=============================

.. contents:: Table of Contents
   :local:
   :depth: 1



Introduction
^^^^^^^^^^^^

This document lists the release notes for AWS Neuron compiler. The
Neuron Compiler is an ahead-of-time compiler that ensures Neuron will
optimally utilize the Inferentia chips.

Operator-support for each input format is provided directly from the
compiler.

::

   neuron-cc list-operators --framework {TENSORFLOW | MXNET | ONNX}

The supported operators are also listed here:

Tensorflow: :ref:`neuron-cc-ops-tensorflow`

Pytorch: :ref:`neuron-cc-ops-pytorch`

MXNet: :ref:`neuron-cc-ops-mxnet`

ONNX: :ref:`neuron-cc-ops-onnx`



Known issues and limitations - updated 02/24/2020
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Control flow** Neuron only supports control flow operators which
   are static at compile time. For example static length RNN, top-k,
   sort.
2. **Data layout** The Neuron compiler supports multiple data layout
   format (NCHW, NHWC, ...). Non-CNHW input/output data-layouts will
   require Neuron to insert additional *transpose* operations, causing a
   degradation in performance.
3. **Object detection models** SSD-300, YOLO v3, YOLO v4 are supported.
   More support is coming in future releases for RCNN-based models.
4. **Primary inputs in NeuronCore Pipeline mode** When a neural network
   is executed in NeuronCore Pipeline mode, only the first operator in a
   neural network can receive primary inputs from the host.
5. **Reduce data type** INT8 data type is not currently supported by the
   Neuron compiler.
6. **ONNX support** Support for ONNX models is limited. If generated
   from other frameworks then please use the native model directly.
7. **NeuronCore Pipeline:** NeuronCorePipeline mode provides low-latency
   and high-throughput for small batch sizes. We recommend to start
   testing with batch=1 and gradually increase batch size to fine tune
   your model throughput and latency performance. Currently there is a
   known issue with a compiler crash on batch size 32 using BERT-Base,
   sequence length=128, --neuroncore-pipeline-cores = 16, but this is
   not the optimal setting for that model.
8. **Conv2d operator** is mapped to Inferentia except for specific cases of extremely large tensors and specific parameters.

[1.2.7.0]
^^^^^^^^^

Date 2/24/2021

Summary
-------

Fix for CVE-2021-3177.

[1.2.2.0]
^^^^^^^^^

Date 1/30/2021

Summary
-------

Added suport for multiple new operators (see operators list) for Tensoflow and MXNET. Improved inference performance of language, object recognition models on single as well as multiple pipelined cores using neuroncore-pipeline. 

Major New Features
------------------

- The following models are now supported: Resnext 224x224, specific BERT variations applied to natural language processing and translation.

- A number of new operators is now supported on Inferentia, see the full lists :ref:`neuron-cc-ops-tensorflow`
 and :ref:`neuron-cc-ops-mxnet`

- Improved inference performance on yolov4 BERT base sequence 64 (on 16 pipelined cores) and openpose 184.

Resolved Issues
---------------

- Corrected a random failure to compile Resnet50 batch 5

- Corrected numerical inaccuracy in RSQRT and related operators for tensors with very large values ( > 1e20)






[1.1.7.0]
^^^^^^^^^

Date 12/23/2020

Summary
-------

Added suport for PyTorch Yolo V4, a new Framework-visible progress bar and improved inference performance. We continue to streamline the compiler usability by removing the need for options passed to control behavior. We are aiming to remove the need for such options entirely. Some tutorials have been updated to reflect this, but Resnet50 remains in need of these options to achieve maximum performance. Other useability improvements have been added, such as the compiler progress bar. As always, please let us know if there are other areas that we can improve.


Major New Features
------------------
- Pytorch Yolo V4 is now supported.

- Added a compiler progress bar when compilation is invoked from the Framework. This allows the user to see that progress continues as compilation proceeds, which is useful when compilation takes several minutes. A dot is printed every 20 seconds.

- Improved inference performance of Tensorflow BERT base seq 256 batch 3 by 10% .

Resolved Issues
---------------
- Resolved issue with depthwise convolution that manifests as a type check error 


.. _10240450:

[1.0.24045.0]
^^^^^^^^^^^^^

Date 11/17/2020

Summary
-------

Improved performance for pipelined execution (NeuronCore Pipeline).

Major New Features
------------------

-  NeuronCore Pipeline: improved partitioning to enable better static
   weights loading to cache.

Resolved Issues
---------------

-  --static-weights : No longer needed. As this is shown in some
   examples, please remove the option since the compiler now performs
   this auto-detection by default.

-  --num-neuroncores renamed to --neuroncore-pipeline-cores. The prior
   option form is still functional (backwards compatible) and will be
   removed in future releases.

-  --batching_en: Resolved compilation failure of ResNet50 FP32 batch 1
   on Ubuntu16 when "--batching_en" was used.


.. _neuron-cc-10206000:

[1.0.20600.0]
^^^^^^^^^^^^^

Date 9/22/2020

Summary
-------

Various performance improvements - both compilation time and inference
speed of object recognition models.

-  Compiler optimization '-O2' option is now enabled by default.

.. _major-new-features-1:

Major New Features
------------------

-  Improved inference performance of YOLO v3, YOLO v4, VGG16, SSD300.
   BERT models were improved by an additional 10%.

-  Modifed such that -O2 is now the default behavior and does not need
   to be specified. Note: some tutorials still explicitly specify "-O2".
   These will be modified in forthcoming updates.

.. _resolved-issues-1:

Resolved Issues
---------------

-  Sped up compilation of large models that were taking hours to sub-40
   minute.


.. _neuron-cc-10180010:

[1.0.18001.0]
^^^^^^^^^^^^^

Date 8/08/2020

.. _summary-1:

Summary
-------

Various performance improvements.

.. _major-new-features-1:

Major New Features
------------------

Improved performance of BERT base with -O2

.. _resolved-issues-1:

Resolved Issues
---------------

-  n/a

.. _neuron-cc-10179370:

[1.0.17937.0]
^^^^^^^^^^^^^

Date 8/05/2020

.. _summary-2:

Summary
-------

Various improvements.

.. _neuron-cc-10168610:

[1.0.16861.0]
^^^^^^^^^^^^^

Date 7/16/2020

.. _summary-3:

Summary
-------

This release has some bug fixes and some functional and performance
improvements to support compilation of several neural networks.

.. _major-new-features-2:

Major New Features
------------------

This release

-  Supports compilation of PoseNet, tested for images of specific
   resolutions upto 736.
-  Update the -O2 with a new memory allocator to reduce spilling to DRAM
-  Improved performance of the '-O2' on BERT base, and openpose pose
   network.

.. _resolved-issues-2:

Resolved Issues
---------------

-  Resolved compilation error in Vgg16 batch 1

Other Notes
-----------

-  Some versions of Inception network may fail to compile in Tensorflow
   on Ubuntu 16 in conda environment. The symptom is neuron-cc backend
   data race error. As a workaround use Ubuntu 18, Amazon Linux 2, or
   virtual env, or use neuron-cc with flag -O2.

.. _neuron-cc-10152750:

[1.0.15275.0]
^^^^^^^^^^^^^

Date 6/11/2020

.. _summary-4:

Summary
-------

This release has some bug fixes and some functional and performance
improvements to support compilation of several neural networks.

.. _major-new-features-3:

Major New Features
------------------

This release

-  Supports compilation of PoseNet for images of specific resolutions
   upto 400x400.
-  Improves performance of resnet152.
-  Supports a new command line option '-O2' that can help with handling
   of large tensor inputs for certain models.
-  increase NEFF versions to 1.0. This means new NEFFs compiled from
   this release forward are not compatible with older versions of Neuron
   Runtime prior to May, 2020 (1.0.6905.0) release. Please update the
   Neuron Runtime when using NEFF version 1.0.

.. _resolved-issues-3:

Resolved Issues
---------------

-  Compilation issues on prosotron encoder, decoder neural networks.

.. _other-notes-1:

Other Notes
-----------

Dependencies
------------

-  This version creates NEFF 1.0 thus may require update of neuron-rtd
   if older than May 2020 release.

dmlc_nnvm==1.0.2574.0 dmlc_topi==1.0.2574.0 dmlc_tvm==1.0.2574.0
inferentia_hwm==1.0.1362.0 islpy==2018.2

.. _neuron-cc-10126960:

[1.0.12696.0]
^^^^^^^^^^^^^

Date 5/11/2020

.. _summary-5:

Summary
-------

Bug fixes and some functional and performance improvements to several
neural networks.

.. _major-new-features-4:

Major New Features
------------------

-  This version supports compilation of unmodified Tensorflow BERT with
   batch size 1, 4, 6 for input sequence 128.
-  Improved Tensorflow BERT batch 4 sequence 128 performance to 45% of
   the accelerator peak (from 34%).
-  Support for MXNET BERT base batch 8 compilation
-  Support for TF Resnet152 batch 2 compilation
-  Most compiler messages are migrated from cout to logging mechanisms
   with verbosity control

.. _resolved-issues-4:

Resolved Issues
---------------

-  Fixed failure to compile unmodified Tensorflow BERT model for small
   batches

-  Fixed run-to-run-variability in OneHot operator implementation

-  Robustness improvements for ParallelWavenet and transformer decoder
   networks

.. _other-notes-2:

Other Notes
-----------

.. _dependencies-1:

Dependencies
------------

::

   dmlc_nnvm==1.0.2356.0
   dmlc_topi==1.0.2356.0
   dmlc_tvm==1.0.2356.0
   inferentia_hwm==1.0.1294.0
   islpy==2018.2

.. _neuron-cc-1094100:

[1.0.9410.0]
^^^^^^^^^^^^

Date 3/26/2020

.. _summary-6:

Summary
-------

Bug fixes and some functional and performance improvements to several
neural networks.

.. _major-new-features-5:

Major New Features
------------------

-  Support compilation of modified SSD-300
   (:ref:`tensorflow-ssd300`)
-  Improved inference performance in natural language processing
   networks (such as prosotron encoder) by 45%

.. _resolved-issues-5:

Resolved Issues
---------------

-  Eliminated redundant fp32 to bfloat16 cast on input and output
   tensors

Known issues and limitations
----------------------------

-  See previous releases.

.. _other-notes-3:

Other Notes
-----------

-  Added support for faster iteration on recurrent networks (aka
   auto-loop)

.. _dependencies-2:

Dependencies
------------

::

   dmlc_nnvm==1.0.2049.0 
   dmlc_topi==1.0.2049.0 
   pip install --upgrade dmlc_tvm==1.0.2049.0
   inferentia_hwm==1.0.897.0
   islpy==2018.2

.. _neuron-cc-1078780:

[1.0.7878.0]
^^^^^^^^^^^^

Date 2/27/2020

.. _summary-7:

Summary
-------

Bug fixes and minor performance improvements.

.. _major-new-features-6:

Major New Features
------------------

None

.. _resolved-issues-6:

Resolved Issues
---------------

-  Corrected image resize operator functionallity
-  Compiler internal enhancements made that will benefit models such as
   BERT

.. _known-issues-and-limitations-1:

Known issues and limitations
----------------------------

-  See previous releases.

.. _other-notes-4:

Other Notes
-----------

.. _dependencies-3:

Dependencies
------------

::

   dmlc_nnvm-1.0.1826.0
   dmlc_topi-1.0.1826.0
   dmlc_tvm-1.0.1826.0
   inferentia_hwm-1.0.897.0
   islpy-2018.2

.. _neuron-cc-1068010:

[1.0.6801.0]
^^^^^^^^^^^^

Date 1/27/2020

.. _summary-8:

Summary
-------

Bug fixes and some performance enhancement related to data movement for
BERT-type neural networks.

.. _major-new-features-7:

Major New Features
------------------

None

.. _resolved-issues-7:

Resolved Issues
---------------

-  Improved throughput for operators processed in the Neuron Runtime
   CPU. As an example: execution of 4 single NeuronCore NEFF models of
   ResNet50 v2 float16 batch = 5 in parallel on an inf1.1xlarge sped up
   by 30%.
-  Corrected shape handling in Gather(TensorFlow)/Take(MXNet) operators
   that are processed by the Neuron Runtime in the Neuron Runtime vCPU,
   which resolves a possible crash in Neuron Compiler when compiling
   models with these operators with some shapes.
-  Added support for TensorFlow *OneHot* operator (as a Neuron Runtime
   CPU operator).
-  Added more internal checking for compiler correctness with newly
   defined error messages for this case.

::

         “Internal ERROR: Data race between Op1 'Name1(...) [...]' and Op2 'Name2(...) [...]'”

-  Fixed out-of-memory issue introduced in 1.0.5939.0 such that some
   large models (BERT) compiled on instances with insufficient host
   memory would cause the runtime to crash with an invalid NEFF. This is
   actually a compiler error, but due to additional script layers
   wrapping this in the :ref:`tensorflow-bert-demo`, this would
   have likely been seen as a runtime error like this:

.. code:: bash

   2020-01-09 13:40:26.002594: E tensorflow/core/framework/op_segment.cc:54] Create kernel failed: Invalid argument: neff is invalid
   2020-01-09 13:40:26.002637: E tensorflow/core/common_runtime/executor.cc:642] Executor failed to create kernel. Invalid argument: neff is invalid
   [[{{node bert/NeuronOp}}]]

.. _known-issues-and-limitations-2:

Known issues and limitations
----------------------------

See previous release notes. Some tutorials show use of specific compiler
options and flags, these are needed to help provide guidance to the
compiler to achieve best performance in specific cases. Please do not
use in cases other than as shown in the specific tutorial as results may
not be defined. These options should be considered experimental and will
be removed over time.

.. _other-notes-5:

Other Notes
-----------

.. _dependencies-4:

Dependencies
------------

::

   dmlc_nnvm-1.0.1619.0
   dmlc_topi-1.0.1619.0
   dmlc_tvm-1.0.1619.0
   inferentia_hwm-1.0.839.0
   islpy-2018.2

.. _1059390:

[1.0.5939.0]
^^^^^^^^^^^^

Date 12/20/2019

.. _summary-9:

Summary
-------

Bug fixes and some performance enhancement for NeuronCore Pipeline.

.. _major-new-features-8:

Major New Features
------------------

.. _resolved-issues-8:

Resolved Issues
---------------

-  Fixed pipeline execution on more than 10 NeuronCores
-  Improved NeuronCores Pipeline execution by improving data exchange
   efficiency between NeuronCores
-  Added warning for unaligned memory access
-  Fixed handling of cast on input FP32 tensor
-  Improved handling of data layouts and transpose
-  Improved dead-code elimination
-  Improved efficiency of compute engine synchronization
-  Improved efficiency of data transfers within the Neuron code

.. _known-issues-and-limitations-3:

Known issues and limitations
----------------------------

See previous release notes. Some tutorials show use of specific compiler
options and flags, these are needed to help provide guidance to the
compiler to achieve best performance in specific cases. Please do not
use in cases other than as shown in the specific tutorial as results may
not be defined. These options should be considered experimental and will
be removed over time.

.. _other-notes-6:

Other Notes
-----------

.. _dependencies-5:

Dependencies
------------

-  dmlc_nnvm-1.0.1416.0

-  dmlc_topi-1.0.1416.0

-  dmlc_tvm-1.0.1416.0

-  inferentia_hwm-1.0.720.0

-  islpy-2018.2

.. _1053010:

[1.0.5301.0]
^^^^^^^^^^^^

Date 12/1/2019

.. _summary-10:

Summary
-------

.. _major-new-features-9:

Major New Features
------------------

.. _resolved-issues-9:

Resolved Issues
---------------

-  Added warning for unsupported operators and convolution sizes
-  Added warning for unsupported layout / upsampling
-  Added support for Relu6, AddV2, BatchMatmulV2 operators
-  Added support for default MXNet outputs in –io-config
-  Improved performance of batched inference for convolutional networks
-  Fixed MatMult column size 1
-  Fixed bf16 constant loading
-  Fixed Conv2D tile accumulation

.. _known-issues-and-limitations-4:

Known Issues and Limitations
----------------------------

See previous release notes. Resolved issues are shown in Resolved
Issues.

.. _other-notes-7:

Other Notes
-----------

Please install g++ on AMIs without g++ pre-installed (i.e. server AMIs):

.. code:: bash

   # Ubuntu
   sudo apt-get install -y g++

.. code:: bash

   # Amazon Linux
   sudo yum nstall -y gcc-c++

Supported Python versions:

-  3.5, 3.6, 3.7

Supported Linux distributions:

-  Ubuntu 16, Ubuntu 18, Amazon Linux 2

.. _dependencies-6:

Dependencies
------------

-  dmlc_nnvm-1.0.1328.0
-  dmlc_topi-1.0.1328.0
-  dmlc_tvm-1.0.1328.0
-  inferentia_hwm-1.0.674.0
-  islpy-2018.2

.. _1046800:

[1.0.4680.0]
^^^^^^^^^^^^

Date: 11/25/2019

.. _major-new-features-10:

Major new features
------------------

N/A, this is the first release.

.. _resolved-issues-10:

Resolved issues
---------------

N/A, this is the first release.

.. _known-issues-and-limitations-5:

Known issues and limitations
----------------------------

1. **Control flow** Inferentia has a limited support for control flow.
   In general, Neuron can only support control flow operators which are
   static at compile time, i.e. static length RNN, top-k, sort, ...
2. **Size of neural network** The size of neural network is influenced
   by a) type of neural network (CNN, LSTM, MLP) , b) number of layers,
   c) sizes of input (dimension of the tensors, batch size, ...). The
   current Neuron compiler release has a limitation in terms of the size
   of neural network it could effectively optimize. As a result, we
   limit CNN models (e.g. ResNet) to have an input size of up to 480x480
   FP16, batch size of 4; LSTM models (e.g. GNMT) are limited to a time
   step limit of up to 900; MLP models (like BERT) are limited up to
   sequence-length equal 128, batch=8.
3. **Data layout** The Neuron compiler supports multiple data layout
   format (NCHW, NHWC, ...). Non-CNHW input/output data-layouts will
   require Neuron to insert additional *transpose* operations, causing a
   degradation in performance.
4. **Object detection models** Computer-vision object detection and
   segmentation models are not supported by the current release.
5. **Reduce data type** INT8 data type is not currently supported by the
   Neuron compiler.
6. **Tensor residency** When a sub-graph that is executed on the host is
   communicating with a sub-graph that is executing on Neuron cores,
   tensors are copied via the communication queues between the host and
   Inferentia memory for each inference, which may result in end-to-end
   performance degradation.
7. **Primary inputs in NeuronCore Pipeline mode** When a neural network
   is executed in NeuronCore Pipeline mode, only the first operator in a
   neural network can receive primary inputs from the host.

.. _other-notes-8:

Other Notes
-----------

.. _dependencies-7:

Dependencies
------------

-  nnvm: dmlc_nnvm-1.0.1219.0
-  topi: dmlc_topi-1.0.1219.0
-  tvm: dmlc_tvm-1.0.1219.0
-  hwm: inferentia_hwm-1.0.602.0
-  islpy: islpy-2018.2+aws2018.x.73.0
