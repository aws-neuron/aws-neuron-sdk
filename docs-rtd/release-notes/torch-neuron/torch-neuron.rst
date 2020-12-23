.. _pytorch-neuron-rn:

PyTorch Neuron release notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This document lists the release notes for the Pytorch-Neuron package.

Known Issues and Limitations - Updated 12/23/2020
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following are not torch-neuron limitations, but may impact models
you can successfully torch.neuron.trace

-  Torchvision has dropped support for Python 3.5
-  HuggingFace transformers has dropped support for Python 3.5
-  There are known issues when customer use a mixture of conda and pip
   packages. We strongly recommend that you install aws neuron conda
   packages if you are using a conda environment, and use the pip
   installation if you are working in a base python environment (or a
   native python virtual environment) as recommended in our installation
   notes at :ref:`neuron-install-conda-packages`
   -  When using the most recent DLAMI and 'aws_neuron_pytorch_p36' you may
   see lower performance than expected in :ref:`pytorch-getting-started`.
   This issue will be corrected in the v37 DLAMI release.
-  aten::max only correctly implements the simplest versions of that
   operator, the variants that return a tuple with arg max now return
   NotImplementedError during compilation
-  There is a dependency between versions of torchvision and the torch package that customers should be aware of when compiling torchvision models.  These dependency rules can be managed through pip.  At the time of writing torchvision==0.6.1 matched the torch==1.5.1 release, and torchvision==0.8.2 mathced the torch==1.7.1 release


[1.1.7.0]
^^^^^^^^^^^^

Date: 12/23/2020

Summary
-------

-  torch.neuron.trace behavior will now throw a RuntimeError in the case that no operators are compiled for neuron hardware
-  torch.neuron.trace will now display compilation progress indicators (dots) as default behavior (neuron-cc must updated to the December release to greater to see this feature)
-  Added new operator support. Please see :ref:`neuron-cc-ops-pytorch` for the complete list of operators.
-  Extended the BERT pretrained tutorial to demonstrate execution on multiple cores and batch modification, updated the tutorial to accomodate changes in the Hugging Face Transformers code for version 4.0
-  Added a tutorial for torch-serve which extends the BERT tutorial
-  Added support for PyTorch 1.7

.. _neuron-torch-1019780:

[1.0.1978.0]
^^^^^^^^^^^^

Date: 11/17/2020

Summary
-------

-  Fixed bugs in comparison operators, and added remaining variantes
   (eq, ne, gt, ge, lt, le)
-  Added support for prim::PythonOp - note that this must be run on CPU
   and not Neuron. We recommend you replace this code with PyTorch
   operators if possible
-  Support for a series of new operators. Please see :ref:`neuron-cc-ops-pytorch` for the
   complete list of operators.
-  Performance improvements to the runtime library
-  Correction of a runtime library bug which caused models with large
   tensors to generate incorrect results in some cases



.. _neuron-torch-1017210:

[1.0.1721.0]
^^^^^^^^^^^^

Date: 09/22/2020

Summary
-------

-  Various minor improvements to the Pytorch autopartitioner feature
-  Support for the operators aten::constant_pad_nd, aten::meshgrid
-  Improved performance on various torchvision models. Of note are
   resnet50 and vgg16

.. _neuron-torch-1015320:

[1.0.1532.0]
^^^^^^^^^^^^

Date: 08/08/2020

.. _summary-1:

Summary
-------

-  Various minor improvements to the Pytorch autopartitioner feature
-  Support for the aten:ones operator

.. _neuron-torch-1015220:

[1.0.1522.0]
^^^^^^^^^^^^

Date: 08/05/2020

.. _summary-2:

Summary
-------

Various minor improvements.

.. _neuron-torch-1013860:

[1.0.1386.0]
^^^^^^^^^^^^

Date: 07/16/2020

.. _summary-3:

Summary
-------

This release adds auto-partitioning, model analysis and PyTorch 1.5.1
support, along with a number of new operators

Major New Features
------------------

-  Support for Pytorch 1.5.1
-  Introduce an automated operator device placement mechanism in
   torch.neuron.trace to run sub-graphs that contain operators that are
   not supported by the neuron compiler in native PyTorch. This new
   mechanism is on by default and can be turned off by adding argument
   fallback=False to the compiler arguments.
-  Model analysis to find supported and unsupported operators in a model

Resolved Issues
---------------

.. _neuron-torch-1011680:

[1.0.1168.0]
^^^^^^^^^^^^

Date 6/11/2020

.. _summary-4:

Summary
-------

.. _major-new-features-1:

Major New Features
------------------

.. _resolved-issues-1:

Resolved Issues
---------------

Known Issues and Limitations
----------------------------

.. _neuron-torch-1010010:

[1.0.1001.0]
^^^^^^^^^^^^

Date: 5/11/2020

.. _summary-5:

Summary
-------

Additional PyTorch operator support and improved support for model
saving and reloading.

.. _major-new-features-2:

Major New Features
------------------

-  Added Neuron Compiler support for a number of previously unsupported
   PyTorch operators. Please see :ref:`neuron-cc-ops-pytorch`for the
   complete list of operators.
-  Add support for torch.neuron.trace on models which have previously
   been saved using torch.jit.save and then reloaded.

.. _resolved-issues-2:

Resolved Issues
---------------

.. _known-issues-and-limitations-1:

Known Issues and Limitations
----------------------------

.. _neuron-torch-108250:

[1.0.825.0]
^^^^^^^^^^^

Date: 3/26/2020

.. _summary-6:

Summary
-------

.. _major-new-features-3:

Major New Features
------------------

.. _resolved-issues-3:

Resolved Issues
---------------

.. _known-issues-and-limitations-2:

Known Issues and limitations
----------------------------

.. _neuron-torch-107630:

[1.0.763.0]
^^^^^^^^^^^

Date: 2/27/2020

.. _summary-7:

Summary
-------

Added Neuron Compiler support for a number of previously unsupported
PyTorch operators. Please see :ref:`neuron-cc-ops-pytorch` for the complete
list of operators.

.. _major-new-features-4:

Major new features
------------------

-  None

.. _resolved-issues-4:

Resolved issues
---------------

-  None

.. _neuron-torch-106720:

[1.0.672.0]
^^^^^^^^^^^

Date: 1/27/2020

.. _summary-8:

Summary
-------

.. _major-new-features-5:

Major new features
------------------

.. _resolved-issues-5:

Resolved issues
---------------

-  Python 3.5 and Python 3.7 are now supported.

.. _known-issues-and-limitations-3:

Known issues and limitations
----------------------------

Other Notes
-----------

.. _neuron-torch-106270:

[1.0.627.0]
^^^^^^^^^^^

Date: 12/20/2019

.. _summary-9:

Summary
-------

This is the initial release of torch-neuron. It is not distributed on
the DLAMI yet and needs to be installed from the neuron pip repository.

Note that we are currently using a TensorFlow as an intermediate format
to pass to our compiler. This does not affect any runtime execution from
PyTorch to Neuron Runtime and Inferentia. This is why the neuron-cc
installation must include [tensorflow] for PyTorch.

.. _major-new-features-6:

Major new features
------------------

.. _resolved-issues-6:

Resolved issues
---------------

.. _known-issues-and-limitations-4:

Known issues and limitations
----------------------------

Models TESTED
-------------

The following models have successfully run on neuron-inferentia systems

1. SqueezeNet
2. ResNet50
3. Wide ResNet50

Pytorch Serving
---------------

In this initial version there is no specific serving support. Inference
works correctly through Python on Inf1 instances using the neuron
runtime. Future releases will include support for production deployment
and serving of models

Profiler support
----------------

Profiler support is not provided in this initial release and will be
available in future releases

Automated partitioning
----------------------

Automatic partitioning of graphs into supported and non-supported
operations is not currently supported. A tutorial is available to
provide guidance on how to manually parition a model graph. Please see
:ref:`pytorch-manual-partitioning-jn-tutorial`

PyTorch dependency
------------------

Currently PyTorch support depends on a Neuron specific version of
PyTorch v1.3.1. Future revisions will add support for 1.4 and future
releases.

Trace behavior
--------------

In order to trace a model it must be in evaluation mode. For examples
please see :ref:`pytorch-getting-started`

Six pip package is required
---------------------------

The Six package is required for the torch-neuron runtime, but it is not
modeled in the package dependencies. This will be fixed in a future
release.

Multiple NeuronCore support
---------------------------

If the num-neuroncores options is used the number of cores must be
manually set in the calling shell environment variable for compilation
and inference.

For example: Using the keyword argument
compiler_args=['—num-neuroncores', '4'] in the trace call, requires
NEURONCORE_GROUP_SIZES=4 to be set in the environment at compile time
and runtime

CPU execution
-------------

At compilation time a constant output is generated for the purposes of
tracing. Running inference on a non neuron instance will generate
incorrect results. This must not be used. The following error message is
generated to stderr:

::

   Warning: Tensor output are ** NOT CALCULATED ** during CPU execution and only
   indicate tensor shape

.. _other-notes-1:

Other notes
-----------

-  Python version(s) supported:

   -  3.6

-  Linux distribution supported:

   -  DLAMI Conda 26.0 and beyond running on Ubuntu 16, Ubuntu 18,
      Amazon Linux 2 (using Python 3.6 Conda environments)
   -  Other AMIs based on Ubuntu 16, 18
   -  For Amazon Linux 2 please install Conda and use Python 3.6 Conda
      environment
