.. _pytorch-neuron-rn:

PyTorch Neuron (``torch-neuron``) release notes
===============================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for the Pytorch-Neuron package.



Known Issues and Limitations - Updated 11/23/2022
-------------------------------------------------

Min & Max Accuracy
~~~~~~~~~~~~~~~~~~

The index outputs of the ``aten::argmin``, ``aten::argmax``, ``aten::min``, and
``aten::max`` operator implementations are sensitive to precision. For models
that contain these operators and have ``float32`` inputs, we recommend using the
``--fp32-cast=matmult --fast-math no-fast-relayout`` compiler option to avoid
numerical imprecision issues. Additionally, the ``aten::min`` and ``aten::max``
operator implementations do not currently support ``int64`` inputs when
``dim=0``. For more information on precision and performance-accuracy tuning,
see :ref:`neuron-cc-training-mixed-precision`.

Python 3.5
~~~~~~~~~~

If you attempt to import torch.neuron from Python 3.5 you will see this error
in 1.1.7.0 - please use Python 3.6 or greater:

.. code-block::

   File "/tmp/install_test_env/lib/python3.5/site-packages/torch_neuron/__init__.py", line 29
      f'Invalid dependency version torch=={torch.__version__}. '
                                                             ^
   SyntaxError: invalid syntax

-  Torchvision has dropped support for Python 3.5
-  HuggingFace transformers has dropped support for Python 3.5

Torchvision
~~~~~~~~~~~

When versions of ``torchvision`` and ``torch`` are mismatched, this
can result in exceptions when compiling ``torchvision`` based
models. Specific versions of ``torchvision`` are built against each release
of ``torch``. For example:

- ``torch==1.5.1`` matches ``torchvision==0.6.1``
- ``torch==1.7.1`` matches ``torchvision==0.8.2``
- etc.

Simultaneously installing both ``torch-neuron`` and ``torchvision`` is the
recommended method of correctly resolving versions.


Dynamic Batching
~~~~~~~~~~~~~~~~

Dynamic batching does not work properly for some models that use the
``aten::size`` operator. When this issue occurs, the input batch sizes are not
properly recorded at inference time, resulting in an error such as:

.. code-block:: text

    RuntimeError: The size of tensor a (X) must match the size of tensor b (Y) at non-singleton dimension 0.

This error typically occurs when ``aten::size`` operators are partitioned to
CPU. We are investigating a fix for this issue.


PyTorch Neuron release [2.9.0.0]
--------------------------------------------------

Date: XX/XX/2023

New in this release
~~~~~~~~~~~~~~~~~~~

* Added support for ``torch==1.13.1``
* New versions of ``torch-neuron`` no longer includes versions for ``torch==1.7`` and ``torch==1.8``
* Added support for Neuron runtime 2.12
* Added support for new operators:
    * ``aten::tensordot``
    * ``aten::adaptive_avg_pool1d``
    * ``aten::prelu``
    * ``aten::reflection_pad2d``
    * ``aten::baddbmm``
    * ``aten::repeat``
* Added a ``separate_weights`` flag to :func:`torch_neuron.trace` to support
  models that are larger than 2GB


Bug fixes
~~~~~~~~~

* Fixed ``aten::_convolution`` with grouping for:
    * :class:`torch.nn.Conv1d`
    * :class:`torch.nn.Conv3d`
    * :class:`torch.nn.ConvTranspose2d`
* Fixed ``aten::linear`` to support 1d input tensors
* Fixed an issue where an input could not be directly returned from the network


PyTorch Neuron release [2.5.0.0]
--------------------------------------------------

Date: 11/23/2022

New in this release
~~~~~~~~~~~~~~~~~~~

* Added PyTorch 1.12 support
* Added Python 3.8 support
* Added new operators support. See :ref:`neuron-cc-ops-pytorch`
* Added support for ``aten::lstm``. See: :ref:`torch_neuron_lstm_support`
* Improved logging:
    * Improved error messages for specific compilation failure modes, including out-of-memory errors
    * Added a warning to show the code location of ``prim::PythonOp`` operations
    * Removed overly-verbose tracing messages
    * Added improved error messages for ``neuron-cc`` and ``tensorflow`` dependency issues
    * Added more debug information when an invalid dynamic batching configuration is used
* Added new experimental explicit NeuronCore placement API. See: :ref:`torch_neuron_core_placement_api`
* Added new guide for NeuronCore placement. See: :ref:`torch_neuron_core_placement_guide`
* Improved :func:`torch_neuron.trace` performance when using large graphs
* Reduced host memory usage of loaded models in ``libtorchneuron.so``
* Added ``single_fusion_ratio_threshold`` argument to :func:`torch_neuron.trace`
  to give more fine-grained control of partitioned graphs



Bug fixes
~~~~~~~~~

* Improved handling of tensor mutations which previously caused accuracy issues on certain models (i.e. yolor, yolov5)
* Fixed an issue where ``inf`` and ``-inf`` values would cause unexpected ``NaN`` values. This could occur with newer versions of ``transformers``
* Fixed an issue where :func:`torch.neuron.DataParallel` would not fully utilize all NeuronCores for specific batch sizes
* Fixed and improved operators:
    * ``aten::upsample_bilinear2d``: Improved error messages in cases where the operation cannot be supported
    * ``aten::_convolution``: Added support for ``output_padding`` argument
    * ``aten::div``: Added support for ``rounding_mode`` argument
    * ``aten::sum``: Fixed to handle non-numeric data types
    * ``aten::expand``: Fixed to handle scalar tensors
    * ``aten::permute``: Fixed to handle negative indices
    * ``aten::min``: Fixed to support more input types
    * ``aten::max``: Fixed to support more input types
    * ``aten::max_pool2d``: Fixed to support both 3-dimensional and 4-dimensional input tensors
    * ``aten::Int``: Fixed an issue where long values would incorrectly lose precision
    * ``aten::constant_pad_nd``: Fixed to correctly use non-0 padding values
    * ``aten::pow``: Fixed to support more input types & values
    * ``aten::avg_pool2d``: Added support for ``count_include_pad`` argument. Added support for ``ceil_mode`` argument if padding isn’t specified
    * ``aten::zero``: Fixed to handle scalars correctly
    * ``prim::Constant``: Fixed an issue where ``-inf`` was incorrectly handled
    * Improved handling of scalars in arithmetic operators


PyTorch Neuron release [2.3.0.0]
--------------------------------------------------

Date: 04/29/2022

New in this release
~~~~~~~~~~~~~~~~~~~

* Added support PyTorch 1.11.
* Updated PyTorch 1.10 to version 1.10.2.
* End of support for torch-neuron 1.5, see :ref:`eol-pt-15`.
* Added support for new operators:

    * ``aten::masked_fill_``
    * ``aten::new_zeros``
    * ``aten::frobenius_norm``

Bug fixes
~~~~~~~~~

* Improved ``aten::gelu`` accuracy
* Updated ``aten::meshgrid`` to support optional indexing argument introduced in ``torch 1.10`` , see  `PyTorch issue 50276 <https://github.com/pytorch/pytorch/issues/50276>`_



PyTorch Neuron release [2.2.0.0]
--------------------------------------------------

Date: 03/25/2022

New in this release
~~~~~~~~~~~~~~~~~~~

* Added full support for  ``aten::max_pool2d_with_indices`` -  (Was previously supported only when indices were unused).
* Added new torch-neuron packages compiled with ``-D_GLIBCXX_USE_CXX11_ABI=1``, the new packages support PyTorch 1.8, PyTorch 1.9, and PyTorch 1.10.
  To install the additional packages compiled with ``-D_GLIBCXX_USE_CXX11_ABI=1`` please change the package repo index to ``https://pip.repos.neuron.amazonaws.com (https://pip.repos.neuron.amazonaws.com/)/cxx11/``
  

PyTorch Neuron release [2.1.7.0]
--------------------------------------------------

Date: 01/20/2022

New in this release
~~~~~~~~~~~~~~~~~~~

* Added PyTorch 1.10 support
* Added new operators support, see :ref:`neuron-cc-ops-pytorch`
* Updated ``aten::_convolution`` to support 2d group convolution
* Updated ``neuron::forward`` operators to allocate less dynamic memory. This can increase performance on models with many input & output tensors.
* Updated ``neuron::forward`` to better handle batch sizes when ``dynamic_batch_size=True``. This can increase performance at 
  inference time when the input batch size is exactly equal to the traced model batch size.

Bug fixes
~~~~~~~~~

* Added the ability to ``torch.jit.trace`` a ``torch.nn.Module`` where a submodule has already been traced with :func:`torch_neuron.trace` on a CPU-type instance.
  Previously, if this had been executed on a CPU-type instance, an initialization exception would have been thrown.
* Fixed ``aten::matmul`` behavior on 1-dimensional by n-dimensional multiplies. Previously, this would cause a validation error.
* Fixed binary operator type promotion. Previously, in unusual situations, operators like ``aten::mul`` could produce incorrect results due to invalid casting.
* Fixed ``aten::select`` when index was -1. Previously, this would cause a validation error.
* Fixed ``aten::adaptive_avg_pool2d`` padding and striding behavior. Previously, this could generate incorrect results with specific configurations.
* Fixed an issue where dictionary inputs could be incorrectly traced when the tensor values had gradients.


PyTorch Neuron release [2.0.536.0]
--------------------------------------------------

Date: 01/05/2022


New in this release
~~~~~~~~~~~~~~~~~~~

* Added new operator support for specific variants of operations (See :ref:`neuron-cc-ops-pytorch`)
* Added optional ``optimizations`` keyword to :func:`torch_neuron.trace` which accepts a list of :class:`~torch_neuron.Optimization` passes.


PyTorch Neuron release [2.0.468.0]
--------------------------------------------------

Date: 12/15/2021


New in this release
~~~~~~~~~~~~~~~~~~~

* Added support for ``aten::cumsum`` operation.
* Fixed ``aten::expand`` to correctly handle adding new dimensions.


PyTorch Neuron release [2.0.392.0]
--------------------------------------------------

Date: 11/05/2021

* Updated Neuron Runtime (which is integrated within this package) to ``libnrt 2.2.18.0`` to fix a container issue that was preventing
  the use of containers when /dev/neuron0 was not present. See details here :ref:`neuron-runtime-release-notes`.

PyTorch Neuron release [2.0.318.0]
--------------------------------------------------

Date: 10/27/2021

New in this release
~~~~~~~~~~~~~~~~~~~

-  PyTorch Neuron 1.x now support Neuron Runtime 2.x (``libnrt.so`` shared library) only.

     .. important::

        -  You must update to the latest Neuron Driver (``aws-neuron-dkms`` version 2.1 or newer)
           for proper functionality of the new runtime library.
        -  Read :ref:`introduce-libnrt`
           application note that describes :ref:`why are we making this
           change <introduce-libnrt-why>` and
           how :ref:`this change will affect the Neuron
           SDK <introduce-libnrt-how-sdk>` in detail.
        -  Read :ref:`neuron-migrating-apps-neuron-to-libnrt` for detailed information of how to
           migrate your application.

-  Introducing PyTorch 1.9.1 support (support for ``torch==1.9.1)``
-  Added ``torch_neuron.DataParallel``, see ResNet-50 tutorial :ref:`[html] </src/examples/pytorch/resnet50.ipynb>` and
   :ref:`torch-neuron-dataparallel-app-note` application note.
-  Added support for tracing on GPUs
-  Added support for ``ConvTranspose1d``
-  Added support for new operators:

   -  ``aten::empty_like``
   -  ``aten::log``
   -  ``aten::type_as``
   -  ``aten::movedim``
   -  ``aten::einsum``
   -  ``aten::argmax``
   -  ``aten::min``
   -  ``aten::argmin``
   -  ``aten::abs``
   -  ``aten::cos``
   -  ``aten::sin``
   -  ``aten::linear``
   -  ``aten::pixel_shuffle``
   -  ``aten::group_norm``
   -  ``aten::_weight_norm``

-  Added ``torch_neuron.is_available()``


Resolved Issues
~~~~~~~~~~~~~~~

-  Fixed a performance issue when using both the
   ``dynamic_batch_size=True`` trace option and
   ``--neuron-core-pipeline`` compiler option. Dynamic batching now uses
   ``OpenMP`` to execute pipeline batches concurrently.
-  Fixed ``torch_neuron.trace`` issues:

   -  Fixed a failure when the same submodule was traced with multiple
      inputs
   -  Fixed a failure where some operations would fail to be called with
      the correct arguments
   -  Fixed a failure where custom operators (torch plugins) would cause
      a trace failure

-  Fixed variants of ``aten::upsample_bilinear2d`` when
   ``scale_factor=1``
-  Fixed variants of ``aten::expand`` using ``dim=-1``
-  Fixed variants of ``aten::stack`` using multiple different input data
   types
-  Fixed variants of ``aten::max`` using indices outputs


[1.8.1.1.5.21.0]
--------------------------------------------------

Date: 08/12/2021

Summary
~~~~~~~

- Minor updates.


.. _neuron-torch-1570:

[1.8.1.1.5.7.0]
--------------------------------------------------

Date: 07/02/2021

Summary
~~~~~~~

- Added support for dictionary outputs using ``strict=False`` flag. See
  :ref:`/neuron-guide/neuron-frameworks/pytorch-neuron/troubleshooting-guide.rst`.
- Updated ``aten::batch_norm`` to correctly implement the ``affine`` flag.
- Added support for ``aten::erf`` and ``prim::DictConstruct``. See
  :ref:`neuron-cc-ops-pytorch`.
- Added dynamic batch support. See
  :ref:`/neuron-guide/neuron-frameworks/pytorch-neuron/api-compilation-python-api.rst`.


.. _neuron-torch-1410:

[1.8.1.1.4.1.0]
--------------------------------------------------

Date: 5/28/2021

Summary
~~~~~~~~

* Added support for PyTorch 1.8.1

    * Models compatibility

        * Models compiled with previous versions of PyTorch Neuron (<1.8.1) are compatible with PyTorch Neuron 1.8.1.
        * Models compiled with PyTorch Neuron 1.8.1 are not backward compatible with previous versions of PyTorch Neuron (<1.8.1) .

    * Updated  tutorials to use Hugging Face Transformers 4.6.0.
    * Added a new set of forward operators (forward_v2)
    * Host memory allocation when loading the same model on multiple NeuronCores is significantly reduced
    * Fixed an issue where models would not deallocate all memory within a python session after being garbage collected.
    * Fixed a TorchScript/C++ issue where loading the same model multiple times would not use multiple NeuronCores by default.


* Fixed logging to no longer configure the root logger.
* Removed informative messages that were produced during compilations as warnings.  The number of warnings reduced significantly.
* Convolution operator support has been extended to include ConvTranspose2d variants.
* Reduce the amount of host memory usage during inference.


.. _neuron-torch-1350:

[1.7.1.1.3.5.0]
--------------------------------------------------

Date: 4/30/2021

Summary
~~~~~~~

- ResNext models now functional with new operator support
- Yolov5 support refer to https://github.com/aws/aws-neuron-sdk/issues/253 note https://github.com/ultralytics/yolov5/pull/2953 which optimized YoloV5 for AWS Neuron
- Convolution operator support has been extended to include most Conv1d and Conv3d variants
- New operator support.  Please see :ref:`neuron-cc-ops-pytorch` for the complete list of operators.

.. _neuron-torch-12160:

[1.7.1.1.2.16.0]
--------------------------------------------------

Date: 3/4/2021

Summary
~~~~~~~~

-  Minor enhancements.

.. _neuron-torch-12150:

[1.7.1.1.2.15.0]
--------------------------------------------------

Date: 2/24/2021

Summary
~~~~~~~

-  Fix for CVE-2021-3177.

.. _neuron-torch-1230:

[1.7.1.1.2.3.0]
--------------------------------------------------

Date: 1/30/2021

Summary
~~~~~~~~

-  Made changes to allow models with -inf scalar constants to correctly compile
-  Added new operator support. Please see :ref:`neuron-cc-ops-pytorch` for the complete list of operators.

.. _neuron-torch-11170:

[1.1.7.0]
--------------------------------------------------

Date: 12/23/2020

Summary
~~~~~~~~

-  We are dropping support for Python 3.5 in this release
-  torch.neuron.trace behavior will now throw a RuntimeError in the case that no operators are compiled for neuron hardware
-  torch.neuron.trace will now display compilation progress indicators (dots) as default behavior (neuron-cc must updated to the December release to greater to see this feature)
-  Added new operator support. Please see :ref:`neuron-cc-ops-pytorch` for the complete list of operators.
-  Extended the BERT pretrained tutorial to demonstrate execution on multiple cores and batch modification, updated the tutorial to accomodate changes in the Hugging Face Transformers code for version 4.0
-  Added a tutorial for torch-serve which extends the BERT tutorial
-  Added support for PyTorch 1.7

.. _neuron-torch-1019780:

[1.0.1978.0]
--------------------------------------------------

Date: 11/17/2020

Summary
~~~~~~~

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
--------------------------------------------------

Date: 09/22/2020

Summary
~~~~~~~

-  Various minor improvements to the Pytorch autopartitioner feature
-  Support for the operators aten::constant_pad_nd, aten::meshgrid
-  Improved performance on various torchvision models. Of note are
   resnet50 and vgg16

.. _neuron-torch-1015320:

[1.0.1532.0]
--------------------------------------------------

Date: 08/08/2020

.. _summary-1:

Summary
~~~~~~~

-  Various minor improvements to the Pytorch autopartitioner feature
-  Support for the aten:ones operator

.. _neuron-torch-1015220:

[1.0.1522.0]
--------------------------------------------------

Date: 08/05/2020

.. _summary-2:

Summary
~~~~~~~~

Various minor improvements.

.. _neuron-torch-1013860:

[1.0.1386.0]
--------------------------------------------------

Date: 07/16/2020

.. _summary-3:

Summary
~~~~~~~

This release adds auto-partitioning, model analysis and PyTorch 1.5.1
support, along with a number of new operators

Major New Features
~~~~~~~~~~~~~~~~~~

-  Support for Pytorch 1.5.1
-  Introduce an automated operator device placement mechanism in
   torch.neuron.trace to run sub-graphs that contain operators that are
   not supported by the neuron compiler in native PyTorch. This new
   mechanism is on by default and can be turned off by adding argument
   fallback=False to the compiler arguments.
-  Model analysis to find supported and unsupported operators in a model

Resolved Issues
~~~~~~~~~~~~~~~~

.. _neuron-torch-1011680:

[1.0.1168.0]
--------------------------------------------------

Date 6/11/2020

.. _summary-4:

Summary
~~~~~~~

.. _major-new-features-1:

Major New Features
~~~~~~~~~~~~~~~~~~

.. _resolved-issues-1:

Resolved Issues
~~~~~~~~~~~~~~~

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _neuron-torch-1010010:

[1.0.1001.0]
--------------------------------------------------

Date: 5/11/2020

.. _summary-5:

Summary
~~~~~~~~

Additional PyTorch operator support and improved support for model
saving and reloading.

.. _major-new-features-2:

Major New Features
~~~~~~~~~~~~~~~~~~

-  Added Neuron Compiler support for a number of previously unsupported
   PyTorch operators. Please see :ref:`neuron-cc-ops-pytorch`for the
   complete list of operators.
-  Add support for torch.neuron.trace on models which have previously
   been saved using torch.jit.save and then reloaded.

.. _resolved-issues-2:

Resolved Issues
~~~~~~~~~~~~~~~~

.. _known-issues-and-limitations-1:

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _neuron-torch-108250:

[1.0.825.0]
--------------------------------------------------

Date: 3/26/2020

.. _summary-6:

Summary
~~~~~~~

.. _major-new-features-3:

Major New Features
~~~~~~~~~~~~~~~~~

.. _resolved-issues-3:

Resolved Issues
~~~~~~~~~~~~~~~

.. _known-issues-and-limitations-2:

Known Issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _neuron-torch-107630:

[1.0.763.0]
--------------------------------------------------

Date: 2/27/2020

.. _summary-7:

Summary
~~~~~~~

Added Neuron Compiler support for a number of previously unsupported
PyTorch operators. Please see :ref:`neuron-cc-ops-pytorch` for the complete
list of operators.

.. _major-new-features-4:

Major new features
~~~~~~~~~~~~~~~~~~

-  None

.. _resolved-issues-4:

Resolved issues
~~~~~~~~~~~~~~~~~

-  None

.. _neuron-torch-106720:

[1.0.672.0]
--------------------------------------------------

Date: 1/27/2020

.. _summary-8:

Summary
~~~~~~~~

.. _major-new-features-5:

Major new features
~~~~~~~~~~~~~~~~~~

.. _resolved-issues-5:

Resolved issues
~~~~~~~~~~~~~~~~

-  Python 3.5 and Python 3.7 are now supported.

.. _known-issues-and-limitations-3:

Known issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other Notes
~~~~~~~~~~~

.. _neuron-torch-106270:

[1.0.627.0]
--------------------------------------------------

Date: 12/20/2019

.. _summary-9:

Summary
~~~~~~~~

This is the initial release of torch-neuron. It is not distributed on
the DLAMI yet and needs to be installed from the neuron pip repository.

Note that we are currently using a TensorFlow as an intermediate format
to pass to our compiler. This does not affect any runtime execution from
PyTorch to Neuron Runtime and Inferentia. This is why the neuron-cc
installation must include [tensorflow] for PyTorch.

.. _major-new-features-6:

Major new features
~~~~~~~~~~~~~~~~~~

.. _resolved-issues-6:

Resolved issues
~~~~~~~~~~~~~~~

.. _known-issues-and-limitations-4:

Known issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models TESTED
~~~~~~~~~~~~~~

The following models have successfully run on neuron-inferentia systems

1. SqueezeNet
2. ResNet50
3. Wide ResNet50

Pytorch Serving
~~~~~~~~~~~~~~~

In this initial version there is no specific serving support. Inference
works correctly through Python on Inf1 instances using the neuron
runtime. Future releases will include support for production deployment
and serving of models

Profiler support
~~~~~~~~~~~~~~~~

Profiler support is not provided in this initial release and will be
available in future releases

Automated partitioning
~~~~~~~~~~~~~~~~~~~~~~

Automatic partitioning of graphs into supported and non-supported
operations is not currently supported. A tutorial is available to
provide guidance on how to manually parition a model graph. Please see
:ref:`pytorch-manual-partitioning-jn-tutorial`

PyTorch dependency
~~~~~~~~~~~~~~~~~~

Currently PyTorch support depends on a Neuron specific version of
PyTorch v1.3.1. Future revisions will add support for 1.4 and future
releases.

Trace behavior
~~~~~~~~~~~~~~

In order to trace a model it must be in evaluation mode. For examples
please see :ref:`/src/examples/pytorch/resnet50.ipynb`

Six pip package is required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Six package is required for the torch-neuron runtime, but it is not
modeled in the package dependencies. This will be fixed in a future
release.

Multiple NeuronCore support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the num-neuroncores options is used the number of cores must be
manually set in the calling shell environment variable for compilation
and inference.

For example: Using the keyword argument
compiler_args=['—num-neuroncores', '4'] in the trace call, requires
NEURONCORE_GROUP_SIZES=4 to be set in the environment at compile time
and runtime

CPU execution
~~~~~~~~~~~~~~

At compilation time a constant output is generated for the purposes of
tracing. Running inference on a non neuron instance will generate
incorrect results. This must not be used. The following error message is
generated to stderr:

::

   Warning: Tensor output are ** NOT CALCULATED ** during CPU execution and only
   indicate tensor shape

.. _other-notes-1:

Other notes
~~~~~~~~~~~

-  Python version(s) supported:

   -  3.6

-  Linux distribution supported:

   -  DLAMI Ubuntu 18 and Amazon Linux 2 (using Python 3.6 Conda environments)
   -  Other AMIs based on Ubuntu 18
   -  For Amazon Linux 2 please install Conda and use Python 3.6 Conda
      environment
