.. _tensorflow-neuron-rn:

Tensorflow-Neuron Release Notes
===============================


This document lists the release notes for the TensorFlow-Neuron package.

.. _tf-known-issues-and-limitations:

Known Issues and Limitations - updated 08/12/2021
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Support on serialized TensorFlow 2.x custom operators is currently limited. Serializing some operators registered from tensorflow-text through `TensorFlow Hub <https://tfhub.dev/>`_ is going to cause failure in tensorflow.neuron.trace.


-  Issue: When compiling large models, user might run out of memory and
   encounter this fatal error.

::

   terminate called after throwing an instance of 'std::bad_alloc'

Solution: run compilation on a c5.4xlarge instance type or larger.

-  Issue: When upgrading ``tensorflow-neuron`` with
   ``pip install tensorflow-neuron --upgrade``, the following error
   message may appear, which is caused by ``pip`` version being too low.

::

     Could not find a version that satisfies the requirement tensorflow<1.16.0,>=1.15.0 (from tensorflow-neuron)

Solution: run a ``pip install pip --upgrade`` before upgrading
``tensorflow-neuron``.

-  Issue: Some Keras routines throws the following error:

::

   AttributeError: 'str' object has no attribute 'decode'.

Solution: Please downgrade `h5py` by `pip install 'h5py<3'`. This is caused by https://github.com/tensorflow/tensorflow/issues/44467.


.. _2501680:

[2.5.0.1.6.8.0]
^^^^^^^^^^^^^^^^

Date: 08/12/2021

Summary
-------

* TensorFlow 2.x - First release of TensorFlow 2.x integration, Neuron support now TensorFlow versions 2.1.4, 2.2.3, 2.3.3, 2.4.2, and 2.5.0.

* TensorFlow 2.x - New public API tensorflow.neuron.trace: trace a TensorFlow 2.x keras.Model or a Python callable that can be decorated by tf.function, and return an AWS-Neuron-optimized keras.Model that can execute on AWS Machine Learning Accelerators.
 **Please note** that TensorFlow 1.x SavedModel compilation API tensorflow.neuron.saved_model.compile is not supported in tensorflow-neuron 2.x . It continues to function in tensorflow-neuron 1.15.x .

.. _11551510:

[1.15.5.1.5.1.0]
^^^^^^^^^^^^^^^^

Date: 07/02/2021

Summary
-------

* Bug fixes regarding scalar inputs/outputs.
* Minor performance improvements when dynamic batch size is turned on or when model is small.

.. _11551400:

[1.15.5.1.4.0.0]
^^^^^^^^^^^^^^^^

Date: 05/28/2021

Summary
-------

* Reduce the amount of input/output data movement during inference.
* Improve parallelism for dynamic batch size inference by adopting a new sharding mechanism.
* Reduce the amount of host memory usage during inference.
* tfn.saved_model.compile now generates correct code when operator Split is used as output.
* tfn.saved_model.compile now properly reads input tensor shape information from SignatureDef proto.
* tfn.saved_model.compile now terminates properly when neuron-cc compiler argument is passed but there is no successful compilation.
* Fix bug on some wrong internal tensor names when neuron-cc compiler crashes.
* Other minor bug fixes.

.. _11551330:

[1.15.5.1.3.3.0]
^^^^^^^^^^^^^^^^

Date: 05/01/2021

Summary
-------

1. Minor enhancements.

.. _11551290:

[1.15.5.1.2.9.0]
^^^^^^^^^^^^^^^^

Date: 03/04/2021

Summary
-------

1. Minor enhancements.


.. _11551280:

[1.15.5.1.2.8.0]
^^^^^^^^^^^^^^^^

Date: 02/24/2021

Summary
-------

1. Fix for CVE-2021-3177.


.. _11551220:

[1.15.5.1.2.2.0]
^^^^^^^^^^^^^^^^

Date: 01/30/2021

Summary
-------

1. Bug fixes and internal refactor.

2. Bump tensorflow base package version to 1.15.5.

3. Introduced a new argument ``convert_constants_to_variables`` to the compilation API ``tfn.saved_model.compile``. Setting it to ``True`` can address the issue of large constants consuming too much memory in the tensorflow runtime.




.. _11541130:

[1.15.4.1.1.3.0]
^^^^^^^^^^^^^^^^

Date: 12/23/2020

Summary
-------

1. Improved logging during `tfn.saved_model.compile` to display `neuron-cc` compilation progress.

2. Small performance improvement in some edge cases by optimizing the NeuronCore-executable assignment mechanism.




.. _11541021680:

[1.15.4.1.0.2168.0]
^^^^^^^^^^^^^^^^^^^

Date: 11/17/2020

Summary
-------

1. tensorflow-neuron is now a plugin package that can be used together
   with tensorflow~=1.15.0 built with ``GLIBCXX_USE_CXX11_ABI=0``.

2. Improved logging during ``tfn.saved_model.compile`` to display
   ``neuron-cc`` logging file path, which is useful for tracking
   ``neuron-cc`` compilation progress.

3. Small performance improvement by utilizing shared memory more
   efficiently.


.. _11531020430:

[1.15.3.1.0.2043.0]
^^^^^^^^^^^^^^^^^^^

Date: 09/22/2020

Summary
-------

1. tensorflow-neuron now automatically enables data parallel mode on
   four cores in one Inferentia. In ``tensorflow-model-server-neuron``,
   most models can now fully utilize four cores automatically. In Python
   tensorflow, running threaded inference using ``>=4`` Python threads
   in the same tensorflow Session lead to full utilization of four
   cores.

2. tensorflow-neuron now tries to enable dynamic batch size
   automatically for a limited number of models, such as ResNet50.

3. Improved logging during ``tfn.saved_model.compile`` to display
   input/output information about subgraphs that are going to be
   compiled by ``neuron-cc``.

.. _11531019650:

[1.15.3.1.0.1965.0]
^^^^^^^^^^^^^^^^^^^

Date: 08/08/2020

.. _summary-1:

Summary
-------

Various minor improvements.

.. _11531019530:

[1.15.3.1.0.1953.0]
^^^^^^^^^^^^^^^^^^^

Date: 08/05/2020

.. _summary-2:

Summary
-------

Various minor improvements.

.. _11531018910:

[1.15.3.1.0.1891.0]
^^^^^^^^^^^^^^^^^^^

Date: 07/16/2020

.. _summary-3:

Summary
-------

This version contains a few bug fixes and user experience improvements.

Dependency change
-----------------

1. Bump tensorflow base package version number to 1.15.3
2. Add ``tensorflow >= 1.15.0, < 1.16.0`` as an installation dependency
   so that packages depending on tensorflow can be installed together
   with tensorflow-neuron without error

New Features
------------

1. ``tensorflow-neuron`` now displays a summary of model performance
   when profiling is enable by setting environment variable
   ``NEURON_PROFILE``

Resolved Issues
---------------

1. Environment variable ``NEURON_PROFILE`` can now be set to a
   non-existing path which will be automatically created
2. Fixed a bug in ``tfn.saved_model.compile`` that causes compilation
   failure when ``dynamic_batch_size=True`` is specified on a SavedModel
   with unknown rank inputs.

.. _11521017960:

[1.15.2.1.0.1796.0]
^^^^^^^^^^^^^^^^^^^

Date 6/11/2020

.. _summary-4:

Summary
-------

This version contains a few bug fixes.

Major New Features
------------------

.. _resolved-issues-1:

Resolved Issues
---------------

1. Fixed a bug related with device placement. Now models with device
   information hardcoded to GPU can be successfully compiled with
   ``tfn.saved_model.compile``
2. Fixed a bug in ``tfn.saved_model.compile`` that causes models
   containing Reshape operators not functioning correctly when it is
   compiled with ``dynamic_batch_size=True``
3. Fixed a bug in ``tfn.saved_model.compile`` that causes models
   containing Table related operators to initialize incorrectly after
   compilation.

Known Issues and limitations
----------------------------

.. _11521015720:

[1.15.2.1.0.1572.0]
^^^^^^^^^^^^^^^^^^^

Date: 5/11/2020

.. _summary-5:

Summary
-------

This version contains some bug fixes and new features.

.. _major-new-features-1:

Major New Features
------------------

-  Tensorflow-Neuron is now built on TensorFlow 1.15.2 instead of
   TensorFlow 1.15.0

.. _resolved-issues-2:

Resolved Issues
---------------

-  Fixed a bug that caused Neuron runtime resources to not all be
   released when a tensorflow-neuron process terminated with in-flight
   inferences
-  Inference timeout value set at compile time is now correctly
   recognized at runtime


Known Issues and limitations
----------------------------

.. _11501013330:

[1.15.0.1.0.1333.0]
^^^^^^^^^^^^^^^^^^^

Date: 3/26/2020

.. _summary-6:

Summary
-------

.. _major-new-features-2:

Major New Features
------------------

-  Improved performance between Tensorflow to Neuron runtime.

.. _resolved-issues-3:

Resolved Issues
---------------

-  Fixed a bug in Neuron runtime adaptor operator's shape function when
   dynamic batch size inference is enabled
-  Framework method (tensorflow.neuron.saved-model.compile) improved
   handling of compiler timeout termination by letting it clean up
   before exiting.

.. _known-issues-and-limitations-2:

Known Issues and limitations
----------------------------

.. _11501012400:

[1.15.0.1.0.1240.0]
^^^^^^^^^^^^^^^^^^^

Date: 2/27/2020

.. _summary-7:

Summary
-------

.. _major-new-features-3:

Major New Features
------------------

-  Enabled runtime memory optimizations by default to improve inference
   performance, specifically in cases with large input/output tensors
-  tfn.saved_model.compile now displays warning message instead of
   "successfully compiled" if less than 30% of operators are mapped to
   Inferentia
-  Improve error messages. Runtime failure error messages are now more
   descriptive and also provide instructions to restart neuron-rtd when
   necessary.

.. _resolved-issues-4:

Resolved Issues
---------------

.. _known-issues-and-limitations-3:

Known Issues and Limitations
----------------------------

-  Issue: When compiling a large model, may encounter.

::

   terminate called after throwing an instance of 'std::bad_alloc'

Solution: run compilation on c5.4xlarge instance type or larger.

Other Notes
-----------

.. _1150109970:

[1.15.0.1.0.997.0]
^^^^^^^^^^^^^^^^^^

Date: 1/27/2020

.. _summary-8:

Summary
-------

.. _major-new-features-4:

Major New Features
------------------

-  Added support for NCHW pooling operators in tfn.saved_model.compile.

.. _resolved-issues-5:

Resolved Issues
---------------

-  Fixed GRPC transient status error issue.
-  Fixed a graph partitioner issue with control inputs.

.. _known-issues-and-limitations-4:

Known Issues and Limitations
----------------------------

-  Issue: When compiling a large model, may encounter.

::

   terminate called after throwing an instance of 'std::bad_alloc'

Solution: run compilation on c5.4xlarge instance type or larger.

.. _other-notes-1:

Other Notes
-----------

.. _1150108030:

[1.15.0.1.0.803.0]
^^^^^^^^^^^^^^^^^^

Date: 12/20/2019

.. _summary-9:

Summary
-------

.. _major-new-features-5:

Major New Features
------------------

.. _resolved-issues-6:

Resolved Issues
---------------

-  Improved handling of ``tf.neuron.saved_model.compile`` arguments

.. _known-issues-and-limitations-5:

Known Issues and Limitations
----------------------------

.. _other-notes-2:

Other Notes
-----------

.. _1150107490:

[1.15.0.1.0.749.0]
^^^^^^^^^^^^^^^^^^

Date: 12/1/2019

.. _summary-10:

Summary
-------

.. _major-new-features-6:

Major New Features
------------------

.. _resolved-issues-7:

Resolved Issues
---------------

-  Fix race condition between model load and model unload when the
   process is killed
-  Remove unnecessary GRPC calls when the process is killed

.. _known-issues-and-limitations-6:

Known Issues and Limitations
----------------------------

-  When compiling a large model, may encounter “terminate called after
   throwing an instance of 'std::bad_alloc'”. Solution: run compilation
   on c5.4xlarge instance type or larger.

-  The pip package ``wrapt`` may have a conflicting version in some
   installations. This is seen when this error occurs:

.. code:: bash

   ERROR: Cannot uninstall 'wrapt'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.

To solve this, you can update wrapt to the newer version:

.. code:: bash

   python3 -m pip install wrapt --ignore-installed
   python3 -m pip install tensorflow-neuron

Within a Conda environment:

.. code:: bash

   conda update wrapt
   conda update tensorflow-neuron

.. _other-notes-3:

Other Notes
-----------

.. _1150106630:

[1.15.0.1.0.663.0]
^^^^^^^^^^^^^^^^^^

Date: 11/25/2019

.. _summary-11:

Summary
-------

This version is available only in released DLAMI v26.0 and is based on
TensorFlow version 1.15.0. Please
:ref:`update <dlami-rn-known-issues>` to latest version.

.. _major-new-features-7:

Major New Features
------------------

.. _resolved-issues-8:

Resolved Issues
---------------

Known Issues and Limits
-----------------------

Models Supported
----------------

The following models have successfully run on neuron-inferentia systems

1. BERT_LARGE and BERT_BASE
2. Transformer
3. Resnet50 V1/V2
4. Inception-V2/V3/V4

.. _other-notes-4:

Other Notes
-----------

-  Python versions supported:

   -  3.5, 3.6, 3.7

-  Linux distribution supported:

   -  Ubuntu 18, Amazon Linux 2



