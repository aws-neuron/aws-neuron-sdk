.. _tensorflow-neuron-rn-v2:

TensorFlow Neuron (``tensorflow-neuron (TF2.x)``) Release Notes
===============================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for the tensorflow-neuron 2.x packages.

.. _tf-known-issues-and-limitations:

Known Issues and Limitations - updated 08/12/2021
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Support on serialized TensorFlow 2.x custom operators is currently limited. Serializing some operators registered from tensorflow-text through `TensorFlow Hub <https://tfhub.dev/>`_ is going to cause failure in tensorflow.neuron.trace.

- Memory leak exists on latest releases of TensorFlow Neuron for versions 2.1, 2.2, 2.3, and 2.4.


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

Solution: Please downgrade `h5py` by `pip install 'h5py<3'`. This is caused by https://github.com/TensorFlow/TensorFlow/issues/44467.

tensorflow-neuron 2.x release [2.10.1.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 09/15/2023

* Minor updates.

tensorflow-neuron 2.x release [2.9.3.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 7/19/2023

* Minor updates.


tensorflow-neuron 2.x release [2.8.9.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 6/14/2023

* Added Python 3.10 support.

tensorflow-neuron 2.x release [2.8.1.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 05/01/2023

* Added support for tracing models larger than 2 GB through the environment variable ``NEURON_CC_FLAGS='--extract-weights INSTANCE_TYPE'`` for all inf1 instance types.
* Neuron release 2.10 release will be the last release that will include support for tensorflow-neuron version 2.7. Future Neuron releases will not include tensorflow-neuron version 2.7.

tensorflow-neuron 2.x release [2.7.4.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 04/19/2023

* Minor updates.

tensorflow-neuron 2.x release [2.7.3.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 03/28/2023

* Introduce the ``tfn.analyze_model`` function that displays information about the supported and unsupported operators of a traceable model.
* Introduce the ``on_neuron_ratio`` attribute of AWS Optimized Neuron Models returned by ``tfn.trace``, which is the percentage of ops on neuron after compilation. 

tensorflow-neuron 2.x release [2.6.5.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 02/24/2023

* Minor updates.

tensorflow-neuron 2.x release [2.6.0.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 2/24/2023

* Minor bug fixes.

tensorflow-neuron 2.x release [2.4.0.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 11/22/2022

* Experimental support for tracing models larger than 2 GB through environment variable ``NEURON_CC_FLAGS='--extract-weights'``.
* Introduce ``tfn.auto_multicore`` Python API to enable automatic data parallel on multiple NeuronCores.
* Introduce ``tf-neuron-auto-multicore`` tool to enable automatic data parallel on multiple NeuronCores.
* Deprecated the NEURONCORE_GROUP_SIZES environment variable.
* Minor bug fixes.


tensorflow-neuron 2.x release [2.3.0.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 04/29/2022

* Added support for Tensorflow 2.8.0.
* Added support for Slice operator
* The graph partitioner now prefers to place less compute intensive operators on CPU if the model already contains a large amount of compute intensive operators.
* Fixed `Github issue #408 <https://github.com/aws/aws-neuron-sdk/issues/408>`_, the fix solves data type handling bug in ``tfn.trace`` when the model contains Conv2D operators.


tensorflow-neuron 2.x release [2.2.0.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 03/25/2022

* Updated TensorFlow 2.5 to version 2.5.3.
* Added support for TensorFlow 2.6 and 2.7.
* Added a warning message when calling ``tfn.saved_model.compile`` API. In tensorflow-neuron 2.x you should call :ref:`tensorflow.neuron.trace <tensorflow-ref-neuron-tracing-api>`. ``tfn.saved_model.compile`` API supports only partial functionality of :ref:`tensorflow.neuron.trace <tensorflow-ref-neuron-tracing-api>` and will be deprecated in the future.



tensorflow-neuron 2.x release [2.1.14.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 02/17/2022

* Fixed a bug in TensorFlow Neuron versions 2.1, 2.2. 2.3 and 2.4. The fixed bug was causing a memory leak of 128 bytes for each inference.
* Improved warning message when calling deprecated compilation API under tensorflow-neuron 2.x. 


tensorflow-neuron 2.x release [2.1.13.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 02/16/2022

* Fixed a bug that caused a memory leak. The memory leak was approximately 128b for each inference and 
  exists in all versions of TensorFlow Neuron versions part of Neuron 1.16.0 to Neuron 1.17.0 releases. see :ref:`pre-release-content` 
  for exact versions included in each release.  This release only addresses the leak in TensorFlow Neuron 2.5.  Future release of TensorFlow Neuron will fix the leak in other versions as well (2.1, 2.2, 2.3, 2.4).



tensorflow-neuron 2.x release [2.1.6.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 01/20/2022

* Updated TensorFlow 2.5 to version 2.5.2.
* Enhanced auto data parallel (e.g. when using NEURONCORE_GROUP_SIZES=X,Y,Z,W) to support edge cases.
* Fixed a bug that may cause tensorflow-neuron to generate in some cases scalar gather instruction with incorrect arguments.


tensorflow-neuron 2.x release [2.0.4.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 11/05/2021

* Updated Neuron Runtime (which is integrated within this package) to ``libnrt 2.2.18.0`` to fix a container issue that was preventing 
  the use of containers when /dev/neuron0 was not present. See details here :ref:`neuron-runtime-release-notes`.

tensorflow-neuron 2.x release [2.0.3.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 10/27/2021

New in this release
-------------------

* TensorFlow 2.x (``tensorflow-neuron``) now support Neuron Runtime 2.x (``libnrt.so`` shared library) only.

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


* Updated TensorFlow 2.3.x from TensorFlow 2.3.3 to TensorFlow 2.3.4. 
* Updated TensorFlow 2.4.x from TensorFlow 2.4.2 to TensorFlow 2.4.3.
* Updated TensorFlow 2.5.x from TensorFlow 2.5.0 to TensorFlow 2.5.1.


Resolved Issues
---------------

* Fix bug that can cause illegal compiler optimizations
* Fix bug that can cause dynamic-shape operators be placed on Neuron

.. _2501680:

tensorflow-neuron 2.x release [1.6.8.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 08/12/2021

New in this release
-------------------

* First release of TensorFlow 2.x integration, Neuron support now TensorFlow versions 2.1.4, 2.2.3, 2.3.3, 2.4.2, and 2.5.0.

* New public API tensorflow.neuron.trace: trace a TensorFlow 2.x keras.Model or a Python callable that can be decorated by tf.function, and return an AWS-Neuron-optimized keras.Model that can execute on AWS Machine Learning Accelerators.
 **Please note** that TensorFlow 1.x SavedModel compilation API tensorflow.neuron.saved_model.compile is not supported in tensorflow-neuron 2.x . It continues to function in tensorflow-neuron 1.15.x .

* Included versions:

   - tensorflow-neuron-2.5.0.1.6.8.0 
   - tensorflow-neuron-2.4.2.1.6.8.0
   - tensorflow-neuron-2.3.3.1.6.8.0
   - tensorflow-neuron-2.2.3.1.6.8.0
   - tensorflow-neuron-2.1.4.1.6.8.0
