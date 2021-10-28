.. _tensorflow-neuron-rn-v2:

Tensorflow-Neuron 2.x Release Notes
===================================

.. contents::
   :local:
   :depth: 1

This document lists the release notes for the TensorFlow-Neuron 2.x packages.

.. _tf-known-issues-and-limitations:

Known Issues and Limitations - updated 08/12/2021
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Tensorflow-Neuron 2.x release [2.0.3.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 10/27/2021

New in this release
-------------------

* TensorFlow Neuron 2.x now support Neuron Runtime 2.x (``libnrt.so`` shared library) only.

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


* Updated Tensorflow 2.3.x from Tensorflow 2.3.3 to Tensorflow 2.3.4. 
* Updated Tensorflow 2.4.x from Tensorflow 2.4.2 to Tensorflow 2.4.3.
* Updated Tensorflow 2.5.x from Tensorflow 2.5.0 to Tensorflow 2.5.1.


Resolved Issues
---------------

* Fix bug that can cause illegal compiler optimizations
* Fix bug that can cause dynamic-shape operators be placed on Neuron

.. _2501680:

Tensorflow-Neuron 2.x release [1.6.8.0]
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