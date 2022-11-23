.. _tensorflow-modelserver-rn-v2:

TensorFlow-Model-Server-Neuron 2.x Release Notes
================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for the
TensorFlow-Model-Server-Neuron package.

TensorFlow Model Server Neuron 2.x release [2.4.0.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 11/23/2022

* Deprecated the NEURONCORE_GROUP_SIZES environment variable.
* Minor bug fixes.


TensorFlow Model Server Neuron 2.x release [2.3.0.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 04/29/2022

* Added support for tensorflow-model-serving 2.8.0.


TensorFlow Model Server Neuron 2.x release [2.2.0.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 03/25/2022

* Updated tensorflow-serving 2.5 to 2.5.4.
* Add support for tensorflow-model-serving 2.6 and 2.7.



TensorFlow Model Server Neuron 2.x release [2.1.6.0]
----------------------------------------------------

Date: 01/20/2022

* Updated tensorflow-model-server 2.5 to version 2.5.3


TensorFlow Model Server Neuron 2.x release [2.0.4.0]
----------------------------------------------------

Date: 11/05/2021

* Updated Neuron Runtime (which is integrated within this package) to ``libnrt 2.2.18.0`` to fix a container issue that was preventing 
  the use of containers when /dev/neuron0 was not present. See details here :ref:`neuron-runtime-release-notes`.

TensorFlow Model Server Neuron 2.x release [2.0.3.0]
----------------------------------------------------

Date: 10/27/2021

New in this release
^^^^^^^^^^^^^^^^^^^

* TensorFlow Model Server Neuron 2.x now support Neuron Runtime 2.x (``libnrt.so`` shared library) only.

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


.. _2511680:

TensorFlow Model Server Neuron 2.x release [1.6.8.0]
----------------------------------------------------

Date: 08/12/2021

Summary
^^^^^^^

TensorFlow 2.x - tensorflow-model-server-neuron now support TensorFlow 2.x,  tensorflow-model-server-neuron package versions 2.1.4, 2.2.2, 2.3.0, 2.4.1, and 2.5.1 support TensorFlow 2.x.
