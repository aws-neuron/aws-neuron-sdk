.. _tensorflow-modelserver-rn-v2:

TensorFlow-Model-Server-Neuron 2.x Release Notes
================================================

.. contents::
   :local:
   :depth: 1

This document lists the release notes for the
TensorFlow-Model-Server-Neuron package.


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
