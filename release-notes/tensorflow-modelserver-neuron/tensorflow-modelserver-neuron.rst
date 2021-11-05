.. _tensorflow-modelserver-rn:

TensorFlow-Model-Server-Neuron 1.x Release Notes
================================================

.. contents::
   :local:
   :depth: 1

This document lists the release notes for the
TensorFlow-Model-Server-Neuron package.


TensorFlow Model Server Neuron 1.x release [2.0.4.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 11/05/2021

* Updated Neuron Runtime (which is integrated within this package) to ``libnrt 2.2.18.0`` to fix a container issue that was preventing 
  the use of containers when /dev/neuron0 was not present. See details here :ref:`neuron-runtime-release-notes`.

TensorFlow Model Server Neuron 1.x release [2.0.3.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 10/27/2021

New in this release
-------------------

* TensorFlow Model Server Neuron 1.x now support Neuron Runtime 2.x (``libnrt.so`` shared library) only.

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


.. _11501510:

[1.15.0.1.5.1.0]
^^^^^^^^^^^^^^^^

Date: 07/02/2021

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _11501400:

[1.15.0.1.4.0.0]
^^^^^^^^^^^^^^^^

Date: 05/24/2021

Summary
-------

1. Remove SIGINT/SIGTERM handler and rely on mechnisms provided by Neuron runtime for resource cleanup.
2. Uncap protobuf size limit.

.. _11501330:

[1.15.0.1.3.3.0]
^^^^^^^^^^^^^^^^^^^

Date: 05/01/2021

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _11501290:

[1.15.0.1.2.9.0]
^^^^^^^^^^^^^^^^^^^

Date: 03/04/2021

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _11501280:

[1.15.0.1.2.8.0]
^^^^^^^^^^^^^^^^^^^

Date: 02/24/2021

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.


.. _11501220:

[1.15.0.1.2.2.0]
^^^^^^^^^^^^^^^^^^^

Date: 01/30/2021

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.


.. _11501130:

[1.15.0.1.1.3.0]
^^^^^^^^^^^^^^^^^^^

Date: 12/23/2020

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.


.. _11501021680:

[1.15.0.1.0.2168.0]
^^^^^^^^^^^^^^^^^^^

Date: 11/17/2020

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.


.. _11501020430:

[1.15.0.1.0.2043.0]
^^^^^^^^^^^^^^^^^^^

Date: 09/22/2020

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _11501019650:

[1.15.0.1.0.1965.0]
^^^^^^^^^^^^^^^^^^^

Date: 08/08/2020

.. _summary-1:

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _11501019530:

[1.15.0.1.0.1953.0]
^^^^^^^^^^^^^^^^^^^

Date: 08/05/2020

.. _summary-2:

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _11501018910:

[1.15.0.1.0.1891.0]
^^^^^^^^^^^^^^^^^^^

Date: 07/16/2020

.. _summary-3:

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _11501017960:

[1.15.0.1.0.1796.0]
^^^^^^^^^^^^^^^^^^^

Date 6/11/2020

.. _summary-4:

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _11501015720:

[1.15.0.1.0.1572.0]
^^^^^^^^^^^^^^^^^^^

Date 5/11/2020

.. _summary-5:

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _11501013330:

[1.15.0.1.0.1333.0]
^^^^^^^^^^^^^^^^^^^

Date 3/26/2020

.. _summary-6:

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _11501012400:

[1.15.0.1.0.1240.0]
^^^^^^^^^^^^^^^^^^^

Date 2/27/2020

.. _summary-7:

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _1150109970:

[1.15.0.1.0.997.0]
^^^^^^^^^^^^^^^^^^

Date 1/27/2019

.. _summary-8:

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _1150108030:

[1.15.0.1.0.803.0]
^^^^^^^^^^^^^^^^^^

Date 12/20/2019

.. _summary-9:

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _1150107490:

[1.15.0.1.0.749.0]
^^^^^^^^^^^^^^^^^^

Date 12/1/2019

.. _summary-10:

Summary
-------

No change. See :ref:`tensorflow-neuron-release-notes` for related TensorFlow-Neuron release
notes.

.. _1150106630:

[1.15.0.1.0.663.0]
^^^^^^^^^^^^^^^^^^

Date 11/29/2019

.. _summary-11:

Summary
-------

This version is available only in released DLAMI v26.0. See
TensorFlow-Neuron Release Notes. Please
:ref:`update <dlami-rn-known-issues>` to latest version.
