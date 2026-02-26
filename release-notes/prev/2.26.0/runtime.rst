.. _neuron-2-26-0-runtime:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Runtime component, version 2.26.0. Release date: 9/18/2025.

AWS Neuron SDK 2.26.0: Neuron Runtime release notes
===================================================

**Date of release**:  September 18, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.26.0 release notes home <neuron-2-26-0-whatsnew>`

Released versions
-----------------

- Neuron Driver: ``2.24.7.0``
- Neuron Runtime Library: ``2.28.19.0``
- Neuron Collectives: ``2.28.20.0``

Neuron Runtime Library 2.28.19.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Added rank ID to all events emitted from the Profiler 2.0 system trace.
* Improved timestamp alignment of Profiler 2.0 NeuronCore and CPU system trace events enhancing the accuracy of the trace timeline.

Neuron Driver 2.24.7.0
^^^^^^^^^^^^^^^^^^^^^^
* Fixed installation issue causing builds to fail for Linux Kernels 6.13+.

Neuron Runtime Library 2.28.19.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Fixed bug where `nrt_unload` returned `NRT_SUCCESS` even when model stop fails due to Neuron Core lockups.
* Fixed bug where `model_name` was empty in Profiler 2.0 system trace events.

Neuron Collectives 2.28.20.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Fixed bug where error messages were incorrectly being displayed on machines with no EFA devices.

Previous release notes
----------------------

* :ref:`neuron-2-25-0-runtime`
* :ref:`runtime_rn`
* :ref:`runtime_rn`
* :ref:`runtime_rn`
