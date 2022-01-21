.. _neuron-driver-release-notes:

Neuron Driver 2.x Release Notes
===============================

.. contents::
   :local:
   :depth: 1


Neuron Driver release [2.2.13.0]
--------------------------------

Date: 01/20/2022

Summary
-------

- Minor updates


Neuron Driver release [2.2.6.0]
-------------------------------

Date: 10/27/2021

New in this release
^^^^^^^^^^^^^^^^^^^

-  Memory improvements made to ensure all allocations are made with 4K
   alignments.


Resolved issues
^^^^^^^^^^^^^^^

-  No longer delays 1s per NeuronDevice when closing Neuron Tools
   applications.
-  Fixes a Ubuntu 20 build issue


Neuron Driver release [2.1]
---------------------------

-  Support is added for Neuron Runtime 2.x (``libnrt.so``).
-  Support for previous releases of Neuron Runtime 1.x is continued with
   Driver 2.x releases.