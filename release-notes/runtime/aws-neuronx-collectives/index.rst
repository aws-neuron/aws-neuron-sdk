.. _neuron-collectives-rn:

Neuron Collectives Release Notes
================================

Neuron Collectives refers to a set of libraries used to support collective compute operations within the Neuron SDK.  The collectives support is delivered via the aws-neuronx-collectives package and includes a pre-built version of the OFI plugin required for use of collectives with Elastic Fabric Adapter (EFA).

.. contents:: Table of contents
   :local:
   :depth: 1

Neuron Collectives [2.12.22.0]
------------------------------
Date: 03/28/2023

New in this release

* Added support for TRN1N.
* Added support for 16 channels and 16 EFA devices, which is required for enabling EC2 TRN1N instances with Neuron.
* Added support for hierarchical All-Reduce and Reduce-Scatter. These implementations are now used by default and provides up to 75% reduction in latency for 2MB buffers across 256 ranks.


Neuron Collectives [2.11.47.0]
------------------------------
Date: 02/08/2023

New in this release

* Added support for Inf2. 



Neuron Collectives [2.10.20.0]
-----------------------------
Date: 10/10/2022

New in this release

* Improved logging to appear similar in style to Neuron Runtime

Bug Fixes

* Fixed memory registration to support 2GB+ sizes
* Fixed association of network devices to channels (removes previous hard-coding).


Neuron Collectives [2.9.86.0]
-----------------------------
Date: 10/10/2022

New in this release

* Added support for All-Reduce, Reduce-Scatter, All-Gather, and Send/Recv operations.

