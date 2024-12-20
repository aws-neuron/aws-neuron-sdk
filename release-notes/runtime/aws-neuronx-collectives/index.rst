.. _neuron-collectives-rn:

Neuron Collectives Release Notes
================================

Neuron Collectives refers to a set of libraries used to support collective compute operations within the Neuron SDK.  The collectives support is delivered via the aws-neuronx-collectives package and includes a pre-built version of the OFI plugin required for use of collectives with Elastic Fabric Adapter (EFA).

.. contents:: Table of contents
   :local:
   :depth: 1

Neuron Collectives [2.23.133.0]
------------------------------
Date: 12/20/2024


New in this release
^^^^^^^^^^^^^^^^^^^
* Added Trainium2 support

Improvements
^^^^^^^^^^^^
* Improved startup times for large scale training jobs by up to 5 seconds
* Enhanced error logging for bootstrap failures

Neuron Collectives [2.22.26.0]
------------------------------
Date: 09/16/2024

New in this release:
^^^^^^^^^^^^^^^^^^^^
* Added check to print out an error message on invalid ``NEURON_RT_ROOT_COMM_ID`` configurations

Bug fixes
^^^^^^^^^
* Resolved an issue where the ``libnccom.so`` filename was versioned incorrectly as ``libnccom.so.2.y.y``. Will be correctly versioned as ``libnccom.so.2.22.26`` in this release.

Neuron Collectives [2.21.46.0]
------------------------------
Date: 07/03/2024

New in this release:
^^^^^^^^^^^^^^^^^^^^

* Bootstrap changes to improve application startup latency for large-scale workloads
* Logging improvements


Neuron Collectives [2.20.22.0]
------------------------------
Date: 04/01/2024

New in this release:
^^^^^^^^^^^^^^^^^^^^

* minor bug fixes and enhancements


Neuron Collectives [2.20.11.0]
------------------------------
Date: 02/13/2024

Bug Fixes
^^^^^^^^^

* Require “libatomic” for rpm installs

Neuron Collectives [2.19.7.0]
------------------------------
Date: 12/21/2023

New in this release
^^^^^^^^^^^^^^^^^^^

* Improve collectives barrier latency from 500us to 40us

Bug Fixes
^^^^^^^^^

* Fix bug where proxy thread blocks the runtime from adding ops leading to an execution hang

Neuron Collectives [2.18.18.0]
------------------------------
Date: 10/26/2023

New in this release:
* Bumpped compatibility version to 17 to align with struct change in the nec.h header


Neuron Collectives [2.17.9.0]
------------------------------
Date: 9/14/2023

New in this release:
* minor bug fixes and enhancements

Neuron Collectives [2.16.16.0]
------------------------------
Date: 9/01/2023

New in this release:
* minor bug fixes and enhancements



Neuron Collectives [2.16.8.0]
------------------------------
Date: 8/28/2023

New in this release:

* Improved error messages for unsupported topologies
* Improved timeout error messages for bootstrapInit

Bug Fixes:
* Fix bug where Linux kernel version check for SAFE_FORK env variable was incorrectly requiring SAFE_FORK to be set on kernel versions greater than 5


Neuron Collectives [2.15.16.0]
------------------------------
Date: 8/09/2023

New in this release:

* minor bug fixes and enhancements


Neuron Collectives [2.15.13.0]
------------------------------
Date: 7/19/2023

New in this release:

* AllReduce with All-to-all communication pattern enabled for 16 ranks on TRN1/TRN1N within the instance (intranode); choice of 16 ranks is limited to NeuronCores 0-15 or 16-31.

Bug Fixes:

* Fix incorrect mask calculation for 16 ranks when using NeuronCores 16-31
* Fix channels for 16 ranks to avoid failures in the runtime; restrict participating ranks to 0-15 or 16-31



Neuron Collectives [2.14.9.0]
------------------------------
Date: 6/14/2023

New in this release

* Added check for FI_EFA_FORK_SAFE environment variable; now forcing the flag to be set to 1 for multinode runs executing on Linux kernels older than 5.15. 


Neuron Collectives [2.13.7.0]
------------------------------
Date: 05/01/2023

New in this release

* Added support for dma_buf - required for future EFA and Linux kernel updates. 
* Reduced benign reporting of timeouts. Previous implementations reported “Timeout waiting for incoming connection” too frequently (log spam).



Neuron Collectives [2.12.35.0]
------------------------------
Date: 04/19/2023

Bug Fixes

* Fixed support for SOCKET_IFNAME config that was affecting EKS users at scale on large training jobs.



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

