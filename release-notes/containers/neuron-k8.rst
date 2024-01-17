.. _neuron-k8-rn:

Neuron K8 Release Notes
^^^^^^^^^^^^^^^^^^^^^^^

.. contents:: Table of contents
   :local:
   :depth: 1


Introduction
============

This document lists the current release notes for AWS Neuron Kubernetes
(k8) components. Neuron K8 components include a device plugin and a
scheduler extension to assist with deployment and management of inf/trn
nodes within Kubernetes clusters. Both components are offered as
pre-built containers in Public ECR and ready for deployment.

-  **Device Plugin:**
   public.ecr.aws/neuron/neuron-device-plugin:2.x.y.z
-  **Neuron Scheduler:**
   public.ecr.aws/neuron/neuron-scheduler:2.x.y.z

It's recommended to pin the version of the components used and to never
use the "latest" tag. To get the list of image tags (2.x.y.z), please refer to
these notes or check the image tags on the repo directly.


To Pull the Images from ECR:

::

   docker pull  public.ecr.aws/neuron/neuron-device-plugin:2.x.y.z
   docker pull  public.ecr.aws/neuron/neuron-scheduler:2.x.y.z

.. _1622:

Neuron K8 release [2.19.16.0]
===========================

Date: 01/18/2024

Critical Security Patch
---------

We updated the dependencies used by the Neuron Device Plugin and the Neuron Kubernetes Scheduler to fix several important
security vulnerabilities.
This update fixes all security vulnerabilities reported in https://github.com/aws-neuron/aws-neuron-sdk/issues/817.
Please see the ticket for all impacted dependencies and their associated vulnerabilities.


.. _1622:

Neuron K8 release [2.16.18.0]
===========================

Date: 09/01/2023

Major New Features
------------------

- Previously, the Neuron Device indexing was assigned randomly, which made programming difficult.  Changed to using 0-based indexing for Neuron Devices and NeuronCores in EKS container environments; requires Neuron Driver version 2.12.14 or newer.  
- Improved logging when Neuron Driver not installed/present.

Bug Fixes
---------

- Fixed Neuron Device Plugin crash when Neuron Driver is not installed/present on the host.
- Fixed issue where pods fail to deploy when multiple containers are requesting Neuron resources.
- Fixed issue where launching many pods each requesting Neuron cores fails to deploy.


.. _1622:

Neuron K8 release [2.1.0.0]
===========================

Date: 10/27/2022

Summary
-------

- Added support for NeuronCore based scheduling to the Neuron Kubernetes Scheduler.  Learn more about how to use NeuronCores for finer grain control over container scheduling by following the K8 tutorials documentation in the :ref:`containers section <neuron_containers>`.

.. _1622:

Neuron K8 release [2.0.0.0]
===========================

Date: 10/10/2022

Summary
-------

- Added support for TRN1 and INF1 EC2 instance types.


Neuron K8 release [1.9.3.0]
===========================

Date: 08/02/2022

Summary
-------

- Minor updates.


Neuron K8 release [1.9.2.0]
===========================

Date: 05/27/2022

Summary
-------

- Minor updates.


Neuron K8 release [1.9.0.0]
===========================

Date: 04/29/2022

Summary
-------

- Minor updates.


Neuron K8 release [1.8.2.0]
===========================

Date: 03/25/2022

Summary
-------

- Minor updates.


Neuron K8 release [1.7.7.0]
===========================

Date: 01/20/2022

Summary
-------

Minor updates

Neuron K8 release [1.7.3.0]
===========================

Date: 10/27/2021

Summary
-------

Minor updates


[1.6.22.0]
=========

Date: 08/30/2021

Summary
-------

Minor updates.


.. _1615:

[1.6.15.0]
=========

Date: 08/06/2021

Summary
-------

Minor updates.



.. _1670:

[1.6.7.0]
=========

Date: 07/26/2021

Summary
-------

Minor internal enhancements.

.. _1600:

[1.6.0.0]
=========

Date: 07/02/2021

Summary
-------

Minor internal enhancements.

.. _1530:

[1.5.3.0]
=========

Date: 05/01/2021

Summary
-------

Minor internal enhancements.


.. _1410:

[1.4.1.0]
=========

Date: 01/30/2021

Summary
-------

Minor internal enhancements.


.. _1320:

[1.3.2.0]
=========

Date: 12/23/2020

Summary
-------

Minor internal enhancements.

.. _1200:

[1.2.0.0]
=========

Date: 11/17/2020

Summary
-------

Minor internal enhancements.

.. _11230:

[1.1.23.0]
==========

Date: 10/22/2020

.. _summary-1:

Summary
-------

Support added for use with Neuron Runtime 1.1. More details in the
Neuron Runtime release notes at :ref:`neuron-runtime-release-notes`.


.. _11170:

[1.1.17.0]
==========

Date: 09/22/2020

Summary
-------

Minor internal enhancements.

.. _10110000:

[1.0.11000.0]
=============

Date: 08/08/2020

.. _summary-1:

Summary
-------

First release of the Neuron K8 Scheduler extension.

Major New Features
------------------

-  New scheduler extension is provided to ensure that kubelet is
   scheduling pods on inf1 with contiguous device ids. Additional
   details about the new scheduler are provided :ref:`neuron-k8-scheduler-ext`.
   including instructions on how to apply it.

   -  NOTE: The scheduler is only required when using inf1.6xlarge
      and/or inf1.24xlarge

-  With this release the device plugin now requires RBAC permission
   changes to get/patch NODE/POD objects. Please apply the 
   :github:`k8s-neuron-device-plugin-rbac.yml </src/k8/k8s-neuron-device-plugin-rbac.yml>`
   before using the new device plugin.

Resolved Issues
---------------

-  Scheduler is intended to address
   https://github.com/aws/aws-neuron-sdk/issues/110
