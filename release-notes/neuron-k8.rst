.. _neuron-k8-rn:

Neuron K8 Release Notes
^^^^^^^^^^^^^^^^^^^^^^^

Introduction
============

This document lists the current release notes for AWS Neuron Kubernetes
(k8) components. Neuron K8 components include a device plugin and a
scheduler extension to assist with deployment and management of inf1
nodes within Kubernetes clusters. Both components are offered as
pre-built containers in Public ECR and ready for deployment.

-  **Device Plugin:**
   public.ecr.aws/neuron/neuron-device-plugin:1.5.3.0
-  **Neuron Scheduler:**
   public.ecr.aws/neuron/neuron-scheduler:1.5.3.0

It's recommended to pin the version of the components used and to never
use the "latest" tag. To get the list of image tags, please refer to
these notes or check the image tags on the repo directly.


To Pull the Images from ECR:

::

   docker pull  public.ecr.aws/neuron/neuron-device-plugin:1.5.3.0
   docker pull  public.ecr.aws/neuron/neuron-scheduler:1.5.3.0

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
   :neuron-deploy:`k8s-neuron-device-plugin-rbac.yml <k8s-neuron-device-plugin-rbac.yml>`
   before using the new device plugin.

Resolved Issues
---------------

-  Scheduler is intended to address
   https://github.com/aws/aws-neuron-sdk/issues/110
