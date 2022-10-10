.. _sdk-maintenance-policy:
.. _neuron-maintenance-policy:

SDK Maintenance Policy
======================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

This document outlines the maintenance policy for AWS Neuron Software Development Kit (SDK) and its underlying dependencies. AWS regularly provides the Neuron SDK with updates that may contain support for new or updated APIs, new features, enhancements, bug fixes, security patches, or documentation updates. Updates may also address changes with dependencies, language runtimes, and operating systems. Neuron SDK releases are available as Conda ( up to :ref:`Neuron 1.13.0 <eol-conda-packages>` ) and Pip Packages that can be installed within Amazon Machine Images (AMIs). 

We recommend users to stay up-to-date with SDK releases to keep up with the latest features, security updates, and underlying dependencies. Continued use of an unsupported SDK version is not recommended and is done at the user’s discretion.

Neuron SDK
----------

AWS Neuron is the SDK for `AWS Inferentia <https://aws.amazon.com/machine-learning/inferentia/>`__, the custom designed machine learning chips enabling high-performance deep learning inference applications on `EC2 Inf1 instances <https://aws.amazon.com/ec2/instance-types/inf1/>`__. Neuron includes a deep learning compiler, runtime and tools that are natively integrated into TensorFlow, PyTorch and MXNet. With Neuron, you can develop, profile, and deploy high-performance inference applications on top of `EC2 Inf1 instances <https://aws.amazon.com/ec2/instance-types/inf1/>`__.

The Neuron SDK release versions are in the form of X.Y.Z where X represents the major version and Y represent the minor version. Increasing the major version of an SDK indicates that this SDK underwent significant and substantial changes, and some of those changes may not maintain the same programming model. 
Increasing the minor version of an SDK indicates that this SDK underwent addition of new features, support of new dependency software versions, end-of-support of certain dependency software, enhacement and/or bugfixes.
Applications may need to be updated in order for them to work with the newest SDK version. It is important to update major versions carefully and in accordance with the upgrade guidelines provided by AWS.


Dependency Software
-------------------

Neuron SDK has underlying dependencies, such as language runtimes, operating systems, or third party libraries and machine learning frameworks. These dependencies are typically tied to the language community or the vendor who owns that particular component. The following terms are used to classify underlying dependencies:

* Operating system (OS): Examples include Amazon Linux AMI, Amazon Linux 2.

* Language runtime: Examples include Python.

* Third party library / framework: Examples include PyTorch, TensorFlow, MXNet and ONNX.

Each community or vendor maintains their own versioning policy and publishes their own end-of-support schedule for their product.


Neuron SDK version life-cycle
-----------------------------

The life-cycle for Neuron SDK version consists of 3 phases, which are outlined below.

- **Supported (Phase 1)**
  
  During this phase, AWS will provide critical bugfixes and security patches. Usually AWS will support each Neuron SDK version for at least 12 months, but AWS reserves the right to stop supporting an SDK version before the 12 months period.

  .. note::

   AWS will address new features or Dependency Software updates by publishing a new version with an increment in the Neuron SDK minor version.


- **End-of-Support Announcement (Phase 2)**
  
  AWS will announce the End-of-Support phase at least 3 months before a specific Neuron SDK version enters End-of-Support phase. During this period, the SDK will continue to be supported.

- **End-of-Support (Phase 3)**
  
  When a Neuron SDK version reaches end-of support, it will no longer receive critical bugfixes and security patches. Previously published Neuron SDK versions will continue to be available via Conda ( up to :ref:`Neuron 1.13.0 <eol-conda-packages>` ) or Pip packages.
  Use of an SDK version which has reached end-of-support is done at the user’s discretion. We recommend users to upgrade to the latest Neuron SDK version.


Dependency Software version life-cycle
--------------------------------------

The life-cycle for Dependency Software version consists of 4 phases, but there may not be a Phase 3 (Maintenance) period in some cases. The phases are outlined below.

- **Supported (Phase 1)**
  
  During this phase, Dependency Software version is supported. AWS will provide regular updates, bug fixes and/or security patches to the Dependency Software version, AWS will address those updates and bug fixes by including them in a new Neuron SDK version with an increment in the Neuron SDK minor version.  There is no minimum support period for a Dependency Software version.

- **Maintenance and/or End-of-Support Announcement (Phase 2)**
  
  AWS will announce the Maintenance phase or the End-of-Support phase of Dependency Software version.
  
  Since each community or vendor maintains their own versioning policy and publishes their own end-of-support schedule for their product, there is no minimum duration to do the announcement before Dependency Software version enters Maintenance phase or End-of-Support phase and in some cases the announcement can happen at the same time when the Dependency Software version enters Maintenance phase or End-of-Support phase.
  
  During this period, the Dependency Software version will continue to be supported.

- **Maintenance (Phase 3)**
  
  During the maintenance phase, AWS limits Dependency Software version to address critical bug fixes and security issues only. There is no minimum Maintenance period.

  This phase is optional and AWS will reserve the right to skip it for specific Dependency Software products.

- **End-of-Support (Phase 4)**
  
  When a Dependency Software version reaches end-of support, it will no longer receive updates or releases. Previously published releases will continue to be available via Conda ( up to :ref:`Neuron 1.13.0 <eol-conda-packages>` ) or Pip packages. Use of an SDK which has reached end-of-support is done at the user’s discretion. We recommend users to upgrade to the new major version.

  When a Dependency Software version reaches end-of support, it will no longer receive critical bugfixes and security patches. Previously published Dependency Software versions will continue to be available via Neuron SDK Conda ( up to :ref:`Neuron 1.13.0 <eol-conda-packages>` ) or Pip packages.

  Use of a Dependency Software version which has reached end-of-support is done at the user’s discretion. We recommend users to upgrade to the latest Neuron SDK version that include the latest Dependency Software versions.


.. note::

   AWS reserves the right to stop support for an underlying dependency without a maintenance phase.

Communication
-------------

Maintenance and End-Of-Support announcements are communicated as follows:

* Neuron SDK documentation.

To see the list of available Neuron SDK versions and supported Dependency Software versions see :ref:`neuron-release-content` and :ref:`neuron-whatsnew` in latest Neuron version.
