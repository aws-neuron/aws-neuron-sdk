.. meta::
   :description: Operating system, Linux kernel, and GLIBC versions tested with the AWS Neuron Runtime library and the Neuron Driver on Inf2, Trn1, Trn2, and Trn3 instances.
   :keywords: Neuron Runtime, Neuron Driver, compatibility, operating system, Linux kernel, GLIBC, Ubuntu 26.04, Ubuntu 24.04, Amazon Linux 2023, Rocky Linux, Red Hat Enterprise Linux, Debian, aws-neuronx-dkms, aws-neuronx-runtime-lib
   :date-modified: 2026-10-08

.. _neuron-runtime-compatibility:

=================================================
Neuron Runtime and Driver compatibility reference
=================================================

This topic lists the operating system, Linux kernel, and GLIBC versions that were tested with the AWS Neuron Runtime library and the Neuron Driver. Use this page to make sure that a host uses a tested configuration before you install or upgrade Neuron.

Applies to
----------

- Neuron SDK 2.33.0
- Neuron Runtime library 2.35 (``aws-neuronx-runtime-lib``)
- Neuron Driver 2.31 (``aws-neuronx-dkms``)
- Inf2, Trn1, Trn2, and Trn3 instances

Overview
--------

The Neuron Runtime library and the Neuron Driver are tested separately, so this page has one table for each. Each row is one configuration that was tested for the current Neuron SDK release. A configuration that is not in a table can work, but it was not tested.

The Neuron Driver is a kernel module that DKMS compiles for the running kernel when you install ``aws-neuronx-dkms``. For this reason, the kernel version is the most important value for the driver. If you upgrade the kernel on a host, make sure that the new kernel is in the Neuron Driver table, and install the kernel headers before you reinstall the driver. For the steps, see :ref:`Neuron Runtime troubleshooting <nrt-troubleshooting>`.

Neuron Deep Learning AMIs (DLAMIs) and Deep Learning Containers (DLCs) support a smaller set of operating systems. For those lists, see :ref:`Neuron DLAMIs <neuron-dlami-overview>`.

This page shows the configurations for the current release. For the configurations that were tested with an earlier release, see the Compatibility Support Table for that release in the :ref:`Neuron Runtime release notes <runtime_rn>`.

Usage
-----

Use this reference when you:

- Choose an operating system and kernel for a new Neuron host or a custom AMI.
- Plan a kernel or operating system upgrade on hosts that already run Neuron.
- Diagnose a Neuron Driver installation that fails after a kernel upgrade.

.. TODO(SDM): Confirm in review whether Inf2 and Trn1 are tested on Ubuntu 26.04 for Neuron 2.33.0. The Neuron DLAMI page lists Ubuntu 26.04 DLAMIs for Inf2, Trn1, Trn1n, Trn2, and Trn3, but these tables list Ubuntu 26.04 only for Trn2 and Trn3. Update the tables and remove this comment.

Neuron Runtime library tested configurations
--------------------------------------------

The Neuron Runtime library 2.35 was tested with these configurations:

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Instance family
     - Operating system
     - Linux kernel
     - GLIBC
   * - Inf2, Trn1
     - Ubuntu 24.04
     - 6.17
     - 2.39
   * - Inf2, Trn1
     - Debian 12
     - 6.1
     - 2.36
   * - Inf2, Trn1
     - Amazon Linux 2023
     - 6.18
     - 2.34
   * - Inf2, Trn1
     - Amazon Linux 2023
     - 6.1
     - 2.34
   * - Trn2, Trn3
     - Ubuntu 26.04
     - 7.0
     - 2.43
   * - Trn2, Trn3
     - Ubuntu 24.04
     - 6.17
     - 2.39
   * - Trn2, Trn3
     - Rocky Linux 10
     - 6.12
     - 2.39
   * - Trn2, Trn3
     - Debian 12
     - 6.1
     - 2.36
   * - Trn2, Trn3
     - Amazon Linux 2023
     - 6.18
     - 2.34

Neuron Driver tested configurations
-----------------------------------

The Neuron Driver 2.31 was tested with these configurations:

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Instance family
     - Operating system
     - Linux kernel
     - GLIBC
   * - Inf2, Trn1
     - Ubuntu 24.04
     - 6.17
     - 2.39
   * - Inf2, Trn1
     - Red Hat Enterprise Linux 10
     - 6.12
     - 2.39
   * - Inf2, Trn1
     - Debian 12
     - 6.1
     - 2.36
   * - Inf2, Trn1
     - Amazon Linux 2023
     - 6.18
     - 2.34
   * - Inf2, Trn1
     - Amazon Linux 2023
     - 6.1
     - 2.34
   * - Trn2, Trn3
     - Ubuntu 26.04
     - 7.0
     - 2.43
   * - Trn2, Trn3
     - Ubuntu 24.04
     - 6.17
     - 2.39
   * - Trn2, Trn3
     - Rocky Linux 10
     - 6.12
     - 2.39
   * - Trn2, Trn3
     - Red Hat Enterprise Linux 10
     - 6.12
     - 2.39
   * - Trn2, Trn3
     - Debian 12
     - 6.1
     - 2.36
   * - Trn2, Trn3
     - Amazon Linux 2023
     - 6.18
     - 2.34
   * - Trn2, Trn3
     - Amazon Linux 2023
     - 6.1
     - 2.34

Related reference
-----------------

- :ref:`Neuron Runtime release notes <runtime_rn>`: changes in each release, and the configurations tested with earlier releases.
- :ref:`Neuron Runtime troubleshooting <nrt-troubleshooting>`: fix Neuron Driver installation failures, including missing kernel headers.
- :ref:`Neuron SDK maintenance policy <sdk-maintenance-policy>`: how Neuron adds and ends support for operating systems.
- :ref:`Neuron DLAMIs <neuron-dlami-overview>`: operating systems for the Neuron Deep Learning AMIs.

Learn more
----------

- :ref:`About the Neuron Runtime <neuron-runtime-about>`
- :ref:`Install PyTorch for Neuron <pytorch-setup>`
- :ref:`Install JAX for Neuron <jax-setup>`
