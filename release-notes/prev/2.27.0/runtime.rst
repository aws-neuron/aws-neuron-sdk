.. _neuron-2-27-0-runtime:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Runtime component, version 2.27.0. Release date: 12/19/2025.

AWS Neuron SDK 2.27.0: Neuron Runtime release notes
===================================================

**Date of release**:  December 19, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.27.0 release notes home <neuron-2-27-0-whatsnew>`

Released versions
-----------------

- Neuron Driver: ``2.25.4.0``
- Neuron Runtime Library: ``2.29.40.0``
- Neuron Collectives: ``2.29.41.0``

Compatibility Support Tables
----------------------------
**Runtime Version:** 2.29.40.0

The Neuron runtime was tested for the following EC2 instances and configurations:

=========================== ============= ============== ================= ===============
Instance Family               OS Type       OS Version     Kernel Version    GLIBC Version
=========================== ============= ============== ================= ===============
``Inf2``                    Ubuntu        U24            6.14              2.39
``Inf2``                    Ubuntu        U22            6.8               2.35
``Inf2``                    Rocky Linux   RL9            5.14              2.34
``Inf2``                    Amazon Linux  AL2023         6.12              2.34
``Trn1``                    Ubuntu        U24            6.14              2.39
``Trn1``                    Ubuntu        U22            6.8               2.35
``Trn1``                    Rocky Linux   RL9            5.14              2.34
``Trn1``                    Amazon Linux  AL2023         6.12              2.34
``Trn2``                    Ubuntu        U24            6.14              2.39
``Trn2``                    Ubuntu        U22            6.8               2.35
``Trn2``                    Amazon Linux  AL2023         6.12              2.34
=========================== ============= ============== ================= ===============

**Driver Version:** 2.25.4.0

The Neuron driver was tested for the following EC2 instances and configurations:

=========================== ============= ============== ================= ===============
Instance Family               OS Type       OS Version     Kernel Version    GLIBC Version
=========================== ============= ============== ================= ===============
``Inf2``                    Ubuntu        U24            6.14              2.39
``Inf2``                    Ubuntu        U22            6.8               2.35
``Inf2``                    Rocky Linux   RL9            5.14              2.34
``Inf2``                    Amazon Linux  AL2023         6.12              2.34
``Inf2``                    Amazon Linux  AL2            5.10              2.26
``Trn1``                    Ubuntu        U24            6.14              2.39
``Trn1``                    Ubuntu        U22            6.8               2.35
``Trn1``                    Rocky Linux   RL9            5.14              2.34
``Trn1``                    Amazon Linux  AL2023         6.12              2.34
``Trn1``                    Amazon Linux  AL2            5.10              2.26
``Trn2``                    Ubuntu        U24            6.14              2.39
``Trn2``                    Ubuntu        U22            6.8               2.35
``Trn2``                    Amazon Linux  AL2023         6.12              2.34
``Trn2``                    Amazon Linux  AL2            5.10              2.26
=========================== ============= ============== ================= ===============

What's New
----------

Neuron Runtime Library 2.29.40.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Added support for Trainium3 (single node mode)

Neuron Driver 2.25.4.0
^^^^^^^^^^^^^^^^^^^^^^^^

* Added support for Trainium3

Improvements
------------

Neuron Runtime Library 2.29.40.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Reduced the overhead of reprogramming the Collectives Engine by up to 100x for NEFFs compiled with the ``-O1`` flag. This improves end-to-end performance of these NEFFs by up to 15%.
* Reduced NeuronCore branch overhead by up to 3x, decreasing the overhead of starting a NEFF program by up to 5%.
* Reduced the overhead of starting a NEFF program by up to 50% with an on-device hardware barrier between ranks.
* Improved all-gather latency by up to 35% for messages greater than 1MB in TP8 (LNC2) and TP16 (LNC1) collectives.
* Added support for :ref:`NRT Debug Stream APIs <nrt-debug-stream-api>`.

Bug fixes
---------

Neuron Runtime Library 2.29.40.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Fixed scratchpad page allocation bug that caused excessive page allocations due to page rounding error.
* Fixed segfault that occurred when freeing an empty tensor.

Previous release notes
----------------------

* :ref:`neuron-2-26-0-runtime`