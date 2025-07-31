.. _neuron-2-25-0-compiler:

.. meta::
   :description: The official release notes for the AWS Neuron SDK compiler component, version 2.25.0. Release date: 7/31/2025.

AWS Neuron SDK 2.25.0: Neuron Compiler release notes
====================================================

**Date of release**: July 31, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.25.0 release notes home <neuron-2-25-0-whatsnew>`

Announcements
-------------
The Neuron Compiler default for the ``--auto-cast`` option will change from ``--auto-cast=matmult`` to ``--auto-cast=none`` in a future release.

Behavioral changes
------------------

*Behavioral changes are small, user-facing changes that you may notice after upgrading to this version.*

* Minor bug fixes and performance enhancements for both the ``trn1`` and ``trn2`` platforms.


Known issues
------------

* The Llama3 70B test has a compile time increase of 16% and 18%, for 16 and 32 nodes respectively. We are investigating the cause of this increase and will provide an update in the future.
