.. _neuron-2-25-0-nki:

.. meta::
   :description: The official release notes for the AWS Neuron Kernel Interface (NKI) component, version 2.25.0. Release date: 7/31/2025.

AWS Neuron SDK 2.25.0: Neuron Kernel Interace (NKI) release notes
=================================================================

**Date of release**: July 31, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.25.0 release notes home <neuron-2-25-0-whatsnew>`


Behavioral changes
------------------

*Behavioral changes are small, user-facing changes that you may notice after upgrading to this version*.

* Added a new NKI ISA ``select_reduce`` instruction that combines selection and reduction operations in a single instruction. For more details, see :ref:`the NKI API reference <nki_api_reference>`.
* Improved ``nl.store`` ergonomics to perform a copy to sbuf prior to the store to HBM if the value tile is in PSUM. For more details, see :doc:`nl_store <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.store.html>`.
* Added support for ``dict``, ``list``, and ``dataclass`` arguments containing tensors as inputs to NKI kernels.
