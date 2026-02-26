.. _neuron-2-27-0-nki:

.. meta::
   :description: The official release notes for the AWS Neuron Kernel Interface (NKI) component, version 2.27.0. Release date: 12/19/2025.

AWS Neuron SDK 2.27.0: Neuron Kernel Interface (NKI) release notes
===================================================================

**Date of release**: December 19, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.27.0 release notes home <neuron-2-27-0-whatsnew>`

What's New
----------

This release introduces NKI Beta 2, featuring the new :doc:`NKI Compiler </nki/deep-dives/nki-compiler>`
and significant enhancements to the NKI language constructs and APIs, including changes to existing APIs. 
For information about the different NKI Beta versions, see :doc:`About NKI Beta Versions </nki/deep-dives/nki-beta-versions>`.

To take advantage of Beta 2 with the new compiler, import the ``nki.*`` namespace in your code
and annotate your top-level kernel function with ``@nki.jit``.

**Backward Compatibility and Migration**

Neuron 2.27 supports both the ``neuronxcc.nki.*`` and ``nki.*`` namespaces side by side,
allowing existing Beta 1 kernels to continue working seamlessly. However, Neuron 2.27 will
be the last release to include support for the ``neuronxcc.nki.*`` namespace. Starting with
Neuron 2.28, this namespace will no longer be supported.

The new ``nki.*`` namespace introduces changes to NKI APIs and language constructs. We
encourage customers to migrate existing kernels from ``neuronxcc.nki.*`` to the new ``nki.*``
namespace. A kernel migration guide is available in the Neuron 2.27 documentation to assist
with this transition.

New nki.language APIs
^^^^^^^^^^^^^^^^^^^^^

* :doc:`nki.language.device_print </nki/api/generated/nki.language.device_print>`

New nki.isa APIs
^^^^^^^^^^^^^^^^

* :doc:`nki.isa.dma_compute </nki/api/generated/nki.isa.dma_compute>`
* :doc:`nki.isa.quantize_mx </nki/api/generated/nki.isa.quantize_mx>`
* :doc:`nki.isa.nc_matmul </nki/api/generated/nki.isa.nc_matmul>`
* :doc:`nki.isa.nc_n_gather </nki/api/generated/nki.isa.nc_n_gather>` [used to be ``nl.gather_flattened`` with free partition limited to 512]
* :doc:`nki.isa.rand2 </nki/api/generated/nki.isa.rand2>`
* :doc:`nki.isa.rand_set_state </nki/api/generated/nki.isa.rand_set_state>`
* :doc:`nki.isa.rand_get_state </nki/api/generated/nki.isa.rand_get_state>`
* :doc:`nki.isa.set_rng_seed </nki/api/generated/nki.isa.set_rng_seed>`
* :doc:`nki.isa.rng </nki/api/generated/nki.isa.rng>`

New dtypes
^^^^^^^^^^^^^^

* :doc:`nki.language.float8_e5m2_x4 </nki/api/generated/nki.language.float8_e5m2_x4>`
* :doc:`nki.language.float4_e2m1fn_x4 </nki/api/generated/nki.language.float4_e2m1fn_x4>`
* :doc:`nki.language.float8_e4m3fn_x4 </nki/api/generated/nki.language.float8_e4m3fn_x4>`

Changes to Existing APIs
^^^^^^^^^^^^^^^^^^^^^^^^

* Several nki.language APIs have been removed in NKI Beta 2
* All nki.isa APIs have ``dst`` as an input param
* All nki.isa APIs removed ``dtype`` and ``mask`` support
* :doc:`nki.isa.memset </nki/api/generated/nki.isa.memset>` — removed ``shape`` positional arg , since we have ``dst``
* :doc:`nki.isa.affine_select </nki/api/generated/nki.isa.affine_select>` — instead of ``pred``, we now take ``pattern`` and ``cmp_op`` params
* :doc:`nki.isa.iota </nki/api/generated/nki.isa.iota>` — ``expr`` replaced with ``pattern`` and ``offset``
* :doc:`nki.isa.nc_stream_shuffle </nki/api/generated/nki.isa.nc_stream_shuffle>` - ``src`` and ``dst`` order changed

Documentation Updates
^^^^^^^^^^^^^^^^^^^^^^

* Restructured NKI Documentation to align with workflows
* Added :doc:`Trainium3 Architecture Guide for NKI </nki/guides/architecture/trainium3_arch>`
* Added :doc:`About Neuron Kernel Interface (NKI) </nki/get-started/about/index>`
* Added :doc:`NKI Environment Setup Guide </nki/get-started/setup-env>`
* Added :doc:`Get Started with NKI </nki/get-started/quickstart-implement-run-kernel>`
* Added :doc:`NKI Language Guide </nki/get-started/nki-language-guide>`
* Added :doc:`About the NKI Compiler </nki/deep-dives/nki-compiler>`
* Added :doc:`About NKI Beta Versions </nki/deep-dives/nki-beta-versions>`
* Added :doc:`MXFP Matrix Multiplication with NKI </nki/deep-dives/mxfp-matmul>`
* Updated :doc:`Matrix Multiplication Tutorial </nki/guides/tutorials/matrix_multiplication>`
* Updated :doc:`Profile a NKI Kernel </nki/deep-dives/use-neuron-profile>`
* Updated :doc:`NKI APIs </nki/api/index>`
* Updated :doc:`NKI Library docs </nki/library/index>`
* Removed NKI Error Guide

Known issues
------------

* :doc:`nki.isa.nc_matmul </nki/api/generated/nki.isa.nc_matmul>` - ``is_moving_onezero`` was incorrectly named ``is_moving_zero`` in this release
* NKI ISA semantic checks are not available with Beta 2, workaround is to reference the API docs
* NKI Collectives are not available with Beta 2
* nki.benchmark and nki.profile are not available
