.. _neuron-2-27-0-nkilib:

.. meta::
   :description: The official release notes for the NKI Library component, version 2.27.0. Release date: 12/19/2025.

AWS Neuron SDK 2.27.0: NKI Library release notes
=================================================

**Date of release**: December 19, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.27.0 release notes home <neuron-2-27-0-whatsnew>`

What's New
----------

This release introduces the NKI Library, which provides pre-built kernels you can use to optimize
the performance of your models. The NKI Library offers ready-to-use, pre-optimized kernels that
leverage the full capabilities of AWS Trainium hardware.

NKI Library kernels are published in the `NKI Library GitHub repository <https://github.com/aws-neuron/nki-library>`_.
In Neuron 2.27, these kernels are also shipped as part of neuronx-cc under the ``nkilib.*`` namespace.

Accessing NKI Library Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can access NKI Library kernels in two ways:

* **Shipped version**: Import from the ``nkilib.*`` namespace (included with neuronx-cc in Neuron 2.27)
* **Open source repository**: Clone and use kernels from the GitHub repository under the ``nkilib_standalone.nkilib.*`` namespace

New Kernels
^^^^^^^^^^^

This release includes the following pre-optimized kernels:

* **Attention CTE Kernel** — Implements attention with support for multiple variants and optimizations
* **Attention TKG Kernel** — Implements attention specifically optimized for token generation scenarios
* **MLP Kernel** — Implements a Multi-Layer Perceptron with optional normalization fusion and various optimizations
* **Output Projection CTE Kernel** — Computes the output projection operation optimized for Context Encoding use cases
* **Output Projection TKG Kernel** — Computes the output projection operation optimized for Token Generation use cases
* **QKV Kernel** — Performs Query-Key-Value projection with optional normalization fusion
* **RMSNorm-Quant Kernel** — Performs optional RMS normalization followed by quantization to fp8

NKI Library Kernel Migration to New nki.* Namespace in Neuron 2.28
-------------------------------------------------------------------

Some NKI Library kernels currently use the legacy ``neuronxcc.nki.*`` namespace. Starting with
Neuron 2.28, all NKI Library kernels will migrate to the new ``nki.*`` namespace.

The new ``nki.*`` namespace introduces changes to NKI APIs and language constructs. Customers
using NKI Library kernels should review the migration guide for any required changes.

NKI Library Namespace Changes in Neuron 2.28
---------------------------------------------

Starting with Neuron 2.28, the open source repository namespace will change from
``nkilib_standalone.nkilib.*`` to ``nkilib.*``, providing a consistent namespace between
the open source repository and the shipped version.

Customers who want to add or modify NKI Library kernels can build and install them to
replace the default implementation without changing model imports.

