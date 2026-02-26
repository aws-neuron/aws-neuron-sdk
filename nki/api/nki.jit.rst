.. _nki-jit-reference:

nki.jit Decorator Reference
============================

.. currentmodule:: nki

This topic provides reference for using NKI JIT decorators and kernel execution APIs
with the AWS Neuron SDK. Use this page to look up the available decorators for compiling
and running NKI kernels on NeuronDevices.

Overview
--------

NKI provides decorators and APIs for compiling and executing kernels on NeuronDevices.
The primary entry point is :doc:`nki.jit <generated/nki.jit>`, which automatically detects the
ML framework in use and compiles the kernel accordingly.

Compilation and Execution
--------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   jit

Related reference
------------------

* :ref:`nki_api_reference` - Full NKI API reference
* :ref:`nki-language` - NKI language operations and data types
