.. meta::
   :description: NKI API reference for the AWS Neuron Kernel Interface — the nki, nki.language, nki.isa, and nki.collectives modules for writing and compiling custom kernels on Trainium and Inferentia.
   :keywords: NKI, Neuron Kernel Interface, API reference, nki.language, nki.isa, nki.collectives, nki.jit, nki.simulate, Trainium, Inferentia, custom kernels
   :date-modified: 2026-07-30

.. _nki_api_reference:

========================
NKI API reference manual
========================

The NKI (Neuron Kernel Interface) API reference documents every module you use to
write, compile, and run custom kernels on AWS Trainium and Inferentia. It covers the
high-level kernel language, the low-level hardware ISA, multi-core collectives, and
the decorators that compile and simulate kernels. Use the sections below to jump to
the module you need.

Compile and run kernels
-----------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: nki
      :link: /nki/api/nki
      :link-type: doc

      Top-level entry points for compiling and running NKI kernels, including the
      ``jit`` decorator and the ``simulate`` CPU simulator for debugging.

Write kernels
-------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: nki.language
      :link: /nki/api/nki.language
      :link-type: doc

      High-level constructs for writing kernels: tensor creation, indexing, type
      casting, math operations, and loop constructs the compiler maps to hardware.

Hardware ISA
------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: nki.isa
      :link: /nki/api/nki.isa
      :link-type: doc

      Low-level ISA instructions for compute, data movement, and synchronization,
      mapping to Tensor, Vector, Scalar, and DMA engine operations.

   .. grid-item-card:: NKI ISA common fields
      :link: /nki/api/nki.api.shared
      :link-type: doc

      Shared fields used across ``nki.isa`` APIs, including the supported NKI data
      types accepted by the ``dtype`` parameter.

Multi-core communication
------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: nki.collectives
      :link: /nki/api/nki.collectives
      :link-type: doc

      Collective communication operations such as all-reduce and all-gather for
      multi-rank kernels that run across NeuronCores.

.. toctree::
    :maxdepth: 2
    :hidden:

    nki
    nki.isa
    nki.language
    nki.collectives
    nki.api.shared
