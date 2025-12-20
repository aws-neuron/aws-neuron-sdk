.. meta::
    :description: Introduction to AWS Neuron Kernel Interface (NKI), a bare-metal programming interface for direct access to AWS NeuronDevices on Inf2, Trn1, Trn2, and Trn3 instances.
    :keywords: NKI, AWS Neuron, Kernel Interface, Trainium, Inferentia
    :date-modified: 12/13/2025

.. _neuron-nki:

Neuron Kernel Interface (NKI) Documentation
===========================================

.. admonition:: NKI Beta Versions

      NKI is currently in beta, with Beta 2 as the current version. Read more about :doc:`NKI beta versions </nki/deep-dives/nki-beta-versions>`.

The Neuron Kernel Interface (NKI) is a Python-embedded Domain Specific Language (DSL) that gives developers direct access to Neuron’s Instruction Set Architecture (NISA). NKI provides the ease-of-programming offered by tile-level operations and full access to the Neuron Instruct Set Architecture within a familiar pythonic programming environment. It provides the flexibility to implement architecture-specific optimizations rapidly, at a speed difficult to achieve in higher-level DSLs and frameworks. This has enabled developers to achieve optimal performance across a wide spectrum of machine learning models on Trainium, including Transformers, Mixture-of-Experts, State Space Models, and more. 

In addition to directly exposing NISA, NKI provides easy-to-use APIs for controlling instruction scheduling, memory management across the memory hierarchy, software pipelining, and other optimization techniques. The APIs are carefully designed to help simplify the code while providing more control and flexibility to developers. This gives developers fine-grained tuning optimizations that work in concert with the capabilities provided by the compiler.

NKI currently supports multiple NeuronDevice generations:

* Trainium/Inferentia2, available on AWS ``trn1``, ``trn1n`` and ``inf2`` instances
* Trainium2, available on AWS ``trn2`` instances and UltraServers
* Trainium3, available on AWS ``trn3`` instances and UltraServers

Explore the comprehensive guides below to learn how to implement and optimize your kernels for AWS Neuron accelerators:

.. grid:: 1
      :margin: 2

      .. grid-item::

            .. card:: About NKI
                  :link: nki_about_home
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

                  Learn about Neuron Kernel Interface (NKI) and core concepts essential for working with it.

      .. grid-item::

            .. card:: NKI Language Guide
                  :link: nki-language-guide
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

                  Developer guide for NKI's Pythonic language syntax.

      .. grid-item::

            .. card:: NKI Compiler Documentation
                  :link: /nki/deep-dives/nki-compiler
                  :link-type: doc
                  :class-body: sphinx-design-class-title-small

                  Documentation for the NKI compiler and its integration with the Neuron Compiler.        
 
      .. grid-item::

            .. card:: NKI Library Documentation
                  :link: nkl_home
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

                  API documentation for the set of pre-built kernels in the NKI Library.

.. _functional_docs:

Writing NKI Kernels
^^^^^^^^^^^^^^^^^^^

.. grid:: 2
      :margin: 2 2 1 1

      .. grid-item::

            .. card:: Getting Started with NKI
                  :link: quickstart-run-nki-kernel
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

      .. grid-item::

            .. card:: NKI Tutorials
                  :link: /nki/guides/tutorials/index
                  :link-type: doc
                  :class-body: sphinx-design-class-title-small


.. _performant_docs:

Optimizing NKI Kernels
^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 2
      :margin: 4 1 0 0

      .. grid-item::

            .. card:: Profiling a NKI Kernel with Neuron Explorer
                  :link: /nki/deep-dives/use-neuron-profile
                  :link-type: doc
                  :class-body: sphinx-design-class-title-small

      .. grid-item::

            .. card:: NKI Performance Optimizations
                  :link: nki_perf_guide
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

.. toctree::
      :maxdepth: 1
      :hidden:

      NKI Release Notes </nki/nki_rn>
      NKI FAQ <nki_faq>
