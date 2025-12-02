.. meta::
    :description: Introduction to AWS Neuron Kernel Interface (NKI), a bare-metal programming interface for direct access to AWS NeuronDevices on Inf2, Trn1, Trn2, and Trn3 instances.
    :keywords: NKI, AWS Neuron, Kernel Interface, Trainium, Inferentia
    :date-modified: 12/01/2025

.. _neuron-nki:

About the Neuron Kernel Interface (NKI) - Beta 2
================================================

The Neuron Kernel Interface (NKI) is a bare-metal programming interface that enables direct access to AWS NeuronDevices available on Inf2, Trn1, Trn2, and Trn3 instances. NKI empowers ML developers to write high-performance kernel functions that can be integrated into PyTorch and JAX models, allowing fine-grained control over hardware resources while maintaining a familiar programming model.

.. admonition:: NKI Beta versions

      NKI is currently in beta, with Beta 2 as the current shipped version. Read more about :doc:`NKI beta versions here </nki/about/nki-beta-versions>`.

With NKI, you can develop, optimize, and run custom operators directly on NeuronCores, making full use of available compute engines and memory resources. This interface bridges the gap between high-level machine learning frameworks and the specialized hardware capabilities of AWS Neuron accelerators, enabling you to self-serve and invent new ways to use the NeuronCore hardware.

NKI currently supports multiple NeuronDevice generations:

* Trainium/Inferentia2, available on AWS ``trn1``, ``trn1n`` and ``inf2`` instances
* Trainium2, available on AWS ``trn2`` instances and UltraServers
* Trainium3, available on AWS ``trn3`` instances and UltraServers

NKI provides a Python-based programming environment with syntax and tile-level semantics similar to `Triton <https://triton-lang.org/main/index.html>`_ and `NumPy <https://numpy.org/doc/stable/>`_, enabling you to get started quickly while still having full control of the underlying hardware. At the hardware level, NeuronCore's tensorized memory access capability enables efficient reading and writing of multi-dimensional arrays on a per-instruction basis, making NKI's tile-based programming highly suitable for the NeuronCore instruction set.

For comparison, before NKI was introduced, the only way to program NeuronDevices was through defining high-level ML models in frameworks such as `PyTorch <https://pytorch.org/>`_ and `JAX <https://jax.readthedocs.io/en/latest/index.html>`_. The :doc:`Neuron Graph Compiler </compiler/index>` takes such high-level model definitions as input, performs multiple rounds of optimization, and eventually generates a NEFF (Neuron Executable File Format) that is executable on NeuronDevices. At a high level, the Graph Compiler runs the following optimization stages in order:

1. **Hardware-agnostic graph-level optimizations.** These transformations are done in the Graph Compiler's front-end, using `XLA <https://openxla.org/xla>`_ and including optimizations like constant propagation, re-materialization and operator fusion.

2. **Loop-level optimization.** THe Graph Compiler turns the optimized graph from stage 1 into a series of loop nests and performs layout, tiling and loop fusion optimizations.

3. **Hardware intrinsics mapping.** The Graph Compiler maps the architecture-agnostic loop nests from stage 2 into architecture-specific instructions.

4. **Hardware-specific optimizations.** These optimizations are mainly done at the instruction level in the Graph Compiler's back-end, with a key goal of reducing memory pressure and improving instruction-level parallelism. For example, memory allocation and instruction scheduling are done in this stage.

**NKI kernels bypass the first 3 stages through the specialized NKI Compiler, which translates kernel code directly into IRs (intermediate representations) that the Neuron Compiler's back-end can immediately process**. The NKI Compiler serves as a critical bridge, converting high-level NKI code into optimized low-level representations while preserving developer-specified optimizations. This direct path to lower-level compilation provides significant performance advantages and preserves fine-grained control. 

Advanced features in NKI, such as direct allocation, further enable programmers to bypass specific compiler passes in stage 4, giving developers precise control over NeuronDevices down to the instruction level. The NKI Compiler's targeted optimizations complement the Neuron Compiler's back-end capabilities, creating a powerful toolchain for hardware-specific acceleration. For optimal kernel performance, Neuron strongly recommends studying the underlying hardware architecture before optimization. 

Explore the comprehensive guides below to learn how to optimize your kernels for AWS Neuron hardware:

.. grid:: 1
      :margin: 2

      .. grid-item::

            .. card:: NKI Language Guide (Beta 2)
                  :link: /nki/deep-dives/nki-language-guide
                  :link-type: doc
                  :class-body: sphinx-design-class-title-small

                  Developer guide for NKI's Pythonic language syntax.

      .. grid-item::

            .. card:: NKI Compiler Documentation
                  :link: nki_compiler_home
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

                  Documentation for the NKI compiler and its integration with the Neuron Compiler.        
 
      .. grid-item::

            .. card:: NKI Library Documentation
                  :link: nkl_home
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

                  API documentation for the set of pre-built kernels in the NKI Library .

.. _api_reference_guide:

API Reference Guides
^^^^^^^^^^^^^^^^^^^^^

.. grid:: 2
      :margin: 4 1 0 0

      .. grid-item::

            .. card:: NKI API Reference Documentation
                  :link: nki_api_reference
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

                  API documentation for the Neuron Kernel Interface (NKI) programming model.
 
      .. grid-item::

            .. card:: Neuron Kernel Reference Documentation
                  :link: nkl_api_ref_home
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

                  API documentation for the set of pre-built kernels in the NKI Library.

.. _functional_docs:

Write Functional NKI Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 2
      :margin: 2 2 1 1

      .. grid-item::

            .. card:: Getting Started with NKI
                  :link: nki_getting_started
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small


      .. grid-item::

            .. card:: NKI Programming Model (Beta 1 and earlier)
                  :link: nki_programming_model
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

      .. grid-item::

            .. card:: NKI Kernel as Framework Custom-Operator
                  :link: nki_framework_custom_op
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

      .. grid-item::

            .. card:: NKI Tutorials
                  :link: /nki/tutorials/index
                  :link-type: doc
                  :class-body: sphinx-design-class-title-small


.. _performant_docs:

Write Performant NKI Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 2
      :margin: 4 1 0 0

      .. grid-item::

            .. card:: NeuronDevice Architecture Guide
                  :link: nki_arch_guides
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

      .. grid-item::

            .. card:: Profiling NKI kernels with Neuron Profile
                  :link: neuron_profile_for_nki
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

      .. grid-item::

            .. card:: NKI Performance Guide
                  :link: nki_perf_guide
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

      .. grid-item::

            .. card:: Direct Allocation Developer Guide
                  :link: nki_direct_allocation_guide
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small


.. _general_resources:

General Resources
^^^^^^^^^^^^^^^^^^^^

.. grid:: 2
      :margin: 4 1 0 0

      .. grid-item::

            .. card:: NKI FAQ
                  :link: nki_faq
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small


      .. grid-item::

            .. card:: NKI What's New
                  :link: nki_rn
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

      .. grid-item::

            .. card:: NKI Known Issues
                  :link: nki_known_issues
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

..
      migration_guide

.. toctree::
      :maxdepth: 1
      :hidden:

      NKI Release Notes </nki/nki_rn>
