.. _neuron-nki:

Neuron Kernel Interface (NKI) - Beta
====================================

Neuron Kernel Interface (NKI) is a bare-metal language and compiler for directly programming NeuronDevices
available on AWS Trn/Inf instances. You can use NKI to develop, optimize and run new operators directly on
NeuronCores while making full use of available compute and memory resources. NKI empowers ML developers to
self-serve and invent new ways to use the NeuronCore hardware, starting NeuronCores v2 (Trainium1) and beyond.

NKI provides developers with direct access to the NeuronCore ISA (Instruction Set Architecture), accessible from a
Python-based programming environment, which has syntax and tile-level semantics that are similar to
`Triton <https://triton-lang.org/main/index.html>`_ and `NumPy <https://numpy.org/doc/stable/>`_.
This enables developers to get started quickly and optimize performance in a familiar environment, while at the same
time get full control of the underlying hardware. At the hardware level, NeuronCore's tensorized memory access
capability enables efficient reading and writing of multi-dimensional arrays on a per instruction basis,
which makes NKI's tile-based programming highly suitable for the NeuronCore instruction set.

For comparison, before NKI was introduced, the only way to program NeuronDevices was through defining high-level ML
models in frameworks such as `PyTorch <https://pytorch.org/>`_
and `JAX <https://jax.readthedocs.io/en/latest/index.html>`_.
Neuron Compiler takes such high-level model definitions as input,
performs multiple rounds of optimization, and eventually generates a NEFF (Neuron Executable File Format) that
is executable on NeuronDevices. At a high level, Neuron Compiler runs the following optimization stages in order:

1. **Hardware-agnostic graph-level optimizations.** These transformations are done in the compiler front-end,
   using `XLA <https://openxla.org/xla>`_, including optimizations like constant propagation, re-materialization
   and operator fusion.

2. **Loop-level optimization.** Compiler turns the optimized graph from Step 1 into a series of loop nests
   and performs layout, tiling and loop fusion optimizations.

3. **Hardware intrinsics mapping.** Compiler maps the architecture-agnostic loop nests from Step 2 into
   architecture-specific instructions.

4. **Hardware-specific optimizations.** These optimizations are mainly
   done at the instruction level [#]_ in compiler back-end,
   with a key goal of reducing memory pressure and improving instruction-level parallelism. For example, memory
   allocation and instruction scheduling are done in this stage.

NKI kernels bypass the first 3 steps, and are compiled into IRs (intermediate representations) that the compiler's
back-end (Step 4 above) can directly consume. Advanced features in NKI, such as direct allocation, also allow programmers
to bypass certain compiler passes in Step 4. As a result, NKI developers can now have great control over NeuronDevices down to
the instruction level. We highly recommend developers to study the underlying hardware architecture before
optimizing performance of their NKI kernels. See the NKI guide below to learn more!

.. .. contents:: Table of contents
.. 	:local:
.. 	:depth: 1

Guide
--------------

NKI guide is organized in four parts:

1. :ref:`API Reference Guide <api_reference_guide>` has the NKI API reference manual.
2. :ref:`Writing Functional NKI Kernels <functional_docs>` includes guides that are designed for
   NKI beginners to learn NKI key concepts and implement kernels to meet functionality requirements.
3. :ref:`Writing Performant NKI Kernels <performant_docs>` includes a deep dive of NeuronDevice architecture
   and programmer's guides to optimize performance of NKI kernels.
4. :ref:`General Resources <general_resources>` include any miscellaneous guides.

.. _api_reference_guide:

API Reference Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 2
      :margin: 4 1 0 0

      .. grid-item::

            .. card:: NKI API Reference Manual
                  :link: nki_api_reference
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small


.. _functional_docs:

Writing Functional NKI Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 2
      :margin: 4 1 0 0

      .. grid-item::

            .. card:: Getting Started with NKI
                  :link: nki_getting_started
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small


      .. grid-item::

            .. card:: NKI Programming Model
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
                  :link: nki_tutorials
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small

      .. grid-item::

            .. card:: NKI Kernels
                  :link: nki_kernels
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small


.. _performant_docs:

Writing Performant NKI Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. rubric:: Footnotes

.. [#] A small number of loop-level optimizations are performed after hardware intrinsic mappings in the current
       Beta release. Subject to future changes.
