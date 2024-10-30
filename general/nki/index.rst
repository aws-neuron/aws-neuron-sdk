.. _neuron-nki:

Neuron Kernel Interface (NKI) - Beta
====================================

The Neuron Kernel Interface (NKI) is a bare-metal language and
compiler for directly programming NeuronDevices available on AWS Trn/Inf instances.
You can use NKI to develop and run new operators directly on
NeuronCores while making full use of available compute and memory resources.
We envision NKI to empower ML researchers and practitioners to self-serve and innovate creative ways
to use the hardware independently.
NKI is designed to work on NeuronCores v2 and beyond.

With a Python-based programming environment, the NKI language adopts syntax and tile-level semantics that are aligned
with `Triton <https://triton-lang.org/main/index.html>`_ and `NumPy <https://numpy.org/doc/stable/>`_. This enables
developers to get started quickly and optimize performance in a familiar environment.
At the hardware level,
NeuronCore's tensorized memory access capability enables efficient reading and writing
of multi-dimensional arrays on a per instruction basis. This makes tile-based programming highly suitable
for the NeuronCore instruction set, which also operate on tiles.

Before NKI was introduced, the only way to program NeuronDevices was through defining high-level ML models in
ML frameworks such as PyTorch and JAX. Neuron Compiler takes such high-level model definitions as input,
performs multiple rounds of optimization, and eventually
generates a NEFF (Neuron Executable File Format) that is executable on NeuronDevices.
At a high level, Neuron Compiler runs the following optimization stages in order,

1. *Hardware-agnostic graph-level optimizations.* These transformations are done in the compiler
   front-end, using `HLO <https://openxla.org/xla/architecture>`__.
   Some examples include constant propagation and operator fusion.

2. *Loop-level optimization.* Compiler turns the optimized graph from Step 1 into a series of loop nests and
   performs layout, tiling and loop fusion optimizations.

3. *Hardware intrinsics mapping.* Compiler maps the architecture-agnostic loop nests from Step 2 into
   architecture-specific instructions.

4. *Hardware-specific optimizations.* These optimizations are done at the instruction level in compiler back-end,
   with a key goal of reducing memory pressure and improving instruction-level parallelism.
   For example, memory allocation and instruction scheduling are done in this stage.

NKI kernels are compiled into IRs (intermediate representations)
that Step 4 can directly consume, bypassing almost all of the compilation stages
in Step 1-3 to provide NKI programmers great control over NeuronDevices down to the instruction level.
To achieve good kernel performance, it is crucial to select proper layouts and tile sizes, along with implementing
efficient loop schedule. We also highly recommended users to study the underlying hardware architecture
before diving into performance optimizations. See NKI guide below to learn more!

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

            .. card:: Trainium/Inferentia2 Architecture Guide
                  :link: trainium_inferentia2_arch
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



.. toctree::
      :maxdepth: 1
      :hidden:

      API Reference Manual <api/index>
      Developer Guide <developer_guide>
      Tutorials <tutorials>
      Kernels <api/nki.kernels>
      Misc <misc>

..
      migration_guide
