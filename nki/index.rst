.. _neuron-nki:

.. meta::
   :description: Neuron Kernel Interface (NKI) - Low-level programming interface for custom kernel development on AWS Trainium and Inferentia with direct NeuronCore ISA access.
   :keywords: NKI, Neuron Kernel Interface, custom kernels, NeuronCore, AWS Neuron, Trainium, Inferentia, ISA, tile programming, torch.compile
   :date-modified: 2026-06-15

Neuron Kernel Interface (NKI)
=============================

NKI is a bare-metal language and compiler for programming AWS Trainium and
Inferentia NeuronDevices directly, giving you instruction-level control to
develop, optimize, and run custom operators on NeuronCores.

Before you write a kernel, check the :ref:`NKI Library <nkl_home>` — it may
already have an optimized kernel for your operation. New to NKI? Start with
:ref:`Get started <nki-get-started>`. For how NKI fits into the Neuron
compilation stack, see :ref:`How NKI works <nki-how-it-works>` below.

Start here
----------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Get started
      :link: nki-get-started
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Set up your environment, then implement and run your first NKI kernel.

   .. grid-item-card:: NKI Library
      :link: nkl_home
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Use prebuilt, optimized kernels for common operations. Check here before
      writing your own.

Learn and build
---------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Tutorials
      :link: nki-tutorials
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Step-by-step walkthroughs for building NKI kernels, from basics to
      advanced patterns.

   .. grid-item-card:: How-to guides
      :link: nki-guides
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Task-focused guides: insert kernels into models, use the CPU simulator,
      profile, and control scheduling.

   .. grid-item-card:: Deep dives
      :link: nki_deep-dives_home
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Performance optimization, access patterns, DMA, dynamic loops, and the
      NKI compiler.

   .. grid-item-card:: NKI FAQ
      :link: nki_faq
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Common questions about programming with NKI.

.. _api_reference_guide:

Reference
---------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: NKI API reference manual
      :link: nki_api_reference
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Complete API reference for ``nki``, ``nki.language``, ``nki.isa``, and
      ``nki.collectives``.

.. _nki-how-it-works:

How NKI works
-------------

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
   done at the instruction level in compiler back-end,
   with a key goal of reducing memory pressure and improving instruction-level parallelism. For example, memory
   allocation and instruction scheduling are done in this stage.

NKI kernels bypass the first 3 steps, and are compiled into IRs (intermediate representations) that the compiler's
back-end (Step 4 above) can directly consume. Advanced features in NKI, such as direct allocation, also allow programmers
to bypass certain compiler passes in Step 4. As a result, NKI developers can now have great control over NeuronDevices down to
the instruction level.

.. note::
   Neuron highly recommends developers study the underlying hardware architecture before optimizing performance of their NKI kernels. To start, read :doc:`the NKI architecture guides for Trainium </nki/guides/architecture/index>` and :doc:`the NKI performance guide </nki/deep-dives/nki_perf_guide>`.

.. toctree::
   :maxdepth: 1
   :hidden:

   Get started <get-started/index>
   Guides <guides/index>
   Deep dives <deep-dives/index>
   Migration <migration/index>
   NKI FAQ <nki_faq>
