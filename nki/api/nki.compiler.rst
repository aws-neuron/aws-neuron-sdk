nki.compiler
============

.. currentmodule:: nki.compiler


The ``nki.compiler`` module provides the public per-kernel configuration passed
to :func:`nki.jit`.

.. _nki-compiler-config:

Per-kernel config
-----------------

Passed to the :func:`nki.jit` decorator as
``@nki.jit(compile_config=CompileConfig(...), specialization_config=SpecializationConfig(...))``.
``CompileConfig`` holds options that impact downstream compiler behavior;
``SpecializationConfig`` holds options that tune the specialization phase.

.. autosummary::
   :toctree: generated
   :template: nki-custom-class-template.rst
   :nosignatures:

   CompileConfig
   SpecializationConfig
   AddressRotation
