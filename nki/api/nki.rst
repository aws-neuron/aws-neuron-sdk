.. _nki-reference:

nki
============

.. currentmodule:: nki

The ``nki`` module provides the top-level entry points for compiling and running NKI kernels.
Use the :func:`jit` decorator to compile a kernel for NeuronDevices, or :func:`simulate` to
run a kernel in the CPU simulator for debugging.

.. _nki_decorators:


.. autosummary::
   :toctree: generated
   :nosignatures:

   jit
   simulate
