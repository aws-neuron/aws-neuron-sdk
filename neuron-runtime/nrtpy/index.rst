.. meta::
   :description: nrtpy — a Pythonic runtime layer for loading, executing, and benchmarking compiled NEFF models on AWS Neuron NeuronCores.
   :date-modified: 2026-07-31
   :keywords: Neuron, nrtpy, Neuron Runtime, NRT, libnrt, Python, NEFF, NeuronCore

.. _nrtpy-guide:

=============================
nrtpy (Neuron Runtime Python)
=============================

Overview
--------

``nrtpy`` is a Pythonic runtime layer for AWS Neuron that lifts ``libnrt``
(the Neuron Runtime Library) into an idiomatic Python interface. It provides:

- **Pythonic user experience**: Core runtime concepts (tensors, models,
  execution) with idiomatic Python patterns for resource management, data
  movement, and error handling.
- **Minimal performance overhead**: Performance equivalent to C++, achieved
  through zero-copy buffer protocol and efficient nanobind C++ bindings.

Use ``nrtpy`` when you need to load and execute compiled Neuron Executable File Format (NEFF) files directly on NeuronCores from Python (for
example, benchmarking a compiled model or building test harnesses) without going through a higher-level framework.

Prerequisites
-------------

- A Trainium or Inferentia EC2 instance
- Python 3.11, 3.12, 3.13, or 3.14
- AWS NeuronX Runtime (``aws-neuronx-runtime-lib``) installed at matching
  version (see :ref:`NRT installation requirements <nrt_reqs>`)
- Compiled NEFF(s)

.. warning::

   The standalone ``nrtpy`` wheel and the ``nki`` wheel cannot be installed in
   the same Python environment due to a namespace conflict. In a future release,
   this limitation will be resolved. Until then, install only one per
   environment. Use your NKI environment to compile kernels to NEFFs, and a
   separate ``nrtpy`` environment to load and execute them.

Architecture
------------

``nrtpy`` follows a single-runtime execution model:

- **1 Python process** holds **1 nrtpy singleton** managing **1 libnrt instance**.
- The singleton is created lazily on first use (for example, when you construct
  an ``NrtpyTensor`` or load an ``NrtpyModel``).
- Call :py:func:`nrtpy.configure` before first use to set visible NeuronCores.
- Call :py:func:`nrtpy.reset` to close the runtime and allow reconfiguration.

By default, when :py:func:`nrtpy.configure` is not called, all NeuronCores on
the instance are visible to the runtime but operations target ``core_id=0``
(a single core). For workloads requiring
multiple cores (for example, collective communication), configure multiple
visible cores and specify ``core_id`` when loading models and allocating
tensors.

.. code-block:: text

   +---------------------------+
   |    Python Application     |
   +---------------------------+
                |
                v
   +---------------------------+
   |   nrtpy Python API        |
   |  NrtpyModel, NrtpyTensor  |
   +---------------------------+
                |
                v
   +---------------------------+
   |   nrtpy C++ (nanobind)    |
   +---------------------------+
                |
                v
   +---------------------------+
   |   libnrt (Neuron Runtime) |
   +---------------------------+
                |
                v
   +---------------------------+
   |     Neuron Hardware       |
   +---------------------------+

Getting started and tutorials
-----------------------------

.. toctree::
   :maxdepth: 1

   Getting started with nrtpy <nrtpy-getting-started>
   Test and debug a NKI kernel end-to-end <tutorial-debug-neff-nrtpy>
   Validate, benchmark, and trace a kernel <tutorial-validate-benchmark>
   Troubleshooting <nrtpy-troubleshooting>

API reference
-------------

.. toctree::
   :maxdepth: 1

   Model API <model>
   Tensor API <tensor>
   Configuration <configuration>
   Error handling <errors>

Related information
-------------------

- :ref:`nrt_api_reference` — the C Neuron Runtime API that ``nrtpy`` wraps.
- :ref:`neuron-compiler-cli-reference-guide` — the ``neuronx-cc`` compiler that
  produces NEFF files.
