.. meta::
   :description: How to generate a NEFF file with the Neuron Graph Compiler (neuronx-cc) by compiling an XLA HLO graph for a Trainium or Inferentia target.
   :keywords: neuronx-cc, NEFF, compile, HLO, XLA, Trainium, Inferentia, Neuron compiler
   :date-modified: 2026-07-30

.. _neuronx-cc-generate-neff-how-to:

===========================
How to generate a NEFF file
===========================

Task overview
-------------

This topic shows how to generate a NEFF (Neuron Executable File Format) file with
the Neuron Graph Compiler (``neuronx-cc``). A NEFF is the compiled artifact that the
:doc:`Neuron Runtime </neuron-runtime/index>` loads to run your model on
NeuronCores. You produce one by compiling an XLA HLO graph for a target instance
family.

Prerequisites
-------------

- **The Neuron compiler is installed:** ``neuronx-cc`` is on your ``PATH`` (activate
  the Neuron virtual environment if you use one). Run ``neuronx-cc --help`` to check.
- **An XLA HLO file:** a ``hlo.pb`` graph exported from your framework. Frameworks
  such as PyTorch NeuronX usually emit this for you during tracing.
- **A target instance family:** one of ``trn1``, ``trn1n``, ``inf2``, or ``trn2``.
  You can run the compilation step on any EC2 instance or on-premises host.

Instructions
------------

**1:** Verify the compiler is available.

.. code-block:: shell

   neuronx-cc --help

**2:** Compile the HLO graph into a NEFF.

.. code-block:: shell

   neuronx-cc compile model.hlo \
     --framework XLA \
     --target trn1 \
     --output model.neff

If you omit ``--output``, the compiler writes ``file.neff`` by default.

**3:** (Optional) Apply optimizations while compiling.

.. code-block:: shell

   neuronx-cc compile model.hlo \
     --framework XLA \
     --target trn1 \
     --model-type transformer \
     --auto-cast matmult \
     --auto-cast-type bf16 \
     --output model.neff

.. note::

   When you compile through a framework, you do not run these commands directly.
   Pass the same options through the ``NEURON_CC_FLAGS`` environment variable and the
   framework forwards them to ``neuronx-cc``.

Confirm your work
-----------------

The compiler returns exit status ``0`` on success. Confirm the NEFF file was written:

.. code-block:: shell

   echo $?          # prints 0 on success
   ls -lh model.neff

Common issues
-------------

.. rubric:: ``neuronx-cc: command not found``

- **Possible solution:** The compiler is not on your ``PATH``. Activate your Neuron
  virtual environment, or install the Neuron compiler, then try again.

.. rubric:: Compilation fails on an unsupported operator

- **Possible solution:** List the operators the compiler supports with
  ``neuronx-cc list-operators --framework XLA``, then partition the model in your
  framework to remove unsupported operations before compiling.

.. rubric:: Wrong target instance family

- **Possible solution:** Set ``--target`` to the instance family where the NEFF will
  run (``trn1``, ``trn1n``, ``inf2``, or ``trn2``). A NEFF is built for a specific
  target.

Related information
-------------------

- :ref:`What is the Neuron Graph Compiler (neuronx-cc)? <neuronx-cc-overview>`
- :ref:`Neuron Compiler CLI reference <neuron-compiler-cli-reference-guide>`
- :doc:`Neuron Runtime </neuron-runtime/index>`
