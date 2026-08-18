.. meta::
   :description: Install and configure nrtpy on a Neuron instance.
   :date-modified: 2026-07-31
   :keywords: Neuron, nrtpy, install, setup, NeuronCore, pip, venv

.. _nrtpy-getting-started:

========================
Get started with nrtpy
========================

This page walks you through setting up a Python environment and installing
``nrtpy`` on a Neuron instance. When you are done, you will have ``nrtpy``
imported and ready to load NEFFs.

Before you begin
----------------

You need a Trainium or Inferentia2 EC2 instance with the Neuron SDK installed.
The simplest way to get started is to launch an instance using the latest
`Neuron Deep Learning AMI (DLAMI) <https://docs.aws.amazon.com/dlami/latest/devguide/>`_.
For detailed instructions on launching and connecting to a Neuron instance, see
:ref:`training-quickstart`.

Set up your environment
-----------------------

Create a dedicated virtual environment for ``nrtpy``:

.. code-block:: bash

   python3 -m venv ~/nrtpy_venv
   source ~/nrtpy_venv/bin/activate

Set the Neuron Runtime library path so ``nrtpy`` can find ``libnrt``:

.. code-block:: bash

   export LD_LIBRARY_PATH=/opt/aws/neuron/lib:$LD_LIBRARY_PATH

.. tip::

   Add the ``export LD_LIBRARY_PATH`` line to ``~/nrtpy_venv/bin/activate``
   so it is set automatically each time you activate the environment.

Install nrtpy
-------------

Install ``nrtpy`` from the Neuron pip repository:

.. code-block:: bash

   pip install nrtpy --extra-index-url=https://pip.repos.neuron.amazonaws.com

For general information about installing Neuron packages, see
:ref:`training-quickstart`.

Using nrtpy with NKI
---------------------

.. warning::

   If you plan to use ``nrtpy`` to execute NEFFs compiled from NKI kernels, you
   will also need a separate NKI environment for compilation. The standalone
   ``nrtpy`` wheel and the ``nki`` wheel cannot be installed in the same Python
   environment due to a namespace conflict. In a future release, this limitation
   will be resolved.

Recommended setup:

- **NKI environment** (for compiling kernels to NEFFs): see
  :ref:`how-to-set-up-nki-env`.
- **nrtpy environment** (for loading and executing NEFFs): the venv created
  above.

Next steps
----------

* :ref:`nrtpy-tutorial-debug-kernel` — load, execute, and debug a compiled
  NEFF with ``nrtpy``.
* :ref:`nrtpy-tutorial-validate-benchmark` — validate correctness and
  benchmark kernel performance.
* :ref:`nrtpy-model-ref` — full ``NrtpyModel`` API documentation.

Further reading
---------------

* :ref:`nrtpy-guide` — nrtpy overview and architecture.
* :ref:`nrtpy-troubleshooting` — common issues and solutions.
