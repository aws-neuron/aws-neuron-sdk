.. _transformers-neuronx-setup:

Transformers Neuron Setup (``transformers-neuronx``)
====================================================

--------------------
Stable Release
--------------------

To install the most rigorously tested stable release, use the PyPI pip wheel:

::

    pip install transformers-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com

--------------------
Development Version
--------------------

To install the development version with the latest features and improvements, use ``git`` to install from the
`Transformers Neuron repository <https://github.com/aws-neuron/transformers-neuronx>`_:

::

   pip install git+https://github.com/aws-neuron/transformers-neuronx.git

.. raw:: html

   <details>
   <summary>Installation Alternatives</summary>
   <br>

Without ``git``, save the `Transformers Neuron repository <https://github.com/aws-neuron/transformers-neuronx>`_ package contents locally and use:

::

   pip install transformers-neuronx/ # This directory contains `setup.py`

Similarly, a standalone wheel can be created using the ``wheel`` package
with the local repository contents:

::

   pip install wheel
   cd transformers-neuronx/  # This directory contains `setup.py`
   python setup.py bdist_wheel
   pip install dist/transformers_neuronx*.whl

This generates an installable ``.whl`` package under the ``dist/``
folder.

.. raw:: html

   </details>

.. warning::
    The development version may contain breaking changes. Please use it with caution.
    Additionally, the APIs and functionality in the development version are
    subject to change without warning.