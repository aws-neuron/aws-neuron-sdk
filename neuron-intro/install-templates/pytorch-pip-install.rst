
.. code:: bash

   # NOTE: Make sure [tensorflow] option is provided during installation of neuron-cc for PyTorch-Neuron compilation; this is not necessary for PyTorch-Neuron inference.
   pip install neuron-cc[tensorflow]
   pip install torch-neuron


.. admonition:: Previous Versions
    :class: hint

    Installing the default version of ``torch-neuron`` will use the latest
    *supported* ``torch`` version. Previous versions of ``torch`` may
    be supported using specific ``torch-neuron`` versions.

    .. code:: bash

        # Example: Install latest torch-neuron compatible with PyTorch 1.5
        pip install "torch-neuron==1.5.*"

        # Example: Install latest torch-neuron compatible with PyTorch 1.7
        pip install "torch-neuron==1.7.*"