.. code:: bash

   # If you are installing Torch-Neuron plus Neuron-Compiler
   conda install torch-neuron

.. admonition:: Previous Versions
    :class: hint

    Installing the default version of ``torch-neuron`` will use the latest
    *supported* ``torch`` version. Previous versions of ``torch`` may
    be supported using specific ``torch-neuron`` versions.

    To view the supported versions of the given package, use:

    .. code:: bash

        # Show python/framework versions supported by torch-neuron
        conda search torch-neuron

    To install a specific version of a package use a version specifier:

    .. code:: bash

        # Example: Install latest torch-neuron compatible with PyTorch 1.5
        conda install torch-neuron=1.5

        # Example: Install latest torch-neuron compatible with PyTorch 1.7
        conda install torch-neuron=1.7
