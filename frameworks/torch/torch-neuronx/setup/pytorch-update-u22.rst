
.. _pytorch-neuronx-ubuntu22-update:

.. Update PyTorch NeuronX to the latest release on Ubuntu 22.04

If you already have a previous Neuron release installed, select the PyTorch version tab below to get the update commands for your environment.


.. important::
   ``torch-neuronx`` (PyTorch/XLA) is not included in Neuron 2.32.0. To update to
   ``torch-neuronx``, use the latest Neuron release that ships it, **Neuron 2.31**.
   The commands below install the Neuron 2.31 package versions.

.. tab-set::

    .. tab-item:: PyTorch 2.9.0

        .. include:: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. code-block:: bash

           # Activate your existing Python virtual environment
           source aws_neuron_venv_pytorch/bin/activate

           # Install Jupyter notebook kernel
           pip install ipykernel
           python3.13 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
           pip install jupyter notebook
           pip install environment_kernels

           # Set pip repository pointing to the Neuron repository
           python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

           # Install wget, awscli
           python -m pip install wget
           python -m pip install awscli

           # Update Neuron Compiler and Framework (Neuron 2.31 versions)
           python -m pip install --upgrade neuronx-cc==2.26.6360.0 torch-neuronx==2.9.0.2.15.32035

    .. tab-item:: PyTorch 2.8.0

        .. include:: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. note::
            PyTorch versions 2.7 and 2.8 are no longer supported on Neuron. If you are looking for setup instructions specific to PyTorch 2.7 and 2.8 on Amazon Linux 2023, Ubuntu 24.04, or Ubuntu 22.04, see `the Neuron release 2.28.0 version of the setup docs <https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html#setup-torch-neuronx-ubuntu22>`__.
