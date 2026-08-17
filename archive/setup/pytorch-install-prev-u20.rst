
.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-08-04

.. _pytorch-neuronx-install-prev-u20:

Install previous PyTorch NeuronX releases for Ubuntu 20.04 — Archived
======================================================================

.. warning::

   This document is archived. Ubuntu 20.04 is no longer a supported install
   target for the AWS Neuron SDK, and these Neuron releases are superseded.
   It is provided here for reference only. For current setup, see
   :doc:`Set up environments </setup/index>`.

Use the tabs below to install a specific previous Neuron SDK release of PyTorch NeuronX on Ubuntu 20.04. Select the Neuron version you need.

.. tab-set::

    .. tab-item:: Neuron 2.21.0

        .. code-block:: bash

            # Install Python venv
            sudo apt-get install -y python3.8-venv g++

            # Create Python venv
            python3.8 -m venv aws_neuron_venv_pytorch

            # Activate Python venv
            source aws_neuron_venv_pytorch/bin/activate
            python -m pip install -U pip

            # Install Jupyter notebook kernel
            pip install ipykernel
            python3.8 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
            pip install jupyter notebook
            pip install environment_kernels

            # Set pip repository pointing to the Neuron repository
            python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

            # Install wget, awscli
            python -m pip install wget
            python -m pip install awscli

            # Install Neuron Compiler and Framework
            python -m pip install neuronx-cc==2.16.345.0 torch-neuronx==2.1.2.2.4.0

    .. tab-item:: Neuron 2.20.0

        .. code-block:: bash

            # Install Python venv
            sudo apt-get install -y python3.8-venv g++

            # Create Python venv
            python3.8 -m venv aws_neuron_venv_pytorch

            # Activate Python venv
            source aws_neuron_venv_pytorch/bin/activate
            python -m pip install -U pip

            # Install Jupyter notebook kernel
            pip install ipykernel
            python3.8 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
            pip install jupyter notebook
            pip install environment_kernels

            # Set pip repository pointing to the Neuron repository
            python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

            # Install wget, awscli
            python -m pip install wget
            python -m pip install awscli

            # Install Neuron Compiler and Framework
            python -m pip install neuronx-cc==2.15.128.0 torch-neuronx==2.1.2.2.3.0

    .. tab-item:: Neuron 2.19.0

        .. code-block:: bash

            # Install Python venv
            sudo apt-get install -y python3.8-venv g++

            # Create Python venv
            python3.8 -m venv aws_neuron_venv_pytorch

            # Activate Python venv
            source aws_neuron_venv_pytorch/bin/activate
            python -m pip install -U pip

            # Install Jupyter notebook kernel
            pip install ipykernel
            python3.8 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
            pip install jupyter notebook
            pip install environment_kernels

            # Set pip repository pointing to the Neuron repository
            python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

            # Install wget, awscli
            python -m pip install wget
            python -m pip install awscli

            # Install Neuron Compiler and Framework
            python -m pip install neuronx-cc==2.14.213.0 torch-neuronx==2.1.2.2.2.0
