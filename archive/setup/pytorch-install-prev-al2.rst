.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :keywords: AWS Neuron, PyTorch, Trainium, Inferentia, setup, torch-neuronx, previous releases, Amazon Linux 2, AL2
   :date-modified: 2026-08-04

.. _pytorch-neuronx-install-prev-al2:

Install previous PyTorch NeuronX releases for Amazon Linux 2 — Archived
========================================================================

.. warning::

   This document is archived. Amazon Linux 2 is no longer a supported install
   target for the AWS Neuron SDK, and these Neuron releases are superseded.
   It is provided here for reference only. For current setup, see
   :doc:`Set up environments </setup/index>`.

Use the tabs below to install a specific previous Neuron SDK release of PyTorch NeuronX on Amazon Linux 2. Select the Neuron version you need.


.. tab-set::

    .. tab-item:: Neuron 2.18.0

        .. code-block:: bash

            # Install Python venv
            sudo dnf install -y python3.8-venv gcc-c++

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
            python -m pip install neuronx-cc==2.13.66.0 torch-neuronx==1.13.1.1.14.0

    .. tab-item:: Neuron 2.17.0

        .. code-block:: bash

            # Install Python venv
            sudo dnf install -y python3.8-venv gcc-c++

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
            python -m pip install neuronx-cc==2.12.68.0 torch-neuronx==1.13.1.1.13.1

    .. tab-item:: Neuron 2.16.0

        .. code-block:: bash

            # Install Python venv
            sudo dnf install -y python3.8-venv gcc-c++

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
            python -m pip install neuronx-cc==2.12.54.0 torch-neuronx==1.13.1.1.13.0