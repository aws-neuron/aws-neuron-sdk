.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :keywords: AWS Neuron, Inferentia, PyTorch, Trainium, setup, torch-neuronx, Neuron 2.7.0, previous release
   :date-modified: 2026-08-04

.. _install-neuronx-2.7.0-pytorch:

Install PyTorch NeuronX (Neuron 2.7.0) — Archived
=================================================

.. warning::

   This document is archived. Neuron 2.7.0 is a superseded release provided
   here for reference only. For current setup, see
   :doc:`Set up environments </setup/index>`.

.. tab-set::

    .. tab-item:: PyTorch 1.13.0

        .. tab-set::

            .. tab-item:: Amazon Linux 2 AMI

                .. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

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
                    python -m pip install neuronx-cc==2.4.0.21 torch-neuronx==1.13.0.1.4.0

            .. tab-item:: Ubuntu 20 AMI

                .. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

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
                    python -m pip install neuronx-cc==2.4.0.21 torch-neuronx==1.13.0.1.4.0
