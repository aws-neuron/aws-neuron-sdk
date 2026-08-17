
.. _pytorch-neuronx-install-prev-u24:

.. Install previous PyTorch NeuronX releases for Ubuntu 24.04

Use the tabs below to install a specific previous Neuron SDK release of PyTorch NeuronX on Ubuntu 24.04. Select the Neuron version you need.

.. tab-set::

    .. tab-item:: Neuron 2.31.0

        .. code-block:: bash

            # Install Python venv
            sudo apt-get install -y python3.12-venv g++

            # Create Python venv
            python3.12 -m venv aws_neuron_venv_pytorch

            # Activate Python venv
            source aws_neuron_venv_pytorch/bin/activate
            python -m pip install -U pip

            # Install Jupyter notebook kernel
            pip install ipykernel
            python3.12 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
            pip install jupyter notebook
            pip install environment_kernels

            # Set pip repository pointing to the Neuron repository
            python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

            # Install wget, awscli
            python -m pip install wget
            python -m pip install awscli

            # Install Neuron Compiler and Framework
            python -m pip install neuronx-cc==2.26.6360.0 torch-neuronx==2.9.0.2.15.32035

    .. tab-item:: Neuron 2.30.0

        .. code-block:: bash

            # Install Python venv
            sudo apt-get install -y python3.12-venv g++

            # Create Python venv
            python3.12 -m venv aws_neuron_venv_pytorch

            # Activate Python venv
            source aws_neuron_venv_pytorch/bin/activate
            python -m pip install -U pip

            # Install Jupyter notebook kernel
            pip install ipykernel
            python3.12 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
            pip install jupyter notebook
            pip install environment_kernels

            # Set pip repository pointing to the Neuron repository
            python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

            # Install wget, awscli
            python -m pip install wget
            python -m pip install awscli

            # Install Neuron Compiler and Framework
            python -m pip install neuronx-cc==2.25.3371.0 torch-neuronx==2.9.0.2.14.27725

    .. tab-item:: Neuron 2.29.1

        .. code-block:: bash

            # Install Python venv
            sudo apt-get install -y python3.12-venv g++

            # Create Python venv
            python3.12 -m venv aws_neuron_venv_pytorch

            # Activate Python venv
            source aws_neuron_venv_pytorch/bin/activate
            python -m pip install -U pip

            # Install Jupyter notebook kernel
            pip install ipykernel
            python3.12 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
            pip install jupyter notebook
            pip install environment_kernels

            # Set pip repository pointing to the Neuron repository
            python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

            # Install wget, awscli
            python -m pip install wget
            python -m pip install awscli

            # Install Neuron Compiler and Framework
            python -m pip install neuronx-cc==2.24.8799.0 torch-neuronx==2.9.0.2.13.26312

