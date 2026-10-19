.. _tensorflow-neuronx-install-prev-al2:

.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11

Install Previous TensorFlow Neuron Releases for Amazon Linux (``tensorflow-neuronx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   This document is archived. TensorFlow is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/archive/frameworks/index`.


.. toctree::
   :maxdepth: 1



This section will assist you in installing previous Neuron releases.

.. tab-set::


    .. tab-item:: Neuron 2.18.0

        .. code-block:: bash


           # Install Python venv 
           sudo dnf install -y python3.8-venv gcc-c++ 

           # Create Python venv
           python3.8 -m venv aws_neuron_venv_tensorflow 

           # Activate Python venv 
           source aws_neuron_venv_tensorflow/bin/activate 
           python -m pip install -U pip 

           # Install Jupyter notebook kernel
           pip install ipykernel 
           python3.8 -m ipykernel install --user --name aws_neuron_venv_tensorflow --display-name "Python (tensorflow-neuronx)"
           pip install jupyter notebook
           pip install environment_kernels

           # Set pip repository pointing to the Neuron repository 
           python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

           # Install wget, awscli 
           python -m pip install wget 
           python -m pip install awscli 

           # Install Neuron Compiler and Framework
           python -m pip install neuronx-cc==2.13.66.0 tensorflow-neuronx==2.10.1.2.1.0

    .. tab-item:: Neuron 2.17.0

        .. code-block:: bash


           # Install Python venv 
           sudo dnf install -y python3.8-venv gcc-c++ 

           # Create Python venv
           python3.8 -m venv aws_neuron_venv_tensorflow 

           # Activate Python venv 
           source aws_neuron_venv_tensorflow/bin/activate 
           python -m pip install -U pip 

           # Install Jupyter notebook kernel
           pip install ipykernel 
           python3.8 -m ipykernel install --user --name aws_neuron_venv_tensorflow --display-name "Python (tensorflow-neuronx)"
           pip install jupyter notebook
           pip install environment_kernels

           # Set pip repository pointing to the Neuron repository 
           python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

           # Install wget, awscli 
           python -m pip install wget 
           python -m pip install awscli 

           # Install Neuron Compiler and Framework
           python -m pip install neuronx-cc==2.12.68.0 tensorflow-neuronx==2.10.1.2.1.0

    .. tab-item:: Neuron 2.16.0

        .. code-block:: bash


           # Install Python venv 
           sudo dnf install -y python3.8-venv gcc-c++ 

           # Create Python venv
           python3.8 -m venv aws_neuron_venv_tensorflow 

           # Activate Python venv 
           source aws_neuron_venv_tensorflow/bin/activate 
           python -m pip install -U pip 

           # Install Jupyter notebook kernel
           pip install ipykernel 
           python3.8 -m ipykernel install --user --name aws_neuron_venv_tensorflow --display-name "Python (tensorflow-neuronx)"
           pip install jupyter notebook
           pip install environment_kernels

           # Set pip repository pointing to the Neuron repository 
           python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

           # Install wget, awscli 
           python -m pip install wget 
           python -m pip install awscli 

           # Install Neuron Compiler and Framework
           python -m pip install neuronx-cc==2.12.54.0 tensorflow-neuronx==2.10.1.2.1.0
