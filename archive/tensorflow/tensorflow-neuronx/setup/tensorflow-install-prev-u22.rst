.. _tensorflow-neuronx-install-prev-u22:

.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11

Install Previous TensorFlow Neuron Releases for Ubuntu (``tensorflow-neuronx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   This document is archived. TensorFlow is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/archive/frameworks/index`.


.. toctree::
   :maxdepth: 1



This section will assist you in installing previous Neuron releases.

.. tab-set::

    .. tab-item:: Neuron 2.21.0

        .. code-block:: bash


           # Install Python venv 
           sudo apt-get install -y python3.10-venv g++ 

           # Create Python venv
           python3.10 -m venv aws_neuron_venv_tensorflow 

           # Activate Python venv 
           source aws_neuron_venv_tensorflow/bin/activate 
           python -m pip install -U pip 

           # Install Jupyter notebook kernel
           pip install ipykernel 
           python3.10 -m ipykernel install --user --name aws_neuron_venv_tensorflow --display-name "Python (tensorflow-neuronx)"
           pip install jupyter notebook
           pip install environment_kernels

           # Set pip repository pointing to the Neuron repository 
           python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

           # Install wget, awscli 
           python -m pip install wget 
           python -m pip install awscli 

           # Install Neuron Compiler and Framework
           python -m pip install neuronx-cc==2.16.345.0 tensorflow-neuronx==2.10.1.2.1.0

    .. tab-item:: Neuron 2.20.0

        .. code-block:: bash


           # Install Python venv 
           sudo apt-get install -y python3.10-venv g++ 

           # Create Python venv
           python3.10 -m venv aws_neuron_venv_tensorflow 

           # Activate Python venv 
           source aws_neuron_venv_tensorflow/bin/activate 
           python -m pip install -U pip 

           # Install Jupyter notebook kernel
           pip install ipykernel 
           python3.10 -m ipykernel install --user --name aws_neuron_venv_tensorflow --display-name "Python (tensorflow-neuronx)"
           pip install jupyter notebook
           pip install environment_kernels

           # Set pip repository pointing to the Neuron repository 
           python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

           # Install wget, awscli 
           python -m pip install wget 
           python -m pip install awscli 

           # Install Neuron Compiler and Framework
           python -m pip install neuronx-cc==2.15.128.0 tensorflow-neuronx==2.10.1.2.1.0
    
    .. tab-item:: Neuron 2.19.0

        .. code-block:: bash


           # Install Python venv 
           sudo apt-get install -y python3.10-venv g++ 

           # Create Python venv
           python3.10 -m venv aws_neuron_venv_tensorflow 

           # Activate Python venv 
           source aws_neuron_venv_tensorflow/bin/activate 
           python -m pip install -U pip 

           # Install Jupyter notebook kernel
           pip install ipykernel 
           python3.10 -m ipykernel install --user --name aws_neuron_venv_tensorflow --display-name "Python (tensorflow-neuronx)"
           pip install jupyter notebook
           pip install environment_kernels

           # Set pip repository pointing to the Neuron repository 
           python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

           # Install wget, awscli 
           python -m pip install wget 
           python -m pip install awscli 

           # Install Neuron Compiler and Framework
           python -m pip install neuronx-cc==2.14.213.0 tensorflow-neuronx==2.10.1.2.1.0
