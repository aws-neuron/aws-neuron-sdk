.. _install-neuronx-2.8.0-tensorflow:

.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11

Install Tensorflow Neuron (Neuron 2.8.0)
========================================

.. warning::

   This document is archived. TensorFlow is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/archive/frameworks/index`.


.. tab-set::

    .. tab-item:: Tensorflow 2.10.0

        .. tab-set::

            .. tab-item:: Amazon Linux 2 AMI

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
                   python -m pip install neuronx-cc==2.4.0.21 tensorflow-neuronx==2.10.1.1.0.0

            .. tab-item:: Ubuntu 20 AMI

                .. code-block:: bash


                   # Install Python venv 
                   sudo apt-get install -y python3.8-venv g++ 

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
                   python -m pip install neuronx-cc==2.4.0.21 tensorflow-neuronx==2.10.1.1.0.0
