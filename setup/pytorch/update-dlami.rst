.. meta::
   :description: Update PyTorch Neuron on an existing AWS Deep Learning AMI
   :keywords: pytorch, neuron, dlami, update, upgrade, driver
   :framework: pytorch
   :installation-method: dlami
   :instance-types: trn2, trn3
   :os: ubuntu-26.04, ubuntu-24.04, al2023
   :content-type: installation-guide
   :date-modified: 2026-10-16

Update PyTorch on a Deep Learning AMI
=======================================

Update PyTorch and Neuron components on an existing DLAMI to the latest release.

.. note::
   Starting with Neuron SDK 2.33.0, the **Deep Learning AMI Neuron torch-neuronx 2.14.0** DLAMI
   provides TorchNeuron (``torch-neuronx``) for PyTorch 2.14 on Ubuntu 24.04 and Ubuntu 26.04, in
   the ``/opt/aws_neuronx_venv_torch_neuronx_2_14_0`` virtual environment. DLAMIs for PyTorch 2.9
   (PyTorch/XLA) are no longer published after Neuron SDK 2.31.0. For more information on how to
   query for SDK-specific DLAMIs, see :ref:`the DLAMI User Guide <neuron-dlami-overview>`.

.. contents:: On this page
   :local:
   :depth: 2


Update PyTorch on Ubuntu 24.04
-------------------------------

If you already have a previous Neuron release installed, select the PyTorch version tab below to get the update commands for your environment.

.. tab-set::

    .. tab-item:: PyTorch 2.14

        On the **Deep Learning AMI Neuron torch-neuronx 2.14.0** DLAMI, activate the
        pre-configured virtual environment, then update TorchNeuron, the Neuron Compiler,
        and NKI. TorchNeuron supports Trn2 and Trn3 instances.

        .. code-block:: bash

           source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate

           python -m pip install --upgrade "torch-neuronx==2.14.*" \
             "neuronx-cc==2.*" nki \
             --extra-index-url https://pip.repos.neuron.amazonaws.com \
             --extra-index-url https://download.pytorch.org/whl/cpu

    .. tab-item:: PyTorch 2.9.0

        .. include:: /archive/frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. code-block:: bash

           # Activate your existing Python virtual environment
           source aws_neuron_venv_pytorch/bin/activate

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

           # Update Neuron Compiler and Framework (Neuron 2.31 versions)
           python -m pip install --upgrade neuronx-cc==2.26.6360.0 torch-neuronx==2.9.0.2.15.32035

    .. tab-item:: PyTorch 2.8.0

        .. include:: /archive/frameworks/torch/torch-neuronx/setup/note-setup-general.rst


Update PyTorch on Ubuntu 26.04
-------------------------------

On the **Deep Learning AMI Neuron torch-neuronx 2.14.0** DLAMI for Ubuntu 26.04, activate the
pre-configured virtual environment, then update TorchNeuron, the Neuron Compiler, and NKI.
TorchNeuron supports Trn2 and Trn3 instances.

.. code-block:: bash

   source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate

   python -m pip install --upgrade "torch-neuronx==2.14.*" \
     "neuronx-cc==2.*" nki \
     --extra-index-url https://pip.repos.neuron.amazonaws.com \
     --extra-index-url https://download.pytorch.org/whl/cpu


Update PyTorch on Amazon Linux 2023
-------------------------------------

If you already have a previous Neuron release installed, select the PyTorch version tab below to get the update commands for your environment.

.. tab-set::

    .. tab-item:: PyTorch 2.8.0

        .. include:: /archive/frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. note::
            PyTorch versions 2.7 and 2.8 are no longer supported on Neuron. If you are looking for setup instructions specific to PyTorch 2.7 and 2.8 on Amazon Linux 2023, Ubuntu 24.04, or Ubuntu 22.04, see `the Neuron release 2.28.0 version of the setup docs <https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/setup/neuron-setup/pytorch/neuronx/amazon-linux/torch-neuronx-al2023.html#id2>`__.

    .. tab-item:: PyTorch 2.7.0

        .. note::
            PyTorch versions 2.7 and 2.8 are no longer supported on Neuron. If you are looking for setup instructions specific to PyTorch 2.7 and 2.8 on Amazon Linux 2023, Ubuntu 24.04, or Ubuntu 22.04, see `the Neuron release 2.28.0 version of the setup docs <https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/setup/neuron-setup/pytorch/neuronx/amazon-linux/torch-neuronx-al2023.html#id2>`__.


Update Neuron driver and runtime
---------------------------------

Update the Neuron driver, runtime, and tools on your DLAMI host. This is recommended
when updating to a new Neuron SDK release.

.. tab-set::

   .. tab-item:: Ubuntu 24.04

      .. code-block:: bash

         sudo apt-get update
         sudo apt-get install -y aws-neuronx-dkms
         sudo apt-get install -y aws-neuronx-runtime-lib
         sudo apt-get install -y aws-neuronx-collectives
         sudo apt-get install -y aws-neuronx-tools

   .. tab-item:: Ubuntu 26.04

      .. code-block:: bash

         sudo apt-get update
         sudo apt-get install -y aws-neuronx-dkms
         sudo apt-get install -y aws-neuronx-runtime-lib
         sudo apt-get install -y aws-neuronx-collectives
         sudo apt-get install -y aws-neuronx-tools

   .. tab-item:: Amazon Linux 2023

      .. code-block:: bash

         sudo dnf install -y aws-neuronx-dkms
         sudo dnf install -y aws-neuronx-runtime-lib
         sudo dnf install -y aws-neuronx-collectives
         sudo dnf install -y aws-neuronx-tools


Verify the update
------------------

After updating, activate your virtual environment. The venv name includes the framework version (for example, ``torch_neuronx_2_14_0``):

.. code-block:: bash

   # List available Neuron virtual environments
   ls /opt/aws_neuronx_venv_*

   # Activate the one matching your framework version
   source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate


And verify the update: 

.. code-block:: bash

   python3 -c "import torch; import torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
   neuron-ls

You should see output similar to this (the versions, instance IDs, and details should match your expected ones, not the ones in this example):
      
**Expected output**:

.. code-block:: text
   
   PyTorch 2.14.0+cpu, torch-neuronx 2.14.3.0
   
   +--------+--------+--------+-----------+
   | DEVICE | CORES  | MEMORY | CONNECTED |
   +--------+--------+--------+-----------+
   | 0      | 2      | 32 GB  | Yes       |
   | 1      | 2      | 32 GB  | Yes       |
   +--------+--------+--------+-----------+


Previous releases
------------------

To install a specific previous Neuron SDK release:

- :doc:`Previous releases for Ubuntu 24.04 </archive/frameworks/torch/torch-neuronx/setup/pytorch-install-prev-u24>`
- :doc:`Previous releases for Ubuntu 22.04 </archive/frameworks/torch/torch-neuronx/setup/pytorch-install-prev-u22>`
- :doc:`Previous releases for Amazon Linux 2023 </archive/frameworks/torch/torch-neuronx/setup/pytorch-install-prev-al2023>`

