.. meta::
   :noindex:
   :nofollow:
   :description: Archived legacy PyTorch installation guide for AWS Inferentia 1 (Inf1) instances
   :keywords: pytorch, neuron, inf1, legacy, installation, torch-neuron
   :framework: pytorch
   :instance-types: inf1
   :status: legacy
   :content-type: legacy-guide
   :date-modified: 2026-08-04

.. _legacy-inf1-pytorch:

PyTorch on Inf1 (legacy) — Archived
====================================

.. warning::

   This document is archived and no longer maintained. Inf1 instances use
   NeuronCore v1 with PyTorch 1.x (``torch-neuron``), which is superseded.

   For new projects, use **Inf2, Trn1, Trn2, or Trn3** with PyTorch 2.x+
   (``torch-neuronx``). See :doc:`/setup/pytorch/index` for current PyTorch
   setup. The commands below are pinned to the last Inf1-supporting Neuron
   release and are provided for reference only.

Key differences from current PyTorch
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Inf1 (torch-neuron)
     - Inf2, Trn1, Trn2, Trn3 (torch-neuronx)
   * - PyTorch version
     - 1.x
     - 2.9+
   * - Backend
     - PyTorch/XLA (``torch_neuron``)
     - Native Neuron (``torch_neuronx``)
   * - Compilation
     - ``torch_neuron.trace()``
     - ``torch.compile(backend='neuronx')``
   * - Training support
     - No
     - Yes
   * - NeuronCore version
     - v1
     - v2

Setup instructions
------------------

Launch an Inf1 instance with the AMI for your chosen operating system, connect
to it, then run the matching install commands below.

.. tab-set::

   .. tab-item:: Ubuntu 20.04

      .. code-block:: bash

         # Configure Linux for Neuron repository updates
         . /etc/os-release
         sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
         deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
         EOF
         wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

         # Update OS packages
         sudo apt-get update -y

         # Install OS headers
         sudo apt-get install linux-headers-$(uname -r) -y

         # Install git
         sudo apt-get install git -y

         # Install Neuron Driver
         sudo apt-get install aws-neuronx-dkms=2.21.* -y

         # Install Neuron Tools
         sudo apt-get install aws-neuronx-tools=2.* -y

         # Add PATH
         export PATH=/opt/aws/neuron/bin:$PATH

         # Install Python (torch-neuron requires Python 3.9, not the Ubuntu 20.04 default)
         sudo add-apt-repository ppa:deadsnakes/ppa
         sudo apt-get install python3.9

         # Install Python venv
         sudo apt-get install -y python3.9-venv g++

         # Create Python venv
         python3.9 -m venv aws_neuron_venv_pytorch_inf1

         # Activate Python venv
         source aws_neuron_venv_pytorch_inf1/bin/activate
         python -m pip install -U pip

         # Install Jupyter notebook kernel
         pip install ipykernel
         python3.9 -m ipykernel install --user --name aws_neuron_venv_pytorch_inf1 --display-name "Python (torch-neuron)"
         pip install jupyter notebook
         pip install environment_kernels

         # Set pip repository pointing to the Neuron repository
         python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

         # Install PyTorch Neuron
         python -m pip install torch-neuron neuron-cc[tensorflow] "protobuf" torchvision

   .. tab-item:: Ubuntu 22.04

      .. code-block:: bash

         # Configure Linux for Neuron repository updates
         . /etc/os-release
         sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
         deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
         EOF
         wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

         # Update OS packages
         sudo apt-get update -y

         # Install OS headers
         sudo apt-get install linux-headers-$(uname -r) -y

         # Install git
         sudo apt-get install git -y

         # Install Neuron Driver
         sudo apt-get install aws-neuronx-dkms=2.21.* -y

         # Install Neuron Tools
         sudo apt-get install aws-neuronx-tools=2.* -y

         # Add PATH
         export PATH=/opt/aws/neuron/bin:$PATH

         # Install Python venv
         sudo apt-get install -y python3.10-venv g++

         # Create Python venv
         python3.10 -m venv aws_neuron_venv_pytorch_inf1

         # Activate Python venv
         source aws_neuron_venv_pytorch_inf1/bin/activate
         python -m pip install -U pip

         # Install Jupyter notebook kernel
         pip install ipykernel
         python3.10 -m ipykernel install --user --name aws_neuron_venv_pytorch_inf1 --display-name "Python (torch-neuron)"
         pip install jupyter notebook
         pip install environment_kernels

         # Set pip repository pointing to the Neuron repository
         python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

         # Install PyTorch Neuron
         python -m pip install torch-neuron neuron-cc[tensorflow] "protobuf" torchvision

   .. tab-item:: Amazon Linux 2023

      .. code-block:: bash

         # Configure Linux for Neuron repository updates
         sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
         [neuron]
         name=Neuron YUM Repository
         baseurl=https://yum.repos.neuron.amazonaws.com
         enabled=1
         metadata_expire=0
         EOF
         sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB

         # Update OS packages
         sudo dnf update -y

         # Install OS headers
         sudo dnf install -y "kernel-devel-uname-r = $(uname -r)"

         # Install git
         sudo dnf install git -y

         # Install Neuron Driver
         sudo dnf install aws-neuronx-dkms-2.21.* -y

         # Install Neuron Tools
         sudo dnf install aws-neuronx-tools-2.* -y

         # Add PATH
         export PATH=/opt/aws/neuron/bin:$PATH

         # Install External Dependency
         sudo dnf install -y libxcrypt-compat

         # Install Python venv
         sudo dnf install -y gcc-c++

         # Create Python venv
         python3.9 -m venv aws_neuron_venv_pytorch_inf1

         # Activate Python venv
         source aws_neuron_venv_pytorch_inf1/bin/activate
         python -m pip install -U pip

         # Install Jupyter notebook kernel
         pip install ipykernel
         python3.9 -m ipykernel install --user --name aws_neuron_venv_pytorch_inf1 --display-name "Python (torch-neuron)"
         pip install jupyter notebook
         pip install environment_kernels

         # Set pip repository pointing to the Neuron repository
         python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

         # Install PyTorch Neuron
         python -m pip install torch-neuron neuron-cc[tensorflow] "protobuf" torchvision

.. note::

   Update and previous-release install flows for Inf1 (``torch-neuron``) are
   no longer maintained. For historical previous-release install pages, see
   the :doc:`Archived setup guides </archive/setup/index>`.

Verification
------------

After installation, verify with:

.. code-block:: python

   import torch
   import torch_neuron

   print(f"torch-neuron version: {torch_neuron.__version__}")

.. code-block:: bash

   neuron-ls

Next steps
----------

- :doc:`/archive/torch-neuron/api-reference-guide-torch-neuron` - torch-neuron API reference
- :ref:`setup-guide-index` - Current setup options (Inf2, Trn1, Trn2, Trn3)
