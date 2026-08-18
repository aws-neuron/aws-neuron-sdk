.. meta::
   :description: Manual installation of PyTorch Neuron on Inf2, Trn1, Trn2, Trn3 instances
   :keywords: pytorch, neuron, manual, installation, pip
   :framework: pytorch
   :installation-method: manual
   :instance-types: inf2, trn1, trn2, trn3
   :os: ubuntu-24.04, ubuntu-22.04, al2023
   :python-versions: 3.11, 3.12, 3.13
   :content-type: installation-guide
   :estimated-time: 15 minutes
   :date-modified: 2026-08-18

Install PyTorch via manual installation
========================================

Install PyTorch with Neuron support on a bare OS AMI or existing system.

⏱️ **Estimated time**: 15 minutes

.. note::
   For a faster setup, consider using the :doc:`DLAMI-based installation <dlami>` instead.

.. include:: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

----

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Requirement
     - Details
   * - Instance Type
     - Inf2, Trn1, Trn2, or Trn3
   * - Operating System
     - Ubuntu 24.04, Ubuntu 22.04, or Amazon Linux 2023
   * - Python Version
     - Python 3.11, 3.12 or 3.13
   * - AWS Account
     - With EC2 permissions
   * - SSH Key Pair
     - For instance access

Installation steps
------------------

.. tab-set::

   .. tab-item:: Ubuntu 24.04
      :sync: ubuntu-24-04

      **Step 1: Launch instance**

      * Follow the instructions to `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_.
      * Select Ubuntu Server 24 AMI.
      * For Trn1, adjust your primary EBS volume size to a minimum of 512GB.
      * `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_.

      **Step 2: Install drivers and tools**

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

         # Install Neuron Driver (Neuron 2.32 versions)
         sudo apt-get install aws-neuronx-dkms=2.30.2.0* -y

         # Install Neuron Runtime
         sudo apt-get install aws-neuronx-collectives=2.34.10.0* -y
         sudo apt-get install aws-neuronx-runtime-lib=2.34.10.0* -y

         # Install Neuron Tools
         sudo apt-get install aws-neuronx-tools=2.32.28.0* -y

         # Add PATH
         export PATH=/opt/aws/neuron/bin:$PATH

      **Step 3: Install EFA** (Trn1/Trn1n/Trn2/Trn3 only)

      .. code-block:: bash

         # Install EFA Driver (only required for multi-instance training)
         curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
         wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
         cat aws-efa-installer.key | gpg --fingerprint
         wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
         tar -xvf aws-efa-installer-latest.tar.gz
         cd aws-efa-installer && sudo bash efa_installer.sh --yes
         cd
         sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

      **Step 4: Install PyTorch and Neuron packages**

      .. important::
         ``torch-neuronx`` (PyTorch/XLA) is not included in Neuron 2.32.0. To install
         ``torch-neuronx``, use the latest Neuron release that ships it, **Neuron 2.31**.
         The commands below install the Neuron 2.32 compiler with the Neuron 2.31
         ``torch-neuronx`` package.

      .. tab-set::

          .. tab-item:: PyTorch 2.9.0

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

                 # Install Neuron Compiler and Framework (torch-neuronx pinned to Neuron 2.31 as XLA support ended with that release)
                 python -m pip install neuronx-cc==2.27.5334.0 torch-neuronx==2.9.0.2.15.32035

          .. tab-item:: PyTorch 2.8.0

              .. note::
                PyTorch versions 2.7 and 2.8 are no longer supported on Neuron. If you are looking for setup instructions specific to PyTorch 2.7 and 2.8 on Amazon Linux 2023, Ubuntu 24.04, or Ubuntu 22.04, see `the Neuron release 2.28.0 version of the setup docs <https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu24.html#setup-torch-neuronx-ubuntu24>`__.

      **Step 5: Verify installation**

      .. code-block:: bash

         python3 -c "import torch; import torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
         neuron-ls

      You should see output similar to this (the versions, instance IDs, and details should match your expected ones, not the ones in this example):
      
      **Expected output**:
      
      .. code-block:: text
         
         PyTorch 2.9.0+cpu, torch-neuronx 2.9.0.1.0
         
         +--------+--------+--------+-----------+
         | DEVICE | CORES  | MEMORY | CONNECTED |
         +--------+--------+--------+-----------+
         | 0      | 2      | 32 GB  | Yes       |
         | 1      | 2      | 32 GB  | Yes       |
         +--------+--------+--------+-----------+

   .. tab-item:: Ubuntu 22.04
      :sync: ubuntu-22-04

      **Step 1: Launch instance**

      .. important::
         Ubuntu 22.04 has reached end-of-support on Neuron. Neuron no longer provides Ubuntu 22.04 DLAMIs or container images. New deployments should use Ubuntu 24.04. See :ref:`announce-eos-ubuntu-22-04-dlami-dlc`.

      * Follow the instructions to `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_.
      * Select Ubuntu Server 22 AMI.
      * For Trn1, adjust your primary EBS volume size to a minimum of 512GB.
      * `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_.

      **Step 2: Install drivers and tools**

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

         # Install Neuron Driver (Neuron 2.32 versions)
         sudo apt-get install aws-neuronx-dkms=2.30.2.0* -y

         # Install Neuron Runtime
         sudo apt-get install aws-neuronx-collectives=2.34.10.0* -y
         sudo apt-get install aws-neuronx-runtime-lib=2.34.10.0* -y

         # Install Neuron Tools
         sudo apt-get install aws-neuronx-tools=2.32.28.0* -y

         # Add PATH
         export PATH=/opt/aws/neuron/bin:$PATH

      **Step 3: Install EFA** (Trn1/Trn1n/Trn2/Trn3 only)

      .. code-block:: bash

         # Install EFA Driver (only required for multi-instance training)
         curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
         wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
         cat aws-efa-installer.key | gpg --fingerprint
         wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
         tar -xvf aws-efa-installer-latest.tar.gz
         cd aws-efa-installer && sudo bash efa_installer.sh --yes
         cd
         sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

      **Step 4: Install PyTorch and Neuron packages**

      .. important::
         ``torch-neuronx`` (PyTorch/XLA) is not included in Neuron 2.32.0. To install
         ``torch-neuronx``, use the latest Neuron release that ships it, **Neuron 2.31**.
         The commands below install the Neuron 2.32 compiler with the Neuron 2.31
         ``torch-neuronx`` package.

      .. tab-set::

          .. tab-item:: PyTorch 2.9.0

              .. code-block:: bash

                 # Install Python (PyTorch 2.9 requires Python 3.11-3.13, newer than the Ubuntu 22.04 default)
                 sudo add-apt-repository ppa:deadsnakes/ppa
                 sudo apt-get install python3.13

                 # Install Python venv
                 sudo apt-get install -y python3.13-venv g++

                 # Create Python venv
                 python3.13 -m venv aws_neuron_venv_pytorch

                 # Activate Python venv
                 source aws_neuron_venv_pytorch/bin/activate
                 python -m pip install -U pip

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

                 # Install Neuron Compiler and Framework (torch-neuronx pinned to Neuron 2.31 as XLA support ended with that release)
                 python -m pip install neuronx-cc==2.27.5334.0 torch-neuronx==2.9.0.2.15.32035

          .. tab-item:: PyTorch 2.8.0

              .. note::
                  PyTorch versions 2.7 and 2.8 are no longer supported on Neuron. If you are looking for setup instructions specific to PyTorch 2.7 and 2.8 on Amazon Linux 2023, Ubuntu 24.04, or Ubuntu 22.04, see `the Neuron release 2.28.0 version of the setup docs <https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html#setup-torch-neuronx-ubuntu22>`__.

      **Step 5: Verify installation**

      .. code-block:: bash

         python3 -c "import torch; import torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
         neuron-ls

      You should see output similar to this (the versions, instance IDs, and details should match your expected ones, not the ones in this example):
      
      **Expected output**:
      
      .. code-block:: text
         
         PyTorch 2.9.0+cpu, torch-neuronx 2.9.0.1.0
         
         +--------+--------+--------+-----------+
         | DEVICE | CORES  | MEMORY | CONNECTED |
         +--------+--------+--------+-----------+
         | 0      | 2      | 32 GB  | Yes       |
         | 1      | 2      | 32 GB  | Yes       |
         +--------+--------+--------+-----------+

   .. tab-item:: Amazon Linux 2023
      :sync: al2023

      .. note::
         Currently, PyTorch 2.9 is not available on Amazon Linux 2023 and PyTorch 2.7 and 2.8 are no longer supported for Neuron. Use Ubuntu 24.04 for PyTorch 2.9 support. If you are using Neuron 2.28.0, `see the Amazon Linux 2023 setup documentation in the 2.28.0 version of the Neuron docs <https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/setup/neuron-setup/pytorch/neuronx/amazon-linux/torch-neuronx-al2023.html>`__.


.. tip:: **vLLM for LLM inference**

   After completing the manual installation, you can add vLLM for inference serving
   using the ``vllm-neuron`` plugin:

   .. code-block:: bash

      git clone https://github.com/vllm-project/vllm-neuron.git
      cd vllm-neuron
      pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .

   Or use the pre-configured vLLM DLC image for a containerized deployment.
   See :doc:`/libraries/nxd-inference/vllm/index` for all deployment options.

Update an existing installation
--------------------------------

To update PyTorch versions or Neuron drivers on an existing manual installation, see
:doc:`update-manual`.

Next steps
----------

- :doc:`Training with torch-neuronx [archived content] </archive/nxd-training/index>` - Training on Trn1/Trn2
- :doc:`/frameworks/torch/inference-torch-neuronx` - Inference on Inf2/Trn1/Trn2
- :doc:`/tools/neuron-explorer/index` - Profile your workloads
- :doc:`/tools/neuron-sys-tools/neuron-top-user-guide` - Monitor system resources

Advanced
--------

- :doc:`/frameworks/torch/torch-neuronx/setup/pytorch-neuronx-install-cxx11` - Build torch-xla from source with CXX11 ABI

Additional resources
--------------------

- :doc:`dlami` - Use pre-configured DLAMI instead
- :doc:`dlc` - Use pre-configured Docker containers
- :doc:`/deploy/index` - Container-based deployment
- :doc:`../troubleshooting` - Common issues and solutions
- :doc:`/release-notes/index` - Version compatibility information
