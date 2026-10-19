.. meta::
   :description: Manual installation of PyTorch Neuron (TorchNeuron) on Trn2 and Trn3 instances
   :keywords: pytorch, neuron, manual, installation, pip
   :framework: pytorch
   :installation-method: manual
   :instance-types: trn2, trn3
   :os: ubuntu-26.04, ubuntu-24.04, al2023
   :python-versions: 3.12, 3.13, 3.14
   :content-type: installation-guide
   :estimated-time: 15 minutes
   :date-modified: 2026-10-18

Install PyTorch via manual installation
========================================

Install PyTorch with Neuron support on a bare OS AMI or existing system.

⏱️ **Estimated time**: 15 minutes

.. note::
   For a faster setup, consider using the :doc:`DLAMI-based installation <dlami>` instead.

.. include:: /archive/frameworks/torch/torch-neuronx/setup/note-setup-general.rst

----

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Requirement
     - Details
   * - Instance Type
     - Trn2 or Trn3
   * - Operating System
     - Ubuntu 26.04, Ubuntu 24.04, or Amazon Linux 2023. For the tested Linux kernel and
       GLIBC versions, see :ref:`neuron-runtime-compatibility`.
   * - Python Version
     - Python 3.12, 3.13, or 3.14
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
         sudo install -m 0755 -d /etc/apt/keyrings
         wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/neuron.gpg
         sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
         deb [signed-by=/etc/apt/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
         EOF

         # Update OS packages
         sudo apt-get update -y

         # Install OS headers
         sudo apt-get install linux-headers-$(uname -r) -y

         # Install git
         sudo apt-get install git -y

         # Install Neuron Driver (Neuron 2.33 versions)
         sudo apt-get install aws-neuronx-dkms=2.31.* -y

         # Install Neuron Runtime
         sudo apt-get install aws-neuronx-collectives=2.35.* -y
         sudo apt-get install aws-neuronx-runtime-lib=2.35.* -y

         # Install Neuron Tools
         sudo apt-get install aws-neuronx-tools=2.33.* -y

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

      Starting with Neuron 2.33.0, ``torch-neuronx`` for PyTorch 2.13 and 2.14 is TorchNeuron,
      the native PyTorch backend for Trainium. TorchNeuron supports Trn2 and Trn3 instances.
      To install it, select the **PyTorch 2.14** tab. The **PyTorch 2.9.0** tab installs the
      earlier PyTorch/XLA-based ``torch-neuronx`` for existing workloads.

      .. tab-set::

          .. tab-item:: PyTorch 2.14

              .. code-block:: bash

                 # Install Python venv
                 sudo apt-get install -y python3.12-venv g++

                 # Create and activate a Python venv
                 python3.12 -m venv aws_neuron_venv_pytorch_2_14
                 source aws_neuron_venv_pytorch_2_14/bin/activate
                 python -m pip install --upgrade pip

                 # Install TorchNeuron for PyTorch 2.14, the Neuron Compiler, and NKI
                 python -m pip install "torch-neuronx==2.14.*" \
                   "neuronx-cc==2.*" nki \
                   --extra-index-url https://pip.repos.neuron.amazonaws.com \
                   --extra-index-url https://download.pytorch.org/whl/cpu

          .. tab-item:: PyTorch 2.9.0

              .. note::
                 PyTorch/XLA-based ``torch-neuronx`` for PyTorch 2.9 last shipped in Neuron 2.31.
                 These commands install the Neuron 2.33 compiler with the Neuron 2.31
                 ``torch-neuronx`` package.

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
                 python -m pip install neuronx-cc==2.28.* torch-neuronx==2.9.0.2.15.32035

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

   .. tab-item:: Ubuntu 26.04
      :sync: ubuntu-26-04

      **Step 1: Launch instance**

      * Follow the instructions to `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_.
      * Select Ubuntu Server 26 AMI.
      * For Trn1, adjust your primary EBS volume size to a minimum of 512GB.
      * `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_.

      **Step 2: Install drivers and tools**

      .. include:: /src/helperscripts/installationScripts/python_instructions.txt
          :start-line: 320
          :end-line: 321

      **Step 3: Install EFA** (Trn1/Trn1n/Trn2/Trn3 only)

      .. include:: /src/helperscripts/installationScripts/python_instructions.txt
          :start-line: 311
          :end-line: 312

      **Step 4: Install PyTorch and Neuron packages**

      On Ubuntu 26.04, install TorchNeuron (``torch-neuronx``) for PyTorch 2.14. TorchNeuron
      supports Trn2 and Trn3 instances.

      .. code-block:: bash

         # Install Python venv (Python 3.14 is the default on Ubuntu 26.04)
         sudo apt-get install -y python3-venv g++

         # Create and activate a Python venv
         python3 -m venv aws_neuron_venv_pytorch_2_14
         source aws_neuron_venv_pytorch_2_14/bin/activate
         python -m pip install --upgrade pip

         # Install TorchNeuron for PyTorch 2.14, the Neuron Compiler, and NKI
         python -m pip install "torch-neuronx==2.14.*" \
           "neuronx-cc==2.*" nki \
           --extra-index-url https://pip.repos.neuron.amazonaws.com \
           --extra-index-url https://download.pytorch.org/whl/cpu

      **Step 5: Verify installation**

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
   See :doc:`vLLM Neuron </vllm-neuron/docs/index>` for all deployment options.

Update an existing installation
--------------------------------

To update PyTorch versions or Neuron drivers on an existing manual installation, see
:doc:`update-manual`.

Next steps
----------

- :ref:`Build your first model with TorchNeuron <torchneuron-quickstart-build-first-model>` - Run a PyTorch model in eager mode and with ``torch.compile``
- :doc:`TorchNeuron tutorials </torch/tutorials/index>` - Training and inference tutorials for native PyTorch on Trn2 and Trn3
- :doc:`TorchTitan on Neuron </training/torchtitan/index>` - Pre-train large models on Trainium
- :doc:`vLLM Neuron online serving quickstart </vllm-neuron/docs/getting-started/quickstart-online-serving>` - Serve LLMs with an OpenAI-compatible API
- :doc:`/tools/neuron-explorer/index` - Profile your workloads
- :doc:`/tools/neuron-sys-tools/neuron-top-user-guide` - Monitor system resources

Additional resources
--------------------

- :doc:`dlami` - Use pre-configured DLAMI instead
- :doc:`dlc` - Use pre-configured Docker containers
- :doc:`/deploy/index` - Container-based deployment
- :doc:`../troubleshooting` - Common issues and solutions
- :doc:`/release-notes/index` - Version compatibility information
