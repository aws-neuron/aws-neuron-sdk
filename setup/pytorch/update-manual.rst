.. meta::
   :description: Update a manual PyTorch Neuron installation to the latest release
   :keywords: pytorch, neuron, manual, update, upgrade, driver, pip
   :framework: pytorch
   :installation-method: manual
   :instance-types: trn2, trn3
   :os: ubuntu-26.04, ubuntu-24.04, al2023
   :content-type: installation-guide
   :date-modified: 2026-10-18

Update a manual PyTorch installation
======================================

Update PyTorch and Neuron components on an existing manual installation to the latest release.

.. contents:: On this page
   :local:
   :depth: 2


Update PyTorch on Ubuntu 24.04
-------------------------------

Select the PyTorch version that your existing installation uses.

.. tab-set::

   .. tab-item:: PyTorch 2.14

      TorchNeuron (``torch-neuronx``) for PyTorch 2.14 supports Trn2 and Trn3 instances.
      Activate your existing virtual environment, then update TorchNeuron, the Neuron
      Compiler, and NKI:

      .. code-block:: bash

         source aws_neuron_venv_pytorch_2_14/bin/activate

         python -m pip install --upgrade "torch-neuronx==2.14.*" \
           "neuronx-cc==2.*" nki \
           --extra-index-url https://pip.repos.neuron.amazonaws.com \
           --extra-index-url https://download.pytorch.org/whl/cpu

   .. tab-item:: PyTorch 2.9 (PyTorch/XLA)

      .. include:: /archive/frameworks/torch/torch-neuronx/setup/pytorch-update-u24.rst
         :start-after: _pytorch-neuronx-ubuntu24-update:


Update PyTorch on Ubuntu 26.04
-------------------------------

On Ubuntu 26.04, Neuron supports TorchNeuron (``torch-neuronx``) for PyTorch 2.14 on Trn2 and
Trn3 instances. Activate your existing virtual environment, then update TorchNeuron, the Neuron
Compiler, and NKI:

.. code-block:: bash

   source aws_neuron_venv_pytorch_2_14/bin/activate

   python -m pip install --upgrade "torch-neuronx==2.14.*" \
     "neuronx-cc==2.*" nki \
     --extra-index-url https://pip.repos.neuron.amazonaws.com \
     --extra-index-url https://download.pytorch.org/whl/cpu


Update PyTorch on Amazon Linux 2023
-------------------------------------

.. include:: /archive/frameworks/torch/torch-neuronx/setup/pytorch-update-al2023.rst
   :start-after: _pytorch-neuronx-al2023-update:


Update Neuron driver and runtime
---------------------------------

Update the Neuron driver, runtime, and tools on your host. This is recommended
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

      .. note::
            PyTorch versions 2.9 and earlier are not supported by Neuron for Amazon Linux 2023. If you are looking for setup instructions specific to PyTorch 2.9 on Amazon Linux 2023, see `the Neuron release 2.28.0 version of the setup docs <https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/setup/neuron-setup/pytorch/neuronx/amazon-linux/torch-neuronx-al2023.html>`__.

      .. code-block:: bash

         sudo dnf install -y aws-neuronx-dkms
         sudo dnf install -y aws-neuronx-runtime-lib
         sudo dnf install -y aws-neuronx-collectives
         sudo dnf install -y aws-neuronx-tools


Verify the update
------------------

After updating, activate your virtual environment and verify:

.. code-block:: bash

   source aws_neuron_venv_pytorch_2_14/bin/activate

.. code-block:: python

   python3 << EOF
   import torch
   import torch_neuronx

   print(f"PyTorch version: {torch.__version__}")
   print(f"torch-neuronx version: {torch_neuronx.__version__}")

   import subprocess
   result = subprocess.run(['neuron-ls'], capture_output=True, text=True)
   print(result.stdout)
   EOF

You should see output similar to the following example from a ``trn2.48xlarge`` instance. Your
versions, instance type, and device details depend on your installation and instance.

.. code-block:: text

   PyTorch version: 2.14.0+cpu
   torch-neuronx version: 2.14.3.0
   instance-type: trn2.48xlarge
   logical-neuroncore-config: 2
   +--------+--------+----------+--------+------+
   | NEURON | NEURON |  NEURON  | NEURON | NUMA |
   | DEVICE | CORES  | CORE IDS | MEMORY | NODE |
   +--------+--------+----------+--------+------+
   | 0      | 4      | 0-3      | 96 GB  | 1    |
   | ...    | ...    | ...      | ...    | ...  |
   | 15     | 4      | 60-63    | 96 GB  | 1    |
   +--------+--------+----------+--------+------+

----

.. _install-prev-releases:

Install previous releases
-------------------------

If you need to install older Neuron releases on your instance, follow the instructions below.
Neuron supports Ubuntu 26.04 starting with Neuron 2.33.0, so there are no previous releases for
Ubuntu 26.04.

Install previous releases on Ubuntu 24.04
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /archive/frameworks/torch/torch-neuronx/setup/pytorch-install-prev-u24.rst
   :start-after: _pytorch-neuronx-install-prev-u24:


Install previous releases on Ubuntu 22.04
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important::
   Ubuntu 22.04 has reached end-of-support on Neuron. Neuron no longer provides Ubuntu 22.04 DLAMIs or container images. New deployments should use Ubuntu 24.04. See :ref:`announce-eos-ubuntu-22-04-dlami-dlc`.

.. include:: /archive/frameworks/torch/torch-neuronx/setup/pytorch-install-prev-u22.rst
   :start-after: _pytorch-neuronx-install-prev-u22:


Install previous releases on Amazon Linux 2023
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /archive/frameworks/torch/torch-neuronx/setup/pytorch-install-prev-al2023.rst
   :start-after: _pytorch-neuronx-install-prev-al2023:
