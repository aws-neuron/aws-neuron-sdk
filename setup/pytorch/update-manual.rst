.. meta::
   :description: Update a manual PyTorch Neuron installation to the latest release
   :keywords: pytorch, neuron, manual, update, upgrade, driver, pip
   :framework: pytorch
   :installation-method: manual
   :instance-types: inf2, trn1, trn2, trn3
   :os: ubuntu-24.04, ubuntu-22.04, al2023
   :content-type: installation-guide
   :date-modified: 2026-03-30

Update a manual PyTorch installation
======================================

Update PyTorch and Neuron components on an existing manual installation to the latest release.

.. contents:: On this page
   :local:
   :depth: 2


Update PyTorch on Ubuntu 24.04
-------------------------------

.. include:: /frameworks/torch/torch-neuronx/setup/pytorch-update-u24.rst
   :start-after: _pytorch-neuronx-ubuntu24-update:


Update PyTorch on Ubuntu 22.04
-------------------------------

.. important::
   Ubuntu 22.04 has reached end-of-support on Neuron. Neuron no longer provides Ubuntu 22.04 DLAMIs or container images. New deployments should use Ubuntu 24.04. See :ref:`announce-eos-ubuntu-22-04-dlami-dlc`.

.. include:: /frameworks/torch/torch-neuronx/setup/pytorch-update-u22.rst
   :start-after: _pytorch-neuronx-ubuntu22-update:


Update PyTorch on Amazon Linux 2023
-------------------------------------

.. include:: /frameworks/torch/torch-neuronx/setup/pytorch-update-al2023.rst
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

   .. tab-item:: Ubuntu 22.04

      .. important::
         Ubuntu 22.04 has reached end-of-support on Neuron. Neuron no longer provides Ubuntu 22.04 DLAMIs or container images. New deployments should use Ubuntu 24.04. See :ref:`announce-eos-ubuntu-22-04-dlami-dlc`.

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

   source ~/neuron_venv/bin/activate

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

You should see output similar to this (the instance IDS and details should match your expected ones, not the ones in this example):

.. code-block::

   PyTorch version: 2.9.1+cu128, torch-neuronx version: 2.9.0.2.13.23887+8e870898
   $ neuron-ls
   instance-type: trn1.2xlarge
   instance-id: i-0bea223b1afb7e159
   +--------+--------+----------+--------+--------------+----------+------+
   | NEURON | NEURON |  NEURON  | NEURON |     PCI      |   CPU    | NUMA |
   | DEVICE | CORES  | CORE IDS | MEMORY |     BDF      | AFFINITY | NODE |
   +--------+--------+----------+--------+--------------+----------+------+
   | 0      | 2      | 0-1      | 32 GB  | 0000:00:1e.0 | 0-7      | -1   |
   +--------+--------+----------+--------+--------------+----------+------+

----

.. _install-prev-releases:

Install previous releases
-------------------------

If you need to install older Neuron releases on your instance, follow the instructions below.

Install previous releases on Ubuntu 24.04
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/setup/pytorch-install-prev-u24.rst
   :start-after: _pytorch-neuronx-install-prev-u24:


Install previous releases on Ubuntu 22.04
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important::
   Ubuntu 22.04 has reached end-of-support on Neuron. Neuron no longer provides Ubuntu 22.04 DLAMIs or container images. New deployments should use Ubuntu 24.04. See :ref:`announce-eos-ubuntu-22-04-dlami-dlc`.

.. include:: /frameworks/torch/torch-neuronx/setup/pytorch-install-prev-u22.rst
   :start-after: _pytorch-neuronx-install-prev-u22:


Install previous releases on Amazon Linux 2023
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/setup/pytorch-install-prev-al2023.rst
   :start-after: _pytorch-neuronx-install-prev-al2023:
