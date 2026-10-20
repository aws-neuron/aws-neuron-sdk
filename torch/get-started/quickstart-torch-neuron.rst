.. meta::
   :description: Install torch-neuronx and verify it on AWS Trainium.
   :keywords: TorchNeuron, torch-neuronx, install, verify, quickstart, Trainium, PyTorch, Neuron
   :date-modified: 2026-10-20

.. _torchneuron-quickstart-install-verify:

Quickstart: Install and verify torch-neuronx
============================================

This topic guides you through installing ``torch-neuronx`` on a Trainium instance and
verifying that PyTorch can run operations on Neuron hardware. When you have completed
it, you will have a working ``torch-neuronx`` installation confirmed end-to-end.

**This quickstart is for**: ML engineers setting up TorchNeuron for the first time

**Time to complete**: 5 minutes with a single-framework DLAMI or 15 minutes with
an Ubuntu 26.04 AMI. A source build takes longer.

Prerequisites
-------------

TorchNeuron supports the Trn2 and Trn3 instance families. You need these resources:

* An instance from the Trn2 or Trn3 family, for example ``trn2.3xlarge`` or
  ``trn2.48xlarge``
* A :doc:`TorchNeuron single-framework DLAMI </setup/pytorch/dlami>` for
  Ubuntu 24.04 or Ubuntu 26.04, or an Ubuntu 22.04, 24.04, or 26.04 AMI
* SSH access to the system

Use the single-framework DLAMI for the fastest setup. It includes ``torch-neuronx``
and the required Neuron packages.

The standalone wheels in this release support Ubuntu 26.04 only. On Ubuntu 24.04,
use the single-framework DLAMI for the fastest setup. For a manual setup on Ubuntu
22.04 or Ubuntu 24.04, `build torch-neuronx from source
<https://github.com/aws-neuron/torch-neuronx?tab=contributing-ov-file#setup-development-environment>`_.

For a container setup, see
:doc:`Install PyTorch via Deep Learning Container </setup/pytorch/dlc>`.

Step 1: Install the Neuron system packages
------------------------------------------

Choose the instructions for your system.

Use the TorchNeuron single-framework DLAMI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Ubuntu 24.04 and Ubuntu 26.04 single-framework DLAMIs include
``torch-neuronx`` and the required Neuron packages. Activate the environment:

.. code-block:: bash

   source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate

Continue to Step 2.

Use an Ubuntu 26.04 AMI
~~~~~~~~~~~~~~~~~~~~~~~

An Ubuntu AMI does not include Neuron packages. Configure the Neuron repository:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y curl gnupg
   curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
     | sudo gpg --dearmor --yes -o /usr/share/keyrings/neuron-archive-keyring.gpg
   . /etc/os-release
   printf 'deb [signed-by=/usr/share/keyrings/neuron-archive-keyring.gpg] https://apt.repos.neuron.amazonaws.com %s main\n' \
     "${VERSION_CODENAME}" \
     | sudo tee /etc/apt/sources.list.d/neuron.list

Install the Neuron driver and tools:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y linux-headers-$(uname -r)
   sudo apt-get install -y \
       aws-neuronx-dkms=2.* \
       aws-neuronx-tools=2.*

Add the Neuron tools to the current shell:

.. code-block:: bash

   export PATH=/opt/aws/neuron/bin:$PATH

Install the common packages:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y \
       python3-venv \
       aws-neuronx-runtime-lib=2.* \
       aws-neuronx-collectives=2.*

Use an Ubuntu 22.04 or Ubuntu 24.04 AMI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standalone wheels in this release do not support Ubuntu 22.04 or Ubuntu 24.04.
`Build torch-neuronx from source
<https://github.com/aws-neuron/torch-neuronx?tab=contributing-ov-file#setup-development-environment>`_
for a manual setup.

Step 2: Check the Neuron devices
--------------------------------

Connect to the system. Then run this command:

.. code-block:: bash

   neuron-ls

For example, a ``trn2.48xlarge`` instance returns output similar to this:

.. code-block:: text

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

You should see output that lists the available Neuron devices. If no devices appear,
check the instance type and Neuron driver.

Step 3: Install torch-neuronx
-----------------------------

Skip this step on a single-framework DLAMI. It already includes ``torch-neuronx``.
Also skip this step after you build ``torch-neuronx`` from source.

Run this step only on an Ubuntu 26.04 AMI.

Check the Python version. Then create a virtual environment:

.. code-block:: bash

   python3 --version
   python3 -m venv ~/torch-neuronx-venv
   source ~/torch-neuronx-venv/bin/activate
   python -m pip install --upgrade pip

Install TorchNeuron and its compiler dependencies:

.. code-block:: bash

   python -m pip install "torch-neuronx==2.14.*" \
     "neuronx-cc==2.*" nki \
     --extra-index-url https://pip.repos.neuron.amazonaws.com \
     --extra-index-url https://download.pytorch.org/whl/cpu

This command installs the PyTorch 2.14 CPU wheel and its matching ``torch-neuronx``
package. This release also supports PyTorch 2.13. Both package families support
Python 3.12, 3.13, and 3.14 on Linux x86_64.

Step 4: Verify the installation
-------------------------------

Create a file named ``verify_torch_neuronx.py``:

.. code-block:: python

   import torch
   import torch_neuronx


   print(f"PyTorch: {torch.__version__}")
   print(f"torch-neuronx: {torch_neuronx.__version__}")

   assert torch.neuron.is_available(), "No Neuron device is available."
   print(f"PyTorch Neuron devices: {torch.neuron.device_count()}")

   torch.manual_seed(0)
   cpu_input = torch.randn(2, 3)
   expected = cpu_input @ cpu_input.T

   neuron_input = cpu_input.to("neuron")
   actual = neuron_input @ neuron_input.T

   torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
   print("torch-neuronx verification passed")

Run the file:

.. code-block:: bash

   python verify_torch_neuronx.py

The first Neuron operation can include compilation time. A successful run prints
output similar to this:

.. code-block:: text

   PyTorch: 2.14.0+cpu
   torch-neuronx: 2.14.3.0
   PyTorch Neuron devices: 64
   torch-neuronx verification passed

The exact ``torch-neuronx`` version can differ.

The test compares the Neuron result with a CPU result. This comparison verifies
device execution and result transfer.

Common issues
-------------

``ModuleNotFoundError: No module named 'torch_neuronx'``
   Activate the virtual environment. Then run the installation commands again.

``No Neuron device is available.``
   Run ``neuron-ls``. Confirm that you use an instance from the Trn2 or Trn3 family.

Package installation fails
   Run ``python --version``. Use Python 3.12, 3.13, or 3.14. Use both package
   repository options from Step 3.

.. include:: /includes/torch/trn3-sizing-note.txt

Next steps
----------

* :doc:`/torch/tutorials/tutorial-simple-training` - Run a training loop in eager
  mode or with ``torch.compile``.
* :doc:`/torch/get-started/quickstart-nki-kernel` - Run an NKI kernel from
  ``torch-neuronx``.
* :doc:`/setup/pytorch/index` - Choose a DLAMI, a DLC, or a manual installation.
* :doc:`/torch/about-pytorch-neuron` - Learn how TorchNeuron works.
* :doc:`/torch/troubleshooting` - Find solutions for installation and execution
  problems.
