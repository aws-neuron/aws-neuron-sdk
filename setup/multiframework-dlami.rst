.. meta::
   :description: Get started with the Neuron Multi-Framework Deep Learning AMI for PyTorch, JAX, and vLLM on Inf2, Trn1, Trn2, Trn3
   :keywords: neuron, dlami, multi-framework, pytorch, jax, vllm, installation
   :instance-types: inf2, trn1, trn2, trn3
   :content-type: installation-guide
   :date-modified: 2026-03-30

.. _setup-multiframework-dlami:

Get started with the Neuron multi-framework DLAMI
===================================================

The Neuron multi-framework Deep Learning AMI (DLAMI) provides a pre-configured environment
with multiple frameworks and libraries ready to use. Each framework has its own virtual
environment with all Neuron components pre-installed.

The multi-framework DLAMI supports Inf2, Trn1, Trn1n, Trn2, and Trn3 instances and is
updated with each Neuron SDK release.

.. contents:: On this page
   :local:
   :depth: 2

Step 1: Launch the instance
----------------------------

.. important::
   Currently, only Ubuntu 24.04 is supported for multi-framework DLAMIs.

.. tab-set::

   .. tab-item:: Ubuntu 24.04
      :sync: ubuntu-24-04

      Open the `EC2 Console <https://console.aws.amazon.com/ec2>`_, select your desired
      AWS region, and choose "Launch Instance". Under AMI selection, choose "Quick Start"
      then "Ubuntu", and select **Deep Learning AMI Neuron (Ubuntu 24.04)**.

      .. image:: /images/neuron-multi-framework-dlami-U24-quick-start.png
         :scale: 20%
         :align: center

      Select your desired Neuron instance type (Inf2, Trn1, Trn1n, Trn2, or Trn3),
      configure disk size (minimum 512 GB for Trn instances), and launch the instance.

.. note::

   To retrieve the latest DLAMI ID programmatically for automation flows, use
   :ref:`SSM parameters <ssm-parameter-neuron-dlami>`.

Step 2: Activate a virtual environment
----------------------------------------

The multi-framework DLAMI includes pre-configured virtual environments for each
supported framework and library. Activate the one that matches your use case:

1. Find the virtual environment name for your framework or library in the
   :ref:`Neuron DLAMI overview <neuron-dlami-multifw-venvs>`.

2. Activate the virtual environment:

   .. code-block:: bash

      source /opt/<name_of_virtual_environment>/bin/activate

Common virtual environments include:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Framework
     - Virtual environment
     - Use case
   * - PyTorch 2.9
     - ``aws_neuronx_venv_pytorch``
     - Training and inference
   * - PyTorch vLLM
     - ``aws_neuronx_venv_pytorch_inference_vllm``
     - LLM inference serving
   * - JAX
     - ``aws_neuronx_venv_jax``
     - Training and inference

.. note::

   Virtual environment names and available frameworks may vary by DLAMI version.
   See :ref:`neuron-dlami-multifw-venvs` for the complete list.

Step 3: Verify and start
--------------------------

After activating a virtual environment, verify the installation:

.. tab-set::

   .. tab-item:: PyTorch

      .. code-block:: bash

         python3 -c "import torch; import torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
         neuron-ls

      You should see output similar to this (the framework, versions, instance IDs, and details should match your expected ones, not the ones in this example):

      .. code-block::

         PyTorch 2.9.1+cu128, torch-neuronx 2.9.0.2.13.23887+8e870898
         $ neuron-ls
         instance-type: trn1.2xlarge
         instance-id: i-0bea223b1afb7e159
         +--------+--------+----------+--------+--------------+----------+------+
         | NEURON | NEURON |  NEURON  | NEURON |     PCI      |   CPU    | NUMA |
         | DEVICE | CORES  | CORE IDS | MEMORY |     BDF      | AFFINITY | NODE |
         +--------+--------+----------+--------+--------------+----------+------+
         | 0      | 2      | 0-1      | 32 GB  | 0000:00:1e.0 | 0-7      | -1   |
         +--------+--------+----------+--------+--------------+----------+------+

   .. tab-item:: JAX

      .. code-block:: bash

         python3 -c "import jax; print(f'JAX {jax.__version__}'); print(f'Devices: {jax.devices()}')"
         neuron-ls

      You should see output similar to this (the framework, versions, instance IDs, and details should match your expected ones, not the ones in this example):

      .. code-block::

         JAX 0.6.2.1.0.1, torch-neuronx 2.9.0.2.13.23887+8e870898
         $ neuron-ls
         instance-type: trn1.2xlarge
         instance-id: i-0bea223b1afb7e159
         +--------+--------+----------+--------+--------------+----------+------+
         | NEURON | NEURON |  NEURON  | NEURON |     PCI      |   CPU    | NUMA |
         | DEVICE | CORES  | CORE IDS | MEMORY |     BDF      | AFFINITY | NODE |
         +--------+--------+----------+--------+--------------+----------+------+
         | 0      | 2      | 0-1      | 32 GB  | 0000:00:1e.0 | 0-7      | -1   |
         +--------+--------+----------+--------+--------------+----------+------+






Next steps
----------

After setup, explore the framework documentation:

- :doc:`/frameworks/torch/index` - PyTorch on Neuron
- :doc:`/frameworks/jax/index` - JAX on Neuron
- :doc:`/libraries/nxd-inference/vllm/index` - vLLM on Neuron
- :doc:`/deploy/environments/dlami` - Full DLAMI documentation and SSM parameters
