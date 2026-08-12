.. meta::
   :description: Legacy PyTorch installation guide for AWS Inferentia 1 (Inf1) instances
   :keywords: pytorch, neuron, inf1, legacy, installation, torch-neuron
   :framework: pytorch
   :instance-types: inf1
   :status: legacy
   :content-type: legacy-guide
   :date-modified: 2026-03-30

PyTorch on Inf1 (legacy)
=========================

.. warning::
   
   **Legacy hardware**: Inf1 instances use NeuronCore v1 with PyTorch 1.x (``torch-neuron``).
   
   For new projects, use **Inf2, Trn1, Trn2, or Trn3** with PyTorch 2.9+ (``torch-neuronx``).
   See :doc:`/setup/pytorch/index` for current PyTorch setup.

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

.. tab-set::

   .. tab-item:: Ubuntu 20.04

      **Launch Instance**

      * `Launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ and select an Inf1 instance type.
      * Select Ubuntu Server 20 AMI.
      * `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_.

      **Install Drivers and Tools**

      .. note::

         Inf1 instances are no longer supported. The installation and upgrade
         commands that were previously generated on this page have been removed.
         This document is retained for archival reference only.

      .. include:: /includes/setup/tab-inference-torch-neuron-u20.txt

   .. tab-item:: Ubuntu 22.04

      **Launch Instance**

      * `Launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ and select an Inf1 instance type.
      * Select Ubuntu Server 22 AMI.
      * `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_.

      **Install Drivers and Tools**

      .. note::

         Inf1 instances are no longer supported. The installation and upgrade
         commands that were previously generated on this page have been removed.
         This document is retained for archival reference only.

      .. include:: /includes/setup/tab-inference-torch-neuron-u22.txt

   .. tab-item:: Amazon Linux 2023

      **Launch Instance**

      * `Launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ and select an Inf1 instance type.
      * Select Amazon Linux 2023 AMI.
      * `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_.

      **Install Drivers and Tools**

      .. note::

         Inf1 instances are no longer supported. The installation and upgrade
         commands that were previously generated on this page have been removed.
         This document is retained for archival reference only.

      .. include:: /includes/setup/tab-inference-torch-neuron-al2023.txt

Update an Existing Installation
-------------------------------

.. tab-set::

   .. tab-item:: Ubuntu 20.04

      .. include:: /archive/torch-neuron/setup/pytorch-update-u20.rst

   .. tab-item:: Ubuntu 22.04

      .. include:: /archive/torch-neuron/setup/pytorch-update-u22.rst

   .. tab-item:: Amazon Linux 2023

      .. include:: /archive/torch-neuron/setup/pytorch-update-al2023.rst

Previous Versions
-----------------

.. tab-set::

   .. tab-item:: Ubuntu 20.04

      .. include:: /archive/torch-neuron/setup/pytorch-install-prev-u20.rst

   .. tab-item:: Ubuntu 22.04

      .. include:: /archive/torch-neuron/setup/pytorch-install-prev-u22.rst

   .. tab-item:: Amazon Linux 2023

      .. include:: /archive/torch-neuron/setup/pytorch-install-prev-al2023.rst

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
- :doc:`/frameworks/torch/inference-torch-neuronx` - Inference guides
- :ref:`setup-guide-index` - Current setup options (Inf2, Trn1, Trn2, Trn3)
