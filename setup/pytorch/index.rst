.. meta::
   :description: Install PyTorch for AWS Neuron on Inf2, Trn1, Trn2, Trn3 instances
   :keywords: pytorch, neuron, installation, trn1, trn2, trn3, inf2
   :framework: pytorch
   :instance-types: inf2, trn1, trn2, trn3
   :content-type: framework-setup-hub
   :date-modified: 2026-03-03

.. _pytorch-setup:

Install PyTorch for Neuron
===========================

Install PyTorch with AWS Neuron support for training and inference on Inferentia and Trainium instances.

**Supported Instances**: Inf2, Trn1, Trn2, Trn3

**PyTorch Version**: 2.9+ with Native Neuron backend

Choose installation method
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Best for
     - Considerations
   * - :doc:`DLAMI <dlami>`
     - Getting started quickly, prototyping, single-user development
     - Pre-configured with tested dependency versions; launch a new AMI to update
   * - :doc:`DLC <dlc>`
     - Production deployments, CI/CD pipelines, multi-tenant environments
     - Requires Docker and Neuron driver on host; portable across EC2, ECS, EKS
   * - :doc:`Manual <manual>`
     - Custom OS images, shared clusters, integrating into existing environments
     - Full control over versions and dependencies; requires manual dependency management

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: 🚀 AWS Deep Learning AMI
      :link: dlami
      :link-type: doc
      :class-card: sd-border-2
      
      **Recommended for most users**
      
      Pre-configured environment with all dependencies
      
      ✅ All dependencies included
      
      ✅ Tested configurations
      
      ✅ Multiple Python versions
      
      ⏱️ **Setup time**: ~5 minutes

   .. grid-item-card:: 🐳 Deep Learning Container
      :link: dlc
      :link-type: doc
      :class-card: sd-border-2
      
      **For containerized deployments**
      
      Pre-configured Docker images from AWS ECR
      
      ✅ Docker-based isolation
      
      ✅ Training and inference images
      
      ✅ vLLM-ready images available
      
      ⏱️ **Setup time**: ~10 minutes

   .. grid-item-card:: 🔧 Manual Installation
      :link: manual
      :link-type: doc
      :class-card: sd-border-2
      
      **For custom environments**
      
      Install on existing systems or custom setups
      
      ✅ Existing system integration
      
      ✅ Custom Python versions
      
      ✅ Full control over dependencies
      
      ⏱️ **Setup time**: ~15 minutes

Prerequisites
-------------

Before installing, ensure you have:

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Requirement
     - Details
   * - Instance Type
     - Inf2, Trn1, Trn2, or Trn3 instance
   * - Operating System
     - Ubuntu 24.04, Ubuntu 22.04, or Amazon Linux 2023
   * - Python Version
     - Python 3.11 or 3.12
   * - AWS Account
     - With EC2 launch permissions
   * - SSH Access
     - Key pair for instance connection

What You'll Get
---------------

After installation, you'll have:

- **PyTorch 2.9+** with Native Neuron backend
- **torch-neuronx** package for Neuron-specific operations
- **neuronx-cc** compiler for model optimization
- **Neuron Runtime** for model execution
- **Neuron Tools** for profiling and debugging

Version Information
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Component
     - Version
   * - PyTorch
     - 2.9.0+
   * - torch-neuronx
     - 2.9.0+
   * - neuronx-cc
     - 2.15.0+
   * - Python
     - 3.11, 3.12

Next Steps
----------

After installation:

1. **Verify Installation**: Run verification commands in the installation guide
2. **Try a Tutorial**: 
   
   * **Inference**: :doc:`/libraries/nxd-inference/vllm/quickstart-vllm-online-serving`
   * **Training**: :doc:`/frameworks/torch/torch-neuronx/tutorials/training/mlp`
  
3. **Read the torch-neuronx Programming Guide**: :doc:`/frameworks/torch/torch-neuronx/programming-guide/training/index`
4. **Explore Examples**: :doc:`/frameworks/torch/index`

Update an existing installation
--------------------------------

Already have PyTorch Neuron installed and need to update to a newer PyTorch version or Neuron SDK release? Select the guide that matches your installation method.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: 🔄 Update DLAMI
      :link: update-dlami
      :link-type: doc
      :class-card: sd-border-2
      
      Update PyTorch and drivers on an existing Deep Learning AMI

   .. grid-item-card:: 🔄 Update DLC
      :link: update-dlc
      :link-type: doc
      :class-card: sd-border-2
      
      Pull the latest container image and update the host driver

   .. grid-item-card:: 🔄 Update manual install
      :link: update-manual
      :link-type: doc
      :class-card: sd-border-2
      
      Update PyTorch packages and drivers on a manual installation

.. toctree::
   :hidden:
   :maxdepth: 1
   
   New DLAMI <dlami>
   Update Existing DLAMI <update-dlami>
   New DLC <dlc>
   Update Existing DLC <update-dlc>
   New Manual Configuration <manual>
   Update Manual Configuration <update-manual>
