.. _jax-setup:

.. meta::
   :description: Install JAX for AWS Neuron on Inf2, Trn1, Trn2, Trn3 instances
   :keywords: jax, neuron, installation, trn1, trn2, trn3, inf2
   :framework: jax
   :instance-types: inf2, trn1, trn2, trn3
   :content-type: framework-setup-hub
   :date-modified: 2026-03-03

Install JAX for Neuron
=======================

Install JAX with AWS Neuron support for training and inference on Inferentia and Trainium instances.

**Supported Instances**: Inf2, Trn1, Trn2, Trn3

**JAX Version**: 0.7+ with Neuron PJRT plugin

.. admonition:: Beta Release
   :class: note

   JAX NeuronX is currently in beta. Some JAX functionality may not be fully supported. We welcome your feedback and contributions.

Choose Installation Method
---------------------------

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
      
      ✅ Training images available
      
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

- **JAX 0.7+** with Neuron PJRT plugin
- **jax-neuronx** package for Neuron-specific features
- **libneuronxla** PJRT plugin for native JAX device integration
- **neuronx-cc** compiler for model optimization
- **Neuron Runtime** for model execution

Version Information
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Component
     - Version
   * - JAX
     - 0.7.0+
   * - jax-neuronx
     - 0.7.0+
   * - libneuronxla
     - latest
   * - neuronx-cc
     - 2.15.0+
   * - Python
     - 3.11, 3.12

Next Steps
----------

After installation:

1. **Verify Installation**: Run verification commands in the installation guide
2. **Read the Guide**: :doc:`/frameworks/jax/setup/jax-setup`
3. **Explore JAX on Neuron**: :doc:`/frameworks/jax/index`
4. **API Reference**: :doc:`/frameworks/jax/api-reference-guide/index`

.. toctree::
   :hidden:
   :maxdepth: 1
   
   dlami
   dlc
   manual
