.. meta::
   :description: Install AWS Neuron SDK for PyTorch and JAX on Inferentia and Trainium instances
   :keywords: neuron, installation, setup, pytorch, jax, inferentia, trainium, inf2, trn1, trn2, trn3
   :instance-types: inf2, trn1, trn2, trn3, inf1
   :content-type: navigation-hub
   :date-modified: 2026-10-13

.. _setup-guide-index:

Install AWS Neuron SDK
======================

Install the AWS Neuron SDK to enable deep learning acceleration on Inferentia and Trainium instances.

.. note::
   
   **New to Neuron?** Start with the :doc:`quickstart guide </quickstarts>` 
   for a complete end-to-end tutorial.

Quick Start Decision Tree
--------------------------

Answer these questions to find your installation path:

**1. Which framework?**

- :ref:`PyTorch <pytorch-setup>` (recommended for most users)
- :ref:`JAX <jax-setup>`

**2. Installation method?**

- **AWS Deep Learning AMI** — fastest setup, pre-configured with all dependencies. Best for getting started and single-user development.
- **Deep Learning Container** — Docker-based, portable across EC2, ECS, EKS. Best for production deployments and CI/CD pipelines.
- **Manual installation** — full control over packages and versions. Best for custom OS images and shared clusters.

Instance Comparison
-------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 35 15
   
   * - Instance
     - NeuronCore
     - Use Case
     - Status
   * - Trn3
     - :doc:`v4 </about-neuron/arch/neuron-hardware/neuron-core-v4>`
     - Training and inference (latest generation)
     - Current
   * - Trn2
     - :doc:`v3 </about-neuron/arch/neuron-hardware/neuron-core-v3>`
     - Training and inference
     - Current
   * - Trn1
     - :doc:`v2 </about-neuron/arch/neuron-hardware/neuron-core-v2>`
     - Training and inference
     - Current
   * - Inf2
     - :doc:`v2 </about-neuron/arch/neuron-hardware/neuron-core-v2>`
     - Inference
     - Current
   * - Inf1
     - :doc:`v1 </about-neuron/arch/neuron-hardware/neuron-core-v1>`
     - Legacy inference
     - Legacy

Installation by Framework
--------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: PyTorch
      :link: pytorch/index
      :link-type: doc
      :class-card: sd-border-2
      
      **Recommended for most users**
      
      - PyTorch 2.13 or 2.14 with TorchNeuron, the native Neuron backend
      - Eager mode and torch.compile
      - Supports: Trn2, Trn3
      
      :bdg-success:`Most Popular`

   .. grid-item-card:: JAX
      :link: jax/index
      :link-type: doc
      :class-card: sd-border-2
      
      **For JAX users**
      
      - JAX 0.7.0 - 0.10.0 with Neuron backend
      - XLA compilation
      - Supports: Inf2, Trn1, Trn2, Trn3

Multi-framework DLAMI
----------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 🚀 Neuron multi-framework DLAMI
      :link: multiframework-dlami
      :link-type: doc
      :class-card: sd-border-2

      Pre-configured AMI with PyTorch, JAX, and vLLM virtual environments ready to use. The fastest way to get started with any framework.

Common Issues
-------------

.. dropdown:: ⚠️ Module not found errors
   :color: info
   :animate: fade-in
   
   If you see "No module named 'torch_neuronx'" or similar:
   
   1. Verify virtual environment is activated
   2. Check Python version: ``python --version`` (3.12 or later for TorchNeuron)
   3. Reinstall: ``pip install --force-reinstall torch-neuronx``
   
   See :doc:`troubleshooting` for more details.

.. dropdown:: ⚠️ Instance type not recognized
   :color: info
   :animate: fade-in
   
   Ensure you're using a Neuron-supported instance:
   
   - Check with: ``aws ec2 describe-instance-types --instance-types <type>``
   - Verify Neuron devices: ``neuron-ls``
   
   See :doc:`troubleshooting` for more details.

.. dropdown:: ⚠️ Version compatibility issues
   :color: info
   :animate: fade-in
   
   Check version compatibility:
   
   - TorchNeuron requires PyTorch 2.13 or 2.14 and Python 3.12 to 3.14; see the :doc:`TorchNeuron release notes </release-notes/components/torchneuron>` for the support matrix
   - See :doc:`/release-notes/index` for other components
   
   See :doc:`troubleshooting` for more details.

.. _legacy-inf1-support:

Legacy Inf1 Support
-------------------

.. warning::
   
   **Inf1 uses the legacy NeuronCore v1 architecture.** For new projects, use Trn2 or Trn3.
   
   - Current-generation NeuronCores (v3 on Trn2, v4 on Trn3)
   - Broader framework support (TorchNeuron, JAX, vLLM Neuron)
   - Active development and feature updates

.. grid:: 1

   .. grid-item-card:: Inf1 Installation (Legacy, Archived)
      :link: /archive/setup/legacy-inf1/index
      :link-type: doc
      :class-card: sd-border-2
      
      Install Neuron SDK for Inferentia 1 instances (archived)
      
      :bdg-danger:`Archived`

Additional Resources
--------------------

- :doc:`/deploy/ec2/index` - Launch Inf/Trn instances on Amazon EC2
- :doc:`/deploy/index` - Use Deep Learning Containers
- :doc:`troubleshooting` - Installation troubleshooting guide
- :doc:`/release-notes/index` - Version compatibility information

.. toctree::
   :hidden:
   :maxdepth: 1
   
   PyTorch <pytorch/index>
   JAX <jax/index>
   Multi-framework <multiframework-dlami>
   Troubleshooting <troubleshooting>
