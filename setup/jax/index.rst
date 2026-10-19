.. _jax-setup:
.. _jax-neuron-setup:

.. meta::
   :description: Install JAX for AWS Neuron on Inf2, Trn1, Trn2, Trn3 instances
   :keywords: jax, neuron, installation, trn1, trn2, trn3, inf2
   :framework: jax
   :instance-types: inf2, trn1, trn2, trn3
   :content-type: framework-setup-hub
   :date-modified: 2026-09-30

Install JAX for Neuron
=======================

Install JAX with AWS Neuron support for training and inference on Inferentia and Trainium instances.

**Supported Instances**: Inf2, Trn1, Trn2, Trn3

**JAX Version**: 0.10+ with Neuron PJRT plugin

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
     - Ubuntu 26.04, Ubuntu 24.04, or Amazon Linux 2023. For the tested Linux kernel and
       GLIBC versions, see :ref:`neuron-runtime-compatibility`.
   * - Python Version
     - Python 3.11, 3.12, 3.13, or 3.14
   * - AWS Account
     - With EC2 launch permissions
   * - SSH Access
     - Key pair for instance connection

What You'll Get
---------------

After installation, you'll have:

- **JAX 0.10+** with Neuron PJRT plugin
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
     - 0.10.0+
   * - jax-neuronx
     - 0.10.0+
   * - libneuronxla
     - latest
   * - neuronx-cc
     - 2.28.0+
   * - Python
     - 3.11, 3.12, 3.13, 3.14

.. _jax-version-compatibility:

JAX and jaxlib version compatibility
------------------------------------

The ``jax-neuronx[stable]`` meta-package is the recommended way to install JAX for Neuron. It pins a tested combination of ``jax``, ``jaxlib``, ``libneuronxla``, and ``neuronx-cc``, so you get a known-good environment without choosing versions yourself. The DLAMI, DLC, and manual installation methods above all use it.

If you need to bring your own ``jax`` and ``jaxlib`` versions, install the packages separately instead. Because ``libneuronxla`` supports a broad range of ``jaxlib`` versions through the PJRT C-API mechanism, this gives you flexibility when choosing ``jax`` and ``jaxlib`` versions and lets you add the Neuron plugin to an existing JAX environment:

.. code-block:: bash

   python3 -m pip install jax==0.10.0 jaxlib==0.10.0
   python3 -m pip install jax-neuronx libneuronxla neuronx-cc==2.* --extra-index-url=https://pip.repos.neuron.amazonaws.com

Replace ``0.10.0`` with the ``jax`` and ``jaxlib`` version tested for your Neuron release. See the :doc:`JAX NeuronX release notes </release-notes/components/jax>` for the version shipped in each release.

To check which client versions your installed ``libneuronxla`` supports, use the ``libneuronxla.supported_clients`` API:

.. code-block:: python

   import libneuronxla
   libneuronxla.supported_clients()

It returns a list of strings in the form ``"<package> <version> (PJRT C-API <c-api version>)"`` — one entry per known-supported ``jaxlib`` build. Run it in your own environment to see the exact versions for your installed ``libneuronxla``.

The list covers known versions only and may be incomplete. Other versions can work, including future ``jaxlib`` releases, as long as the PJRT C-API stays compatible with your installed ``libneuronxla``. For this reason, Neuron does not pin a strict ``libneuronxla`` –> ``jaxlib`` dependency, which gives you more freedom when coordinating ``jax`` and ``libneuronxla`` installations. Compatibility between ``jaxlib`` and ``libneuronxla`` is determined by the `PJRT C-API version <https://github.com/openxla/xla/blob/60871f5a775904482bfde635966279dab2e07012/xla/pjrt/c/pjrt_c_api.h>`__. For more information, see the `PJRT integration guide <https://github.com/openxla/xla/blob/60871f5a775904482bfde635966279dab2e07012/docs/pjrt/pjrt_integration.md>`__.

Next Steps
----------

After installation:

1. **Verify Installation**: Run verification commands in the installation guide
2. **Explore JAX on Neuron**: :doc:`/jax/index`
3. **API Reference**: :doc:`/jax/api-reference-guide/index`

.. toctree::
   :hidden:
   :maxdepth: 1
   
   dlami
   dlc
   manual
