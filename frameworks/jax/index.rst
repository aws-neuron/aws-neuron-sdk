.. meta::
   :description: JAX support on AWS Neuron SDK - JAX NeuronX for training and inference on Trn1, Trn2, and Inf2 instances with native JAX device integration.
   :keywords: JAX, jax-neuronx, libneuronxla, AWS Neuron, Trainium, Inferentia, PJRT, machine learning
   :date-modified: 01/22/2026

.. _jax-neuron-main:

JAX Support on Neuron
=====================

JAX running on Neuron unlocks high-performance and cost-effective deep learning acceleration on AWS Trainium-based and AWS Inferentia-based Amazon EC2 instances.

The JAX NeuronX plugin is a set of modularized JAX plugin packages that integrate AWS Trainium and Inferentia machine learning accelerators into JAX as pluggable devices using the PJRT (Plugin Runtime) mechanism. This enables native JAX device support for Neuron accelerators with minimal code changes.

JAX NeuronX includes the following key components:

* **libneuronxla**: Neuron's integration into JAX's runtime PJRT, built using the PJRT C-API plugin mechanism. Installing this package enables using Trainium and Inferentia natively as JAX devices.
* **jax-neuronx**: A package containing Neuron-specific JAX features, such as the Neuron NKI JAX interface. It also serves as a meta-package for providing a tested combination of ``jax-neuronx``, ``jax``, ``jaxlib``, ``libneuronxla``, and ``neuronx-cc`` packages.

Key capabilities of JAX NeuronX include:

* **Native JAX device integration**: Seamless integration with JAX through the PJRT C-API plugin mechanism
* **Flexible installation**: Choose between a production-ready meta-package or custom package combinations
* **NKI support**: Access to Neuron Kernel Interface (NKI) through the JAX interface for custom kernel development
* **Broad compatibility**: Support for multiple JAX and jaxlib versions through the PJRT C-API mechanism
* **Training and inference**: Full support for both training and inference workloads on Trainium and Inferentia instances

.. admonition:: Beta Release
   :class: note

   JAX NeuronX is currently in beta. Some JAX functionality may not be fully supported. We welcome your feedback and contributions.

.. grid:: 1 
   :gutter: 3

   .. grid-item-card:: JAX NeuronX Component Release Notes
      :link: /release-notes/components/jax
      :link-type: doc

      Review the JAX NeuronX release notes for all versions of the Neuron SDK.

.. grid:: 1 
   :gutter: 3

   .. grid-item-card:: Setup Guide
      :link: jax-neuron-setup
      :link-type: ref

      Install and configure JAX NeuronX for Trn1, Trn2, and Inf2 instances

   .. grid-item-card:: API Reference Guide
      :link: jax-neuronx-api-reference-guide
      :link-type: ref

      Comprehensive API reference for JAX NeuronX features and environment variables

   .. grid-item-card:: Known Issues
      :link: /frameworks/jax/setup/jax-neuronx-known-issues
      :link-type: doc

      Review known issues and limitations in the current JAX NeuronX release

   .. grid-item-card:: Neuron Kernel Interface (NKI)
      :link: /nki/index
      :link-type: doc

      Learn about NKI for custom kernel development with JAX

.. toctree::
   :maxdepth: 1
   :hidden:

   /frameworks/jax/setup/jax-setup
   /frameworks/jax/setup/jax-neuronx-known-issues
   /frameworks/jax/api-reference-guide/index
   Release Notes </release-notes/components/jax>
